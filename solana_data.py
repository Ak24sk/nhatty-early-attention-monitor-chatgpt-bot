import time
from collections import defaultdict
import requests

class SolanaRPC:
    def __init__(self,url,timeout=20): self.url=url; self.timeout=timeout
    def call(self,method,params):
        r=requests.post(self.url,json={'jsonrpc':'2.0','id':1,'method':method,'params':params},timeout=self.timeout); r.raise_for_status(); d=r.json()
        if d.get('error'): raise RuntimeError(str(d['error']))
        return d.get('result')
    def signatures(self,address,limit=25): return self.call('getSignaturesForAddress',[address,{'limit':int(limit),'commitment':'confirmed'}]) or []
    def transaction(self,signature): return self.call('getTransaction',[signature,{'encoding':'jsonParsed','commitment':'confirmed','maxSupportedTransactionVersion':0}])
    def account_info(self,address): return self.call('getAccountInfo',[address,{'encoding':'jsonParsed','commitment':'confirmed'}])
    def token_supply(self,mint): return self.call('getTokenSupply',[mint,{'commitment':'confirmed'}])
    def largest_accounts(self,mint): return self.call('getTokenLargestAccounts',[mint,{'commitment':'confirmed'}])

def _balances(tx,field):
    out=defaultdict(float); meta=(tx or {}).get('meta') or {}
    for b in meta.get(field) or []:
        mint,owner=b.get('mint'),b.get('owner')
        if mint and owner: out[(owner,mint)]+=float((b.get('uiTokenAmount') or {}).get('uiAmount') or 0)
    return out

def _sol_change(tx,wallet):
    meta=(tx or {}).get('meta') or {}; keys=((tx or {}).get('transaction') or {}).get('message',{}).get('accountKeys',[])
    idx=next((i for i,k in enumerate(keys) if (k.get('pubkey') if isinstance(k,dict) else k)==wallet),None)
    if idx is None: return 0.0
    pre,post=meta.get('preBalances') or [],meta.get('postBalances') or []
    return (float(post[idx])-float(pre[idx]))/1e9 if idx<len(pre) and idx<len(post) else 0.0

def analyze_wallet(rpc,wallet,limit=25,mint_filter=None):
    sigs=rpc.signatures(wallet,limit); events=[]; errors=[]; observed=set(); summary={'wallet':wallet,'transactions_scanned':len(sigs),'successful_transactions':0,'token_events':0,'likely_buys':0,'likely_sells':0,'unique_tokens':0,'first_observed':None,'activity_score':0,'confidence':'LOW'}
    for s in sigs:
        if s.get('err') is not None: continue
        try:
            tx=rpc.transaction(s['signature'])
            if not tx: continue
            summary['successful_transactions']+=1; bt=tx.get('blockTime'); stamp=time.strftime('%Y-%m-%d %H:%M:%S',time.gmtime(bt)) if bt else ''
            pre,post=_balances(tx,'preTokenBalances'),_balances(tx,'postTokenBalances'); sol=_sol_change(tx,wallet)
            for owner,mint in set(pre)|set(post):
                if owner!=wallet or (mint_filter and mint!=mint_filter): continue
                delta=post.get((owner,mint),0)-pre.get((owner,mint),0)
                if abs(delta)<1e-12: continue
                cls='LIKELY BUY' if delta>0 and sol<0 else ('LIKELY SELL' if delta<0 and sol>0 else 'TOKEN BALANCE CHANGE')
                summary['likely_buys']+=cls=='LIKELY BUY'; summary['likely_sells']+=cls=='LIKELY SELL'; observed.add(mint)
                events.append({'wallet':wallet,'token':mint,'time':stamp,'signature':s['signature'],'classification':cls,'token_delta':delta,'sol_change':sol})
        except Exception as e: errors.append({'wallet':wallet,'signature':s.get('signature',''),'error':str(e)})
    summary['token_events']=len(events); summary['unique_tokens']=len(observed); times=[e['time'] for e in events if e['time']]; summary['first_observed']=min(times) if times else None
    summary['activity_score']=min(100,round(summary['successful_transactions']*2+len(events)*3,2))
    if summary['successful_transactions']>=20 and len(events)>=5: summary['confidence']='MEDIUM'
    if summary['successful_transactions']>=50 and len(events)>=10: summary['confidence']='HIGH'
    return {'summary':summary,'events':events,'errors':errors}

def enrich_token(rpc,mint):
    supply=rpc.token_supply(mint); largest=rpc.largest_accounts(mint); account=rpc.account_info(mint); data=((account or {}).get('value') or {}).get('data') or {}; parsed=data.get('parsed',{}) if isinstance(data,dict) else {}; info=parsed.get('info',{}) if isinstance(parsed,dict) else {}
    return {'mint':mint,'supply':(supply or {}).get('value'),'largest_accounts':(largest or {}).get('value'),'mint_authority':info.get('mintAuthority'),'freeze_authority':info.get('freezeAuthority'),'note':'RPC token metadata does not by itself establish LP lock, sellability, DEX route, wash trading, bundle risk, or exact PnL.'}
