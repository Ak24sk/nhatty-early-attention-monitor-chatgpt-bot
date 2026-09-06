import numpy as np
import pandas as pd

def safety_gate(row, t):
    req=['liquidity_usd','lp_locked_days','mint_revoked','freeze_revoked','sellable','top10_pct','dev_pct','cluster_pct','wash_trade_risk','bundle_risk','dev_risk','contract_risk']
    unknown=[c for c in req if c not in row.index or pd.isna(row[c])]
    if unknown: return 'UNVERIFIED', 'Missing: '+', '.join(unknown)
    checks=[(float(row.liquidity_usd)>=t['min_liquidity'],'liquidity'),(float(row.lp_locked_days)>=t['min_lp_locked_days'],'LP lock'),(bool(row.mint_revoked),'mint authority'),(bool(row.freeze_revoked),'freeze authority'),(bool(row.sellable),'sellability'),(float(row.top10_pct)<=t['max_top10_pct'],'top10'),(float(row.dev_pct)<=t['max_dev_pct'],'dev'),(float(row.cluster_pct)<=t['max_cluster_pct'],'cluster'),(str(row.wash_trade_risk).upper()=='LOW','wash trade'),(str(row.bundle_risk).upper()=='LOW','bundle'),(str(row.dev_risk).upper()=='LOW','dev risk'),(str(row.contract_risk).upper()=='CLEAN','contract')]
    failed=[n for ok,n in checks if not ok]
    return ('PASSED','All configured checks passed') if not failed else ('FAILED','Failed: '+', '.join(failed))

def score_traders(df):
    d=df.copy()
    cols=['early_entry','runner_rate','consistency','independence']
    for c in cols: d[c]=pd.to_numeric(d[c],errors='coerce').fillna(0).clip(0,100)
    d['elite_score']=(d.early_entry*.35+d.runner_rate*.30+d.consistency*.20+d.independence*.15).round(2)
    d['tier']=np.select([d.elite_score>=85,d.elite_score>=75,d.elite_score>=65],['ELITE','STRONG','WATCH'],default='LOW')
    return d.sort_values('elite_score',ascending=False)

def attention_engine(df):
    d=df.copy(); d['time']=pd.to_datetime(d['time'],errors='coerce'); d=d.sort_values(['topic','time'])
    for c in ['mentions','unique_accounts','engagement','influencer_event']: d[c]=pd.to_numeric(d[c],errors='coerce').fillna(0)
    d['mention_velocity']=d.groupby('topic').mentions.diff().fillna(0)
    d['unique_velocity']=d.groupby('topic').unique_accounts.diff().fillna(0)
    d['engagement_velocity']=d.groupby('topic').engagement.diff().fillna(0)
    d['attention_score']=(d.mention_velocity.clip(lower=0)+1.5*d.unique_velocity.clip(lower=0)+d.engagement_velocity.clip(lower=0)/10+50*d.influencer_event).round(2)
    d['attention_stage']=np.select([d.attention_score>=1000,d.attention_score>=300,d.attention_score>=50],['SURGING','BUILDING','EARLY'],default='QUIET')
    return d

def trader_convergence(events, window_minutes=15):
    cols=['token','wallet_count','wallets','first_buy','last_buy','window_minutes']
    if events is None or len(events)==0: return pd.DataFrame(columns=cols)
    d=events[events.classification.astype(str).str.upper()=='LIKELY BUY'].copy()
    if d.empty: return pd.DataFrame(columns=cols)
    d['time']=pd.to_datetime(d['time'],errors='coerce'); rows=[]
    for token,g in d.groupby('token'):
        g=g.sort_values('time')
        for _,r in g.iterrows():
            w=g[(g.time>=r.time)&(g.time<=r.time+pd.Timedelta(minutes=window_minutes))]
            wallets=sorted(set(w.wallet.astype(str)))
            if len(wallets)>=2:
                rows.append({'token':token,'wallet_count':len(wallets),'wallets':', '.join(wallets),'first_buy':w.time.min(),'last_buy':w.time.max(),'window_minutes':window_minutes}); break
    return pd.DataFrame(rows)

def build_signal(tokens,traders,x_scores,convergence,t):
    rows=[]; strong=bool((traders.tier.isin(['ELITE','STRONG'])).any())
    for _,r in tokens.iterrows():
        safety,reason=safety_gate(r,t); token=str(r.token)
        xs=x_scores[x_scores.topic.astype(str)==token] if not x_scores.empty else pd.DataFrame()
        xstage=xs.attention_stage.iloc[-1] if not xs.empty else 'QUIET'; xscore=float(xs.attention_score.iloc[-1]) if not xs.empty else 0
        cv=convergence[convergence.token.astype(str)==token] if not convergence.empty else pd.DataFrame(); wc=int(cv.wallet_count.iloc[0]) if not cv.empty else 0
        signal='BLOCKED' if safety!='PASSED' else ('ALERT' if wc>=3 and xstage in ['BUILDING','SURGING'] and strong else ('WATCH' if wc>=2 or xstage in ['BUILDING','SURGING'] else 'BUILDING'))
        rows.append({'token':token,'safety':safety,'safety_reason':reason,'x_stage':xstage,'x_score':round(xscore,2),'converging_wallets':wc,'signal':signal})
    return pd.DataFrame(rows)
