import streamlit as st
import pandas as pd
from detector import score_traders,attention_engine,trader_convergence,build_signal
from solana_data import SolanaRPC,analyze_wallet,enrich_token
st.set_page_config(page_title='Early Runner Intelligence v2.2',layout='wide'); st.title('🚀 Early Runner Intelligence v2.2'); st.caption('Safety-gated early-runner research dashboard — no trade execution, no private keys.')
mode=st.sidebar.selectbox('Data source',['Demo','CSV','Live Solana Token','Live Wallet Scanner'])
t={'min_liquidity':st.sidebar.number_input('Minimum liquidity ($)',0,10000000,50000,5000),'min_lp_locked_days':st.sidebar.number_input('Minimum LP lock (days)',0,3650,30),'max_top10_pct':st.sidebar.slider('Max top-10 concentration (%)',1,100,40),'max_dev_pct':st.sidebar.slider('Max dev concentration (%)',1,100,10),'max_cluster_pct':st.sidebar.slider('Max related-wallet cluster (%)',1,100,50)}
def demo():
    return (pd.DataFrame([{'token':'DEMO1','liquidity_usd':125000,'lp_locked_days':90,'mint_revoked':True,'freeze_revoked':True,'sellable':True,'top10_pct':28,'dev_pct':4,'cluster_pct':18,'wash_trade_risk':'LOW','bundle_risk':'LOW','dev_risk':'LOW','contract_risk':'CLEAN'},{'token':'DEMO2','liquidity_usd':35000,'lp_locked_days':10,'mint_revoked':True,'freeze_revoked':True,'sellable':True,'top10_pct':45,'dev_pct':14,'cluster_pct':60,'wash_trade_risk':'MEDIUM','bundle_risk':'MEDIUM','dev_risk':'MEDIUM','contract_risk':'CLEAN'}]),pd.DataFrame([{'wallet':'RunnerA','early_entry':94,'runner_rate':88,'consistency':91,'independence':90},{'wallet':'RunnerB','early_entry':89,'runner_rate':84,'consistency':86,'independence':87},{'wallet':'RunnerC','early_entry':81,'runner_rate':79,'consistency':82,'independence':84},{'wallet':'TraderD','early_entry':62,'runner_rate':55,'consistency':68,'independence':70}]),pd.DataFrame([{'time':'2026-09-05 10:00','topic':'DEMO1','mentions':120,'unique_accounts':85,'engagement':950,'influencer_event':0},{'time':'2026-09-05 10:05','topic':'DEMO1','mentions':210,'unique_accounts':140,'engagement':1800,'influencer_event':1},{'time':'2026-09-05 10:10','topic':'DEMO1','mentions':390,'unique_accounts':245,'engagement':4200,'influencer_event':1}]),pd.DataFrame([{'wallet':'RunnerA','token':'DEMO1','time':'2026-09-05 10:06','classification':'LIKELY BUY','token_delta':100000,'sol_change':-1.2},{'wallet':'RunnerB','token':'DEMO1','time':'2026-09-05 10:08','classification':'LIKELY BUY','token_delta':75000,'sol_change':-0.9},{'wallet':'RunnerC','token':'DEMO1','time':'2026-09-05 10:12','classification':'LIKELY BUY','token_delta':50000,'sol_change':-0.6}]))
if mode in ['Demo','CSV']:
    if mode=='Demo': tokens,traders,x,w=demo()
    else: tokens,traders,x,w=[pd.read_csv('data/'+f) for f in ['demo_tokens.csv','demo_trades.csv','demo_x.csv','demo_wallets.csv']]
    ts=score_traders(traders); xs=attention_engine(x); cv=trader_convergence(w); board=build_signal(tokens,ts,xs,cv,t)
    st.subheader('Unified Signal Board'); st.dataframe(board,use_container_width=True,hide_index=True)
    a,b,c,d=st.columns(4); a.metric('Tokens',len(tokens)); b.metric('Elite traders',int((ts.tier=='ELITE').sum())); c.metric('Convergence events',len(cv)); d.metric('Alerts',int((board.signal=='ALERT').sum()))
    with st.expander('Elite trader intelligence'): st.dataframe(ts,use_container_width=True,hide_index=True)
    with st.expander('X attention engine'): st.dataframe(xs,use_container_width=True,hide_index=True)
    with st.expander('Wallet convergence'): st.dataframe(cv,use_container_width=True,hide_index=True)
else:
    rpc_url=st.text_input('Solana RPC URL','https://api.mainnet-beta.solana.com')
    if mode=='Live Solana Token':
        mint=st.text_input('Token mint address')
        if st.button('Inspect token') and mint.strip():
            try: st.json(enrich_token(SolanaRPC(rpc_url),mint.strip()))
            except Exception as e: st.error(str(e))
    else:
        wallets_text=st.text_area('Public wallet addresses — one per line'); mint_filter=st.text_input('Optional token mint filter'); limit=st.slider('Transactions per wallet',5,100,25)
        if st.button('Scan wallets'):
            wallets=[x.strip() for x in wallets_text.splitlines() if x.strip()]; summaries=[]; events=[]; errors=[]
            for wallet in wallets:
                try:
                    r=analyze_wallet(SolanaRPC(rpc_url),wallet,limit,mint_filter.strip() or None); summaries.append(r['summary']); events+=r['events']; errors+=r['errors']
                except Exception as e: errors.append({'wallet':wallet,'error':str(e)})
            st.metric('Wallets scanned',len(wallets)); st.metric('Observed token events',len(events))
            if events:
                ev=pd.DataFrame(events); st.dataframe(ev,use_container_width=True,hide_index=True); st.subheader('Wallet Convergence'); st.dataframe(trader_convergence(ev),use_container_width=True,hide_index=True)
            if summaries: st.subheader('Wallet Intelligence'); st.dataframe(pd.DataFrame(summaries),use_container_width=True,hide_index=True)
            if errors:
                with st.expander('Scanner errors'): st.dataframe(pd.DataFrame(errors),use_container_width=True,hide_index=True)
st.divider(); st.caption('PASSED means configured checks passed; it does not mean a token is safe or profitable.')
