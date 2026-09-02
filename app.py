import streamlit as st
import pandas as pd
from detector import calculate_attention, safety_gate, trader_scores, trader_convergence, dev_history, build_signal
from solana_data import SolanaRPC, enrich_token_from_rpc

st.set_page_config(page_title="Early Runner Intelligence V2.1", layout="wide")
st.title("Early Runner Intelligence — V2.1")
st.caption("Safety Gate → Early-Runner Traders → Dev Intelligence → X Attention → On-chain Confirmation → Convergence")

st.warning(
    "Research prototype. PASSED means the configured checks passed using the supplied/queried data; "
    "it is not a guarantee of safety or profit. Unknown critical checks remain blocked. No trading is executed."
)

# ---------------- Sidebar ----------------
st.sidebar.header("Mode")
mode = st.sidebar.radio("Data mode", ["Demo", "CSV", "Live Solana + CSV"])

st.sidebar.header("Safety thresholds")
min_liq = st.sidebar.number_input("Minimum liquidity ($)", 0.0, value=50000.0, step=5000.0)
min_lock = st.sidebar.number_input("Minimum LP lock days", 0, value=30, step=1)
max_top10 = st.sidebar.number_input("Maximum top-10 holder %", 0.0, 100.0, 40.0, step=1.0)
max_dev = st.sidebar.number_input("Maximum dev holder %", 0.0, 100.0, 10.0, step=1.0)
max_cluster = st.sidebar.number_input("Maximum related-cluster %", 0.0, 100.0, 50.0, step=1.0)

def demo_data():
    tokens = pd.DataFrame([
        ["ALPHA", "DemoMintAlpha", 125000, True, 90, False, False, "LOW", "LOW", True, 32, 5, 18, "LOW", "CLEAN"],
        ["BETA", "DemoMintBeta", 18000, False, 0, True, True, "HIGH", "HIGH", False, 62, 19, 71, "HIGH", "SUSPICIOUS"],
        ["GAMMA", "DemoMintGamma", 92000, True, 45, False, False, "LOW", "LOW", True, 37, 7, 24, "LOW", "CLEAN"],
    ], columns=[
        "token","mint","liquidity_usd","lp_locked","lp_lock_days",
        "mint_authority_active","freeze_authority_active","wash_trade_risk","bundle_risk",
        "sellable","top10_holder_pct","dev_holder_pct","cluster_holder_pct","dev_risk","contract_risk"
    ])
    trades = pd.DataFrame([
        ["WALLET_A","2026-09-02 20:00","ALPHA","BUY",1800,True,True],
        ["WALLET_B","2026-09-02 20:02","ALPHA","BUY",2400,True,True],
        ["WALLET_C","2026-09-02 20:04","ALPHA","BUY",1500,True,True],
        ["WALLET_D","2026-09-02 20:05","ALPHA","BUY",800,False,True],
        ["WALLET_A","2026-09-02 20:18","GAMMA","BUY",1000,True,False],
        ["WALLET_E","2026-09-02 20:20","GAMMA","BUY",900,True,True],
        ["WALLET_X","2026-09-02 20:01","BETA","BUY",5000,False,False],
    ], columns=["wallet","time","token","side","usd","early_runner","win"])
    x = pd.DataFrame([
        ["2026-09-02 20:00","ALPHA",3,3,120,0],["2026-09-02 20:01","ALPHA",5,4,210,1],
        ["2026-09-02 20:02","ALPHA",9,7,420,1],["2026-09-02 20:03","ALPHA",15,11,700,2],
        ["2026-09-02 20:04","ALPHA",24,16,1100,2],["2026-09-02 20:05","ALPHA",35,22,1700,3],
        ["2026-09-02 20:06","ALPHA",48,29,2300,4],["2026-09-02 20:00","GAMMA",4,3,140,0],
        ["2026-09-02 20:01","GAMMA",5,4,180,0],["2026-09-02 20:02","GAMMA",6,5,220,0],
        ["2026-09-02 20:03","GAMMA",7,5,250,0],["2026-09-02 20:04","GAMMA",8,6,290,0],
        ["2026-09-02 20:05","GAMMA",10,7,340,0],
    ], columns=["time","topic","mentions","unique_accounts","engagement","influencer_event"])
    return tokens,trades,x

tokens = trades = x = pd.DataFrame()

if mode == "Demo":
    tokens,trades,x=demo_data()
elif mode == "CSV":
    tf=st.sidebar.file_uploader("Token Safety CSV", type="csv")
    trf=st.sidebar.file_uploader("Trader Trades CSV", type="csv")
    xf=st.sidebar.file_uploader("X Attention CSV (optional)", type="csv")
    if not tf or not trf:
        st.info("Upload Token Safety CSV and Trader Trades CSV.")
        st.stop()
    tokens=pd.read_csv(tf); trades=pd.read_csv(trf)
    x=pd.read_csv(xf) if xf else pd.DataFrame()
else:
    st.sidebar.subheader("Live Solana")
    rpc_url=st.sidebar.text_input("Solana RPC URL", "https://api.mainnet-beta.solana.com")
    mint=st.sidebar.text_input("Token mint address")
    label=st.sidebar.text_input("Token label", "LIVE_TOKEN")
    if not mint:
        st.info("Enter a Solana token mint address.")
        st.stop()

    rpc=SolanaRPC(rpc_url)
    if st.sidebar.button("Fetch on-chain data"):
        try:
            with st.spinner("Querying Solana RPC..."):
                enriched=enrich_token_from_rpc(rpc, mint, label)
            st.session_state["live_token"]=enriched
            st.success("On-chain snapshot loaded.")
        except Exception as e:
            st.error(f"RPC request failed: {e}")

    if "live_token" not in st.session_state:
        st.info("Click 'Fetch on-chain data' to load the mint/account snapshot.")
        st.stop()

    tokens=pd.DataFrame([st.session_state["live_token"]])
    trades=pd.DataFrame(columns=["wallet","time","token","side","usd"])
    x=pd.DataFrame()

# ---------------- Processing ----------------
safety=safety_gate(tokens,min_liquidity=min_liq,min_lp_lock_days=min_lock,max_top10=max_top10,max_dev=max_dev,max_cluster=max_cluster)
traders=trader_scores(trades)
convergence=trader_convergence(trades,traders)
dev=dev_history(safety)
attention=calculate_attention(x) if not x.empty else pd.DataFrame(columns=["topic","attention_score","stage","alert"])
signals=build_signal(safety,traders,convergence,dev,attention)

st.subheader("1. Unified Signal Board")
if signals.empty:
    st.info("No signals available.")
else:
    cols=[c for c in ["token","safety_status","safety_score","elite_buyers","convergence_score","dev_score","attention_score","unified_score","signal","reasons"] if c in signals.columns]
    st.dataframe(signals[cols],use_container_width=True)

st.subheader("2. On-chain / Safety Gate")
st.caption("V2.1 can query basic Solana mint/account data. Full LP, cluster, sellability and historical dev analysis still require specialized/indexed data sources.")
st.dataframe(safety,use_container_width=True)

st.subheader("3. Elite Early-Runner Traders")
st.dataframe(traders,use_container_width=True)

st.subheader("4. Trader Convergence")
st.dataframe(convergence,use_container_width=True)

st.subheader("5. Dev / Associated Wallet Intelligence")
st.dataframe(dev,use_container_width=True)

st.subheader("6. X Attention")
if attention.empty:
    st.info("No X data supplied. This is intentional: V2.1 does not scrape X or bypass access controls.")
else:
    st.dataframe(attention,use_container_width=True)
    if "topic" in attention.columns:
        st.bar_chart(attention[["topic","attention_score"]].set_index("topic"))

with st.expander("V2.1 architecture"):
    st.markdown("""
**New in V2.1:** a legitimate Solana JSON-RPC adapter.

The app can query a supplied RPC endpoint for:
- mint account information
- token supply
- largest token accounts

These are raw on-chain observations. They are not automatically treated as proof of LP safety, honeypot safety, wallet relationships, or profitability.

**Hard rule:** missing/unknown critical safety data does not become PASS.

The next live-data layer should add indexed transaction history, DEX liquidity/LP state, holder relationships, and historical wallet behavior.
""")
