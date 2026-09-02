import streamlit as st
import pandas as pd
from detector import (
    calculate_attention,
    safety_gate,
    trader_scores,
    trader_convergence,
    dev_history,
    build_signal,
)

st.set_page_config(page_title="Early Runner Intelligence V2.0", layout="wide")
st.title("Early Runner Intelligence — V2.0")
st.caption("Safety Gate → Elite Early-Runner Traders → Dev Wallet Intelligence → X Attention → Convergence → Alert")

st.warning(
    "Research prototype only. A PASSED token has passed the configured checks in the supplied data; "
    "that is not a guarantee that the token is safe, legitimate, or profitable. "
    "Unknown/unverified data is blocked by default. No trades are executed."
)

# ---------------- Demo data ----------------
def demo_data():
    tokens = pd.DataFrame([
        ["ALPHA", 125000, True, 90, False, False, False, True, 32, 5, 18, "LOW", "CLEAN"],
        ["BETA", 18000, False, 0, True, True, True, False, 62, 19, 71, "HIGH", "SUSPICIOUS"],
        ["GAMMA", 92000, True, 45, False, False, False, True, 37, 7, 24, "LOW", "CLEAN"],
    ], columns=[
        "token","liquidity_usd","lp_locked","lp_lock_days","mint_authority_active",
        "freeze_authority_active","wash_trade_risk","sellable","top10_holder_pct",
        "dev_holder_pct","cluster_holder_pct","dev_risk","contract_risk"
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
        ["2026-09-02 20:00","ALPHA",3,3,120,0],
        ["2026-09-02 20:01","ALPHA",5,4,210,1],
        ["2026-09-02 20:02","ALPHA",9,7,420,1],
        ["2026-09-02 20:03","ALPHA",15,11,700,2],
        ["2026-09-02 20:04","ALPHA",24,16,1100,2],
        ["2026-09-02 20:05","ALPHA",35,22,1700,3],
        ["2026-09-02 20:06","ALPHA",48,29,2300,4],
        ["2026-09-02 20:00","GAMMA",4,3,140,0],
        ["2026-09-02 20:01","GAMMA",5,4,180,0],
        ["2026-09-02 20:02","GAMMA",6,5,220,0],
        ["2026-09-02 20:03","GAMMA",7,5,250,0],
        ["2026-09-02 20:04","GAMMA",8,6,290,0],
        ["2026-09-02 20:05","GAMMA",10,7,340,0],
    ], columns=["time","topic","mentions","unique_accounts","engagement","influencer_event"])
    return tokens, trades, x

# ---------------- Sidebar ----------------
st.sidebar.header("Data Source")
mode = st.sidebar.radio("Mode", ["Demo", "Upload CSVs"])

st.sidebar.header("Safety thresholds")
min_liq = st.sidebar.number_input("Minimum liquidity ($)", min_value=0.0, value=50000.0, step=5000.0)
min_lock = st.sidebar.number_input("Minimum LP lock days", min_value=0, value=30, step=1)
max_top10 = st.sidebar.number_input("Maximum top-10 holder %", min_value=0.0, max_value=100.0, value=40.0)
max_dev = st.sidebar.number_input("Maximum dev holder %", min_value=0.0, max_value=100.0, value=10.0)
max_cluster = st.sidebar.number_input("Maximum related-cluster %", min_value=0.0, max_value=100.0, value=50.0)

if mode == "Demo":
    tokens, trades, x = demo_data()
else:
    token_file = st.sidebar.file_uploader("Token safety CSV", type=["csv"])
    trade_file = st.sidebar.file_uploader("Trader trades CSV", type=["csv"])
    x_file = st.sidebar.file_uploader("X attention CSV", type=["csv"])

    if not token_file or not trade_file:
        st.info("Upload at least Token Safety CSV and Trader Trades CSV. X CSV is optional.")
        st.stop()

    tokens = pd.read_csv(token_file)
    trades = pd.read_csv(trade_file)
    x = pd.read_csv(x_file) if x_file else pd.DataFrame()

# ---------------- Processing ----------------
safety = safety_gate(
    tokens,
    min_liquidity=min_liq,
    min_lp_lock_days=min_lock,
    max_top10=max_top10,
    max_dev=max_dev,
    max_cluster=max_cluster,
)

traders = trader_scores(trades)
convergence = trader_convergence(trades, traders)
dev = dev_history(safety)

if not x.empty:
    attention = calculate_attention(x)
else:
    attention = pd.DataFrame(columns=["topic","attention_score","stage","alert"])

signals = build_signal(safety, traders, convergence, dev, attention)

# ---------------- Overview ----------------
st.subheader("1. Unified Signal Board")

if signals.empty:
    st.info("No signals available.")
else:
    display_cols = [
        c for c in [
            "token","safety_status","safety_score","elite_buyers","convergence_score",
            "dev_score","attention_score","unified_score","signal","reasons"
        ] if c in signals.columns
    ]
    st.dataframe(signals[display_cols], use_container_width=True)

# ---------------- Safety ----------------
st.subheader("2. Safety Gate")
st.caption("Safety is a hard gate. FAILED and UNVERIFIED tokens cannot generate an ALERT.")

safety_cols = [
    c for c in [
        "token","status","score","liquidity_usd","lp_locked","lp_lock_days",
        "mint_authority_active","freeze_authority_active","sellable",
        "top10_holder_pct","dev_holder_pct","cluster_holder_pct",
        "wash_trade_risk","bundle_risk","dev_risk","contract_risk","reasons"
    ] if c in safety.columns
]
st.dataframe(safety[safety_cols], use_container_width=True)

# ---------------- Traders ----------------
st.subheader("3. Elite Early-Runner Traders")
st.dataframe(traders, use_container_width=True)

st.subheader("4. Trader Convergence")
if convergence.empty:
    st.info("No convergence detected.")
else:
    st.dataframe(convergence, use_container_width=True)

# ---------------- Dev ----------------
st.subheader("5. Dev / Associated Wallet Intelligence")
st.dataframe(dev, use_container_width=True)

# ---------------- X ----------------
st.subheader("6. X Attention")
if attention.empty:
    st.info("X attention data is not supplied. The system can still operate using on-chain/trader signals.")
else:
    st.dataframe(attention, use_container_width=True)
    if "topic" in attention.columns and "attention_score" in attention.columns:
        chart = attention[["topic","attention_score"]].set_index("topic")
        st.bar_chart(chart)

# ---------------- How it works ----------------
with st.expander("How V2.0 decides"):
    st.markdown("""
**Step 1 — Safety Gate**
- Liquidity
- LP lock and duration
- Mint/freeze authority
- Holder concentration
- Dev concentration
- Related-wallet/cluster concentration
- Sellability
- Wash-trading and bundle-risk fields
- Dev and contract risk

**Step 2 — Elite Trader Detection**
The system scores wallets using historical early-runner behavior, runner/win history, consistency, and activity volume.

**Step 3 — Convergence**
Multiple independently behaving high-score wallets buying the same token in a short window can create a convergence signal.

**Step 4 — Dev Intelligence**
Historical dev-associated behavior is separated from the opportunity score.

**Step 5 — X Attention**
Attention velocity and acceleration are measured when authorized X data is supplied.

**Step 6 — Unified Signal**
A token must pass the Safety Gate first. Only then can trader convergence, dev history and X attention raise the opportunity score.
""")

with st.expander("CSV schemas"):
    st.markdown("""
### Token Safety CSV
Required:
`token, liquidity_usd, lp_locked, lp_lock_days, mint_authority_active, freeze_authority_active, sellable, top10_holder_pct, dev_holder_pct, cluster_holder_pct`

Recommended:
`wash_trade_risk, bundle_risk, dev_risk, contract_risk`

### Trader Trades CSV
Required:
`wallet, time, token, side, usd`

Recommended historical labels:
`early_runner, win`

### X Attention CSV
`time, topic, mentions, unique_accounts, engagement, influencer_event`

These are data inputs. V2.0 does **not** pretend that an uploaded field is independently verified.
""")
