
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="X Early Attention Monitor", layout="wide")

st.title("X Early Attention Monitor — Version 1")
st.caption("Research prototype. It measures attention dynamics; it does not provide buy/sell signals.")

# Synthetic demonstration dataset.
# It is deliberately labeled as synthetic and is NOT live X data.
demo = pd.DataFrame({
    "time": pd.date_range("2024-10-30 10:00", periods=14, freq="h"),
    "topic": ["Peanut"] * 14,
    "mentions": [8, 9, 11, 13, 16, 20, 28, 39, 55, 78, 112, 160, 220, 300],
    "unique_accounts": [7, 8, 10, 12, 14, 18, 24, 33, 45, 61, 84, 118, 155, 201],
    "engagement": [90, 105, 140, 175, 230, 310, 470, 700, 1100, 1700, 2800, 4300, 6500, 9200],
    "influencer_event": [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
})

def minmax(series):
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - lo) / (hi - lo) * 100

def calculate_detector(df):
    d = df.copy()
    d["time"] = pd.to_datetime(d["time"], errors="coerce")
    d = d.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

    for col in ["mentions", "unique_accounts", "engagement", "influencer_event"]:
        d[col] = pd.to_numeric(d[col], errors="coerce").fillna(0)

    d["mention_velocity"] = d["mentions"].diff().fillna(0)
    d["mention_acceleration"] = d["mention_velocity"].diff().fillna(0)
    d["unique_account_velocity"] = d["unique_accounts"].diff().fillna(0)
    d["engagement_velocity"] = d["engagement"].pct_change().replace(
        [np.inf, -np.inf], 0
    ).fillna(0)

    # Experimental attention score:
    # 35% mention acceleration
    # 30% unique-account growth
    # 20% engagement growth
    # 15% influencer event
    score = (
        0.35 * minmax(d["mention_acceleration"].clip(lower=0))
        + 0.30 * minmax(d["unique_account_velocity"].clip(lower=0))
        + 0.20 * minmax(d["engagement_velocity"].clip(lower=0))
        + 0.15 * d["influencer_event"].clip(0, 1) * 100
    ).clip(0, 100)

    d["attention_score"] = score

    d["status"] = pd.cut(
        d["attention_score"],
        bins=[-0.01, 25, 50, 75, 100.01],
        labels=["NORMAL", "EMERGING", "HIGH", "EXTREME"],
    )

    return d

st.sidebar.header("Data source")
uploaded = st.sidebar.file_uploader(
    "Upload CSV",
    type=["csv"],
    help=(
        "Required columns: time, topic, mentions, unique_accounts, "
        "engagement, influencer_event"
    ),
)

if uploaded is None:
    raw = demo.copy()
    st.sidebar.warning("Demo mode: synthetic data, not live X data.")
else:
    raw = pd.read_csv(uploaded)

required = {
    "time",
    "topic",
    "mentions",
    "unique_accounts",
    "engagement",
    "influencer_event",
}
missing = required - set(raw.columns)

if missing:
    st.error("Your CSV is missing: " + ", ".join(sorted(missing)))
    st.stop()

result = calculate_detector(raw)

topics = result["topic"].dropna().astype(str).unique().tolist()
if not topics:
    st.error("No topic values were found.")
    st.stop()

selected = st.sidebar.selectbox("Select topic", topics)
view = result[result["topic"].astype(str) == selected].copy()

latest = view.iloc[-1]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Attention score", f"{latest['attention_score']:.0f}/100")
c2.metric("Mentions", f"{int(latest['mentions']):,}")
c3.metric("Unique accounts", f"{int(latest['unique_accounts']):,}")
c4.metric(
    "Influencer event",
    "YES" if int(latest["influencer_event"]) == 1 else "NO",
)

status = str(latest["status"])
if status == "EXTREME":
    st.error("EXTREME attention dynamics detected.")
elif status == "HIGH":
    st.warning("HIGH attention dynamics detected.")
elif status == "EMERGING":
    st.info("EMERGING attention dynamics detected.")
else:
    st.success("NORMAL attention dynamics.")

st.subheader(f"{selected} — attention timeline")

chart = view.set_index("time")[["mentions", "unique_accounts", "engagement"]]
st.line_chart(chart)

st.subheader("Detector calculations")

display_cols = [
    "time",
    "mentions",
    "unique_accounts",
    "engagement",
    "mention_velocity",
    "mention_acceleration",
    "unique_account_velocity",
    "attention_score",
    "status",
    "influencer_event",
]

st.dataframe(
    view[display_cols],
    use_container_width=True,
    hide_index=True,
)

st.subheader("How the score works")
st.markdown(
    """
- **35%** mention acceleration
- **30%** unique-account growth
- **20%** engagement growth
- **15%** influencer-event signal

The weights are experimental. We will later test them against historical examples rather than assuming they are optimal.
"""
)

st.download_button(
    "Download demo CSV",
    data=demo.to_csv(index=False).encode("utf-8"),
    file_name="x_attention_demo.csv",
    mime="text/csv",
)

st.caption(
    "Important: the demo data is synthetic. A real-time detector requires an "
    "appropriate, authorized data source. This app intentionally does not execute trades."
)
