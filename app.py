import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="X Early Attention Monitor V1.3", layout="wide")
st.title("X Early Attention Monitor — Version 1.3")
st.caption("Early-acceleration research prototype. Not a trading or buy/sell system.")

times = pd.date_range("2024-10-30 10:00", periods=18, freq="h")

demo = pd.DataFrame({
    "time": list(times) * 4,
    "topic": ["Peanut"]*18 + ["Squirrel"]*18 + ["Stable Topic"]*18 + ["Burst Then Fade"]*18,
    "mentions": (
        [8,9,10,12,14,17,21,27,35,46,60,79,103,132,166,205,250,300] +
        [18,19,20,22,24,27,30,34,39,44,49,55,61,67,73,79,85,91] +
        [60,61,59,62,61,63,62,64,63,65,64,66,65,67,66,68,67,69] +
        [5,6,8,13,24,40,65,95,120,130,118,100,82,68,56,48,42,38]),
    "unique_accounts": (
        [7,8,9,11,13,16,20,25,31,40,51,65,82,101,123,148,174,201] +
        [15,16,17,18,19,21,23,26,29,33,37,41,45,49,53,57,61,65] +
        [40,41,40,42,41,43,42,44,43,45,44,46,45,47,46,48,47,49] +
        [5,6,8,12,22,35,51,68,78,80,73,64,55,47,40,35,31,28]),
    "engagement": (
        [80,90,105,125,155,190,250,330,450,620,850,1150,1550,2050,2700,3500,4500,5700] +
        [220,230,245,265,290,320,355,395,440,490,545,605,670,740,815,895,980,1070] +
        [850,860,840,875,865,890,880,900,890,915,905,930,920,945,935,955,945,965] +
        [45,55,80,140,300,520,850,1250,1650,1900,1700,1450,1150,900,720,600,510,450]),
    "influencer_event": [0]*18 + [0]*18 + [0]*18 + [0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0]
})

def norm(s):
    s = pd.to_numeric(s, errors="coerce").fillna(0).clip(lower=0)
    cap = s.quantile(.95) if len(s) else 0
    cap = cap if cap > 0 else (s.max() if len(s) else 1)
    return (s.clip(upper=cap) / cap * 100).clip(0,100)

def calculate(df):
    d = df.copy()
    d["time"] = pd.to_datetime(d["time"], errors="coerce")
    d = d.dropna(subset=["time"]).sort_values(["topic","time"]).reset_index(drop=True)
    for c in ["mentions","unique_accounts","engagement","influencer_event"]:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0)

    g = d.groupby("topic")
    d["mention_velocity"] = g["mentions"].diff().fillna(0)
    d["mention_acceleration"] = g["mention_velocity"].diff().fillna(0)
    d["unique_velocity"] = g["unique_accounts"].diff().fillna(0)
    d["unique_acceleration"] = g["unique_velocity"].diff().fillna(0)
    d["engagement_growth"] = g["engagement"].pct_change().replace([np.inf,-np.inf],np.nan).fillna(0)
    d["unique_ratio"] = (d["unique_accounts"]/d["mentions"].replace(0,np.nan)).fillna(0).clip(0,1)

    d["velocity_component"] = g["mention_velocity"].transform(norm)
    d["acceleration_component"] = g["mention_acceleration"].transform(norm)
    d["unique_component"] = g["unique_velocity"].transform(norm)
    d["unique_acceleration_component"] = g["unique_acceleration"].transform(norm)
    d["engagement_component"] = g["engagement_growth"].transform(norm)
    d["breadth_component"] = d["unique_ratio"]*100
    d["influencer_component"] = d["influencer_event"].clip(0,1)*100

    raw = (
        .35*d["acceleration_component"] +
        .20*d["velocity_component"] +
        .15*d["unique_acceleration_component"] +
        .10*d["unique_component"] +
        .10*d["engagement_component"] +
        .05*d["breadth_component"] +
        .05*d["influencer_component"]
    )
    d["early_acceleration_score"] = (
        raw * (1 + .20/(1+np.log1p(d["mentions"])))
    ).clip(0,100)

    d["early_stage"] = np.select(
        [(d["mentions"]<=30)&(d["mention_acceleration"]>0),
         (d["mentions"]<=100)&(d["mention_acceleration"]>0),
         d["mention_acceleration"]>0],
        ["EARLY","GROWING","LATE GROWTH"], default="QUIET")

    d["alert"] = np.select(
        [d["early_acceleration_score"]>=75,
         d["early_acceleration_score"]>=55,
         d["early_acceleration_score"]>=35],
        ["🚨 EARLY WAVE","🔥 WATCH","🟡 BUILDING"], default="⚪ NORMAL")
    return d

st.sidebar.header("Data source")
uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
raw = pd.read_csv(uploaded) if uploaded else demo
if uploaded is None:
    st.sidebar.warning("Demo mode: synthetic data — not live X data.")

required = {"time","topic","mentions","unique_accounts","engagement","influencer_event"}
missing = required - set(raw.columns)
if missing:
    st.error("Missing columns: " + ", ".join(sorted(missing)))
    st.stop()

result = calculate(raw)
latest = (result.sort_values("time").groupby("topic", as_index=False).tail(1)
          .sort_values("early_acceleration_score", ascending=False))

st.subheader("🚨 Early attention candidates")
table = latest[["topic","early_acceleration_score","early_stage","alert",
                "mentions","mention_velocity","mention_acceleration","unique_accounts"]].copy()
table["early_acceleration_score"] = table["early_acceleration_score"].round(0)
st.dataframe(table, use_container_width=True, hide_index=True)

top = latest.iloc[0]
st.markdown("### 🏆 Strongest early-wave candidate")
a,b,c,d = st.columns(4)
a.metric("Topic", str(top["topic"]))
b.metric("Early acceleration", f"{top['early_acceleration_score']:.0f}/100")
c.metric("Mentions", f"{int(top['mentions']):,}")
d.metric("Stage", str(top["early_stage"]))

score = top["early_acceleration_score"]
if score >= 75: st.error("🚨 EARLY WAVE — strong acceleration while attention is developing.")
elif score >= 55: st.warning("🔥 WATCH — acceleration deserves attention.")
elif score >= 35: st.info("🟡 BUILDING — some early momentum is visible.")
else: st.success("⚪ No strong early-wave signal.")

topics = result["topic"].astype(str).unique().tolist()
selected = st.sidebar.selectbox("Inspect a topic", topics, index=topics.index(str(top["topic"])))
view = result[result["topic"].astype(str)==selected].copy()
last = view.iloc[-1]

st.subheader(f"🔎 {selected}")
a,b,c,d = st.columns(4)
a.metric("Early acceleration", f"{last['early_acceleration_score']:.0f}/100")
b.metric("Mention velocity", f"{last['mention_velocity']:.0f}")
c.metric("Mention acceleration", f"{last['mention_acceleration']:.0f}")
d.metric("Unique accounts", f"{int(last['unique_accounts']):,}")
st.write("**Current stage:**", last["early_stage"])
st.write("**Alert:**", last["alert"])

st.markdown("### Attention acceleration")
st.line_chart(view.set_index("time")[["velocity_component","acceleration_component",
                                       "unique_acceleration_component","engagement_component"]])
st.markdown("### Raw activity")
st.line_chart(view.set_index("time")[["mentions","unique_accounts","engagement"]])

st.subheader("🧠 V1.3 logic")
st.markdown("""
V1.3 focuses on **change in attention**, not just how popular something already is.

- 35% mention acceleration
- 20% mention velocity
- 15% unique-account acceleration
- 10% unique-account growth
- 10% engagement growth
- 5% discussion breadth
- 5% influencer event

The goal is to detect a possible attention wave **while it is developing**.
The included data is synthetic and is not live X data.
""")

st.download_button("Download V1.3 demo CSV",
                   demo.to_csv(index=False).encode(),
                   "x_attention_demo_v1_3.csv","text/csv")
