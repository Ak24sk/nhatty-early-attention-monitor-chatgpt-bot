import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="X Early Attention Monitor V1.1", layout="wide")
st.title("X Early Attention Monitor — Version 1.1")
st.caption("Research prototype. Detects unusual attention acceleration; it does not provide buy/sell signals.")

times = pd.date_range("2024-10-30 10:00", periods=14, freq="h")
demo = pd.DataFrame({
    "time": list(times)*3,
    "topic": ["Peanut"]*14 + ["Squirrel"]*14 + ["Example Topic"]*14,
    "mentions": [8,9,11,13,16,20,28,39,55,78,112,160,220,300] +
                [20,21,22,24,25,28,31,35,40,43,47,51,56,60] +
                [50,53,51,55,54,56,58,57,59,60,58,61,60,62],
    "unique_accounts": [7,8,10,12,14,18,24,33,45,61,84,118,155,201] +
                       [16,17,18,19,20,22,24,27,30,32,35,38,41,44] +
                       [42,44,43,45,46,47,48,48,49,50,49,51,50,52],
    "engagement": [90,105,140,175,230,310,470,700,1100,1700,2800,4300,6500,9200] +
                  [250,270,285,300,320,350,390,430,480,520,560,600,650,700] +
                  [900,920,910,940,950,960,980,970,990,1000,980,1010,1000,1020],
    "influencer_event": [0]*7 + [1]*7 + [0]*28
})

def norm(s):
    s = pd.to_numeric(s, errors="coerce").fillna(0).clip(lower=0)
    lo, hi = s.min(), s.max()
    return pd.Series(np.zeros(len(s)), index=s.index) if hi <= lo else (s-lo)/(hi-lo)*100

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
    d["engagement_growth"] = g["engagement"].pct_change().replace([np.inf,-np.inf],0).fillna(0)
    d["acceleration_component"] = d.groupby("topic")["mention_acceleration"].transform(norm)
    d["unique_component"] = d.groupby("topic")["unique_velocity"].transform(norm)
    d["engagement_component"] = d.groupby("topic")["engagement_growth"].transform(norm)
    d["attention_score"] = (
        .45*d["acceleration_component"] +
        .30*d["unique_component"] +
        .15*d["engagement_component"] +
        .10*d["influencer_event"].clip(0,1)*100
    ).clip(0,100)
    d["status"] = pd.cut(d["attention_score"], [-.01,25,50,75,100.01],
                         labels=["NORMAL","EMERGING","HIGH","EXTREME"])
    def why(r):
        x=[]
        if r["mention_acceleration"] > 0: x.append("mention acceleration")
        if r["unique_velocity"] > 0: x.append("new-account growth")
        if r["engagement_growth"] > .20: x.append("engagement jump")
        if r["influencer_event"] == 1: x.append("influencer event")
        return x or ["no strong acceleration"]
    d["reasons"] = d.apply(why, axis=1)
    return d

uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
raw = pd.read_csv(uploaded) if uploaded else demo
if uploaded is None:
    st.sidebar.warning("Demo mode: synthetic data, not live X data.")

required = {"time","topic","mentions","unique_accounts","engagement","influencer_event"}
missing = required - set(raw.columns)
if missing:
    st.error("Missing columns: " + ", ".join(sorted(missing)))
    st.stop()

result = calculate(raw)
latest_rows = result.sort_values("time").groupby("topic", as_index=False).tail(1).sort_values("attention_score", ascending=False)

st.subheader("Attention candidates")
cv = latest_rows[["topic","attention_score","mention_velocity","mention_acceleration","unique_velocity","status"]].copy()
cv["attention_score"] = cv["attention_score"].round(0)
st.dataframe(cv, use_container_width=True, hide_index=True)

topics = result["topic"].astype(str).unique().tolist()
selected = st.sidebar.selectbox("Inspect topic", topics)
view = result[result["topic"].astype(str)==selected].copy()
latest = view.iloc[-1]

a,b,c,d = st.columns(4)
a.metric("Attention score", f"{latest['attention_score']:.0f}/100")
b.metric("Mentions", f"{int(latest['mentions']):,}")
c.metric("Unique accounts", f"{int(latest['unique_accounts']):,}")
d.metric("Influencer event", "YES" if latest["influencer_event"] else "NO")

status = str(latest["status"])
if status == "EXTREME": st.error("EXTREME attention acceleration detected.")
elif status == "HIGH": st.warning("HIGH attention acceleration detected.")
elif status == "EMERGING": st.info("EMERGING attention acceleration detected.")
else: st.success("NORMAL attention dynamics.")

st.markdown("**Why did the detector score it this way?**")
for r in latest["reasons"]: st.write("• " + r.capitalize())

st.subheader("Attention components")
st.line_chart(view.set_index("time")[["acceleration_component","unique_component","engagement_component"]])

st.subheader("Activity timeline")
st.line_chart(view.set_index("time")[["mentions","unique_accounts","engagement"]])

st.subheader("Detailed detector output")
cols=["time","mentions","unique_accounts","engagement","mention_velocity","mention_acceleration","unique_velocity","engagement_growth","attention_score","status","influencer_event"]
st.dataframe(view[cols], use_container_width=True, hide_index=True)

st.subheader("V1.1 scoring logic")
st.markdown("- **45%** mention acceleration\n- **30%** unique-account growth\n- **15%** engagement growth\n- **10%** influencer-event signal")
st.caption("Weights are experimental and must be validated against historical cases and false positives.")

st.download_button("Download demo CSV", demo.to_csv(index=False).encode(), "x_attention_demo_v1_1.csv", "text/csv")
st.caption("Demo data is synthetic. Real-time monitoring requires an appropriate, authorized data source. This app does not execute trades or provide buy/sell recommendations.")
