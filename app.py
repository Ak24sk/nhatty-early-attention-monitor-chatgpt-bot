import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timezone

st.set_page_config(page_title="X Early Attention Monitor V1.5", layout="wide")
st.title("X Early Attention Monitor — Version 1.5")
st.caption("Early-attention research prototype with uploaded data analysis, watchlists, and alert queues.")

# Demo data
times = pd.date_range("2024-10-30 10:00", periods=24, freq="h")
demo = pd.DataFrame({
    "time": list(times)*4,
    "topic": ["Squirrel"]*24 + ["Slow Growth"]*24 + ["Stable Topic"]*24 + ["Burst Then Fade"]*24,
    "mentions": [3,4,4,5,5,6,7,8,10,13,17,22,29,38,50,65,82,102,126,154,185,220,260,305] + [8,9,10,11,12,13,15,16,18,20,22,24,26,29,32,35,38,41,44,47,50,53,56,59] + [50,51,49,52,50,51,50,52,51,53,52,51,53,52,54,53,52,54,53,55,54,53,55,54] + [4,5,7,12,22,38,60,88,120,150,165,170,165,150,132,115,98,83,70,60,52,46,41,37],
    "unique_accounts": [3,3,4,4,5,5,6,7,8,10,13,17,22,29,38,49,62,77,93,111,130,152,176,202] + [7,8,8,9,9,10,11,12,13,14,15,16,17,19,20,22,24,25,27,29,31,33,35,37] + [32,33,32,34,33,34,33,35,34,36,35,34,36,35,37,36,35,37,36,38,37,36,38,37] + [4,5,6,10,18,30,45,63,82,98,108,112,108,99,88,77,67,58,51,45,40,36,33,30],
    "engagement": [15,18,19,23,26,30,36,44,55,72,95,128,172,230,305,400,515,650,820,1010,1230,1500,1800,2150] + [80,84,88,92,97,103,110,118,126,135,145,155,166,178,190,203,217,231,246,262,278,295,313,332] + [500,510,490,515,505,520,510,525,515,530,520,515,535,525,540,530,520,545,535,550,540,530,550,540] + [20,25,40,75,150,280,500,800,1150,1500,1750,1900,1850,1700,1500,1300,1120,960,820,700,600,520,460,410],
    "influencer_event": [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0] + [0]*24 + [0]*24 + [0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
})

def norm(s):
    s = pd.to_numeric(s, errors="coerce").fillna(0).clip(lower=0)
    if len(s)==0 or s.max()<=0: return pd.Series(np.zeros(len(s)), index=s.index)
    cap = s.quantile(.95)
    if cap<=0: cap=s.max()
    return (s.clip(upper=cap)/cap*100).clip(0,100)

def calculate(df):
    d=df.copy()
    d["time"]=pd.to_datetime(d["time"],errors="coerce",utc=True)
    d=d.dropna(subset=["time"]).sort_values(["topic","time"]).reset_index(drop=True)
    d["topic"]=d["topic"].astype(str)
    for c in ["mentions","unique_accounts","engagement","influencer_event"]:
        d[c]=pd.to_numeric(d[c],errors="coerce").fillna(0).clip(lower=0)
    g=d.groupby("topic")
    d["mention_velocity"]=g["mentions"].diff().fillna(0)
    d["mention_acceleration"]=g["mention_velocity"].diff().fillna(0)
    d["unique_velocity"]=g["unique_accounts"].diff().fillna(0)
    d["unique_acceleration"]=g["unique_velocity"].diff().fillna(0)
    d["engagement_growth"]=g["engagement"].pct_change().replace([np.inf,-np.inf],np.nan).fillna(0)
    d["mention_baseline"]=g["mentions"].transform(lambda s:s.shift(1).rolling(6,min_periods=3).mean())
    d["mention_baseline_std"]=g["mentions"].transform(lambda s:s.shift(1).rolling(6,min_periods=3).std()).fillna(0)
    d["mention_anomaly"]=((d["mentions"]-d["mention_baseline"])/d["mention_baseline_std"].replace(0,np.nan)).replace([np.inf,-np.inf],np.nan).fillna(0)
    d["unique_ratio"]=(d["unique_accounts"]/d["mentions"].replace(0,np.nan)).fillna(0).clip(0,1)
    d["acceleration_component"]=g["mention_acceleration"].transform(norm)
    d["velocity_component"]=g["mention_velocity"].transform(norm)
    d["unique_acceleration_component"]=g["unique_acceleration"].transform(norm)
    d["unique_component"]=g["unique_velocity"].transform(norm)
    d["engagement_component"]=g["engagement_growth"].transform(norm)
    d["anomaly_component"]=d["mention_anomaly"].clip(0,5)/5*100
    d["breadth_component"]=d["unique_ratio"]*100
    d["influencer_component"]=d["influencer_event"].clip(0,1)*100
    raw=.30*d["acceleration_component"]+.18*d["velocity_component"]+.15*d["unique_acceleration_component"]+.10*d["unique_component"]+.10*d["anomaly_component"]+.08*d["engagement_component"]+.05*d["breadth_component"]+.04*d["influencer_component"]
    d["early_wave_score"]=(raw*(1+.15/(1+np.log1p(d["mentions"])))).clip(0,100)
    d["positive_signals"]=((d["mention_acceleration"]>0).astype(int)+(d["unique_acceleration"]>0).astype(int)+(d["engagement_growth"]>.15).astype(int)+(d["mention_anomaly"]>1.5).astype(int)+(d["influencer_event"]>0).astype(int))
    d["signal_quality"]=np.select([d["positive_signals"]>=4,d["positive_signals"]>=3,d["positive_signals"]>=2],["STRONG","GOOD","WEAK"],default="LOW")
    d["stage"]=np.select([(d["mentions"]<=30)&(d["mention_acceleration"]>0),(d["mentions"]<=100)&(d["mention_acceleration"]>0),d["mention_acceleration"]>0],["EARLY","GROWING","LATE GROWTH"],default="QUIET")
    d["alert"]=np.select([(d["early_wave_score"]>=70)&d["signal_quality"].isin(["STRONG","GOOD"]),d["early_wave_score"]>=50,d["early_wave_score"]>=30],["EARLY WAVE","WATCH","BUILDING"],default="NORMAL")
    return d

st.sidebar.header("Data source")
mode=st.sidebar.radio("Choose mode",["Demo data","Upload collected data"])
uploaded=None
if mode=="Upload collected data":
    uploaded=st.sidebar.file_uploader("Upload CSV",type=["csv"])
raw=pd.read_csv(uploaded) if uploaded is not None else demo
if mode=="Demo data": st.sidebar.info("Demo mode: synthetic data, not live X data.")

required={"time","topic","mentions","unique_accounts","engagement","influencer_event"}
missing=required-set(raw.columns)
if missing:
    st.error("Missing columns: "+", ".join(sorted(missing))); st.stop()

result=calculate(raw)
latest=result.sort_values("time").groupby("topic",as_index=False).tail(1).sort_values("early_wave_score",ascending=False).reset_index(drop=True)

st.subheader("V1.5 Attention Radar")
a,b,c,d=st.columns(4)
a.metric("Topics monitored",len(latest))
b.metric("Strong signals",int((latest["signal_quality"]=="STRONG").sum()))
c.metric("Watchlist signals",int((latest["alert"]!="NORMAL").sum()))
d.metric("Data rows",len(result))
st.caption("Last calculation: "+datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"))

st.markdown("### Ranked early-attention watchlist")
watch=latest[["topic","early_wave_score","signal_quality","stage","alert","mentions","mention_velocity","mention_acceleration","mention_anomaly","positive_signals"]].copy()
watch["early_wave_score"]=watch["early_wave_score"].round(1)
watch["mention_anomaly"]=watch["mention_anomaly"].round(2)
st.dataframe(watch,use_container_width=True,hide_index=True)

top=latest.iloc[0]
st.markdown("### Current strongest signal")
a,b,c,d,e=st.columns(5)
a.metric("Topic",str(top["topic"])); b.metric("Score",f"{top['early_wave_score']:.0f}/100"); c.metric("Stage",str(top["stage"])); d.metric("Quality",str(top["signal_quality"])); e.metric("Alert",str(top["alert"]))

selected=st.selectbox("Inspect a topic",latest["topic"].astype(str).tolist(),index=0)
view=result[result["topic"].astype(str)==selected].copy()
last=view.iloc[-1]
st.markdown(f"## Topic Inspector: {selected}")
a,b,c,d=st.columns(4)
a.metric("Early-wave score",f"{last['early_wave_score']:.1f}/100"); b.metric("Mention velocity",f"{last['mention_velocity']:.1f}"); c.metric("Acceleration",f"{last['mention_acceleration']:.1f}"); d.metric("Baseline anomaly",f"{last['mention_anomaly']:.2f} sigma")
st.write("**Stage:**",last["stage"]); st.write("**Signal quality:**",last["signal_quality"]); st.write("**Alert:**",last["alert"])

st.markdown("### Attention components")
st.line_chart(view.set_index("time")[["velocity_component","acceleration_component","unique_acceleration_component","anomaly_component"]])
st.markdown("### Mentions vs baseline")
st.line_chart(view.set_index("time")[["mentions","mention_baseline"]])
st.markdown("### Conversation spread")
st.line_chart(view.set_index("time")[["unique_accounts","engagement"]])

st.markdown("### Alert queue")
alerts=latest[latest["alert"]!="NORMAL"]
if alerts.empty: st.success("No active alerts in the current dataset.")
else:
    for _,row in alerts.iterrows():
        st.warning(f"{row['alert']} — {row['topic']} | Score {row['early_wave_score']:.1f} | {row['signal_quality']} | {row['stage']}")

with st.expander("CSV format"):
    st.markdown("""Required columns: `time`, `topic`, `mentions`, `unique_accounts`, `engagement`, `influencer_event`.

Each row should represent one time interval for one topic. V1.5 analyzes uploaded data and does not scrape or bypass X access controls.""")

template=pd.DataFrame(columns=["time","topic","mentions","unique_accounts","engagement","influencer_event"])
st.download_button("Download CSV template",template.to_csv(index=False).encode(),"x_attention_input_template.csv","text/csv")
st.download_button("Download V1.5 demo data",demo.to_csv(index=False).encode(),"x_attention_demo_v1_5.csv","text/csv")

st.subheader("What V1.5 adds")
st.markdown("""- Upload mode for externally collected data
- Ranked multi-topic watchlist
- Alert queue
- Dashboard metrics
- CSV template
- Clear separation between data collection and analysis

Research prototype only. Scores are experimental and are not buy/sell recommendations.""")
