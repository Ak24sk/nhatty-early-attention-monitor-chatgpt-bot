import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timezone

st.set_page_config(page_title="X Early Attention Monitor V1.6", layout="wide")
st.title("X Early Attention Monitor — Version 1.6")
st.caption("Collector-ready architecture: raw events → aggregation → attention analysis → alerts")

REQUIRED = ["time","topic","mentions","unique_accounts","engagement","influencer_event"]

def demo_data():
    t=pd.date_range("2024-10-30 10:00",periods=12,freq="h")
    return pd.DataFrame({
        "time":list(t)*3,
        "topic":["Squirrel"]*12+["Slow Growth"]*12+["Burst Then Fade"]*12,
        "mentions":[3,4,6,8,13,22,38,65,102,154,220,305]+[8,9,11,13,16,20,24,29,35,41,50,59]+[4,7,22,60,120,165,170,150,115,83,60,37],
        "unique_accounts":[3,4,5,7,10,17,29,49,77,111,152,202]+[7,8,9,10,12,14,16,19,22,25,31,37]+[4,6,18,45,82,108,112,99,77,58,45,30],
        "engagement":[15,19,30,44,72,128,230,400,650,1010,1500,2150]+[80,88,103,118,135,155,178,203,231,262,295,332]+[20,40,150,500,1150,1750,1900,1700,1300,960,700,410],
        "influencer_event":[0,0,0,0,0,0,1,0,0,0,0,0]+[0]*12+[0,0,0,0,1,1,1,0,0,0,0,0]
    })

def norm(s):
    s=pd.to_numeric(s,errors="coerce").fillna(0).clip(lower=0)
    if s.max()<=0:return pd.Series(np.zeros(len(s)),index=s.index)
    q=s.quantile(.95)
    if q<=0:q=s.max()
    return (s.clip(upper=q)/q*100).clip(0,100)

def aggregate_events(e):
    e=e.copy()
    if "time" not in e.columns or "topic" not in e.columns:
        raise ValueError("Raw events require time and topic columns")
    e["time"]=pd.to_datetime(e["time"],errors="coerce",utc=True)
    e=e.dropna(subset=["time"])
    if "author_id" not in e.columns:e["author_id"]="unknown"
    if "engagement" not in e.columns:e["engagement"]=0
    if "influencer_event" not in e.columns:e["influencer_event"]=0
    e["bucket"]=e["time"].dt.floor("h")
    return e.groupby(["bucket","topic"],as_index=False).agg(
        mentions=("topic","size"),
        unique_accounts=("author_id","nunique"),
        engagement=("engagement","sum"),
        influencer_event=("influencer_event","max")
    ).rename(columns={"bucket":"time"})

def calculate(df):
    d=df.copy()
    d["time"]=pd.to_datetime(d["time"],errors="coerce",utc=True)
    d=d.dropna(subset=["time"]).sort_values(["topic","time"]).reset_index(drop=True)
    for c in REQUIRED[2:]:
        d[c]=pd.to_numeric(d[c],errors="coerce").fillna(0).clip(lower=0)
    g=d.groupby("topic")
    d["mention_velocity"]=g["mentions"].diff().fillna(0)
    d["mention_acceleration"]=g["mention_velocity"].diff().fillna(0)
    d["unique_velocity"]=g["unique_accounts"].diff().fillna(0)
    d["unique_acceleration"]=g["unique_velocity"].diff().fillna(0)
    d["engagement_growth"]=g["engagement"].pct_change().replace([np.inf,-np.inf],np.nan).fillna(0)
    d["baseline"]=g["mentions"].transform(lambda s:s.shift(1).rolling(4,min_periods=3).mean())
    d["baseline_std"]=g["mentions"].transform(lambda s:s.shift(1).rolling(4,min_periods=3).std()).fillna(0)
    d["anomaly"]=((d["mentions"]-d["baseline"])/d["baseline_std"].replace(0,np.nan)).replace([np.inf,-np.inf],np.nan).fillna(0)
    d["accel_c"]=g["mention_acceleration"].transform(norm)
    d["vel_c"]=g["mention_velocity"].transform(norm)
    d["unique_accel_c"]=g["unique_acceleration"].transform(norm)
    d["eng_c"]=g["engagement_growth"].transform(norm)
    d["anomaly_c"]=d["anomaly"].clip(0,5)/5*100
    d["influencer_c"]=d["influencer_event"].clip(0,1)*100
    raw=.32*d["accel_c"]+.20*d["vel_c"]+.16*d["unique_accel_c"]+.12*d["anomaly_c"]+.12*d["eng_c"]+.08*d["influencer_c"]
    d["score"]=(raw*(1+.12/(1+np.log1p(d["mentions"])))).clip(0,100)
    d["signals"]=((d["mention_acceleration"]>0).astype(int)+(d["unique_acceleration"]>0).astype(int)+(d["engagement_growth"]>.15).astype(int)+(d["anomaly"]>1.5).astype(int)+(d["influencer_event"]>0).astype(int))
    d["quality"]=np.select([d["signals"]>=4,d["signals"]>=3,d["signals"]>=2],["STRONG","GOOD","WEAK"],default="LOW")
    d["stage"]=np.select([(d["mentions"]<=30)&(d["mention_acceleration"]>0),(d["mentions"]<=100)&(d["mention_acceleration"]>0),d["mention_acceleration"]>0],["EARLY","GROWING","LATE"],default="QUIET")
    d["alert"]=np.select([(d["score"]>=70)&d["quality"].isin(["STRONG","GOOD"]),d["score"]>=50,d["score"]>=30],["EARLY WAVE","WATCH","BUILDING"],default="NORMAL")
    return d

st.sidebar.header("V1.6 Data Pipeline")
mode=st.sidebar.radio("Input mode",["Demo aggregated data","Upload aggregated CSV","Upload raw event CSV"])
uploaded=None
if mode=="Upload aggregated CSV":
    uploaded=st.sidebar.file_uploader("Upload aggregated CSV",type=["csv"],key="a")
elif mode=="Upload raw event CSV":
    uploaded=st.sidebar.file_uploader("Upload raw events CSV",type=["csv"],key="r")

if mode=="Demo aggregated data":
    raw=demo_data(); label="Synthetic demo"
elif uploaded is None:
    st.info("Upload a CSV file to begin."); st.stop()
else:
    incoming=pd.read_csv(uploaded)
    try:
        raw=aggregate_events(incoming) if mode=="Upload raw event CSV" else incoming
        label="Raw events automatically aggregated" if mode=="Upload raw event CSV" else "Uploaded aggregated data"
    except Exception as e:
        st.error(str(e)); st.stop()

missing=[c for c in REQUIRED if c not in raw.columns]
if missing:
    st.error("Missing columns: "+", ".join(missing)); st.stop()

result=calculate(raw)
latest=result.sort_values("time").groupby("topic",as_index=False).tail(1).sort_values("score",ascending=False).reset_index(drop=True)

st.subheader("Pipeline Status")
a,b,c,d=st.columns(4)
a.metric("Input",label); b.metric("Topics",len(latest)); c.metric("Rows",len(result)); d.metric("Calculated",datetime.now(timezone.utc).strftime("%H:%M:%S UTC"))

st.markdown("**Architecture:** Authorized collector → Raw events → Hourly aggregation → Detection engine → Ranked alerts")

st.subheader("Early Attention Watchlist")
watch=latest[["topic","score","quality","stage","alert","mentions","mention_velocity","mention_acceleration","anomaly","signals"]].copy()
watch["score"]=watch["score"].round(1); watch["anomaly"]=watch["anomaly"].round(2)
st.dataframe(watch,use_container_width=True,hide_index=True)

st.subheader("Active Alert Queue")
alerts=latest[latest["alert"]!="NORMAL"]
if alerts.empty: st.success("No active alerts.")
else:
    for _,r in alerts.iterrows():
        st.warning(f"{r['alert']} | {r['topic']} | Score {r['score']:.1f} | {r['quality']} | {r['stage']}")

selected=st.selectbox("Inspect topic",latest["topic"].astype(str).tolist())
view=result[result["topic"].astype(str)==selected]
last=view.iloc[-1]
st.subheader(f"Topic Inspector — {selected}")
a,b,c,d,e=st.columns(5)
a.metric("Score",f"{last['score']:.1f}/100"); b.metric("Velocity",f"{last['mention_velocity']:.1f}"); c.metric("Acceleration",f"{last['mention_acceleration']:.1f}"); d.metric("Anomaly",f"{last['anomaly']:.2f}σ"); e.metric("Signals",f"{last['signals']}/5")
st.write(f"**Stage:** {last['stage']} | **Quality:** {last['quality']} | **Alert:** {last['alert']}")
st.line_chart(view.set_index("time")[["vel_c","accel_c","unique_accel_c","anomaly_c"]])
st.markdown("### Mentions vs baseline")
st.line_chart(view.set_index("time")[["mentions","baseline"]])

with st.expander("Input formats"):
    st.markdown("""**Aggregated CSV:** `time, topic, mentions, unique_accounts, engagement, influencer_event`

**Raw event CSV minimum:** `time, topic`

Recommended raw columns: `author_id, engagement, influencer_event`

Raw events are automatically grouped into hourly topic metrics.""")

agg_template=pd.DataFrame(columns=REQUIRED)
raw_template=pd.DataFrame(columns=["time","topic","author_id","engagement","influencer_event"])
st.download_button("Download aggregated CSV template",agg_template.to_csv(index=False).encode(),"aggregated_template.csv","text/csv")
st.download_button("Download raw event CSV template",raw_template.to_csv(index=False).encode(),"raw_event_template.csv","text/csv")
st.download_button("Download V1.6 demo data",demo_data().to_csv(index=False).encode(),"demo_v1_6.csv","text/csv")

st.subheader("What V1.6 adds")
st.markdown("""- Raw event input mode
- Automatic hourly aggregation
- Aggregated-data input mode
- Collector-ready pipeline architecture
- Same detector can work behind a future authorized data source

This prototype does not scrape X, bypass access controls, or execute trades.""")
