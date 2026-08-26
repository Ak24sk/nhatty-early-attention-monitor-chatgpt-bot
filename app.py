import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="X Early Attention Monitor V1.2", layout="wide")
st.title("X Early Attention Monitor — Version 1.2")
st.caption("Research prototype for detecting emerging attention waves. Not a trading or buy/sell system.")

times = pd.date_range("2024-10-30 10:00", periods=18, freq="h")
demo = pd.DataFrame({
    "time": list(times) * 4,
    "topic": ["Peanut"]*18 + ["Squirrel"]*18 + ["Stable Topic"]*18 + ["Burst Then Fade"]*18,
    "mentions": [8,9,10,12,14,17,21,27,35,46,60,79,103,132,166,205,250,300] +
                [18,19,20,22,24,27,30,34,39,44,49,55,61,67,73,79,85,91] +
                [60,61,59,62,61,63,62,64,63,65,64,66,65,67,66,68,67,69] +
                [5,6,8,13,24,40,65,95,120,130,118,100,82,68,56,48,42,38],
    "unique_accounts": [7,8,9,11,13,16,20,25,31,40,51,65,82,101,123,148,174,201] +
                       [15,16,17,18,19,21,23,26,29,33,37,41,45,49,53,57,61,65] +
                       [40,41,40,42,41,43,42,44,43,45,44,46,45,47,46,48,47,49] +
                       [5,6,8,12,22,35,51,68,78,80,73,64,55,47,40,35,31,28],
    "engagement": [80,90,105,125,155,190,250,330,450,620,850,1150,1550,2050,2700,3500,4500,5700] +
                  [220,230,245,265,290,320,355,395,440,490,545,605,670,740,815,895,980,1070] +
                  [850,860,840,875,865,890,880,900,890,915,905,930,920,945,935,955,945,965] +
                  [45,55,80,140,300,520,850,1250,1650,1900,1700,1450,1150,900,720,600,510,450],
    "influencer_event": [0]*9+[1]*9+[0]*18+[0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0]
})

def norm(s):
    s = pd.to_numeric(s, errors="coerce").fillna(0).clip(lower=0)
    if s.max() <= 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    cap = s.quantile(.95)
    return (s.clip(upper=cap) / max(cap, 1) * 100).clip(0,100)

def calc(df):
    d=df.copy()
    d["time"]=pd.to_datetime(d["time"], errors="coerce")
    d=d.dropna(subset=["time"]).sort_values(["topic","time"]).reset_index(drop=True)
    for c in ["mentions","unique_accounts","engagement","influencer_event"]:
        d[c]=pd.to_numeric(d[c],errors="coerce").fillna(0)
    g=d.groupby("topic")
    d["mention_velocity"]=g["mentions"].diff().fillna(0)
    d["mention_acceleration"]=g["mention_velocity"].diff().fillna(0)
    d["unique_velocity"]=g["unique_accounts"].diff().fillna(0)
    d["engagement_growth"]=g["engagement"].pct_change().replace([np.inf,-np.inf],np.nan).fillna(0)
    d["unique_ratio"]=(d["unique_accounts"]/d["mentions"].replace(0,np.nan)).fillna(0).clip(0,1)
    d["velocity_component"]=g["mention_velocity"].transform(norm)
    d["acceleration_component"]=g["mention_acceleration"].transform(norm)
    d["unique_component"]=g["unique_velocity"].transform(norm)
    d["engagement_component"]=g["engagement_growth"].transform(norm)
    d["diversity_component"]=d["unique_ratio"]*100
    d["influencer_component"]=d["influencer_event"].clip(0,1)*100
    raw=(.25*d["acceleration_component"]+.20*d["velocity_component"]+
         .20*d["unique_component"]+.15*d["engagement_component"]+
         .10*d["diversity_component"]+.10*d["influencer_component"])
    size_factor=1/(1+np.log1p(d["mentions"]))
    d["early_wave_score"]=(raw*(.75+.25*size_factor*5)).clip(0,100)
    d["status"]=pd.cut(d["early_wave_score"],[-.01,25,50,75,100.01],
                       labels=["NORMAL","EMERGING","HIGH","EXTREME"])
    def why(r):
        x=[]
        if r["mention_acceleration"]>0:x.append("attention is accelerating")
        if r["unique_velocity"]>0:x.append("new accounts are joining")
        if r["unique_ratio"]>=.60:x.append("discussion is relatively broad")
        if r["engagement_growth"]>.20:x.append("engagement jumped")
        if r["influencer_event"]==1:x.append("influencer activity detected")
        return x or ["no strong emerging-wave signal"]
    d["why"]=d.apply(why,axis=1)
    return d

st.sidebar.header("Data")
uploaded=st.sidebar.file_uploader("Upload CSV",type=["csv"])
raw=pd.read_csv(uploaded) if uploaded else demo
if uploaded is None: st.sidebar.warning("Demo mode: synthetic data — not live X data.")
required={"time","topic","mentions","unique_accounts","engagement","influencer_event"}
missing=required-set(raw.columns)
if missing:
    st.error("Missing columns: "+", ".join(sorted(missing))); st.stop()

result=calc(raw)
latest=result.sort_values("time").groupby("topic",as_index=False).tail(1).sort_values("early_wave_score",ascending=False)

st.subheader("🚨 Emerging attention candidates")
table=latest[["topic","early_wave_score","mentions","mention_velocity","mention_acceleration","unique_accounts","unique_ratio","status"]].copy()
table["early_wave_score"]=table["early_wave_score"].round(0)
table["unique_ratio"]=(table["unique_ratio"]*100).round(0).astype(str)+"%"
table["status"]=table["status"].astype(str)
st.dataframe(table,use_container_width=True,hide_index=True)

selected=st.sidebar.selectbox("Inspect topic",result["topic"].astype(str).unique().tolist())
view=result[result["topic"].astype(str)==selected].copy()
last=view.iloc[-1]

a,b,c,d=st.columns(4)
a.metric("Early-wave score",f"{last['early_wave_score']:.0f}/100")
b.metric("Mentions",f"{int(last['mentions']):,}")
c.metric("Unique accounts",f"{int(last['unique_accounts']):,}")
d.metric("Discussion breadth",f"{last['unique_ratio']*100:.0f}%")

status=str(last["status"])
if status=="EXTREME": st.error("EXTREME emerging-wave dynamics detected.")
elif status=="HIGH": st.warning("HIGH emerging-wave dynamics detected.")
elif status=="EMERGING": st.info("EMERGING attention-wave dynamics detected.")
else: st.success("NORMAL attention dynamics.")

st.markdown("### Why is this topic being flagged?")
for reason in last["why"]: st.write("• "+reason.capitalize())

st.subheader("Early-wave components")
st.line_chart(view.set_index("time")[["velocity_component","acceleration_component","unique_component","engagement_component","diversity_component"]])
st.subheader("Raw activity")
st.line_chart(view.set_index("time")[["mentions","unique_accounts","engagement"]])
st.subheader("Detector details")
cols=["time","mentions","unique_accounts","unique_ratio","engagement","mention_velocity","mention_acceleration","engagement_growth","early_wave_score","status","influencer_event"]
st.dataframe(view[cols],use_container_width=True,hide_index=True)

st.subheader("V1.2 scoring model")
st.markdown("- **25%** mention acceleration\n- **20%** mention velocity\n- **20%** unique-account growth\n- **15%** engagement growth\n- **10%** discussion breadth\n- **10%** influencer event")
st.caption("Weights are experimental and must be validated against historical examples and false positives.")

st.download_button("Download V1.2 demo CSV",demo.to_csv(index=False).encode(),"x_attention_demo_v1_2.csv","text/csv")
st.caption("Synthetic demo data only. Real-time X monitoring requires an appropriate, authorized data source. This tool does not execute trades or provide buy/sell recommendations.")
