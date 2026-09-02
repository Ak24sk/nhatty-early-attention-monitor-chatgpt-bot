import numpy as np
import pandas as pd

def _norm(s):
    s=pd.to_numeric(s,errors="coerce").fillna(0.0)
    if s.max()==s.min(): return pd.Series(0.0,index=s.index)
    return (s-s.min())/(s.max()-s.min())*100

def calculate_attention(df):
    required=["time","topic","mentions","unique_accounts","engagement","influencer_event"]
    if any(c not in df.columns for c in required):
        return pd.DataFrame(columns=["topic","attention_score","stage","alert"])
    d=df.copy()
    d["time"]=pd.to_datetime(d["time"],errors="coerce")
    d=d.dropna(subset=["time"]).sort_values(["topic","time"])
    for c in required[2:]:
        d[c]=pd.to_numeric(d[c],errors="coerce").fillna(0)
    g=d.groupby("topic",group_keys=False)
    d["mention_velocity"]=g["mentions"].diff().fillna(0)
    d["mention_accel"]=g["mention_velocity"].diff().fillna(0)
    d["unique_accel"]=g["unique_accounts"].diff().diff().fillna(0)
    d["engagement_growth"]=g["engagement"].diff().fillna(0)
    d["baseline"]=g["mentions"].transform(lambda s:s.shift(1).rolling(5,min_periods=2).mean()).fillna(0)
    std=g["mentions"].transform(lambda s:s.shift(1).rolling(5,min_periods=2).std()).replace(0,np.nan)
    d["anomaly"]=((d["mentions"]-d["baseline"])/std).replace([np.inf,-np.inf],0).fillna(0)
    d["score"]=.32*_norm(d["mention_accel"])+.20*_norm(d["mention_velocity"])+.16*_norm(d["unique_accel"])+.12*_norm(d["anomaly"])+.12*_norm(d["engagement_growth"])+.08*_norm(d["influencer_event"])
    last=d.groupby("topic",as_index=False).tail(1).copy()
    last["attention_score"]=last["score"].clip(0,100).round(1)
    last["stage"]=np.select([last["attention_score"]>=75,last["attention_score"]>=55,last["attention_score"]>=35],["EARLY WAVE","GROWING","BUILDING"],default="QUIET")
    last["alert"]=np.select([last["attention_score"]>=75,last["attention_score"]>=55],["EARLY WAVE","WATCH"],default="NORMAL")
    return last[["topic","attention_score","stage","alert"]].sort_values("attention_score",ascending=False)

def _bool(v, expected):
    if isinstance(v,(bool,np.bool_)): return bool(v)==expected
    if isinstance(v,str):
        return v.strip().lower() in ({"true","1","yes","active"} if expected else {"false","0","no","revoked"})
    return False

def _clean(v, allowed):
    return str(v).strip().upper() in allowed

def safety_gate(df,min_liquidity=50000,min_lp_lock_days=30,max_top10=40,max_dev=10,max_cluster=50):
    if df.empty: return pd.DataFrame(columns=["token","status","score","reasons"])
    rows=[]
    for _,r in df.iterrows():
        checks=[]; reasons=[]
        def check(ok,label):
            checks.append(bool(ok))
            if not ok: reasons.append(label)

        liq=pd.to_numeric(r.get("liquidity_usd"),errors="coerce")
        lock=pd.to_numeric(r.get("lp_lock_days"),errors="coerce")
        top=pd.to_numeric(r.get("top10_holder_pct"),errors="coerce")
        dev=pd.to_numeric(r.get("dev_holder_pct"),errors="coerce")
        cluster=pd.to_numeric(r.get("cluster_holder_pct"),errors="coerce")

        check(pd.notna(liq) and liq>=min_liquidity,"liquidity")
        check(_bool(r.get("lp_locked"),True),"LP not locked")
        check(pd.notna(lock) and lock>=min_lp_lock_days,"LP lock duration")
        check(not _bool(r.get("mint_authority_active"),True),"mint authority active")
        check(not _bool(r.get("freeze_authority_active"),True),"freeze authority active")
        check(pd.notna(top) and top<=max_top10,"top-10 concentration")
        check(pd.notna(dev) and dev<=max_dev,"dev concentration")
        check(pd.notna(cluster) and cluster<=max_cluster,"related-wallet cluster concentration")
        check(_bool(r.get("sellable"),True),"sellability")
        check(_clean(r.get("wash_trade_risk"),{"LOW","CLEAN","NONE","FALSE","NO"}),"wash-trade risk")
        check(_clean(r.get("bundle_risk"),{"LOW","CLEAN","NONE","FALSE","NO"}),"bundle/sniper risk")
        check(_clean(r.get("dev_risk"),{"LOW","CLEAN","NONE"}),"dev risk")
        check(_clean(r.get("contract_risk"),{"LOW","CLEAN","NONE"}),"contract risk")

        rows.append({
            "token":r.get("token",""),
            "status":"PASSED" if all(checks) else "FAILED",
            "score":round(100*sum(checks)/len(checks),1),
            "liquidity_usd":liq,
            "lp_locked":r.get("lp_locked"),
            "lp_lock_days":lock,
            "mint_authority_active":r.get("mint_authority_active"),
            "freeze_authority_active":r.get("freeze_authority_active"),
            "sellable":r.get("sellable"),
            "top10_holder_pct":top,
            "dev_holder_pct":dev,
            "cluster_holder_pct":cluster,
            "wash_trade_risk":r.get("wash_trade_risk"),
            "bundle_risk":r.get("bundle_risk"),
            "dev_risk":r.get("dev_risk"),
            "contract_risk":r.get("contract_risk"),
            "reasons":"PASS" if not reasons else "; ".join(sorted(set(reasons)))
        })
    return pd.DataFrame(rows)

def trader_scores(trades):
    required=["wallet","time","token","side","usd"]
    if trades.empty or any(c not in trades.columns for c in required):
        return pd.DataFrame(columns=["wallet","trader_score","classification"])
    d=trades.copy()
    d["side"]=d["side"].astype(str).str.upper()
    d["usd"]=pd.to_numeric(d["usd"],errors="coerce").fillna(0)
    for c in ["early_runner","win"]:
        if c not in d.columns: d[c]=False
        d[c]=d[c].astype(str).str.lower().isin(["true","1","yes"])
    rows=[]
    for wallet,g in d.groupby("wallet"):
        b=g[g.side=="BUY"]; n=len(b)
        early=b.early_runner.mean()*100 if n else 0
        win=b.win.mean()*100 if n else 0
        consistency=min(100,np.sqrt(n)*20)
        volume=min(100,np.log1p(b.usd.sum())/np.log1p(100000)*100)
        score=.45*early+.35*win+.15*consistency+.05*volume
        rows.append({"wallet":wallet,"trades":n,"early_runner_pct":round(early,1),"win_pct":round(win,1),"consistency":round(consistency,1),"activity_score":round(volume,1),"trader_score":round(score,1),"classification":"ELITE EARLY-RUNNER" if score>=70 else "PROMISING" if score>=50 else "UNVERIFIED"})
    return pd.DataFrame(rows).sort_values("trader_score",ascending=False)

def trader_convergence(trades,trader_table,window_minutes=15,elite_threshold=70):
    if trades.empty or trader_table.empty: return pd.DataFrame(columns=["token","elite_buyers","avg_trader_score","convergence_score"])
    d=trades.copy(); d["time"]=pd.to_datetime(d["time"],errors="coerce"); d["side"]=d.side.astype(str).str.upper()
    d=d[d.side=="BUY"].dropna(subset=["time"])
    e=trader_table[trader_table.trader_score>=elite_threshold][["wallet","trader_score"]]
    d=d.merge(e,on="wallet",how="inner")
    rows=[]
    for token,g in d.groupby("token"):
        g=g.sort_values("time"); best=[]; best_count=0
        for _,row in g.iterrows():
            w=g[(g.time>=row.time)&(g.time<=row.time+pd.Timedelta(minutes=window_minutes))]
            if w.wallet.nunique()>best_count: best_count=w.wallet.nunique(); best=w.trader_score.tolist()
        avg=float(np.mean(best)) if best else 0
        rows.append({"token":token,"elite_buyers":best_count,"avg_trader_score":round(avg,1),"convergence_score":round(min(100,best_count*20+avg*.6),1)})
    return pd.DataFrame(rows).sort_values("convergence_score",ascending=False)

def dev_history(safety):
    if safety.empty: return pd.DataFrame(columns=["token","dev_score","dev_interpretation"])
    rows=[]
    for _,r in safety.iterrows():
        risk=str(r.dev_risk).upper(); pct=pd.to_numeric(r.dev_holder_pct,errors="coerce")
        score=50+(35 if risk in {"LOW","CLEAN","NONE"} else -40 if risk in {"HIGH","SUSPICIOUS","CRITICAL"} else 0)
        if pd.notna(pct): score+=max(-20,min(10,10-pct))
        score=float(np.clip(score,0,100))
        rows.append({"token":r.token,"dev_score":round(score,1),"dev_interpretation":"LOW-RISK PATTERN" if score>=70 else "REVIEW" if score>=45 else "HIGH-RISK PATTERN"})
    return pd.DataFrame(rows)

def build_signal(safety,traders,convergence,dev,attention):
    if safety.empty:return pd.DataFrame()
    b=safety[["token","status","score"]].rename(columns={"status":"safety_status","score":"safety_score"}).copy()
    for table,cols in [(convergence,["token","elite_buyers","convergence_score"]),(dev,["token","dev_score"])]:
        if not table.empty:b=b.merge(table[cols],on="token",how="left")
    if attention.empty:b["attention_score"]=0
    else:b=b.merge(attention.rename(columns={"topic":"token"})[["token","attention_score"]],on="token",how="left")
    for c in ["elite_buyers","convergence_score","dev_score","attention_score"]: 
        if c not in b:b[c]=0
    b=b.fillna(0)
    b["unified_score"]=(.55*b.convergence_score+.20*b.dev_score+.25*b.attention_score).round(1)
    b["signal"]=np.where(b.safety_status!="PASSED","BLOCKED",np.where(b.unified_score>=75,"ALERT",np.where(b.unified_score>=55,"WATCH","BUILDING")))
    b["reasons"]=[
        ("safety gate failed" if r.safety_status!="PASSED" else "")+
        (f" | {int(r.elite_buyers)} elite wallets converged" if r.elite_buyers>=3 else f" | {int(r.elite_buyers)} elite wallet(s)" if r.elite_buyers>0 else "")+
        (" | X attention wave" if r.attention_score>=75 else " | X attention growing" if r.attention_score>=55 else "")
        for _,r in b.iterrows()
    ]
    return b.sort_values("unified_score",ascending=False)
