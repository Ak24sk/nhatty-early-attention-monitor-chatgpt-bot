import numpy as np
import pandas as pd

def _norm(s):
    s = pd.to_numeric(s, errors="coerce").fillna(0.0)
    if s.max() == s.min():
        return pd.Series(0.0, index=s.index)
    return (s - s.min()) / (s.max() - s.min()) * 100.0

def calculate_attention(df):
    """V1.6-style attention engine, adapted for V2.0."""
    required = ["time","topic","mentions","unique_accounts","engagement","influencer_event"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return pd.DataFrame(columns=["topic","attention_score","stage","alert"])

    d = df.copy()
    d["time"] = pd.to_datetime(d["time"], errors="coerce")
    d = d.dropna(subset=["time"]).sort_values(["topic","time"])

    for c in ["mentions","unique_accounts","engagement","influencer_event"]:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0)

    g = d.groupby("topic", group_keys=False)
    d["mention_velocity"] = g["mentions"].diff().fillna(0)
    d["mention_accel"] = g["mention_velocity"].diff().fillna(0)
    d["unique_velocity"] = g["unique_accounts"].diff().fillna(0)
    d["unique_accel"] = g["unique_velocity"].diff().fillna(0)
    d["engagement_growth"] = g["engagement"].diff().fillna(0)

    d["baseline"] = g["mentions"].transform(lambda s: s.shift(1).rolling(5, min_periods=2).mean())
    d["baseline_std"] = g["mentions"].transform(lambda s: s.shift(1).rolling(5, min_periods=2).std())
    d["baseline"] = d["baseline"].fillna(0)
    d["baseline_std"] = d["baseline_std"].replace(0, np.nan)
    d["anomaly"] = ((d["mentions"] - d["baseline"]) / d["baseline_std"]).replace([np.inf,-np.inf],0).fillna(0)

    d["score"] = (
        0.32 * _norm(d["mention_accel"]) +
        0.20 * _norm(d["mention_velocity"]) +
        0.16 * _norm(d["unique_accel"]) +
        0.12 * _norm(d["anomaly"]) +
        0.12 * _norm(d["engagement_growth"]) +
        0.08 * _norm(d["influencer_event"])
    )

    last = d.groupby("topic", as_index=False).tail(1).copy()
    last["attention_score"] = last["score"].clip(0,100).round(1)

    last["stage"] = np.select(
        [
            last["attention_score"] >= 75,
            last["attention_score"] >= 55,
            last["attention_score"] >= 35
        ],
        ["EARLY WAVE","GROWING","BUILDING"],
        default="QUIET"
    )

    last["alert"] = np.select(
        [
            last["attention_score"] >= 75,
            last["attention_score"] >= 55
        ],
        ["EARLY WAVE","WATCH"],
        default="NORMAL"
    )

    return last[["topic","attention_score","stage","alert"]].sort_values("attention_score", ascending=False)

def _risk_ok(value, clean_values):
    if pd.isna(value):
        return False
    return str(value).strip().upper() in clean_values

def safety_gate(
    df,
    min_liquidity=50000,
    min_lp_lock_days=30,
    max_top10=40,
    max_dev=10,
    max_cluster=50,
):
    """Hard-gate safety model. Missing/unknown critical data fails verification."""
    if df.empty:
        return pd.DataFrame(columns=["token","status","score","reasons"])

    d = df.copy()
    defaults = {
        "wash_trade_risk": "UNKNOWN",
        "bundle_risk": "UNKNOWN",
        "dev_risk": "UNKNOWN",
        "contract_risk": "UNKNOWN",
    }
    for c, val in defaults.items():
        if c not in d.columns:
            d[c] = val

    rows = []
    for _, r in d.iterrows():
        reasons = []
        checks = []

        def req_bool(col, expected=False, label=None):
            val = r.get(col, np.nan)
            ok = isinstance(val, (bool, np.bool_)) and val == expected
            # accept textual true/false values
            if isinstance(val, str):
                low = val.strip().lower()
                if expected is False:
                    ok = low in {"false","0","no","revoked","locked"}
                else:
                    ok = low in {"true","1","yes"}
            checks.append(ok)
            if not ok:
                reasons.append(label or col)

        liq = pd.to_numeric(r.get("liquidity_usd", np.nan), errors="coerce")
        lock_days = pd.to_numeric(r.get("lp_lock_days", np.nan), errors="coerce")
        top10 = pd.to_numeric(r.get("top10_holder_pct", np.nan), errors="coerce")
        dev_pct = pd.to_numeric(r.get("dev_holder_pct", np.nan), errors="coerce")
        cluster = pd.to_numeric(r.get("cluster_holder_pct", np.nan), errors="coerce")

        ok = pd.notna(liq) and liq >= min_liquidity
        checks.append(ok)
        if not ok: reasons.append("liquidity")

        req_bool("lp_locked", False, "LP lock")
        # The boolean logic above expects False for "no"; handle LP explicitly.
        lp_locked = r.get("lp_locked", np.nan)
        lp_ok = False
        if isinstance(lp_locked, (bool, np.bool_)):
            lp_ok = bool(lp_locked)
        elif isinstance(lp_locked, str):
            lp_ok = lp_locked.strip().lower() in {"true","1","yes","locked"}
        checks[-1] = lp_ok
        if not lp_ok:
            reasons.append("LP not locked")

        ok = pd.notna(lock_days) and lock_days >= min_lp_lock_days
        checks.append(ok)
        if not ok: reasons.append("LP lock duration")

        req_bool("mint_authority_active", False, "mint authority active")
        req_bool("freeze_authority_active", False, "freeze authority active")

        ok = pd.notna(top10) and top10 <= max_top10
        checks.append(ok)
        if not ok: reasons.append("top-10 concentration")

        ok = pd.notna(dev_pct) and dev_pct <= max_dev
        checks.append(ok)
        if not ok: reasons.append("dev concentration")

        ok = pd.notna(cluster) and cluster <= max_cluster
        checks.append(ok)
        if not ok: reasons.append("related-wallet cluster concentration")

        req_bool("sellable", True, "sellability")

        # Risk fields must explicitly indicate a clean/low state.
        wash_ok = _risk_ok(r["wash_trade_risk"], {"LOW","CLEAN","NONE","FALSE","NO"})
        bundle_ok = _risk_ok(r["bundle_risk"], {"LOW","CLEAN","NONE","FALSE","NO"})
        dev_ok = _risk_ok(r["dev_risk"], {"LOW","CLEAN","NONE"})
        contract_ok = _risk_ok(r["contract_risk"], {"LOW","CLEAN","NONE"})

        for ok, label in [
            (wash_ok, "wash-trade risk"),
            (bundle_ok, "bundle/sniper risk"),
            (dev_ok, "dev risk"),
            (contract_ok, "contract risk"),
        ]:
            checks.append(ok)
            if not ok: reasons.append(label)

        status = "PASSED" if all(checks) else "FAILED"
        score = round(100 * sum(checks) / max(len(checks),1), 1)

        rows.append({
            "token": r.get("token",""),
            "status": status,
            "score": score,
            "liquidity_usd": liq,
            "lp_locked": lp_ok,
            "lp_lock_days": lock_days,
            "mint_authority_active": r.get("mint_authority_active", np.nan),
            "freeze_authority_active": r.get("freeze_authority_active", np.nan),
            "sellable": r.get("sellable", np.nan),
            "top10_holder_pct": top10,
            "dev_holder_pct": dev_pct,
            "cluster_holder_pct": cluster,
            "wash_trade_risk": r["wash_trade_risk"],
            "bundle_risk": r["bundle_risk"],
            "dev_risk": r["dev_risk"],
            "contract_risk": r["contract_risk"],
            "reasons": "PASS" if not reasons else "; ".join(sorted(set(reasons))),
        })
    return pd.DataFrame(rows)

def trader_scores(trades):
    """Score wallets from supplied historical trade labels/behavior."""
    required = ["wallet","time","token","side","usd"]
    missing = [c for c in required if c not in trades.columns]
    if missing:
        return pd.DataFrame(columns=["wallet","trader_score","classification"])

    d = trades.copy()
    d["time"] = pd.to_datetime(d["time"], errors="coerce")
    d["usd"] = pd.to_numeric(d["usd"], errors="coerce").fillna(0)
    d["side"] = d["side"].astype(str).str.upper()

    if "early_runner" not in d.columns:
        d["early_runner"] = False
    if "win" not in d.columns:
        d["win"] = False

    d["early_runner"] = d["early_runner"].astype(str).str.lower().isin(["true","1","yes"])
    d["win"] = d["win"].astype(str).str.lower().isin(["true","1","yes"])

    rows = []
    for wallet, g in d.groupby("wallet"):
        buys = g[g["side"]=="BUY"]
        n = len(buys)
        early = buys["early_runner"].mean() * 100 if n else 0
        win = buys["win"].mean() * 100 if n else 0
        consistency = min(100, np.sqrt(max(n,0)) * 20)
        volume = min(100, np.log1p(buys["usd"].sum()) / np.log1p(100000) * 100)

        # Historical labels matter most; volume is intentionally low weight.
        score = 0.45*early + 0.35*win + 0.15*consistency + 0.05*volume
        if n == 0:
            score = 0

        classification = (
            "ELITE EARLY-RUNNER" if score >= 70
            else "PROMISING" if score >= 50
            else "UNVERIFIED"
        )

        rows.append({
            "wallet": wallet,
            "trades": n,
            "early_runner_pct": round(early,1),
            "win_pct": round(win,1),
            "consistency": round(consistency,1),
            "activity_score": round(volume,1),
            "trader_score": round(score,1),
            "classification": classification,
        })
    return pd.DataFrame(rows).sort_values("trader_score", ascending=False)

def trader_convergence(trades, trader_table, window_minutes=15, elite_threshold=70):
    if trades.empty or trader_table.empty:
        return pd.DataFrame(columns=["token","elite_buyers","avg_trader_score","convergence_score"])

    d = trades.copy()
    d["time"] = pd.to_datetime(d["time"], errors="coerce")
    d["side"] = d["side"].astype(str).str.upper()
    d = d[d["side"]=="BUY"].dropna(subset=["time"])

    elite = trader_table[trader_table["trader_score"] >= elite_threshold][["wallet","trader_score"]]
    d = d.merge(elite, on="wallet", how="inner")
    if d.empty:
        return pd.DataFrame(columns=["token","elite_buyers","avg_trader_score","convergence_score"])

    rows = []
    for token, g in d.groupby("token"):
        g = g.sort_values("time")
        best_count = 0
        best_scores = []
        best_start = None

        for i, row in g.iterrows():
            start = row["time"]
            end = start + pd.Timedelta(minutes=window_minutes)
            w = g[(g["time"] >= start) & (g["time"] <= end)]
            wallets = w["wallet"].nunique()
            if wallets > best_count:
                best_count = wallets
                best_scores = w["trader_score"].tolist()
                best_start = start

        avg_score = float(np.mean(best_scores)) if best_scores else 0
        convergence_score = min(100, best_count*20 + avg_score*0.6)

        rows.append({
            "token": token,
            "elite_buyers": best_count,
            "avg_trader_score": round(avg_score,1),
            "window_start": best_start,
            "window_minutes": window_minutes,
            "convergence_score": round(convergence_score,1),
        })

    return pd.DataFrame(rows).sort_values("convergence_score", ascending=False)

def dev_history(safety):
    if safety.empty:
        return pd.DataFrame(columns=["token","dev_score","dev_interpretation"])

    rows = []
    for _, r in safety.iterrows():
        risk = str(r.get("dev_risk","UNKNOWN")).upper()
        concentration = pd.to_numeric(r.get("dev_holder_pct", np.nan), errors="coerce")
        score = 50.0
        if risk in {"LOW","CLEAN","NONE"}: score += 35
        elif risk in {"HIGH","SUSPICIOUS","CRITICAL"}: score -= 40
        if pd.notna(concentration):
            score += max(-20, min(10, 10 - concentration))
        score = float(np.clip(score, 0,100))
        interpretation = "LOW-RISK PATTERN" if score >= 70 else "REVIEW" if score >= 45 else "HIGH-RISK PATTERN"
        rows.append({
            "token": r["token"],
            "dev_score": round(score,1),
            "dev_interpretation": interpretation,
            "dev_risk": risk,
        })
    return pd.DataFrame(rows)

def build_signal(safety, traders, convergence, dev, attention):
    if safety.empty:
        return pd.DataFrame()

    base = safety[["token","status","score"]].rename(
        columns={"status":"safety_status","score":"safety_score"}
    ).copy()

    if not convergence.empty:
        base = base.merge(
            convergence[["token","elite_buyers","convergence_score"]],
            on="token", how="left"
        )
    else:
        base["elite_buyers"] = 0
        base["convergence_score"] = 0

    if not dev.empty:
        base = base.merge(dev[["token","dev_score"]], on="token", how="left")
    else:
        base["dev_score"] = 0

    if not attention.empty:
        a = attention.rename(columns={"topic":"token"})
        base = base.merge(a[["token","attention_score"]], on="token", how="left")
    else:
        base["attention_score"] = 0

    base = base.fillna(0)

    base["unified_score"] = (
        0.55*base["convergence_score"] +
        0.20*base["dev_score"] +
        0.25*base["attention_score"]
    ).round(1)

    # Safety is a hard gate, not an additive opportunity score.
    base["signal"] = np.where(
        base["safety_status"] != "PASSED",
        "BLOCKED",
        np.where(base["unified_score"] >= 75, "ALERT",
                 np.where(base["unified_score"] >= 55, "WATCH", "BUILDING"))
    )

    reasons = []
    for _, r in base.iterrows():
        rr = []
        if r["safety_status"] != "PASSED":
            rr.append("safety gate failed")
        if r["elite_buyers"] >= 3:
            rr.append(f"{int(r['elite_buyers'])} elite wallets converged")
        elif r["elite_buyers"] > 0:
            rr.append(f"{int(r['elite_buyers'])} elite wallet(s)")
        if r["attention_score"] >= 75:
            rr.append("X attention wave")
        elif r["attention_score"] >= 55:
            rr.append("X attention growing")
        if r["dev_score"] >= 70:
            rr.append("dev pattern favorable")
        reasons.append(" | ".join(rr) if rr else "insufficient confirmation")

    base["reasons"] = reasons
    return base.sort_values(["signal","unified_score"], ascending=[True,False])
