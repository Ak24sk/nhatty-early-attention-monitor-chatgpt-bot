# Early Runner Intelligence V2.0

V2.0 extends the original X Early Attention Monitor into a multi-signal research system:

**Safety Gate → Elite Early-Runner Traders → Dev Wallet Intelligence → X Attention → Trader Convergence → Alert**

## What changed from V1.6

- Added a hard Safety Gate.
- Added elite early-runner trader scoring.
- Added trader convergence detection.
- Added dev/associated-wallet intelligence.
- Kept the X attention engine.
- Added a unified signal board.
- Added explicit BLOCKED / WATCH / BUILDING / ALERT states.
- Unknown or missing critical safety information is treated as a failure, not as a pass.
- No trade execution.

## Important limitation

This version is **data-source ready**, not falsely presented as a live Axiom/FOMO feed.

It accepts CSV data so the architecture can be tested now. A later version can connect legitimate, authorized public/on-chain data sources. The app does not scrape X, bypass access controls, copy private platform APIs, or execute trades.

A token that passes the configured gate has only passed the supplied/configured checks. It is not guaranteed to be safe or profitable.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## CSV schemas

### Token Safety CSV

Required:
- token
- liquidity_usd
- lp_locked
- lp_lock_days
- mint_authority_active
- freeze_authority_active
- sellable
- top10_holder_pct
- dev_holder_pct
- cluster_holder_pct

Recommended:
- wash_trade_risk
- bundle_risk
- dev_risk
- contract_risk

### Trader Trades CSV

Required:
- wallet
- time
- token
- side
- usd

Recommended historical labels:
- early_runner
- win

### X Attention CSV

- time
- topic
- mentions
- unique_accounts
- engagement
- influencer_event

## Architecture

1. Safety Gate
2. Elite Early-Runner Trader Detection
3. Dev/Associated Wallet Intelligence
4. X Attention
5. On-chain confirmation (future live adapter)
6. Convergence Engine
7. Alert

## V2.0 philosophy

The goal is not to find the token with the most attention.

The goal is to find an **early opportunity with independent confirmation while blocking tokens that fail the safety gate**.
