# Early Runner Intelligence V2.1

V2.1 adds a **real Solana JSON-RPC data adapter** to V2.0.

## Live capability

Enter a Solana token mint and an RPC endpoint. The app queries:
- `getAccountInfo` with parsed data
- `getTokenSupply`
- `getTokenLargestAccounts`

Solana documents these RPC account/token methods and their returned structures. The adapter uses the public RPC interface rather than scraping a trading terminal.

## What V2.1 does NOT claim

The basic RPC snapshot does not automatically prove:
- liquidity amount
- LP lock status/duration
- sellability/honeypot status
- wallet relationships / BubbleMaps-style clusters
- wash trading
- bundle/sniper behavior
- historical dev performance
- trader profitability

Those need additional indexed/on-chain or specialized data sources.

## Safety philosophy

Missing critical data = **FAILED / BLOCKED**.

A `PASSED` status means the configured checks passed; it never means the token is guaranteed safe or profitable.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Next target: V2.2

The next build should add a proper indexed Solana transaction layer:
1. token creation/deployer history
2. wallet transaction history
3. first-entry timing
4. holder concentration over time
5. DEX liquidity/pool discovery
6. buy/sell flow and unique buyers
7. bundle/sniper heuristics
8. reusable wallet reputation database
9. persistent scoring/history
10. alerts
