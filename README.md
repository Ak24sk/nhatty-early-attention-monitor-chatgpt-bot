# Early Runner Intelligence v2.2

V2.2 adds a live Solana wallet scanner, wallet intelligence, token-balance heuristics, wallet convergence, and a unified safety/X/convergence signal board.

Run:
`pip install -r requirements.txt`
`streamlit run app.py`

The wallet scanner uses Solana RPC transaction history. Solana documents `getSignaturesForAddress` as returning confirmed transaction signatures and `getTransaction` as returning confirmed transaction details.

Important: LIKELY BUY/SELL are balance-change heuristics, not guaranteed DEX trades and not exact PnL. Public RPC alone cannot prove LP locks, sellability, holder relationships, wash trading, bundle/sniper certainty, or other unavailable checks. Unknown critical safety data remains UNVERIFIED. PASSED is not a guarantee of safety or profitability.

Never enter a seed phrase or private key. This project does not sign or execute trades.
