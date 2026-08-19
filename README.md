
# X Early Attention Monitor — Version 1

A zero-cost local research prototype for studying unusual attention dynamics in an X-style dataset.

## Included

- Attention score (0–100)
- Mention velocity
- Mention acceleration
- Unique-account growth
- Engagement growth
- Influencer-event signal
- NORMAL / EMERGING / HIGH / EXTREME classification
- Timeline chart
- CSV upload
- Synthetic demonstration data

## Important

The included Peanut example is synthetic demonstration data. It is NOT a historical archive of actual X posts and should not be interpreted as exact PNUT measurements.

The application does not connect to X automatically, does not place trades, and does not provide buy/sell signals.

## CSV format

Your CSV must contain:

time,topic,mentions,unique_accounts,engagement,influencer_event

Example:

2026-08-19 10:00,Example,8,7,90,0

`influencer_event` should be 0 or 1.

## Run locally

1. Install Python.
2. Open a terminal in this folder.
3. Run:

   pip install -r requirements.txt

4. Then run:

   streamlit run app.py

5. Open the local address Streamlit gives you in your browser.

## Next development stage

After the prototype is validated, we can replace the synthetic data with an appropriate authorized X data source and test the scoring model against historical cases.
