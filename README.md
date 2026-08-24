# X Early Attention Monitor — Version 1.1

V1.1 emphasizes attention change and acceleration instead of raw popularity.

New:
- Candidate/topic overview
- Stronger mention-acceleration weighting
- Unique-account and engagement components
- Explanation of why a score was produced
- Component and activity charts
- Three synthetic comparison topics

Score: 45% mention acceleration, 30% unique-account growth, 15% engagement growth, 10% influencer event.

The demo data is synthetic, not historical X data. Required CSV columns:
`time,topic,mentions,unique_accounts,engagement,influencer_event`

Run locally with:
`pip install -r requirements.txt`
`streamlit run app.py`
