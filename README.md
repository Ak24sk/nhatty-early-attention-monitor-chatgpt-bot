# X Early Attention Monitor — Version 1.2

V1.2 focuses on detecting emerging attention waves rather than simply ranking popular topics.

New: candidate ranking, mention velocity, acceleration, unique-account growth, discussion breadth, engagement growth, influencer-event signal, a modest small-topic adjustment, and explanations.

Experimental weights: 25% acceleration, 20% velocity, 20% unique-account growth, 15% engagement growth, 10% discussion breadth, 10% influencer event.

The included demo data is synthetic and is NOT historical X data.

Required CSV columns:
`time,topic,mentions,unique_accounts,engagement,influencer_event`
