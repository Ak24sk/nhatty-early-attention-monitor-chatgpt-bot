"""
Reddit connector for X Early Attention Monitor.

Uses PRAW (official Reddit API wrapper) in read-only mode. Requires a free
Reddit "script" app registered at https://www.reddit.com/prefs/apps
(client_id + client_secret, no user login needed for read-only search).

This does NOT scrape or bypass any access controls — it uses Reddit's
public API within its documented rate limits.
"""

from datetime import datetime, timedelta, timezone
import pandas as pd

try:
    import praw
except ImportError:
    praw = None


def get_reddit_client(client_id: str, client_secret: str, user_agent: str = "x-attention-monitor/1.5"):
    """Create a read-only PRAW client. No password/login required."""
    if praw is None:
        raise RuntimeError("praw is not installed. Add 'praw' to requirements.txt.")
    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
    )
    reddit.read_only = True
    return reddit


def fetch_reddit_data(
    reddit,
    topics: list[str],
    subreddits: list[str] | None = None,
    hours_back: int = 24,
    influencer_min_karma: int = 50000,
    limit_per_topic: int = 500,
) -> pd.DataFrame:
    """
    Search Reddit for each topic keyword, bucket results into hourly rows
    matching the app's schema: time, topic, mentions, unique_accounts,
    engagement, influencer_event.

    subreddits: list of subreddit names to restrict search to (e.g. ["CryptoMoonShots", "SatoshiStreetBets"]).
                If None, searches all of Reddit.
    influencer_min_karma: an author with combined karma above this, posting about
                the topic, marks that hour as an influencer_event.
    """
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=hours_back)
    scope = "+".join(subreddits) if subreddits else "all"

    rows = []
    for topic in topics:
        subreddit = reddit.subreddit(scope)
        buckets = {}  # hour_bucket -> {"mentions":.., "authors": set(), "engagement":.., "influencer":False}

        results = list(subreddit.search(topic, sort="new", time_filter="week", limit=limit_per_topic))
        for post in results:
            created = datetime.fromtimestamp(post.created_utc, tz=timezone.utc)
            if created < cutoff:
                continue
            hour = created.replace(minute=0, second=0, microsecond=0)
            b = buckets.setdefault(hour, {"mentions": 0, "authors": set(), "engagement": 0, "influencer": 0})
            b["mentions"] += 1
            author_name = str(post.author) if post.author else "deleted"
            b["authors"].add(author_name)
            b["engagement"] += (post.score or 0) + (post.num_comments or 0)

            if post.author is not None:
                try:
                    karma = (post.author.link_karma or 0) + (post.author.comment_karma or 0)
                    if karma >= influencer_min_karma:
                        b["influencer"] = 1
                except Exception:
                    pass  # suspended/shadowbanned authors raise on attribute access

        for hour, b in buckets.items():
            rows.append({
                "time": hour,
                "topic": topic,
                "mentions": b["mentions"],
                "unique_accounts": len(b["authors"]),
                "engagement": b["engagement"],
                "influencer_event": b["influencer"],
            })

    if not rows:
        return pd.DataFrame(columns=["time", "topic", "mentions", "unique_accounts", "engagement", "influencer_event"])

    df = pd.DataFrame(rows).sort_values(["topic", "time"]).reset_index(drop=True)
    return df
