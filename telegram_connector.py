"""
Telegram connector for X Early Attention Monitor.

Uses Telethon (official Telegram client library) to read messages from
PUBLIC channels/groups you specify. This does not bypass any access
controls — it reads content Telegram already makes publicly readable,
using your own API credentials from https://my.telegram.org (free).

ONE-TIME SETUP (run locally, not on Streamlit Cloud):
    from telegram_connector import generate_session_string
    generate_session_string(api_id=..., api_hash=...)
This will prompt for your phone number + login code once, then print a
session string. Save that string as a Streamlit secret (TELEGRAM_SESSION)
so the deployed app never needs to log in interactively again.
"""

from datetime import datetime, timedelta, timezone
import pandas as pd

try:
    from telethon.sync import TelegramClient
    from telethon.sessions import StringSession
except ImportError:
    TelegramClient = None
    StringSession = None


def generate_session_string(api_id: int, api_hash: str) -> str:
    """Run this ONCE, locally, interactively. Prints a session string to save as a secret."""
    if TelegramClient is None:
        raise RuntimeError("telethon is not installed. Add 'telethon' to requirements.txt.")
    with TelegramClient(StringSession(), api_id, api_hash) as client:
        session_str = client.session.save()
        print("Save this as your TELEGRAM_SESSION secret:\n")
        print(session_str)
        return session_str


def get_telegram_client(api_id: int, api_hash: str, session_string: str) -> "TelegramClient":
    """Create a Telegram client from a saved session string (no interactive login)."""
    if TelegramClient is None:
        raise RuntimeError("telethon is not installed. Add 'telethon' to requirements.txt.")
    client = TelegramClient(StringSession(session_string), api_id, api_hash)
    client.connect()
    return client


def fetch_telegram_data(
    client,
    topics: list[str],
    channels: list[str],
    hours_back: int = 24,
    influencer_min_subscribers: int = 50000,
    limit_per_channel: int = 1000,
) -> pd.DataFrame:
    """
    Scan given public channels/groups for messages mentioning each topic
    keyword, bucket into hourly rows matching the app schema.

    channels: list of public channel usernames, e.g. ["somecryptochannel", "anothergroup"]
              (public channels only — this will not join or read private groups
              you're not already a member of).
    influencer_min_subscribers: if a message is posted/forwarded from a channel
              with this many subscribers, that hour is flagged as an influencer_event.
    """
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=hours_back)

    rows = []
    for topic in topics:
        buckets = {}
        topic_lower = topic.lower()

        for channel_name in channels:
            try:
                entity = client.get_entity(channel_name)
            except Exception:
                continue  # channel not found / not public / not accessible

            subscriber_count = getattr(entity, "participants_count", 0) or 0

            for message in client.iter_messages(entity, limit=limit_per_channel):
                if message.date < cutoff:
                    break  # messages come back newest-first, so we can stop early
                if not message.text or topic_lower not in message.text.lower():
                    continue

                hour = message.date.replace(minute=0, second=0, microsecond=0)
                b = buckets.setdefault(hour, {"mentions": 0, "senders": set(), "engagement": 0, "influencer": 0})
                b["mentions"] += 1
                sender_id = message.sender_id if message.sender_id else f"anon-{channel_name}"
                b["senders"].add(sender_id)
                views = getattr(message, "views", 0) or 0
                forwards = getattr(message, "forwards", 0) or 0
                b["engagement"] += views + forwards * 5  # forwards weighted higher, similar to a share

                if subscriber_count >= influencer_min_subscribers:
                    b["influencer"] = 1

        for hour, b in buckets.items():
            rows.append({
                "time": hour,
                "topic": topic,
                "mentions": b["mentions"],
                "unique_accounts": len(b["senders"]),
                "engagement": b["engagement"],
                "influencer_event": b["influencer"],
            })

    if not rows:
        return pd.DataFrame(columns=["time", "topic", "mentions", "unique_accounts", "engagement", "influencer_event"])

    df = pd.DataFrame(rows).sort_values(["topic", "time"]).reset_index(drop=True)
    return df
