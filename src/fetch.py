import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import feedparser
import requests
import trafilatura

from config import FEEDS, KEYWORDS, RAW_DIR, USER_AGENT, REQUEST_DELAY_SECONDS


def is_relevant(entry) -> bool:
    haystack = (entry.get("title", "") + " " + entry.get("summary", "")).lower()
    return any(k in haystack for k in KEYWORDS)


def article_id(url: str) -> str:
    return hashlib.sha1(url.encode()).hexdigest()[:16]


def already_fetched(url: str) -> bool:
    return (Path(RAW_DIR) / f"{article_id(url)}.txt").exists()


def fetch_article(url: str) -> str | None:
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=20)
    resp.raise_for_status()
    return trafilatura.extract(resp.text, include_comments=False)


def save(url: str, title: str, published: str | None, text: str) -> None:
    out = Path(RAW_DIR)
    out.mkdir(parents=True, exist_ok=True)
    aid = article_id(url)
    (out / f"{aid}.txt").write_text(text, encoding="utf-8")
    meta = {
        "id": aid,
        "url": url,
        "title": title,
        "published": published,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    (out / f"{aid}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def run() -> None:
    new_count = 0
    for feed_url in FEEDS:
        feed = feedparser.parse(feed_url)
        print(f"{feed_url}: {len(feed.entries)} entries")
        for entry in feed.entries:
            url = entry.get("link")
            if not url or not is_relevant(entry) or already_fetched(url):
                continue
            text = fetch_article(url)
            if not text:
                print(f"  no text extracted: {url}")
                continue
            save(url, entry.get("title", ""), entry.get("published"), text)
            new_count += 1
            print(f"  saved: {entry.get('title')}")
            time.sleep(REQUEST_DELAY_SECONDS)
    print(f"done, {new_count} new articles")


if __name__ == "__main__":
    run()