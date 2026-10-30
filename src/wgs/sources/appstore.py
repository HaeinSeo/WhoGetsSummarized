"""App Store customer reviews via Apple's public RSS feed (no login, no scraping of the storefront HTML).

URL form: https://apps.apple.com/<country>/app/<slug>/id<digits>
Feed:     https://itunes.apple.com/<country>/rss/customerreviews/page=<n>/id=<digits>/sortby=mostrecent/json
The feed returns at most 50 reviews per page and roughly 10 pages. Each page is requested once, with a
pause between requests and an identifying User-Agent. Check Apple's current terms before wider use.
"""
from __future__ import annotations

import re
import time

import requests

from ..text import language_guess

URL_RE = re.compile(r"apps\.apple\.com/([a-z]{2})/app/[^/?#]*/id(\d+)", re.I)
FEED = "https://itunes.apple.com/{country}/rss/customerreviews/page={page}/id={app_id}/sortby=mostrecent/json"
UA = "WhoGetsSummarized-research/0.1 (academic; contact via repository)"
PAGE_SIZE = 50
MAX_PAGES = 10
PAUSE_S = 1.0


def parse_url(url: str) -> tuple[str, str] | None:
    m = URL_RE.search(url.strip())
    return (m.group(1).lower(), m.group(2)) if m else None


def fetch_reviews(url: str, max_reviews: int = 150, english_only: bool = True) -> dict:
    parsed = parse_url(url)
    if not parsed:
        raise ValueError("not an App Store app link (expected apps.apple.com/<country>/app/<name>/id<digits>)")
    country, app_id = parsed
    sess = requests.Session()
    sess.headers["User-Agent"] = UA
    raw, pages = [], 0
    for page in range(1, MAX_PAGES + 1):
        r = sess.get(FEED.format(country=country, page=page, app_id=app_id), timeout=30)
        if r.status_code != 200:
            break
        entries = (r.json().get("feed", {}).get("entry") or [])
        reviews = entries[1:] if entries and "im:name" in entries[0] else entries  # first entry = app info
        if not reviews:
            break
        pages += 1
        raw.extend(reviews)
        if len(reviews) < PAGE_SIZE or len(raw) >= max_reviews:
            break
        time.sleep(PAUSE_S)

    out, dropped_non_en, seen = [], 0, set()
    for e in raw:
        title = (e.get("title", {}) or {}).get("label", "").strip()
        body = (e.get("content", {}) or {}).get("label", "").strip()
        text = " ".join(f"{title}. {body}".split()) if title else " ".join(body.split())
        if len(text) < 20 or text in seen:
            continue
        seen.add(text)
        if english_only and language_guess(text) != "en_likely":
            dropped_non_en += 1
            continue
        out.append({"anon_id": f"R{len(out) + 1:02d}", "title": "", "text": text,
                    "rating": int(e.get("im:rating", {}).get("label", 0) or 0)})
        if len(out) >= max_reviews:
            break
    return {"source": "app_store_rss", "country": country, "app_id": app_id, "pages": pages,
            "fetched": len(raw), "dropped_non_english": dropped_non_en, "reviews": out}
