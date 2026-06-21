"""Append-only event archive — the substrate for event-driven backtests.

The sentiment and filings layers keep only a *rolling window* (last ~14 days of
news, last ~90 days of 8-Ks) and overwrite it each refresh. To study how prices
move *after* events, we need the events to accumulate over time. This module
appends every 8-K filing and every news headline to deduplicated, append-only
CSVs (``data/archive/filings.csv``, ``news.csv``) — written automatically by the
filings/sentiment layers, and back-fillable from EDGAR's history to seed it now.

    filings.csv  ticker, filing_date, report_date, accession, items, first_seen
    news.csv     ticker, datetime, headline, source, provider, sentiment, url, key, first_seen
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib

from . import config

ARCHIVE_DIR = config.DATA_DIR / "archive"
_FILINGS = ARCHIVE_DIR / "filings.csv"
_NEWS = ARCHIVE_DIR / "news.csv"

_FILING_COLS = ["ticker", "filing_date", "report_date", "accession", "items", "first_seen"]
_NEWS_COLS = ["ticker", "datetime", "headline", "source", "provider",
              "provider_sentiment", "url", "key", "first_seen"]


def _ensure(path, cols) -> None:
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(cols)


def _existing_keys(path, key_idx: int) -> set[str]:
    if not path.exists():
        return set()
    keys: set[str] = set()
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        next(r, None)
        for row in r:
            if len(row) > key_idx:
                keys.add(row[key_idx])
    return keys


def append_filings(ticker: str, events: list[dict]) -> int:
    """Append new 8-K events (deduped by accession). Returns count added."""
    if not events:
        return 0
    _ensure(_FILINGS, _FILING_COLS)
    seen = _existing_keys(_FILINGS, 3)          # accession column
    today = dt.date.today().isoformat()
    new = []
    for e in events:
        acc = e.get("accession")
        if not acc or acc in seen:
            continue
        seen.add(acc)
        items = ";".join(it.get("code", "") for it in e.get("items", []))
        new.append([ticker, e.get("filing_date", ""), e.get("report_date", ""), acc, items, today])
    if new:
        with _FILINGS.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(new)
    return len(new)


def _news_key(a: dict) -> str:
    url = (a.get("url") or "").split("?")[0].rstrip("/").lower()
    return url or hashlib.sha1((a.get("headline") or "").lower().encode("utf-8")).hexdigest()[:16]


def append_news(ticker: str, articles: list[dict]) -> int:
    """Append new headlines (deduped by URL/headline). Returns count added."""
    if not articles:
        return 0
    _ensure(_NEWS, _NEWS_COLS)
    seen = _existing_keys(_NEWS, 7)             # key column
    today = dt.date.today().isoformat()
    new = []
    for a in articles:
        k = _news_key(a)
        if not k or k in seen:
            continue
        seen.add(k)
        ps = a.get("provider_sentiment")
        new.append([ticker, (a.get("datetime") or "")[:19], (a.get("headline") or "")[:240],
                    a.get("source") or a.get("provider") or "", a.get("provider") or "",
                    ps if ps is not None else "", a.get("url") or "", k, today])
    if new:
        with _NEWS.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(new)
    return len(new)


def load_filings():
    import pandas as pd
    return pd.read_csv(_FILINGS) if _FILINGS.exists() else pd.DataFrame(columns=_FILING_COLS)


def load_news():
    import pandas as pd
    return pd.read_csv(_NEWS) if _NEWS.exists() else pd.DataFrame(columns=_NEWS_COLS)


def backfill_filings(tickers: list[str], lookback_days: int = 365, limit: int = 40) -> int:
    """Seed the filing archive from EDGAR's history (one submissions call/ticker)."""
    from .data_sources import edgar
    total = 0
    for t in tickers:
        events = edgar.fetch_recent_8k(t, lookback_days=lookback_days, limit=limit)
        added = append_filings(t, events)
        total += added
        print(f"  [archive] {t}: {len(events)} 8-Ks in window, +{added} new")
    print(f"[archive] filing archive: +{total} new events -> {_FILINGS}")
    return total


def append_news_from_sentiment(tickers: list[str]) -> int:
    """Fold any currently-stored sentiment articles into the news archive."""
    from . import sentiment
    total = 0
    for t in tickers:
        arts = (sentiment.load(t) or {}).get("articles") or []
        total += append_news(t, arts)
    print(f"[archive] news archive: +{total} new headlines -> {_NEWS}")
    return total


def run(tickers: list[str], force: bool = False) -> None:
    """Seed/refresh the archive: backfill 8-K history + fold in current news."""
    backfill_filings(tickers, lookback_days=365)
    append_news_from_sentiment(tickers)
