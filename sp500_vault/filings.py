"""Material-events layer: recent 8-K filings per company from SEC EDGAR.

8-K is the form a company must file within ~4 business days of a material event —
earnings releases (Item 2.02), executive changes (5.02), material agreements
(1.01), acquisitions (2.01), impairments (2.06), and so on. The submissions index
gives the event *type* (item codes) and date for free in one request per ticker,
so this is a cheap, structured catalyst signal. The events are stored per ticker
and indexed into the RAG (see ``rag._filings_digest``) so the vault can answer
"what material events has NVDA filed recently?".

Cadence is daily-ish (events are sporadic); like sentiment, a ticker is skipped
if it was already refreshed today unless ``--force``.
"""
from __future__ import annotations

import datetime as dt
import json
from concurrent.futures import ThreadPoolExecutor

from . import archive, config
from .data_sources import edgar


def _path(ticker: str):
    return config.FILINGS_DIR / f"{ticker}.json"


def run_for_ticker(ticker: str) -> dict:
    events = edgar.fetch_recent_8k(ticker)
    record = {
        "ticker": ticker,
        "as_of": dt.date.today().isoformat(),
        "event_count": len(events),
        "events": events,
    }
    _path(ticker).write_text(json.dumps(record, indent=2), encoding="utf-8")
    archive.append_filings(ticker, events)   # accumulate into the append-only event archive
    latest = events[0]["filing_date"] if events else "—"
    print(f"  [8-K] {ticker}: {len(events)} filings (latest {latest})")
    return record


def load(ticker: str) -> dict | None:
    p = _path(ticker)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _is_fresh(ticker: str, today: str) -> bool:
    rec = load(ticker)
    return bool(rec and rec.get("as_of") == today)


def run(tickers: list[str], force: bool = False, workers: int = 4) -> None:
    today = dt.date.today().isoformat()
    todo = tickers if force else [t for t in tickers if not _is_fresh(t, today)]
    skipped = len(tickers) - len(todo)
    print(f"[8-K] fetching material events for {len(todo)} tickers "
          f"({skipped} already fresh, skipped)…")
    # Keep workers modest — SEC rate-limits to ~10 req/s and edgar._get sleeps 0.2s.
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(run_for_ticker, todo))
    print(f"[8-K] done -> {config.FILINGS_DIR}")
