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

# Accession-keyed cache of one-line LLM summaries — so each high-signal 8-K is
# summarized exactly once, ever (the per-ticker record overwrites daily, but the
# events themselves repeat). Shared across tickers within a run.
_SUMMARY_CACHE = config.FILINGS_DIR / "summaries.json"


def _path(ticker: str):
    return config.FILINGS_DIR / f"{ticker}.json"


def _load_summaries() -> dict:
    if _SUMMARY_CACHE.exists():
        try:
            return json.loads(_SUMMARY_CACHE.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {}
    return {}


def _save_summaries(cache: dict) -> None:
    _SUMMARY_CACHE.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def _is_high_signal(event: dict, wanted: set[str]) -> bool:
    return any(it.get("code") in wanted for it in event.get("items", []))


def _attach_summaries(ticker: str, events: list[dict], cache: dict, force: bool = False) -> int:
    """For each high-signal 8-K, attach a cached or freshly-fetched one-line summary.
    Mutates ``events`` (adds ``summary``) and ``cache`` (adds new accessions).
    Returns the number of *new* LLM summaries generated."""
    from . import llm
    wanted = config.high_signal_items()
    added = 0
    for e in events:
        acc = e.get("accession")
        if acc and acc in cache and not force:
            e["summary"] = cache[acc].get("summary", "")
            continue
        if not _is_high_signal(e, wanted) or not e.get("doc_url"):
            continue
        try:
            body = edgar.fetch_8k_body(e["doc_url"])
            labels = "; ".join(it.get("label", "") for it in e.get("items", []))
            summary = llm.summarize_8k(ticker, labels, body)
        except Exception:  # noqa: BLE001 - a single bad doc shouldn't fail the ticker
            summary = ""
        if summary:
            e["summary"] = summary
            added += 1
            if acc:
                cache[acc] = {"ticker": ticker, "filing_date": e.get("filing_date"),
                              "items": [it.get("code") for it in e.get("items", [])],
                              "summary": summary}
    return added


def run_for_ticker(ticker: str, summary_cache: dict | None = None) -> dict:
    events = edgar.fetch_recent_8k(ticker)
    own_cache = summary_cache is None
    cache = _load_summaries() if own_cache else summary_cache
    if config.EDGAR_8K_SUMMARIZE and config.ANTHROPIC_API_KEY:
        _attach_summaries(ticker, events, cache)
    record = {
        "ticker": ticker,
        "as_of": dt.date.today().isoformat(),
        "event_count": len(events),
        "events": events,
    }
    _path(ticker).write_text(json.dumps(record, indent=2), encoding="utf-8")
    if own_cache:
        _save_summaries(cache)
    archive.append_filings(ticker, events)   # accumulate into the append-only event archive
    latest = events[0]["filing_date"] if events else "—"
    n_sum = sum(1 for e in events if e.get("summary"))
    extra = f", {n_sum} summarized" if n_sum else ""
    print(f"  [8-K] {ticker}: {len(events)} filings (latest {latest}{extra})")
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
    cache = _load_summaries()                # shared, so a filing is summarized once across tickers
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(lambda t: run_for_ticker(t, summary_cache=cache), todo))
    _save_summaries(cache)
    print(f"[8-K] done -> {config.FILINGS_DIR} ({len(cache)} cached 8-K summaries)")
