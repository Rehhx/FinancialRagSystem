"""Sentiment layer: per-node news pull + Claude scoring.

Each node runs its own independent pass — a chip supplier's sentiment reflects
*its* business, not the companies it supplies. Score goes in frontmatter (for
Dataview sorting); the raw articles are kept for the linked ``_news_log`` note.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
from concurrent.futures import ThreadPoolExecutor

from . import archive, config, llm
from .data_sources import news
from .universe import BY_TICKER


def _path(ticker: str):
    return config.SENTIMENT_DIR / f"{ticker}.json"


def run_for_ticker(ticker: str) -> dict:
    name = BY_TICKER[ticker].name if ticker in BY_TICKER else ticker
    articles = news.fetch_news(ticker)
    # Feed headline + (short) summary to the scorer for richer signal.
    snippets = [
        (a["headline"] + (f" — {a['summary'][:200]}" if a.get("summary") else ""))
        for a in articles
    ]
    try:
        scored = llm.score_sentiment(ticker, name, snippets)
    except Exception as e:  # noqa: BLE001
        print(f"  [sent] scoring failed for {ticker}: {e}")
        scored = {"score": 0.0, "label": "Neutral", "summary": f"Scoring error: {e}"}

    record = {
        "ticker": ticker,
        "name": name,
        "score": scored["score"],
        "label": scored["label"],
        "summary": scored["summary"],
        "as_of": dt.date.today().isoformat(),
        "article_count": len(articles),
        "articles": articles,
    }
    _path(ticker).write_text(json.dumps(record, indent=2), encoding="utf-8")
    archive.append_news(ticker, articles)   # accumulate headlines into the append-only archive
    provs = ",".join(sorted({a.get("provider", "?") for a in articles})) or "none"
    print(f"  [sent] {ticker}: {scored['label']} ({scored['score']:+.2f}, "
          f"{len(articles)} articles via {provs})")
    return record


def load(ticker: str) -> dict | None:
    p = _path(ticker)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _is_fresh(ticker: str, today: str) -> bool:
    rec = load(ticker)
    return bool(rec and rec.get("as_of") == today)


def _append_history(records: list[dict]) -> None:
    """Append (ticker, date, score, label) to a CSV time series, deduped per day.

    This accumulates the sentiment history that a future *sentiment* lead-lag
    backtest needs (we only have today's snapshot otherwise).
    """
    path = config.SENTIMENT_DIR / "history.csv"
    seen: set[tuple[str, str]] = set()
    if path.exists():
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.reader(f):
                if len(row) >= 2:
                    seen.add((row[0], row[1]))
    new = [r for r in records if r and (r["ticker"], r["as_of"]) not in seen]
    if not new:
        return
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["ticker", "date", "score", "label"])
        for r in new:
            w.writerow([r["ticker"], r["as_of"], r["score"], r["label"]])


def run(tickers: list[str], force: bool = False, workers: int = 5) -> None:
    today = dt.date.today().isoformat()
    todo = tickers if force else [t for t in tickers if not _is_fresh(t, today)]
    skipped = len(tickers) - len(todo)
    print(f"[sent] scoring sentiment for {len(todo)} tickers "
          f"({skipped} already fresh, skipped)…")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        records = list(ex.map(run_for_ticker, todo))
    _append_history(records)
    print(f"[sent] done -> {config.SENTIMENT_DIR}")
