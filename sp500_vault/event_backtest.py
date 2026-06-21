"""Event-driven backtest — how do prices move *after* an 8-K, by event type?

Reads the append-only filing archive (``archive.py``), which accumulates every
8-K over time, and historical daily prices, then runs an **event study**: for
each filing, the market-adjusted forward return over 1/3/5 trading days
(``stock_return − SPY_return``). Aggregated by item code (2.02 earnings, 5.02
executive change, 1.01 material agreement, …) with a hit-rate and a t-stat, this
shows whether — and which — material events carry a tradable drift.

Output: ``data/signals/event_backtest.json`` and ``vault/_EventBacktest.md``.
"""
from __future__ import annotations

import datetime as dt
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import archive, config
from .data_sources import edgar

HORIZONS = [1, 3, 5]
_OUT = config.SIGNALS_DIR / "event_backtest.json"
_MD = config.VAULT_DIR / "_EventBacktest.md"


def _close_series(ticker: str, start, end):
    """Daily adjusted close indexed by date, or None."""
    import yfinance as yf
    try:
        hist = yf.Ticker(ticker).history(start=start, end=end, interval="1d", auto_adjust=True)
        s = hist["Close"].dropna()
        return s if not s.empty else None
    except Exception:  # noqa: BLE001
        return None


def _fetch_closes(tickers, start, end, workers: int = 8) -> dict:
    out: dict = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for t, s in zip(tickers, ex.map(lambda x: _close_series(x, start, end), tickers)):
            if s is not None:
                out[t] = s
    return out


def _forward_return(close, event_date, h: int) -> float | None:
    """Return from the first trading close on/after ``event_date`` to h days later.

    Pure given a price series — unit-tested with a synthetic series.
    """
    import pandas as pd
    idx = close.index
    ed = pd.Timestamp(event_date)
    if idx.tz is not None:
        ed = ed.tz_localize(idx.tz)
    pos = int(idx.searchsorted(ed))
    if pos >= len(idx) or pos + h >= len(idx):
        return None
    p0, p1 = float(close.iloc[pos]), float(close.iloc[pos + h])
    return (p1 / p0 - 1.0) if p0 > 0 else None


def run(tickers: list[str] | None = None, force: bool = False) -> dict:
    import pandas as pd

    fil = archive.load_filings()
    if fil.empty:
        print("[event] filing archive is empty — run `pipeline archive` first")
        return {}
    fil = fil.copy()
    fil["filing_date"] = pd.to_datetime(fil["filing_date"], errors="coerce")
    fil = fil.dropna(subset=["filing_date"])
    if tickers:
        fil = fil[fil["ticker"].isin(set(tickers))]
    if fil.empty:
        return {}

    start = (fil["filing_date"].min() - pd.Timedelta(days=10)).date()
    end = dt.date.today() + dt.timedelta(days=1)
    names = sorted(fil["ticker"].unique())
    print(f"[event] {len(fil)} archived 8-Ks across {len(names)} tickers "
          f"({start} → {dt.date.today()}); fetching prices…")
    closes = _fetch_closes(names + ["SPY"], start, end)
    spy = closes.get("SPY")

    rows = []
    for _, r in fil.iterrows():
        close = closes.get(r["ticker"])
        if close is None:
            continue
        codes = [c for c in str(r["items"]).split(";") if c and c != "9.01"] or ["8-K"]
        for h in HORIZONS:
            ret = _forward_return(close, r["filing_date"], h)
            if ret is None:
                continue
            mkt = _forward_return(spy, r["filing_date"], h) if spy is not None else 0.0
            abn = ret - (mkt or 0.0)
            for code in codes:
                rows.append({"code": code, "h": h, "abn": abn})

    if not rows:
        print("[event] no events had enough price history")
        return {}
    df = pd.DataFrame(rows)

    agg = []
    for (code, h), g in df.groupby(["code", "h"]):
        n = len(g)
        mean = float(g["abn"].mean())
        std = float(g["abn"].std(ddof=1)) if n > 1 else float("nan")
        hit = float((g["abn"] > 0).mean())
        t_stat = (mean / (std / np.sqrt(n))) if (n > 1 and std and std > 0) else None
        agg.append({
            "code": code, "label": edgar._item_label(code) if code != "8-K" else "Any 8-K",
            "h": h, "n": n, "mean_abn_ret": round(mean, 4), "hit_rate": round(hit, 3),
            "t_stat": round(t_stat, 2) if t_stat is not None else None,
        })

    report = {"as_of": dt.date.today().isoformat(), "events": int(len(fil)),
              "tickers": len(names), "horizons": HORIZONS,
              "market_proxy": "SPY" if spy is not None else None, "by_event": agg}
    config.SIGNALS_DIR.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_md(report)
    print(f"[event] wrote {len(agg)} (event-type, horizon) cells -> {_OUT}")
    return report


def _write_md(report: dict) -> None:
    lines = ["# Event-Driven Backtest — forward returns after 8-K filings",
             "",
             f"_As of {report['as_of']} · {report['events']} archived 8-Ks · "
             f"{report['tickers']} tickers · market-adjusted vs "
             f"{report.get('market_proxy') or 'none'}_",
             "",
             "Market-adjusted mean return over H trading days after the filing "
             "(`stock − SPY`), with hit-rate and t-stat. Higher |t| = more reliable drift.",
             "",
             "| Event (8-K item) | H | N | Mean abn. ret | Hit-rate | t |",
             "|---|---:|---:|---:|---:|---:|"]
    for c in sorted(report["by_event"], key=lambda x: (x["code"], x["h"])):
        t = f"{c['t_stat']:+.2f}" if c["t_stat"] is not None else "—"
        lines.append(f"| {c['label']} ({c['code']}) | {c['h']} | {c['n']} | "
                     f"{c['mean_abn_ret'] * 100:+.2f}% | {c['hit_rate'] * 100:.0f}% | {t} |")
    lines += ["", "_Append-only archive grows daily (sentiment/filings layers + the "
              "real-time 8-K poller), so N and significance improve over time._"]
    _MD.write_text("\n".join(lines), encoding="utf-8")
