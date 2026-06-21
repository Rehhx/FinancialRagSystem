"""Sentiment lead-lag — does today's news sentiment predict forward returns?

Builds a daily sentiment panel from the append-only **news archive**
(``archive.py``): per (ticker, day), the mean per-article sentiment that the
providers (Marketaux / Alpha Vantage) attach to each headline, plus news volume.
Then it runs the same forward-return engine as the event study and reports, per
horizon (1/3/5 trading days):

    rank IC        Spearman corr between sentiment and market-adjusted forward
                   return across all (ticker, day) observations — the standard
                   cross-sectional predictive-power metric
    high − low     mean forward return of above-median vs below-median sentiment

Output: ``data/signals/sentiment_backtest.json`` and ``vault/_SentimentBacktest.md``.
The archive accumulates daily (sentiment layer + real-time poller), so N and
significance grow over time.
"""
from __future__ import annotations

import datetime as dt
import json

import numpy as np

from . import archive, config
from .event_backtest import _fetch_closes, _forward_return   # reuse the price engine

HORIZONS = [1, 3, 5]
_OUT = config.SIGNALS_DIR / "sentiment_backtest.json"
_MD = config.VAULT_DIR / "_SentimentBacktest.md"


def _rank_ic(x: list[float], y: list[float]) -> float | None:
    """Spearman rank correlation (information coefficient). Pure + unit-tested."""
    import pandas as pd
    if len(x) < 3:
        return None
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def _daily_sentiment_panel():
    """(ticker, date) -> mean provider sentiment + headline count, from the archive."""
    import pandas as pd
    df = archive.load_news()
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    df["ps"] = pd.to_numeric(df["provider_sentiment"], errors="coerce")
    df["date"] = pd.to_datetime(df["datetime"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date"])
    vol = df.groupby(["ticker", "date"]).size().rename("volume")
    sent = df.dropna(subset=["ps"]).groupby(["ticker", "date"])["ps"].mean().rename("sentiment")
    panel = pd.concat([sent, vol], axis=1).reset_index()
    return panel


def run(tickers: list[str] | None = None, force: bool = False) -> dict:
    import pandas as pd

    panel = _daily_sentiment_panel()
    panel = panel.dropna(subset=["sentiment"]) if not panel.empty else panel
    if panel.empty:
        print("[sent-bt] no scored headlines in the archive yet — run `pipeline archive`/`sentiment`")
        return {}
    if tickers:
        panel = panel[panel["ticker"].isin(set(tickers))]

    start = (panel["date"].min() - pd.Timedelta(days=5)).date()
    end = dt.date.today() + dt.timedelta(days=1)
    names = sorted(panel["ticker"].unique())
    print(f"[sent-bt] {len(panel)} ticker-day sentiment obs across {len(names)} tickers "
          f"({panel['date'].min().date()} → {panel['date'].max().date()}); fetching prices…")
    closes = _fetch_closes(names + ["SPY"], start, end)
    spy = closes.get("SPY")

    # Attach market-adjusted forward returns to each observation, per horizon.
    obs: dict[int, list[tuple[float, float, float]]] = {h: [] for h in HORIZONS}  # (sentiment, vol, abn_ret)
    for _, r in panel.iterrows():
        close = closes.get(r["ticker"])
        if close is None:
            continue
        for h in HORIZONS:
            ret = _forward_return(close, r["date"], h)
            if ret is None:
                continue
            mkt = _forward_return(spy, r["date"], h) if spy is not None else 0.0
            obs[h].append((float(r["sentiment"]), float(r["volume"]), ret - (mkt or 0.0)))

    by_h, vol_by_h = [], []
    for h in HORIZONS:
        rows = obs[h]
        if not rows:
            continue
        sent = [s for s, _, _ in rows]
        vol = [v for _, v, _ in rows]
        ab = [a for _, _, a in rows]
        med = float(np.median(sent))
        hi = [a for s, a in zip(sent, ab) if s > med]
        lo = [a for s, a in zip(sent, ab) if s <= med]
        by_h.append({
            "h": h, "n": len(rows),
            "rank_ic": round(_rank_ic(sent, ab), 3) if _rank_ic(sent, ab) is not None else None,
            "high_mean": round(float(np.mean(hi)), 4) if hi else None,
            "low_mean": round(float(np.mean(lo)), 4) if lo else None,
            "high_minus_low": round(float(np.mean(hi) - np.mean(lo)), 4) if hi and lo else None,
        })
        ic_v = _rank_ic(vol, ab)
        vol_by_h.append({"h": h, "n": len(rows), "rank_ic": round(ic_v, 3) if ic_v is not None else None})

    report = {
        "as_of": dt.date.today().isoformat(),
        "n_obs": int(len(panel)), "tickers": len(names),
        "date_range": [str(panel["date"].min().date()), str(panel["date"].max().date())],
        "market_proxy": "SPY" if spy is not None else None,
        "sentiment_ic": by_h, "volume_ic": vol_by_h,
    }
    config.SIGNALS_DIR.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_md(report)
    print(f"[sent-bt] wrote IC by horizon -> {_OUT}")
    return report


def _write_md(report: dict) -> None:
    lines = ["# Sentiment Lead-Lag — news sentiment vs forward returns", "",
             f"_As of {report['as_of']} · {report['n_obs']} ticker-day obs · "
             f"{report['tickers']} tickers · {report['date_range'][0]}→{report['date_range'][1]} · "
             f"market-adjusted vs {report.get('market_proxy') or 'none'}_", "",
             "Per-article provider sentiment (Marketaux / Alpha Vantage) averaged per "
             "ticker-day, vs market-adjusted forward return. **Rank IC** = Spearman "
             "correlation (predictive power); **high − low** = above- vs below-median "
             "sentiment forward-return spread.", "",
             "| Horizon | N | Rank IC | High-sent ret | Low-sent ret | High − Low |",
             "|---:|---:|---:|---:|---:|---:|"]
    for c in report["sentiment_ic"]:
        def pct(v):
            return f"{v * 100:+.2f}%" if v is not None else "—"
        ic = f"{c['rank_ic']:+.3f}" if c["rank_ic"] is not None else "—"
        lines.append(f"| {c['h']}d | {c['n']} | {ic} | {pct(c['high_mean'])} | "
                     f"{pct(c['low_mean'])} | {pct(c['high_minus_low'])} |")
    lines += ["", "**News-volume** check (does headline count predict returns?):", "",
              "| Horizon | N | Rank IC |", "|---:|---:|---:|"]
    for c in report["volume_ic"]:
        ic = f"{c['rank_ic']:+.3f}" if c["rank_ic"] is not None else "—"
        lines.append(f"| {c['h']}d | {c['n']} | {ic} |")
    lines += ["", "_The news archive accumulates daily (sentiment layer + real-time "
              "poller), so N and significance improve over time._"]
    _MD.write_text("\n".join(lines), encoding="utf-8")
