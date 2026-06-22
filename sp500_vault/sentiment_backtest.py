"""Sentiment lead-lag — does today's sentiment predict forward returns?

Runs the same forward-return engine as the event study over **two sentiment
sources**, side by side, so their predictive power can be compared:

    claude    the dense daily Claude sentiment score (``sentiment/history.csv``) —
              every ticker is scored every refresh day, so this panel grows ~50
              observations/day and quickly supports longer horizons.
    provider  the sparse per-article provider sentiment (Marketaux / Alpha Vantage)
              averaged per ticker-day from the append-only news archive — only the
              headlines that carry a provider score contribute.

For each source and horizon (1/3/5 trading days) it reports:

    rank IC        Spearman corr between sentiment and market-adjusted forward
                   return across all (ticker, day) observations — the standard
                   cross-sectional predictive-power metric
    high - low     mean forward return of above-median vs below-median sentiment

Output: ``data/signals/sentiment_backtest.json`` and ``vault/_SentimentBacktest.md``.
The panels accumulate daily (sentiment layer + news archive), so N and significance
grow over time.
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

_LABELS = {
    "claude": "Claude daily sentiment (dense — every ticker, every day)",
    "provider": "News-provider article sentiment (Marketaux / Alpha Vantage)",
}


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


# ── Panels: (ticker, date) -> sentiment [+ volume] ───────────────────────────


def _normalize_history(df):
    """Pure: a raw ``history.csv`` frame -> a normalized (ticker, date, sentiment,
    volume) panel. Volume is 1/ticker-day (one Claude score), so its IC is skipped."""
    import pandas as pd
    if df.empty:
        return df
    out = df.copy()
    out["sentiment"] = pd.to_numeric(out.get("score"), errors="coerce")
    out["date"] = pd.to_datetime(out.get("date"), errors="coerce").dt.normalize()
    out["volume"] = 1.0
    return out.dropna(subset=["date", "sentiment"])[["ticker", "date", "sentiment", "volume"]]


def _claude_panel():
    """Dense daily Claude sentiment from ``sentiment/history.csv``."""
    import pandas as pd
    path = config.SENTIMENT_DIR / "history.csv"
    if not path.exists():
        return pd.DataFrame()
    return _normalize_history(pd.read_csv(path))


def _provider_panel():
    """Sparse per-article provider sentiment + headline volume, from the news archive."""
    import pandas as pd
    df = archive.load_news()
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    df["ps"] = pd.to_numeric(df["provider_sentiment"], errors="coerce")
    df["d"] = pd.to_datetime(df["datetime"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["d"])
    vol = df.groupby(["ticker", "d"]).size().rename("volume")
    sent = df.dropna(subset=["ps"]).groupby(["ticker", "d"])["ps"].mean().rename("sentiment")
    panel = pd.concat([sent, vol], axis=1).reset_index().rename(columns={"d": "date"})
    return panel.dropna(subset=["sentiment"])


# ── Scoring one source against forward returns ───────────────────────────────


def _score_source(name: str, panel, closes: dict, spy, with_volume: bool) -> dict:
    """Attach market-adjusted forward returns to every obs and compute IC by horizon."""
    obs: dict[int, list[tuple[float, float, float]]] = {h: [] for h in HORIZONS}
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

    sent_ic, vol_ic = [], []
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
        ic = _rank_ic(sent, ab)
        sent_ic.append({
            "h": h, "n": len(rows),
            "rank_ic": round(ic, 3) if ic is not None else None,
            "high_mean": round(float(np.mean(hi)), 4) if hi else None,
            "low_mean": round(float(np.mean(lo)), 4) if lo else None,
            "high_minus_low": round(float(np.mean(hi) - np.mean(lo)), 4) if hi and lo else None,
        })
        if with_volume:
            v = _rank_ic(vol, ab)
            vol_ic.append({"h": h, "n": len(rows), "rank_ic": round(v, 3) if v is not None else None})
    return {
        "name": name, "label": _LABELS.get(name, name),
        "n_obs": int(len(panel)), "tickers": int(panel["ticker"].nunique()),
        "date_range": [str(panel["date"].min().date()), str(panel["date"].max().date())],
        "sentiment_ic": sent_ic, "volume_ic": vol_ic,
    }


def run(tickers: list[str] | None = None, force: bool = False) -> dict:
    import pandas as pd

    # claude first — it's the dense source and the headline signal.
    panels = {"claude": _claude_panel(), "provider": _provider_panel()}
    panels = {k: v for k, v in panels.items() if not v.empty}
    if tickers:
        keep = set(tickers)
        panels = {k: v[v["ticker"].isin(keep)] for k, v in panels.items()}
        panels = {k: v for k, v in panels.items() if not v.empty}
    if not panels:
        print("[sent-bt] no sentiment history yet — run `pipeline sentiment` / `archive`")
        return {}

    all_tickers = sorted({t for v in panels.values() for t in v["ticker"].unique()})
    min_date = min(v["date"].min() for v in panels.values())
    start = (min_date - pd.Timedelta(days=5)).date()
    end = dt.date.today() + dt.timedelta(days=1)
    desc = ", ".join(f"{k} {len(v)} obs" for k, v in panels.items())
    print(f"[sent-bt] panels: {desc} across {len(all_tickers)} tickers "
          f"({min_date.date()} → today); fetching prices…")
    closes = _fetch_closes(all_tickers + ["SPY"], start, end)
    spy = closes.get("SPY")

    sources = [_score_source(name, panel, closes, spy, with_volume=(name == "provider"))
               for name, panel in panels.items()]

    report = {
        "as_of": dt.date.today().isoformat(),
        "horizons": HORIZONS,
        "market_proxy": "SPY" if spy is not None else None,
        "sources": sources,
    }
    config.SIGNALS_DIR.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_md(report)
    for s in sources:
        ic1 = next((c["rank_ic"] for c in s["sentiment_ic"] if c["h"] == 1), None)
        print(f"[sent-bt]   {s['name']}: {s['n_obs']} obs, 1d rank-IC {ic1}")
    print(f"[sent-bt] wrote IC by source/horizon -> {_OUT}")
    return report


def _pct(v):
    return f"{v * 100:+.2f}%" if v is not None else "—"


def _write_md(report: dict) -> None:
    lines = ["# Sentiment Lead-Lag — sentiment vs forward returns", "",
             f"_As of {report['as_of']} · market-adjusted vs "
             f"{report.get('market_proxy') or 'none'}_", "",
             "Two sentiment sources, scored the same way: **rank IC** = Spearman "
             "correlation between sentiment and market-adjusted forward return "
             "(predictive power); **high − low** = above- vs below-median sentiment "
             "forward-return spread."]
    for s in report["sources"]:
        lines += ["", f"## {s['label']}", "",
                  f"_{s['n_obs']} ticker-day obs · {s['tickers']} tickers · "
                  f"{s['date_range'][0]}→{s['date_range'][1]}_", ""]
        if not s["sentiment_ic"]:
            lines.append("_No forward returns scorable yet — these observations are too "
                         "recent (no trading days have elapsed after them). Fills in as "
                         "the panel ages._")
            continue
        lines += ["| Horizon | N | Rank IC | High-sent ret | Low-sent ret | High − Low |",
                  "|---:|---:|---:|---:|---:|---:|"]
        for c in s["sentiment_ic"]:
            ic = f"{c['rank_ic']:+.3f}" if c["rank_ic"] is not None else "—"
            lines.append(f"| {c['h']}d | {c['n']} | {ic} | {_pct(c['high_mean'])} | "
                         f"{_pct(c['low_mean'])} | {_pct(c['high_minus_low'])} |")
        if s["volume_ic"]:
            lines += ["", "_News-volume check (does headline count predict returns?):_", "",
                      "| Horizon | N | Rank IC |", "|---:|---:|---:|"]
            for c in s["volume_ic"]:
                ic = f"{c['rank_ic']:+.3f}" if c["rank_ic"] is not None else "—"
                lines.append(f"| {c['h']}d | {c['n']} | {ic} |")
    lines += ["", "_The Claude panel grows ~50 obs/day and the provider panel grows "
              "with scored headlines, so N and significance improve over time._"]
    _MD.write_text("\n".join(lines), encoding="utf-8")
