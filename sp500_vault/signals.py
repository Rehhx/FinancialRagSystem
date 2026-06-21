"""Signals layer: validate the relationship graph against real price action.

The whole premise of the vault is that supplier/customer/competitor links are
economically meaningful. This layer tests that empirically: pull ~1y of daily
prices (Alpaca IEX, falling back to yfinance), compute return correlations, and
check whether *linked* company pairs co-move more than unlinked ones — overall
and by relation type. It also produces a per-edge correlation weight (addressing
the plan's hard-problem #4: edge strength, not just existence) and a per-node
"connected co-movement" score.

Outputs:
    data/signals/correlations.json   (machine-readable; feeds the graph export)
    vault/_Signals.md                (human-readable analysis note)

No LLM calls — this is pure market-data math, fast and free.
"""
from __future__ import annotations

import datetime as dt
import itertools
import json

import numpy as np
import pandas as pd
import requests

from . import config
from .relationships import get_edges
from .universe import BY_TICKER, TICKERS

_ALPACA_URL = "https://data.alpaca.markets/v2/stocks/bars"


# ── Price fetching ───────────────────────────────────────────────────────────


def _alpaca_prices(tickers: list[str], start: str, end: str) -> dict[str, pd.Series]:
    headers = {
        "APCA-API-KEY-ID": config.ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": config.ALPACA_API_SECRET,
    }
    params = {
        "symbols": ",".join(tickers), "timeframe": "1Day", "start": start, "end": end,
        "limit": 10000, "feed": "iex", "adjustment": "split",
    }
    raw: dict[str, list] = {}
    token = None
    while True:
        if token:
            params["page_token"] = token
        r = requests.get(_ALPACA_URL, headers=headers, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        for sym, bars in (data.get("bars") or {}).items():
            raw.setdefault(sym, []).extend(bars)
        token = data.get("next_page_token")
        if not token:
            break
    out = {}
    for sym, bars in raw.items():
        if not bars:
            continue
        s = pd.Series({b["t"][:10]: b["c"] for b in bars})
        s.index = pd.to_datetime(s.index)
        out[sym] = s.sort_index()
    return out


def _yfinance_prices(tickers: list[str], start: str, end: str) -> dict[str, pd.Series]:
    import yfinance as yf

    df = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)
    close = df["Close"] if "Close" in df else df
    if isinstance(close, pd.Series):  # single ticker
        return {tickers[0]: close.dropna()}
    return {c: close[c].dropna() for c in close.columns}


def fetch_prices(tickers: list[str], days: int) -> tuple[dict[str, pd.Series], str]:
    """Daily closes per ticker; prefers Alpaca IEX, falls back to yfinance."""
    end = dt.date.today()
    start = end - dt.timedelta(days=days)
    s, e = start.isoformat(), end.isoformat()
    source = "alpaca"
    series: dict[str, pd.Series] = {}
    if config.ALPACA_API_KEY and config.ALPACA_API_SECRET:
        try:
            series = _alpaca_prices(tickers, s, e)
        except Exception as exc:  # noqa: BLE001
            print(f"  [signals] Alpaca fetch failed ({exc}); falling back to yfinance")
            series = {}
    have = sum(1 for t in tickers if len(series.get(t, [])) > 50)
    if have < len(tickers) * 0.6:
        print(f"  [signals] only {have}/{len(tickers)} via Alpaca — using yfinance")
        series = _yfinance_prices(tickers, s, e)
        source = "yfinance"
    return series, source


# ── Analysis ─────────────────────────────────────────────────────────────────


def _pair_key(a: str, b: str) -> str:
    return "|".join(sorted((a, b)))


def _linked_pairs() -> tuple[set[str], dict[str, set[str]], dict[str, list[str]]]:
    """Undirected linked pairs, pairs-by-relation, and neighbors-by-node."""
    pairs: set[str] = set()
    by_rel: dict[str, set[str]] = {"supplier": set(), "customer": set(), "competitor": set()}
    neighbors: dict[str, set[str]] = {t: set() for t in TICKERS}
    for src in TICKERS:
        for e in get_edges(src, resolved_only=True):
            t = e["target_ticker"]
            if t in BY_TICKER and t != src:
                key = _pair_key(src, t)
                pairs.add(key)
                by_rel.setdefault(e["relation"], set()).add(key)
                neighbors[src].add(t)
                neighbors[t].add(src)
    return pairs, by_rel, {k: sorted(v) for k, v in neighbors.items()}


def analyze(tickers: list[str], days: int = 365) -> dict:
    series, source = fetch_prices(tickers, days)
    have = [t for t in tickers if t in series and len(series[t]) > 50]
    print(f"  [signals] {len(have)}/{len(tickers)} tickers with price history (source: {source})")
    df = pd.DataFrame({t: series[t] for t in have}).sort_index()
    rets = np.log(df / df.shift(1)).replace([np.inf, -np.inf], np.nan)
    corr = rets.corr(min_periods=30)  # pairwise complete obs

    pairs, by_rel, neighbors = _linked_pairs()

    def pair_corr(key: str):
        a, b = key.split("|")
        if a in corr.index and b in corr.columns:
            v = corr.loc[a, b]
            return None if pd.isna(v) else round(float(v), 3)
        return None

    # All available undirected pairs (both tickers have returns)
    all_pairs = {_pair_key(a, b) for a, b in itertools.combinations(have, 2)}
    linked_with_data = {k: pair_corr(k) for k in pairs if pair_corr(k) is not None}
    unlinked = [pair_corr(k) for k in (all_pairs - pairs) if pair_corr(k) is not None]

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return round(float(np.mean(xs)), 3) if xs else None

    by_rel_mean = {
        rel: mean([pair_corr(k) for k in keys if pair_corr(k) is not None])
        for rel, keys in by_rel.items()
    }
    node_corr = {}
    for t, nbrs in neighbors.items():
        vals = [pair_corr(_pair_key(t, n)) for n in nbrs]
        m = mean(vals)
        if m is not None:
            node_corr[t] = m

    linked_mean = mean(list(linked_with_data.values()))
    unlinked_mean = mean(unlinked)
    result = {
        "as_of": dt.date.today().isoformat(),
        "source": source,
        "window_days": days,
        "trading_days": int(rets.dropna(how="all").shape[0]),
        "summary": {
            "linked_mean": linked_mean,
            "unlinked_mean": unlinked_mean,
            "lift": round(linked_mean - unlinked_mean, 3) if (linked_mean and unlinked_mean is not None) else None,
            "baseline_all": mean([pair_corr(k) for k in all_pairs if pair_corr(k) is not None]),
            "by_relation": by_rel_mean,
            "n_linked_pairs": len(linked_with_data),
            "n_pairs": len(all_pairs),
        },
        "pair_corr": linked_with_data,
        "node_neighbor_corr": node_corr,
    }
    return result


# ── Rendering ────────────────────────────────────────────────────────────────


def _render_note(res: dict) -> None:
    s = res["summary"]
    pc = res["pair_corr"]
    top = sorted(pc.items(), key=lambda kv: kv[1], reverse=True)[:10]
    nc = res["node_neighbor_corr"]
    movers = sorted(nc.items(), key=lambda kv: kv[1], reverse=True)

    def link_pair(key):
        a, b = key.split("|")
        return f"[[{a}]] ↔ [[{b}]]"

    lines = [
        "# 🔗 Signals — does the relationship graph predict co-movement?",
        f"_As of {res['as_of']} · {res['trading_days']} trading days · source: {res['source']}_",
        "",
        "Daily-return correlations between modeled companies. If the supply/competitive "
        "graph is economically real, **linked pairs should co-move more than unlinked pairs**.",
        "",
        "## Headline",
        f"- **Linked pairs** mean correlation: **{s['linked_mean']}**  ({s['n_linked_pairs']} pairs)",
        f"- **Unlinked pairs** mean correlation: {s['unlinked_mean']}",
        f"- **Lift (linked − unlinked): {s['lift']}**  — "
        + ("relationships *do* track co-movement ✅" if (s['lift'] or 0) > 0 else "no positive lift ⚠️"),
        f"- Baseline (all pairs): {s['baseline_all']}",
        "",
        "## Co-movement by relation type",
        "| Relation | Mean return correlation |",
        "|---|---|",
    ]
    for rel in ("supplier", "customer", "competitor"):
        lines.append(f"| {rel} | {s['by_relation'].get(rel, '—')} |")
    lines += [
        f"| unlinked (baseline) | {s['unlinked_mean']} |",
        "",
        "## Most co-moving linked pairs",
    ]
    lines += [f"{i+1}. {link_pair(k)} — **{v}**" for i, (k, v) in enumerate(top)] or ["- *(none)*"]
    lines += ["", "## Connected co-movement by node (top 8)"]
    lines += [f"- [[{t}]] — {v}" for t, v in movers[:8]]
    lines += ["", "## Least connected co-movement (bottom 5)"]
    lines += [f"- [[{t}]] — {v}" for t, v in movers[-5:]]
    lines += ["", "_Edge weights in the graph explorer (line thickness) use these "
              "correlations. Regenerate the graph after re-running signals._"]
    (config.VAULT_DIR / "_Signals.md").write_text("\n".join(lines), encoding="utf-8")


def load() -> dict | None:
    p = config.CORRELATIONS_FILE
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def run(tickers: list[str], days: int = 365, force: bool = False) -> dict:
    print(f"[signals] analyzing price co-movement for {len(tickers)} tickers ({days}d)…")
    res = analyze(tickers, days)
    config.CORRELATIONS_FILE.write_text(json.dumps(res, indent=2), encoding="utf-8")
    _render_note(res)
    s = res["summary"]
    print(f"[signals] linked={s['linked_mean']} vs unlinked={s['unlinked_mean']} "
          f"(lift {s['lift']}) -> {config.CORRELATIONS_FILE.name} + vault/_Signals.md")
    return res
