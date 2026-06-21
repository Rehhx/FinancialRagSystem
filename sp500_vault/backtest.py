"""Lead-lag backtest: does a supplier's move lead its customer's?

The plan's V2 question is "does aggregate supplier sentiment predict the
customer's stock movement with lead time?". A *sentiment* lead-lag needs a
sentiment time series, which we only start accumulating now (see
``sentiment.py`` history). So this engine tests the feasible price analog today:

    signal_t(customer)  = mean trailing k-day return of its modeled SUPPLIERS
    target_t(customer)  = the customer's forward h-day return

If supply-chain health leads, strong supplier momentum should precede customer
out-performance. We measure that with the Information Coefficient (IC = pooled
correlation of signal vs forward return) across a (k, h) grid, and run a simple
cross-sectional, dollar-neutral long/short portfolio (long customers with strong
supplier momentum, short weak) to get an annualized return / Sharpe.

No LLM calls — pure price math on the same feed the signals layer uses.
Outputs: data/signals/backtest.json + vault/_Backtest.md
"""
from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd

from . import config, signals
from .relationships import get_edges
from .universe import BY_TICKER, TICKERS

_K_GRID = [3, 5, 10]          # signal lookback (trading days)
_H_GRID = [1, 3, 5, 10]       # forward horizon (trading days)
_PORTFOLIO_K = 5              # lookback used for the long/short backtest


def _supplier_map() -> dict[str, set[str]]:
    """customer -> {its modeled suppliers}, from both edge directions."""
    sup: dict[str, set[str]] = {}
    for src in TICKERS:
        for e in get_edges(src, resolved_only=True):
            t = e["target_ticker"]
            if t not in BY_TICKER or t == src:
                continue
            if e["relation"] == "supplier":       # src buys from t
                sup.setdefault(src, set()).add(t)
            elif e["relation"] == "customer":     # t buys from src
                sup.setdefault(t, set()).add(src)
    return sup


def _returns(tickers: list[str], days: int) -> tuple[pd.DataFrame, str]:
    series, source = signals.fetch_prices(tickers, days)
    have = [t for t in tickers if t in series and len(series[t]) > 50]
    df = pd.DataFrame({t: series[t] for t in have}).sort_index()
    rets = np.log(df / df.shift(1)).replace([np.inf, -np.inf], np.nan)
    return rets, source


def _signal_frame(rk: pd.DataFrame, sup: dict[str, set[str]]) -> tuple[pd.DataFrame, list[str]]:
    """Per-customer supplier-momentum signal (mean of suppliers' trailing returns)."""
    customers = [c for c, ss in sup.items()
                 if c in rk.columns and any(s in rk.columns for s in ss)]
    sig = pd.DataFrame(index=rk.index, columns=customers, dtype=float)
    for c in customers:
        cols = [s for s in sup[c] if s in rk.columns]
        if cols:
            sig[c] = rk[cols].mean(axis=1)
    return sig, customers


def _ic(signal: pd.DataFrame, target: pd.DataFrame) -> tuple[float | None, int]:
    a, b = signal.values.flatten(), target.values.flatten()
    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() < 30:
        return None, int(mask.sum())
    a, b = a[mask], b[mask]
    if a.std() == 0 or b.std() == 0:
        return None, int(mask.sum())
    return round(float(np.corrcoef(a, b)[0, 1]), 4), int(mask.sum())


def _sparkline(series: list[float]) -> str:
    bars = "▁▂▃▄▅▆▇█"
    if not series:
        return ""
    lo, hi = min(series), max(series)
    rng = (hi - lo) or 1.0
    return "".join(bars[min(7, int((v - lo) / rng * 7))] for v in series)


def analyze(tickers: list[str], days: int = 400) -> dict:
    rets, source = _returns(tickers, days)
    sup = _supplier_map()

    # IC grid over (k, h)
    grid = {}
    for k in _K_GRID:
        rk = rets.rolling(k).sum()
        sig, customers = _signal_frame(rk, sup)
        for h in _H_GRID:
            fwd = rets.rolling(h).sum().shift(-h)[sig.columns]
            ic, n = _ic(sig, fwd)
            grid[f"k{k}_h{h}"] = {"k": k, "h": h, "ic": ic, "obs": n}

    # Per-customer IC at the portfolio lookback / h=5, to find the best lead pairs
    rk = rets.rolling(_PORTFOLIO_K).sum()
    sig, customers = _signal_frame(rk, sup)
    fwd5 = rets.rolling(5).sum().shift(-5)[sig.columns]
    per_customer = {}
    for c in customers:
        ic, n = _ic(sig[[c]], fwd5[[c]])
        if ic is not None:
            per_customer[c] = ic

    # Cross-sectional, dollar-neutral long/short, daily rebalanced
    w = sig.sub(sig.mean(axis=1), axis=0)
    w = w.div(w.abs().sum(axis=1).replace(0, np.nan), axis=0)
    pnl = (w * rets[customers].shift(-1)).sum(axis=1).dropna()
    ann_ret = float(pnl.mean() * 252)
    ann_vol = float(pnl.std() * np.sqrt(252))
    sharpe = round(ann_ret / ann_vol, 2) if ann_vol else None
    equity = (1 + pnl).cumprod()
    total_ret = float(equity.iloc[-1] - 1) if len(equity) else 0.0

    best = max((g for g in grid.values() if g["ic"] is not None),
               key=lambda g: g["ic"], default=None)

    return {
        "as_of": dt.date.today().isoformat(),
        "source": source,
        "window_days": days,
        "trading_days": int(rets.dropna(how="all").shape[0]),
        "n_customers": len(customers),
        "signal": "mean trailing-k-day return of modeled suppliers",
        "ic_grid": grid,
        "best": best,
        "portfolio": {
            "lookback_k": _PORTFOLIO_K,
            "annualized_return": round(ann_ret, 4),
            "annualized_vol": round(ann_vol, 4),
            "sharpe": sharpe,
            "hit_rate": round(float((pnl > 0).mean()), 3) if len(pnl) else None,
            "total_return": round(total_ret, 4),
            "equity_curve": [round(float(x), 4) for x in equity.tolist()],
        },
        "top_customers_by_ic": dict(sorted(per_customer.items(), key=lambda kv: kv[1], reverse=True)[:8]),
    }


def _render_note(res: dict) -> None:
    p = res["portfolio"]
    spark = _sparkline(p["equity_curve"])
    lines = [
        "# 📈 Lead-Lag Backtest — do suppliers lead their customers?",
        f"_As of {res['as_of']} · {res['trading_days']} trading days · {res['n_customers']} "
        f"customers · source: {res['source']}_",
        "",
        f"**Signal:** {res['signal']}. **Hypothesis:** strong supplier momentum precedes "
        "customer out-performance.",
        "",
        "## Headline",
    ]
    if res["best"]:
        b = res["best"]
        lines.append(f"- Best lead: **k={b['k']}d signal → h={b['h']}d forward**, "
                     f"Information Coefficient **{b['ic']}** ({b['obs']} obs)")
    lines += [
        f"- Long/short portfolio (k={p['lookback_k']}d, daily rebalance): "
        f"**Sharpe {p['sharpe']}**, ann. return {p['annualized_return']:.1%}, "
        f"hit rate {p['hit_rate']}",
        f"- Equity curve: `{spark}`  (total {p['total_return']:.1%})",
        "",
        "## Information Coefficient grid (IC = corr of supplier signal vs forward return)",
        "| signal \\ forward | " + " | ".join(f"h={h}d" for h in _H_GRID) + " |",
        "|---|" + "---|" * len(_H_GRID),
    ]
    for k in _K_GRID:
        row = [f"**k={k}d**"]
        for h in _H_GRID:
            g = res["ic_grid"].get(f"k{k}_h{h}", {})
            row.append("—" if g.get("ic") is None else f"{g['ic']:+.3f}")
        lines.append("| " + " | ".join(row) + " |")
    lines += ["", "## Customers most predicted by their suppliers (IC, k5→h5)"]
    lines += [f"- [[{c}]] — IC {v:+.3f}" for c, v in res["top_customers_by_ic"].items()]
    lines += [
        "",
        "_Positive IC means supplier momentum leads customer returns. This uses supplier "
        "price momentum as the signal; once enough sentiment history accumulates "
        "(`data/sentiment/history.csv`), the same engine can test supplier **sentiment** "
        "as the leading indicator._",
    ]
    (config.VAULT_DIR / "_Backtest.md").write_text("\n".join(lines), encoding="utf-8")


def load() -> dict | None:
    p = config.SIGNALS_DIR / "backtest.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def run(tickers: list[str], days: int = 400, force: bool = False) -> dict:
    print(f"[backtest] lead-lag (supplier momentum -> customer forward return), {len(tickers)} tickers…")
    res = analyze(tickers, days)
    (config.SIGNALS_DIR / "backtest.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    _render_note(res)
    b, p = res.get("best"), res["portfolio"]
    print(f"[backtest] best IC {b['ic'] if b else '—'} (k{b['k']}->h{b['h']}) · "
          f"L/S Sharpe {p['sharpe']} -> vault/_Backtest.md")
    return res
