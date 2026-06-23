"""Signal engine — a machine-callable LONG / SHORT / FLAT verdict per ticker.

Built to be polled by an *external* trading engine over HTTP, so it can decide
whether to long or short a position using the vault's sentiment + strategies, and
optionally overlay its own signal:

    GET  /signal/{ticker}?horizon=5            one verdict
    GET  /signals?tickers=NVDA,AMD&horizon=5   batch
    overlay your own signal:  ?overlay=0.8&overlay_weight=0.5

It blends the three *validated* strategies already in the vault, each weighted by
its measured Information Coefficient (Grinold–Kahn signal combination; the
Fundamental Law of Active Management, IR = IC·√breadth):

    sentiment_leadlag   today's Claude sentiment  × signed sentiment rank-IC
    supplier_leadlag    suppliers' trailing-k momentum × signed lead-lag IC
    event_drift         recent 8-K item drift (significant only, |t| ≥ EVENT_MIN_T)

For each company we z-score each raw signal cross-sectionally, multiply by that
strategy's *signed* IC (a contrarian signal flips, a strong one dominates), sum
the active components into a blend, then z-score the blend into a conviction.
direction = long/short/flat by |conviction| vs τ, gated on breadth — we never
trade a name with no validated edge (that is the "or not").

``run()`` precomputes a daily snapshot (``data/signals/trades.json`` +
``vault/_Trades.md``); the API reads that snapshot so calls are fast and
deterministic (no market-data or LLM calls per request). The pure scoring
functions take the snapshot + backtest reports as plain dicts, so they unit-test
without any network.
"""
from __future__ import annotations

import bisect
import datetime as dt
import json
import math
import statistics as st

from . import backtest, config, filings, sentiment, signals
from .relationships import get_edges  # noqa: F401  (kept for parity / future use)
from .universe import TICKERS

_OUT = config.SIGNALS_DIR / "trades.json"
_MD = config.VAULT_DIR / "_Trades.md"


# ── Pure math helpers ────────────────────────────────────────────────────────


def _sign(x: float) -> int:
    return 1 if x > 0 else -1 if x < 0 else 0


def _zscore(d: dict[str, float | None]) -> dict[str, float | None]:
    """Cross-sectional z-score (demeaned). Missing values stay None; a degenerate
    spread maps everything to 0. Neutral point = the universe mean — this is what
    IC is measured against, so the blend stays consistent with the backtests."""
    present = {k: float(v) for k, v in d.items() if v is not None}
    out: dict[str, float | None] = {k: None for k in d}
    if len(present) < 2:
        return out | {k: 0.0 for k in present}
    m = st.mean(present.values())
    s = st.pstdev(present.values())
    for k, v in present.items():
        out[k] = round((v - m) / s, 4) if s else 0.0
    return out


def _scale_around_zero(d: dict[str, float]) -> dict[str, float]:
    """Sign-preserving normalization for an event signal whose neutral point is
    *zero* (no event = no signal). Divides by the spread of the nonzero edges, so
    a bullish drift stays positive and a bearish one negative."""
    nz = [v for v in d.values() if v]
    if not nz:
        return {k: 0.0 for k in d}
    s = st.pstdev(nz) if len(nz) > 1 else abs(nz[0])
    return {k: (round(v / s, 4) if (v and s) else 0.0) for k, v in d.items()}


def _sentiment_ic(sentbt: dict, horizon: int) -> tuple[float, str]:
    """Resolve the sentiment lead-lag IC at ``horizon``: prefer the dense Claude
    panel, then the provider panel, then any scorable horizon, then the fallback."""
    def pick(name: str) -> float | None:
        for s in sentbt.get("sources") or []:
            if s.get("name") != name:
                continue
            cells = s.get("sentiment_ic") or []
            for c in cells:
                if c.get("h") == horizon and c.get("rank_ic") is not None:
                    return c["rank_ic"]
            vals = [c["rank_ic"] for c in cells if c.get("rank_ic") is not None]
            if vals:
                return vals[-1]
        return None

    for name in ("claude", "provider"):
        ic = pick(name)
        if ic is not None:
            return float(ic), f"{name} rank-IC"
    return config.SENTIMENT_IC_FALLBACK, "fallback"


def _supplier_ic(bt: dict, k: int, horizon: int) -> tuple[float, str]:
    """Supplier lead-lag IC at (k, horizon) from the backtest grid; fall back to
    the grid's best cell."""
    cell = (bt.get("ic_grid") or {}).get(f"k{k}_h{horizon}")
    if cell and cell.get("ic") is not None:
        return float(cell["ic"]), f"k{k}→h{horizon}"
    best = bt.get("best")
    if best and best.get("ic") is not None:
        return float(best["ic"]), f"best k{best['k']}→h{best['h']}"
    return 0.0, "none"


def _event_index(eventbt: dict) -> dict[tuple[str, int], dict]:
    return {(r["code"], r["h"]): r for r in (eventbt.get("by_event") or [])}


def _event_edge(recent: list[dict], ev_index: dict, horizon: int,
                min_t: float) -> tuple[float, list[str]]:
    """Sum the significant (|t| ≥ min_t) historical drift for the item codes in a
    company's recent 8-Ks. Each code counts once. Returns (edge, human reasons)."""
    edge, reasons, seen = 0.0, [], set()
    for ev in recent:
        code = ev.get("code")
        row = ev_index.get((code, horizon))
        if not row or code in seen:
            continue
        mar, t = row.get("mean_abn_ret"), row.get("t_stat")
        if mar is None or t is None or abs(t) < min_t:
            continue
        seen.add(code)
        edge += mar
        reasons.append(f"8-K {code} {(row.get('label') or '')[:30]} "
                       f"{mar * 100:+.2f}% {horizon}d (t {t:+.1f})")
    return round(edge, 4), reasons


def _recent_events(events: list[dict], as_of: str, lookback_days: int) -> list[dict]:
    """Flatten a ticker's 8-Ks to per-item rows filed within the last
    ``lookback_days`` calendar days of ``as_of`` (drift still 'live')."""
    try:
        ref = dt.date.fromisoformat(as_of)
    except ValueError:
        return []
    out = []
    for e in events or []:
        d = e.get("filing_date")
        try:
            fd = dt.date.fromisoformat(str(d)[:10])
        except (ValueError, TypeError):
            continue
        if not (0 <= (ref - fd).days <= lookback_days):
            continue
        for it in e.get("items") or []:
            # items may be {"code","label"} (filings.load) or bare code strings.
            if isinstance(it, dict):
                out.append({"code": it.get("code"), "label": it.get("label"), "date": str(d)[:10]})
            elif it:
                out.append({"code": str(it), "label": "", "date": str(d)[:10]})
    return out


# ── Blend → verdict (pure: snapshot + reports in, book out) ───────────────────


def _reco_line(v: dict) -> str:
    d = v["direction"]
    if d == "flat":
        why = "no validated edge" if v["breadth"] == 0 else "conviction below threshold"
        return f"FLAT {v['ticker']} — {why} ({v['conviction']:+.2f}σ, {v['horizon_days']}d)"
    agree = " · ".join(sorted(v["components"]))
    return (f"{d.upper()} {v['ticker']} — conviction {v['conviction']:+.2f}σ "
            f"({v['horizon_days']}d, {v['confidence']} confidence); {agree}")


def build_book(snap: dict, bt: dict, sentbt: dict, eventbt: dict,
               horizon: int, tau: float) -> dict[str, dict]:
    """Compute a verdict for every ticker in the snapshot at one (horizon, τ).
    Pure — all inputs are plain dicts, so this is fully unit-testable."""
    sig = snap.get("signals") or {}
    ic_sent, sent_src = _sentiment_ic(sentbt, horizon)
    ic_sup, sup_src = _supplier_ic(bt, snap.get("supmom_k", config.SUPMOM_K), horizon)
    ev_index = _event_index(eventbt)
    floor = config.IC_FLOOR

    edges = {t: _event_edge(v.get("recent_events") or [], ev_index, horizon, config.EVENT_MIN_T)
             for t, v in sig.items()}
    event_norm = _scale_around_zero({t: e for t, (e, _) in edges.items()})

    raw_blend, parts = {}, {}
    for t, v in sig.items():
        comps: dict[str, dict] = {}
        sz, supz = v.get("sentiment_z"), v.get("supplier_z")
        if sz is not None and v.get("sentiment") is not None and abs(ic_sent) >= floor:
            c = round(sz * ic_sent, 4)
            comps["sentiment"] = {
                "signal": v.get("sentiment"), "label": v.get("sentiment_label"),
                "z": sz, "ic": round(ic_sent, 4), "ic_source": sent_src, "contribution": c,
                "reason": f"Claude sentiment {v.get('sentiment_label')} "
                          f"({(v.get('sentiment') or 0):+.2f}); {horizon}d IC {ic_sent:+.3f} ({sent_src})"}
        if supz is not None and v.get("supplier_mom") is not None and abs(ic_sup) >= floor:
            c = round(supz * ic_sup, 4)
            comps["supplier_leadlag"] = {
                "signal": round(v.get("supplier_mom"), 4), "suppliers": v.get("supplier_n", 0),
                "z": supz, "ic": round(ic_sup, 4), "ic_source": sup_src, "contribution": c,
                "reason": f"Suppliers' {snap.get('supmom_k')}d momentum "
                          f"{(v.get('supplier_mom') or 0) * 100:+.2f}% (n={v.get('supplier_n', 0)}); "
                          f"lead-lag IC {ic_sup:+.3f}"}
        edge, ev_reasons = edges[t]
        if edge:
            en = event_norm.get(t, 0.0)
            c = round(en * config.EVENT_IC, 4)
            comps["event_drift"] = {
                "edge": edge, "z": en, "ic": config.EVENT_IC, "events": ev_reasons,
                "contribution": c, "reason": "; ".join(ev_reasons)}
        raw_blend[t] = round(sum(c["contribution"] for c in comps.values()), 4)
        parts[t] = comps

    spread = st.pstdev(raw_blend.values()) if len(raw_blend) > 1 else 0.0
    convs = sorted(((b / spread) if spread else 0.0) for b in raw_blend.values())

    out: dict[str, dict] = {}
    for t, v in sig.items():
        comps, blend = parts[t], raw_blend[t]
        conv = round(blend / spread, 4) if spread else 0.0
        breadth = len(comps)
        agree = (sum(1 for c in comps.values() if _sign(c["contribution"]) == _sign(blend) and c["contribution"])
                 / breadth) if breadth else 0.0
        if breadth == 0:
            direction, confidence = "flat", "none"
        else:
            direction = "long" if conv >= tau else "short" if conv <= -tau else "flat"
            confidence = ("high" if breadth >= 2 and agree == 1
                          else "medium" if breadth >= 2 else "low")
        rank = bisect.bisect_right(convs, conv)
        verdict = {
            "ticker": t, "as_of": snap.get("as_of"), "horizon_days": horizon,
            "direction": direction, "conviction": conv,
            "strength": round(math.tanh(abs(conv)), 3), "confidence": confidence,
            "breadth": breadth, "agreement": round(agree, 2),
            "percentile": round(100 * rank / len(convs)) if convs else None,
            "components": comps, "coverage": True, "price_source": snap.get("price_source"),
        }
        verdict["recommendation"] = _reco_line(verdict)
        out[t] = verdict
    return out


def apply_overlay(v: dict, overlay: float, weight: float, tau: float) -> dict:
    """Blend the caller's own normalized signal (positive = long) into the vault
    conviction: combined = (1−λ)·vault + λ·overlay. Lets an external engine 'overlap'
    its strategies with the vault's. Re-derives direction from the combined score."""
    lam = max(0.0, min(1.0, weight))
    v = dict(v)
    base = v["conviction"]
    combined = round((1 - lam) * base + lam * overlay, 4)
    v["base_direction"], v["base_conviction"] = v["direction"], base
    v["overlay"] = {"signal": overlay, "weight": lam}
    v["conviction"] = combined
    v["strength"] = round(math.tanh(abs(combined)), 3)
    v["direction"] = "long" if combined >= tau else "short" if combined <= -tau else "flat"
    v["overlay_applied"] = True
    v["recommendation"] = _reco_line(v)
    return v


# ── Snapshot building (IO: prices + cached sentiment/filings) ─────────────────


def _current_supplier_momentum(tickers: list[str], k: int, days: int):
    """Each customer's *latest* supplier-momentum signal (mean of its suppliers'
    trailing-k cumulative return), reusing the backtest's signal construction."""
    import numpy as np
    import pandas as pd

    series, source = signals.fetch_prices(tickers, days)
    have = [t for t in tickers if t in series and len(series[t]) > 50]
    df = pd.DataFrame({t: series[t] for t in have}).sort_index()
    rets = np.log(df / df.shift(1)).replace([np.inf, -np.inf], np.nan)
    rk = rets.rolling(k).sum()
    sig, customers = backtest._signal_frame(rk, backtest._supplier_map())
    mom, n = {}, {}
    for c in customers:
        col = sig[c].dropna()
        if col.empty:
            continue
        mom[c] = round(float(col.iloc[-1]), 5)
        n[c] = sum(1 for s in backtest._supplier_map().get(c, set()) if s in rets.columns)
    return mom, n, source


def _build_snapshot(tickers: list[str], days: int) -> dict:
    today = dt.date.today().isoformat()
    mom, supn, source = _current_supplier_momentum(tickers, config.SUPMOM_K, days)
    sig: dict[str, dict] = {}
    for t in tickers:
        s = sentiment.load(t) or {}
        f = filings.load(t) or {}
        sig[t] = {
            "sentiment": s.get("score"),
            "sentiment_label": s.get("label"),
            "supplier_mom": mom.get(t),
            "supplier_n": supn.get(t, 0),
            "recent_events": _recent_events(f.get("events") or [], today, config.EVENT_LOOKBACK_DAYS),
        }
    sent_z = _zscore({t: v["sentiment"] for t, v in sig.items()})
    sup_z = _zscore({t: v["supplier_mom"] for t, v in sig.items()})
    for t, v in sig.items():
        v["sentiment_z"] = sent_z.get(t)
        v["supplier_z"] = sup_z.get(t)
    return {"as_of": today, "price_source": source, "supmom_k": config.SUPMOM_K,
            "lookback_days": days, "universe": sorted(sig), "signals": sig}


# ── Read-side: cached book + verdicts for the API ────────────────────────────


_BOOK_CACHE: dict[tuple, tuple[dict, dict]] = {}


def _load_json(path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (json.JSONDecodeError, OSError):
        return {}


def load_snapshot() -> dict | None:
    return _load_json(_OUT) or None


def _book(horizon: int, tau: float) -> tuple[dict | None, dict | None]:
    """The whole universe's verdicts at (horizon, τ), cached on the snapshot mtime
    so repeated API calls don't re-score."""
    if not _OUT.exists():
        return None, None
    key = (round(_OUT.stat().st_mtime, 3), horizon, round(tau, 4))
    if key not in _BOOK_CACHE:
        snap = _load_json(_OUT)
        bk = build_book(snap, backtest.load() or {},
                        _load_json(config.SIGNALS_DIR / "sentiment_backtest.json"),
                        _load_json(config.SIGNALS_DIR / "event_backtest.json"), horizon, tau)
        _BOOK_CACHE.clear()           # one (horizon, τ) at a time is plenty
        _BOOK_CACHE[key] = (snap, bk)
    return _BOOK_CACHE[key]


def verdict(ticker: str, horizon: int | None = None, tau: float | None = None,
            overlay: float | None = None, overlay_weight: float = 1.0) -> dict:
    """One LONG/SHORT/FLAT verdict for ``ticker`` (the external engine's entry point)."""
    horizon = int(horizon or config.TRADE_HORIZON)
    tau = config.CONVICTION_TAU if tau is None else float(tau)
    t = ticker.upper().strip()
    _snap, bk = _book(horizon, tau)
    if bk is None:
        return {"ticker": t, "coverage": False, "direction": "flat", "confidence": "none",
                "reason": "signal snapshot not built — run `python -m sp500_vault.pipeline trades`"}
    v = bk.get(t)
    if v is None:
        return {"ticker": t, "coverage": False, "direction": "flat", "confidence": "none",
                "reason": "ticker not in the vault universe / no data"}
    if overlay is not None:
        v = apply_overlay(v, float(overlay), float(overlay_weight), tau)
    return v


def book(horizon: int | None = None, tau: float | None = None) -> list[dict]:
    """Every verdict, ranked by conviction (LONG → SHORT). For batch calls + notes."""
    horizon = int(horizon or config.TRADE_HORIZON)
    tau = config.CONVICTION_TAU if tau is None else float(tau)
    _snap, bk = _book(horizon, tau)
    if bk is None:
        return []
    return sorted(bk.values(), key=lambda v: v["conviction"], reverse=True)


# ── Render + run ─────────────────────────────────────────────────────────────


def _render_md(rows: list[dict], snap: dict, horizon: int, tau: float) -> None:
    longs = [r for r in rows if r["direction"] == "long"]
    shorts = [r for r in rows if r["direction"] == "short"]
    lines = [
        "# ⚡ Signal Engine — long / short / flat by blended conviction",
        f"_As of {snap.get('as_of')} · {horizon}-day horizon · τ={tau}σ · "
        f"price source {snap.get('price_source')}_",
        "",
        "IC-weighted blend of the vault's three validated strategies "
        "(sentiment lead-lag, supplier lead-lag, 8-K event drift). Conviction is the "
        "blend z-scored across the universe; a name trades only when |conviction| ≥ τ "
        "**and** it has a validated edge. Poll it from a trading engine via "
        "`GET /signal/{ticker}`.",
        "",
        f"**{len(longs)} long · {len(shorts)} short · "
        f"{len(rows) - len(longs) - len(shorts)} flat** of {len(rows)} names.",
        "",
        "| Dir | Ticker | Conviction (σ) | Conf | Breadth | Why |",
        "|---|---|---:|---|---:|---|",
    ]
    icon = {"long": "🟢 LONG", "short": "🔴 SHORT", "flat": "⚪ flat"}
    for r in longs + shorts:
        why = "; ".join(c["reason"] for c in r["components"].values())
        lines.append(f"| {icon[r['direction']]} | [[{r['ticker']}]] | {r['conviction']:+.2f} | "
                     f"{r['confidence']} | {r['breadth']} | {why[:90]} |")
    lines += ["", "_Snapshot refreshes daily (`pipeline trades`). Horizon and τ are "
              "query params on the API, so the same snapshot serves any holding period._"]
    _MD.write_text("\n".join(lines), encoding="utf-8")


def run(tickers: list[str] | None = None, days: int = 400, force: bool = False) -> dict:
    tickers = list(tickers or TICKERS)
    print(f"[engine] building signal snapshot for {len(tickers)} tickers "
          f"(supplier-mom k={config.SUPMOM_K}, {days}d prices)…")
    snap = _build_snapshot(tickers, days)
    config.SIGNALS_DIR.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(json.dumps(snap, indent=2), encoding="utf-8")
    _BOOK_CACHE.clear()
    rows = book(config.TRADE_HORIZON, config.CONVICTION_TAU)
    _render_md(rows, snap, config.TRADE_HORIZON, config.CONVICTION_TAU)
    n_long = sum(1 for r in rows if r["direction"] == "long")
    n_short = sum(1 for r in rows if r["direction"] == "short")
    print(f"[engine] {config.TRADE_HORIZON}d horizon: {n_long} long, {n_short} short, "
          f"{len(rows) - n_long - n_short} flat -> {_OUT.name} + vault/_Trades.md")
    return snap
