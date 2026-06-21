"""Market fundamentals: Financial Modeling Prep (primary) with yfinance fallback.

yfinance is free and fine for v1, but its fundamentals are scraped and patchy.
FMP gives cleaner TTM ratios/margins from filings. Source is chosen by
``config.FUNDAMENTALS_SOURCE``: ``auto`` uses FMP when ``FMP_API_KEY`` is set and
falls back to yfinance otherwise (or when an FMP call fails). Either way the
return is the **same flat dict** — derived metrics / sector percentiles are
computed in ``quant.py`` — and field *scales* are normalized to the yfinance
convention so a mixed run (some FMP, some fallback) stays internally consistent
(notably debt/equity, which downstream `_de` treats as a percent).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import requests
import yfinance as yf

from .. import config

_TIMEOUT = 12
# FMP's "stable" API (the v3 "/api/v3" endpoints were retired Aug-2024 and 403 for
# new keys). Stable takes the ticker as a ``symbol=`` query param.
_FMP_BASE = "https://financialmodelingprep.com/stable"


# ── shared numeric helpers ───────────────────────────────────────────────────


def _safe(d: dict, key: str) -> Any:
    v = d.get(key)
    if v is None:
        return None
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def _fnum(v: Any) -> float | None:
    """Coerce to a finite float, else None (FMP fields can be strings/null)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if (math.isnan(f) or math.isinf(f)) else f


def _iint(v: Any) -> int | None:
    f = _fnum(v)
    return int(f) if f is not None else None


def _cagr(series: list[float], years: int) -> float | None:
    """Compound annual growth rate from an ordered (oldest->newest) value list."""
    vals = [v for v in series if v is not None and v > 0]
    if len(vals) < 2:
        return None
    first, last = vals[0], vals[-1]
    n = min(years, len(vals) - 1)
    if first <= 0 or n <= 0:
        return None
    return (last / first) ** (1 / n) - 1


def _annualized_vol_30d(tk: yf.Ticker) -> float | None:
    try:
        hist = tk.history(period="3mo", interval="1d")["Close"].dropna()
        if len(hist) < 20:
            return None
        returns = np.log(hist / hist.shift(1)).dropna().tail(30)
        if returns.empty:
            return None
        return float(returns.std() * math.sqrt(252))
    except Exception:  # noqa: BLE001
        return None


# ── yfinance source ──────────────────────────────────────────────────────────


def _revenue_eps_cagr(tk: yf.Ticker) -> tuple[float | None, float | None]:
    """3yr revenue and EPS CAGR from the annual income statement."""
    rev_cagr = eps_cagr = None
    try:
        stmt = tk.income_stmt  # columns are periods, newest first
        if stmt is not None and not stmt.empty:
            cols = list(stmt.columns)[::-1]  # oldest -> newest

            def row(label: str) -> list[float] | None:
                if label in stmt.index:
                    return [stmt.loc[label, c] for c in cols]
                return None

            rev = row("Total Revenue")
            if rev:
                rev_cagr = _cagr([float(x) for x in rev if x is not None], 3)
            eps = row("Diluted EPS") or row("Basic EPS")
            if eps:
                eps_cagr = _cagr([float(x) for x in eps if x is not None], 3)
    except Exception:  # noqa: BLE001
        pass
    return rev_cagr, eps_cagr


def _yf_fundamentals(ticker: str) -> dict[str, Any]:
    """Pull a flat dict of fundamentals for one ticker from yfinance."""
    tk = yf.Ticker(ticker)
    info = tk.get_info() if hasattr(tk, "get_info") else tk.info
    rev_cagr, eps_cagr = _revenue_eps_cagr(tk)

    return {
        "ticker": ticker,
        "name": _safe(info, "longName") or _safe(info, "shortName") or ticker,
        "sector": _safe(info, "sector"),
        "industry": _safe(info, "industry"),
        "market_cap": _safe(info, "marketCap"),
        "enterprise_value": _safe(info, "enterpriseValue"),
        "employees": _safe(info, "fullTimeEmployees"),
        # Valuation
        "pe_ttm": _safe(info, "trailingPE"),
        "ps_ttm": _safe(info, "priceToSalesTrailing12Months"),
        "pb": _safe(info, "priceToBook"),
        "ev_ebitda": _safe(info, "enterpriseToEbitda"),
        "peg": _safe(info, "trailingPegRatio") or _safe(info, "pegRatio"),
        # Growth
        "revenue_growth_yoy": _safe(info, "revenueGrowth"),
        "earnings_growth_yoy": _safe(info, "earningsGrowth"),
        "revenue_cagr_3yr": rev_cagr,
        "eps_cagr_3yr": eps_cagr,
        # Profitability
        "gross_margin": _safe(info, "grossMargins"),
        "operating_margin": _safe(info, "operatingMargins"),
        "net_margin": _safe(info, "profitMargins"),
        "roe": _safe(info, "returnOnEquity"),
        "roa": _safe(info, "returnOnAssets"),
        # Leverage / risk
        "debt_to_equity": _safe(info, "debtToEquity"),     # yfinance convention: percent (180.5 == 1.80x)
        "current_ratio": _safe(info, "currentRatio"),
        "beta": _safe(info, "beta"),
        "volatility_30d": _annualized_vol_30d(tk),
        "data_source": "yfinance",
    }


# ── FMP source ───────────────────────────────────────────────────────────────


def _fmp_get(endpoint: str, symbol: str, extra: dict | None = None) -> Any:
    if not config.FMP_API_KEY:
        return None
    params = {"symbol": symbol, "apikey": config.FMP_API_KEY}
    if extra:
        params.update(extra)
    try:
        r = requests.get(f"{_FMP_BASE}/{endpoint}", params=params, timeout=_TIMEOUT)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:  # noqa: BLE001
        return None


def _fmp_growth(income: list[dict]) -> tuple[float | None, float | None, float | None, float | None]:
    """(rev_cagr_3yr, eps_cagr_3yr, rev_yoy, eps_yoy) from annual income statements
    (FMP returns them newest-first)."""
    if not income:
        return None, None, None, None
    revs = [_fnum(x.get("revenue")) for x in income]
    eps = [_fnum(x.get("epsDiluted")) if x.get("epsDiluted") is not None else _fnum(x.get("eps"))
           for x in income]

    def yoy(s: list[float | None]) -> float | None:
        if len(s) >= 2 and s[0] is not None and s[1] not in (None, 0):
            return (s[0] - s[1]) / abs(s[1])
        return None

    rev_cagr = _cagr([v for v in reversed(revs) if v is not None], 3)
    eps_cagr = _cagr([v for v in reversed(eps) if v is not None], 3)
    return rev_cagr, eps_cagr, yoy(revs), yoy(eps)


def _map_fmp(ticker: str, profile: list, key_metrics: list, ratios: list,
             income: list, volatility: float | None) -> dict[str, Any]:
    """Pure mapping of FMP payloads -> the common fundamentals dict.

    Kept free of network/IO so the field mapping and scale normalization are
    unit-testable. Returns ``{}`` if there is no usable profile.
    """
    if not profile:
        return {}
    p = profile[0] if isinstance(profile, list) else profile
    km = (key_metrics[0] if isinstance(key_metrics, list) and key_metrics else {}) or {}
    rt = (ratios[0] if isinstance(ratios, list) and ratios else {}) or {}
    rev_cagr, eps_cagr, rev_yoy, eps_yoy = _fmp_growth(income if isinstance(income, list) else [])

    def pick(*vals):
        for v in vals:
            n = _fnum(v)
            if n is not None:
                return n
        return None

    # FMP reports debt/equity as a ratio (1.80); yfinance/downstream `_de` expects
    # a percent (180.5). Scale to keep both sources on one convention.
    dte = pick(rt.get("debtToEquityRatioTTM"))

    return {
        "ticker": ticker,
        "name": p.get("companyName") or ticker,
        "sector": p.get("sector") or None,
        "industry": p.get("industry") or None,
        "market_cap": pick(p.get("marketCap"), km.get("marketCap")),
        "enterprise_value": pick(km.get("enterpriseValueTTM"), rt.get("enterpriseValueTTM")),
        "employees": _iint(p.get("fullTimeEmployees")),
        # Valuation (PE/PS/PB/PEG live in ratios-ttm on the stable API)
        "pe_ttm": pick(rt.get("priceToEarningsRatioTTM")),
        "ps_ttm": pick(rt.get("priceToSalesRatioTTM")),
        "pb": pick(rt.get("priceToBookRatioTTM")),
        "ev_ebitda": pick(km.get("evToEBITDATTM"), rt.get("enterpriseValueMultipleTTM")),
        "peg": pick(rt.get("priceToEarningsGrowthRatioTTM")),
        # Growth
        "revenue_growth_yoy": rev_yoy,
        "earnings_growth_yoy": eps_yoy,
        "revenue_cagr_3yr": rev_cagr,
        "eps_cagr_3yr": eps_cagr,
        # Profitability (margins in ratios-ttm; ROE/ROA in key-metrics-ttm)
        "gross_margin": pick(rt.get("grossProfitMarginTTM")),
        "operating_margin": pick(rt.get("operatingProfitMarginTTM")),
        "net_margin": pick(rt.get("netProfitMarginTTM"), rt.get("bottomLineProfitMarginTTM")),
        "roe": pick(km.get("returnOnEquityTTM")),
        "roa": pick(km.get("returnOnAssetsTTM")),
        # Leverage / risk
        "debt_to_equity": dte * 100 if dte is not None else None,
        "current_ratio": pick(km.get("currentRatioTTM"), rt.get("currentRatioTTM")),
        "beta": _fnum(p.get("beta")),
        "volatility_30d": volatility,
        "data_source": "fmp",
    }


def _fmp_fundamentals(ticker: str) -> dict[str, Any]:
    """Fetch + map FMP fundamentals; ``{}`` on failure so the caller can fall back."""
    profile = _fmp_get("profile", ticker)
    if not profile or not isinstance(profile, list):
        return {}
    key_metrics = _fmp_get("key-metrics-ttm", ticker) or []
    ratios = _fmp_get("ratios-ttm", ticker) or []
    # FMP's free tier 402s the fundamentals endpoints for most symbols (only a
    # sample set is covered). If both come back empty, treat the ticker as
    # uncovered and return {} so the caller falls back to yfinance — and skip the
    # remaining (income + price-history) calls.
    if not ratios and not key_metrics:
        return {}
    income = _fmp_get("income-statement", ticker, {"period": "annual", "limit": "5"}) or []
    # Volatility is price-derived; yfinance history is free, so reuse it.
    vol = _annualized_vol_30d(yf.Ticker(ticker))
    return _map_fmp(ticker, profile, key_metrics, ratios, income, vol)


# ── dispatcher ───────────────────────────────────────────────────────────────


def fetch_fundamentals(ticker: str) -> dict[str, Any]:
    """Fundamentals for one ticker from the configured source (FMP or yfinance)."""
    src = (config.FUNDAMENTALS_SOURCE or "auto").lower()
    use_fmp = src == "fmp" or (src == "auto" and config.FMP_API_KEY)
    if use_fmp:
        data = _fmp_fundamentals(ticker)
        if data:                 # FMP covered this ticker (has real metrics)
            return data
        # Uncovered (free-tier 402) or failed — fall back to yfinance.
    return _yf_fundamentals(ticker)
