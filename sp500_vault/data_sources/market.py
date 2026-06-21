"""Market fundamentals via yfinance (free, fine for v1).

Returns a flat dict of raw fundamentals per ticker. Derived metrics and sector
percentiles are computed in ``quant.py`` so this module stays a thin fetch layer.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import yfinance as yf


def _safe(d: dict, key: str) -> Any:
    v = d.get(key)
    if v is None:
        return None
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


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
    except Exception:
        return None


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
    except Exception:
        pass
    return rev_cagr, eps_cagr


def fetch_fundamentals(ticker: str) -> dict[str, Any]:
    """Pull a flat dict of fundamentals for one ticker."""
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
        "debt_to_equity": _safe(info, "debtToEquity"),
        "current_ratio": _safe(info, "currentRatio"),
        "beta": _safe(info, "beta"),
        "volatility_30d": _annualized_vol_30d(tk),
    }
