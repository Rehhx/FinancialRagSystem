"""Quant layer: fundamentals -> per-ticker metrics with sector percentiles.

Reads market fundamentals, computes sector-relative context (percentile + sector
average) for each metric, and writes an enriched JSON per ticker that the vault
renderer consumes. This is the source of the "P/E 31.2 | sector avg 24.1 | 78th
percentile" style tables in the plan.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import pandas as pd

from . import config
from .data_sources import market

# Metrics that get a sector-percentile column. "higher_is_better" drives nothing
# in the math (percentile is just rank) but is recorded for downstream display.
PERCENTILE_METRICS = [
    "pe_ttm", "ps_ttm", "pb", "ev_ebitda", "peg",
    "revenue_growth_yoy", "revenue_cagr_3yr", "eps_cagr_3yr",
    "gross_margin", "operating_margin", "net_margin", "roe", "roa",
    "debt_to_equity", "current_ratio", "beta", "volatility_30d",
]


def _quant_path(ticker: str):
    return config.QUANT_DIR / f"{ticker}.json"


def fetch_all(tickers: list[str], workers: int = 8) -> list[dict[str, Any]]:
    """Fetch raw fundamentals for every ticker (network-bound, parallelized)."""
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(market.fetch_fundamentals, t): t for t in tickers}
        for fut in as_completed(futures):
            t = futures[fut]
            try:
                rows.append(fut.result())
                print(f"  [quant] fetched {t}")
            except Exception as e:  # noqa: BLE001 - keep the batch going
                print(f"  [quant] FAILED {t}: {e}")
    return rows


def compute_and_save(rows: list[dict[str, Any]]) -> dict[str, dict]:
    """Compute sector percentiles and persist one enriched JSON per ticker."""
    if not rows:
        return {}
    df = pd.DataFrame(rows).set_index("ticker")
    df["sector"] = df["sector"].fillna("Unknown")

    # Per (sector, metric): percentile rank and sector average.
    pct: dict[str, dict[str, float]] = {m: {} for m in PERCENTILE_METRICS}
    avg: dict[str, dict[str, float]] = {m: {} for m in PERCENTILE_METRICS}
    for metric in PERCENTILE_METRICS:
        if metric not in df.columns:
            continue
        for sector, grp in df.groupby("sector"):
            vals = pd.to_numeric(grp[metric], errors="coerce")
            ranks = vals.rank(pct=True) * 100.0
            mean = vals.mean()
            for tic in grp.index:
                if pd.notna(ranks.get(tic)):
                    pct[metric][tic] = round(float(ranks[tic]), 1)
                if pd.notna(mean):
                    avg[metric][tic] = round(float(mean), 4)

    enriched: dict[str, dict] = {}
    for tic, raw in df.iterrows():
        raw_d = raw.to_dict()
        metrics = {}
        for m in PERCENTILE_METRICS:
            metrics[m] = {
                "value": raw_d.get(m),
                "sector_avg": avg.get(m, {}).get(tic),
                "percentile": pct.get(m, {}).get(tic),
            }
        note = {
            "ticker": tic,
            "name": raw_d.get("name"),
            "sector": raw_d.get("sector"),
            "industry": raw_d.get("industry"),
            "market_cap": raw_d.get("market_cap"),
            "enterprise_value": raw_d.get("enterprise_value"),
            "employees": raw_d.get("employees"),
            "metrics": metrics,
        }
        _quant_path(tic).write_text(json.dumps(note, indent=2, default=str), encoding="utf-8")
        enriched[tic] = note
    return enriched


def load(ticker: str) -> dict | None:
    p = _quant_path(ticker)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def run(tickers: list[str], force: bool = False) -> dict[str, dict]:
    print(f"[quant] fetching fundamentals for {len(tickers)} tickers…")
    rows = fetch_all(tickers)
    enriched = compute_and_save(rows)
    print(f"[quant] wrote {len(enriched)} enriched notes to {config.QUANT_DIR}")
    return enriched
