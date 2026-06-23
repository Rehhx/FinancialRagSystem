"""Phase 7: per-layer refresh scheduler.

The layers age at very different rates — sentiment is perishable (daily/weekly),
fundamentals update on earnings (~quarterly), relationships on the 10-K (~yearly).
This scheduler runs each layer only when it's due, then re-renders the vault,
re-indexes, and re-exports the graph if anything actually changed.

Usage:
    python -m sp500_vault.scheduler tick           # one-shot (cron-friendly)
    python -m sp500_vault.scheduler run --interval 3600   # long-running daemon
    python -m sp500_vault.scheduler status         # show what's due

Wire `tick` into Task Scheduler / cron to keep the vault self-updating.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time

from . import (backtest, config, engine, filings, graph_export, quant, relationships,
               sentiment, signals, vault_render, rag)
from .universe import TICKERS

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

_STATE = config.DATA_DIR / "scheduler_state.json"

# Refresh cadence per layer, in hours.
CADENCE_HOURS = {
    "sentiment": 24,          # daily
    "filings": 24,            # daily (8-K events are sporadic but time-sensitive)
    "signals": 24,            # daily (price-based)
    "backtest": 24,           # daily (price-based)
    "engine": 24,             # daily — rebuild the LONG/SHORT/FLAT trade snapshot
    "quant": 24 * 90,         # ~quarterly
    "relationships": 24 * 365,  # ~annually
}

_RUNNERS = {
    "quant": quant.run,
    "relationships": relationships.run,
    "sentiment": sentiment.run,
    "filings": filings.run,
    "signals": signals.run,
    "backtest": backtest.run,
    "engine": engine.run,
}


def _now() -> dt.datetime:
    """Naive UTC now (avoids the deprecated utcnow; stored timestamps are naive)."""
    return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)


def _load_state() -> dict:
    if _STATE.exists():
        return json.loads(_STATE.read_text(encoding="utf-8"))
    return {}


def _save_state(state: dict) -> None:
    _STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _due(layer: str, state: dict, now: dt.datetime) -> bool:
    last = state.get(layer)
    if not last:
        return True
    elapsed = (now - dt.datetime.fromisoformat(last)).total_seconds() / 3600
    return elapsed >= CADENCE_HOURS[layer]


def status() -> dict:
    state = _load_state()
    now = _now()
    report = {}
    for layer in CADENCE_HOURS:
        report[layer] = {
            "last_run": state.get(layer, "never"),
            "due": _due(layer, state, now),
            "cadence_hours": CADENCE_HOURS[layer],
        }
    for layer, info in report.items():
        flag = "DUE" if info["due"] else "ok"
        print(f"  {layer:14} {flag:4} last={info['last_run']}")
    return report


def tick(force: list[str] | None = None) -> bool:
    """Run any due (or forced) layers, then refresh downstream. Returns True if anything ran."""
    state = _load_state()
    now = _now()
    force = set(force or [])
    ran = False

    # data layers first; engine last so it rebuilds the snapshot from the freshly
    # refreshed sentiment + 8-K caches (and pulls live prices itself).
    for layer in ("quant", "relationships", "sentiment", "filings", "signals", "backtest", "engine"):
        if layer in force or _due(layer, state, now):
            print(f"[sched] running due layer: {layer}")
            _RUNNERS[layer](list(TICKERS), force=(layer in force))
            state[layer] = now.isoformat()
            ran = True

    if ran:
        print("[sched] change detected -> re-rendering vault, index, graph")
        vault_render.run(list(TICKERS))
        rag.index_vault()
        graph_export.run()
        _save_state(state)
        print("[sched] refresh complete")
    else:
        print("[sched] nothing due")
    return ran


def run_daemon(interval: int) -> None:
    print(f"[sched] daemon started (interval {interval}s) — Ctrl-C to stop")
    while True:
        tick()
        time.sleep(interval)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="sp500_vault.scheduler", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status", help="Show which layers are due")
    tp = sub.add_parser("tick", help="Run due layers once")
    tp.add_argument("--force", help="Comma-separated layers to force (quant,relationships,sentiment)")
    rp = sub.add_parser("run", help="Run as a long-lived daemon")
    rp.add_argument("--interval", type=int, default=3600, help="Seconds between ticks")

    args = parser.parse_args(argv)
    if args.cmd == "status":
        status()
    elif args.cmd == "tick":
        tick(force=[s.strip() for s in args.force.split(",")] if args.force else None)
    elif args.cmd == "run":
        run_daemon(args.interval)


if __name__ == "__main__":
    main()
