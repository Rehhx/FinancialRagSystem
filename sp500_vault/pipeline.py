"""Pipeline orchestrator CLI.

Each layer refreshes on its own schedule (quant quarterly, relationships
annually, sentiment daily/weekly) — so every layer is its own subcommand and
they do not depend on each other's freshness.

Examples:
    python -m sp500_vault.pipeline all
    python -m sp500_vault.pipeline quant --tickers NVDA,AMD,AAPL
    python -m sp500_vault.pipeline sentiment --limit 5
    python -m sp500_vault.pipeline query "Which companies are most exposed to NVDA?"
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys

from . import (archive, backtest, config, evaluation, event_backtest, filings, graph_export,
               quant, relationships, sentiment, sentiment_backtest, signals, vault_render, rag)
from .universe import SECTORS, TICKERS, tickers_for_group

# Windows consoles default to cp1252 and choke on em-dashes/arrows in model output.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass


def _select(args) -> list[str]:
    raw = getattr(args, "tickers", None)
    sector = getattr(args, "sector", None)
    if raw:
        tickers = [t.strip().upper() for t in raw.split(",")]
    elif sector:
        tickers = tickers_for_group(sector)
    else:
        tickers = list(TICKERS)
    if getattr(args, "limit", None):
        tickers = tickers[: args.limit]
    return tickers


def _add_selection(p: argparse.ArgumentParser) -> None:
    p.add_argument("--tickers", help="Comma-separated tickers (default: full pilot universe)")
    p.add_argument("--sector", choices=sorted(SECTORS), help="Restrict to one pilot cluster")
    p.add_argument("--limit", type=int, help="Cap to the first N tickers (handy for cheap test runs)")
    p.add_argument("--force", action="store_true",
                   help="Re-run even if cached/fresh (relationships re-extract, sentiment re-score)")


def _stats() -> None:
    """Print a coverage/status snapshot across every layer."""
    print("S&P 500 RAG Vault — status\n")
    print(f"  universe        {len(TICKERS)} tickers across {len(SECTORS)} clusters")
    qfiles = list(config.QUANT_DIR.glob("*.json"))
    src_counts: dict[str, int] = {}
    for p in qfiles:
        try:
            src = json.loads(p.read_text(encoding="utf-8")).get("data_source") or "unknown"
        except Exception:  # noqa: BLE001
            src = "unknown"
        src_counts[src] = src_counts.get(src, 0) + 1
    src_str = ", ".join(f"{v} {k}" for k, v in sorted(src_counts.items())) if src_counts else "—"
    print(f"  quant           {len(qfiles)}/{len(TICKERS)} notes  [{src_str}]")

    sj = list(config.SENTIMENT_DIR.glob("*.json"))
    dates = []
    for p in sj:
        try:
            dates.append(json.loads(p.read_text(encoding="utf-8")).get("as_of"))
        except Exception:  # noqa: BLE001
            pass
    latest = max([d for d in dates if d], default="—")
    print(f"  sentiment       {len(sj)}/{len(TICKERS)} scored (latest {latest})")

    fj = list(config.FILINGS_DIR.glob("*.json"))
    if fj:
        n_events = 0
        for p in fj:
            try:
                n_events += json.loads(p.read_text(encoding="utf-8")).get("event_count", 0)
            except Exception:  # noqa: BLE001
                pass
        print(f"  filings (8-K)   {len(fj)}/{len(TICKERS)} tickers, {n_events} material events")

    af, an = config.DATA_DIR / "archive" / "filings.csv", config.DATA_DIR / "archive" / "news.csv"
    if af.exists() or an.exists():
        nf = max(0, sum(1 for _ in af.open(encoding="utf-8")) - 1) if af.exists() else 0
        nn = max(0, sum(1 for _ in an.open(encoding="utf-8")) - 1) if an.exists() else 0
        print(f"  archive         {nf} filings, {nn} headlines (append-only, for event studies)")

    if config.RELATIONSHIPS_DB.exists():
        c = sqlite3.connect(config.RELATIONSHIPS_DB)
        total = c.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
        resolved = c.execute("SELECT COUNT(*) FROM edges WHERE target_ticker IS NOT NULL").fetchone()[0]
        by_method = dict(c.execute("SELECT extraction_method, COUNT(*) FROM edges GROUP BY extraction_method").fetchall())
        c.close()
        print(f"  relationships   {total} edges, {resolved} resolved  {by_method}")

    sig = signals.load()
    if sig:
        s = sig["summary"]
        print(f"  signals         linked {s['linked_mean']} vs unlinked {s['unlinked_mean']} "
              f"(lift {s['lift']}; {sig['source']}, {sig['as_of']})")

    bt = backtest.load()
    if bt:
        p, b = bt["portfolio"], bt.get("best")
        print(f"  backtest        L/S Sharpe {p['sharpe']}, best IC "
              f"{b['ic'] if b else '—'} (k{b['k']}->h{b['h']}); {bt['as_of']}")

    eb = config.SIGNALS_DIR / "event_backtest.json"
    if eb.exists():
        e = json.loads(eb.read_text(encoding="utf-8"))
        print(f"  event backtest  {e['events']} 8-Ks studied; forward returns by item type")
    sb = config.SIGNALS_DIR / "sentiment_backtest.json"
    if sb.exists():
        d = json.loads(sb.read_text(encoding="utf-8"))
        ic1 = next((c["rank_ic"] for c in d.get("sentiment_ic", []) if c["h"] == 1), None)
        print(f"  sentiment lead-lag  {d['n_obs']} ticker-day obs, 1d rank-IC {ic1}")

    notes = [p for p in config.VAULT_DIR.glob("*.md")
             if not p.stem.endswith("_news_log") and not p.stem.startswith("_")]
    print(f"  vault notes     {len(notes)}")

    print(f"  rag index       {rag.count()} chunks (LangChain + OpenAI embeddings)")

    ev_file = config.BASE_DIR / "eval" / "eval_report.json"
    if ev_file.exists():
        ev = json.loads(ev_file.read_text(encoding="utf-8")).get("aggregate", {})
        print(f"  rag eval        recall@{ev.get('k')} {ev.get('recall_at_k')}, "
              f"MRR {ev.get('mrr')}, hit-rate {ev.get('hit_rate')}")

    gf = config.DATA_DIR / "graph" / "graph.json"
    if gf.exists():
        g = json.loads(gf.read_text(encoding="utf-8"))
        print(f"  graph export    {len(g['nodes'])} nodes, {len(g['edges'])} edges")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="sp500_vault", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    no_selection = {"index", "export", "dashboard", "stats", "eventbacktest", "sentimentbacktest"}
    for name in ("quant", "relationships", "sentiment", "filings", "archive", "signals",
                 "backtest", "eventbacktest", "sentimentbacktest", "vault", "index",
                 "dashboard", "export", "stats", "all"):
        sp = sub.add_parser(name, help=f"Run the {name} layer")
        if name not in no_selection:
            _add_selection(sp)
        elif name == "index":
            sp.add_argument("--force", action="store_true",
                            help="Rebuild the index from scratch (re-embed every chunk)")

    qp = sub.add_parser("query", help="Ask the vault a question")
    qp.add_argument("question")
    qp.add_argument("--k", type=int, default=6, help="Chunks to retrieve")
    qp.add_argument("--ticker", help="Restrict to one ticker")
    qp.add_argument("--sector", help="Restrict to one sector")
    qp.add_argument("--sentiment", help="Restrict to a sentiment label (Bullish/Neutral/Bearish)")

    ep = sub.add_parser("eval", help="Evaluate retrieval quality against the golden set")
    ep.add_argument("--k", type=int, default=8, help="Chunks to retrieve per question")
    ep.add_argument("--judge", action="store_true", help="Also LLM-judge answer faithfulness (uses Claude)")

    args = parser.parse_args(argv)

    if args.cmd == "query":
        result = rag.query(args.question, k=args.k, ticker=args.ticker,
                           sector=args.sector, sentiment=args.sentiment)
        print("\n" + result["answer"] + "\n")
        print("Sources:", ", ".join(f"{s['ticker']}·{s['section']}" for s in result["sources"]))
        return
    if args.cmd == "export":
        graph_export.run()
        return
    if args.cmd == "dashboard":
        vault_render.render_dashboard()
        return
    if args.cmd == "stats":
        _stats()
        return
    if args.cmd == "eval":
        evaluation.run(k=args.k, judge=args.judge)
        return
    if args.cmd == "eventbacktest":
        event_backtest.run()
        return
    if args.cmd == "sentimentbacktest":
        sentiment_backtest.run()
        return

    tickers = _select(args)
    force = getattr(args, "force", False)
    if args.cmd in ("quant", "all"):
        quant.run(tickers, force=force)
    if args.cmd in ("relationships", "all"):
        relationships.run(tickers, force=force)
    if args.cmd in ("sentiment", "all"):
        sentiment.run(tickers, force=force)
    if args.cmd in ("filings", "all"):
        filings.run(tickers, force=force)
    if args.cmd == "archive":
        archive.run(tickers, force=force)
    if args.cmd in ("signals", "all"):
        signals.run(tickers, force=force)
    if args.cmd in ("backtest", "all"):
        backtest.run(tickers, force=force)
    if args.cmd in ("vault", "all"):
        vault_render.run(tickers)
    if args.cmd in ("index", "all"):
        rag.index_vault(force=getattr(args, "force", False))
    if args.cmd == "all":
        graph_export.run()


if __name__ == "__main__":
    main()
