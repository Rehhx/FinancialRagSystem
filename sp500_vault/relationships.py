"""Relationship layer: build the supplier/customer/competitor edge graph.

Three-tier sourcing per the plan:
  Tier 1  Finnhub peers          -> competitor seeds (cheap, structured)
  Tier 2  EDGAR 10-K + Claude    -> the bulk of typed, evidence-backed edges
  Tier 3  manual overrides CSV   -> high-confidence edges tracked separately

The SQLite table is the source of truth for graph structure; markdown notes are
rendered *from* it and can be regenerated freely.
"""
from __future__ import annotations

import csv
import datetime as dt
import sqlite3
from contextlib import closing

from . import config, llm
from .data_sources import edgar, news
from .universe import BY_TICKER, resolve_ticker

_SCHEMA = """
CREATE TABLE IF NOT EXISTS edges (
    source             TEXT NOT NULL,
    target_key         TEXT NOT NULL,
    target_ticker      TEXT,
    target_name        TEXT,
    relation           TEXT NOT NULL,
    confidence         TEXT,
    source_doc         TEXT,
    extraction_method  TEXT,
    evidence           TEXT,
    estimated_revenue_pct REAL,
    created_at         TEXT,
    UNIQUE(source, target_key, relation)
);
"""


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(config.RELATIONSHIPS_DB)
    conn.execute(_SCHEMA)
    return conn


def _upsert(conn: sqlite3.Connection, edge: dict) -> None:
    conn.execute(
        """
        INSERT INTO edges (source, target_key, target_ticker, target_name, relation,
                           confidence, source_doc, extraction_method, evidence,
                           estimated_revenue_pct, created_at)
        VALUES (:source, :target_key, :target_ticker, :target_name, :relation,
                :confidence, :source_doc, :extraction_method, :evidence,
                :estimated_revenue_pct, :created_at)
        ON CONFLICT(source, target_key, relation) DO UPDATE SET
            target_ticker=excluded.target_ticker,
            confidence=excluded.confidence,
            source_doc=excluded.source_doc,
            extraction_method=excluded.extraction_method,
            evidence=excluded.evidence,
            created_at=excluded.created_at
        """,
        edge,
    )


def _make_edge(source: str, target_name: str, target_ticker: str | None, relation: str,
               confidence: str, method: str, source_doc: str = "", evidence: str = "",
               revenue_pct: float | None = None) -> dict:
    key = (target_ticker or target_name or "").strip().upper()
    return {
        "source": source,
        "target_key": key,
        "target_ticker": target_ticker or None,
        "target_name": target_name,
        "relation": relation,
        "confidence": confidence,
        "source_doc": source_doc,
        "extraction_method": method,
        "evidence": evidence,
        "estimated_revenue_pct": revenue_pct,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }


def extract_for_ticker(conn: sqlite3.Connection, ticker: str, force: bool = False) -> int:
    """Run Tier-1 + Tier-2 extraction for a single source ticker. Returns edge count.

    Incremental by default: if this ticker has already been LLM-extracted, skip the
    (expensive) EDGAR fetch + Claude call. Pass ``force=True`` to re-extract.
    """
    if not force:
        row = conn.execute(
            "SELECT 1 FROM edges WHERE source = ? AND extraction_method = 'llm_extraction' LIMIT 1",
            (ticker,),
        ).fetchone()
        if row:
            print(f"  [rel] {ticker}: cached (use --force to re-extract)")
            return 0

    name = BY_TICKER[ticker].name if ticker in BY_TICKER else ticker
    count = 0

    # Tier 1 — Finnhub peers (competitor seeds, only within our universe).
    for peer in news.fetch_peers(ticker):
        resolved = resolve_ticker(peer)
        if resolved and resolved != ticker:
            _upsert(conn, _make_edge(
                ticker, BY_TICKER[resolved].name, resolved, "competitor",
                "medium", "finnhub_peers"))
            count += 1

    # Tier 2 — EDGAR 10-K Business section -> Claude structured extraction.
    try:
        filing = edgar.fetch_business_section(ticker)
    except Exception as e:  # noqa: BLE001
        print(f"  [rel] EDGAR failed for {ticker}: {e}")
        filing = None

    if filing and filing.get("text"):
        source_doc = f"{ticker}_10K_{filing.get('filing_date', '')}"
        try:
            rels = llm.extract_relationships(ticker, name, filing["text"])
        except Exception as e:  # noqa: BLE001
            print(f"  [rel] Claude extraction failed for {ticker}: {e}")
            rels = []
        for r in rels:
            resolved = resolve_ticker(r.get("ticker") or r.get("company_name"))
            _upsert(conn, _make_edge(
                ticker,
                r.get("company_name", ""),
                resolved,
                r.get("relation", "competitor"),
                r.get("confidence", "low"),
                "llm_extraction",
                source_doc,
                r.get("evidence", ""),
            ))
            count += 1

    conn.commit()
    print(f"  [rel] {ticker}: {count} edges")
    return count


def load_manual_overrides(conn: sqlite3.Connection) -> int:
    """Merge Tier-3 manual edges from CSV (auditable, separate from extraction)."""
    path = config.MANUAL_OVERRIDES_CSV
    if not path.exists():
        return 0
    count = 0
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            src = (row.get("source") or "").strip().upper()
            tgt = (row.get("target_ticker") or "").strip().upper()
            rel = (row.get("relation") or "").strip().lower()
            if not (src and tgt and rel):
                continue
            tgt_name = BY_TICKER[tgt].name if tgt in BY_TICKER else tgt
            _upsert(conn, _make_edge(
                src, tgt_name, tgt if tgt in BY_TICKER else None, rel,
                row.get("confidence", "high"), "manual_override",
                evidence=row.get("evidence", "")))
            count += 1
    conn.commit()
    print(f"[rel] merged {count} manual override edges")
    return count


def reresolve_targets() -> int:
    """Re-run ticker resolution over unresolved edges.

    When the universe grows, names that were previously "external / not modeled"
    (e.g. Alphabet, Amazon) may now resolve to a modeled ticker — upgrading them
    to wikilinks without re-paying for Claude extraction.
    """
    updated = 0
    with closing(_connect()) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT rowid, target_name FROM edges WHERE target_ticker IS NULL").fetchall()
        for r in rows:
            tic = resolve_ticker(r["target_name"])
            if tic:
                conn.execute("UPDATE edges SET target_ticker = ? WHERE rowid = ?", (tic, r["rowid"]))
                updated += 1
        conn.commit()
    print(f"[rel] re-resolved {updated} external stubs to modeled tickers")
    return updated


def get_edges(source: str, relation: str | None = None, resolved_only: bool = False) -> list[dict]:
    """Fetch edges for a source ticker, optionally filtered by relation/resolution."""
    q = "SELECT * FROM edges WHERE source = ?"
    params: list = [source]
    if relation:
        q += " AND relation = ?"
        params.append(relation)
    if resolved_only:
        q += " AND target_ticker IS NOT NULL"
    q += " ORDER BY relation, confidence DESC"
    with closing(_connect()) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute(q, params).fetchall()]


def connected_market_cap(source: str, quant_loader) -> float:
    """Sum of resolved supplier/customer/competitor market caps (connected node value)."""
    total = 0.0
    seen: set[str] = set()
    for e in get_edges(source, resolved_only=True):
        t = e["target_ticker"]
        if t in seen:
            continue
        seen.add(t)
        note = quant_loader(t)
        if note and note.get("market_cap"):
            total += float(note["market_cap"])
    return total


def run(tickers: list[str], force: bool = False) -> None:
    print(f"[rel] extracting relationships for {len(tickers)} tickers"
          f"{' (force re-extract)' if force else ' (incremental)'}…")
    with closing(_connect()) as conn:
        for t in tickers:
            extract_for_ticker(conn, t, force=force)
        load_manual_overrides(conn)
    reresolve_targets()
    print(f"[rel] done -> {config.RELATIONSHIPS_DB}")
