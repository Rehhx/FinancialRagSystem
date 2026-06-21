"""Export the relationship graph as JSON for downstream tools.

The markdown vault drives Obsidian's graph view, but a standalone node/edge JSON
is what a future web visualization (React + D3, per the plan's V2 ideas) would
read directly — and it's handy for inspecting graph structure right now.

Output: data/graph/graph.json
    nodes: [{id, label, group, sector, market_cap, sentiment_score, sentiment_label, degree}]
    edges: [{source, target, relation, confidence, method, evidence}]  (modeled only)
"""
from __future__ import annotations

import json
import math

from . import config, quant, relationships, sentiment
from .universe import BY_TICKER, TICKERS


def _clean(obj):
    """Recursively replace NaN/inf floats with None so the JSON is valid.

    yfinance leaves some metrics as NaN; Python would emit a bare `NaN` token
    (invalid JSON — rejected by both the browser's JSON.parse and FastAPI's
    encoder). Coerce those to null.
    """
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    return obj

_CONF_RANK = {"high": 3, "medium": 2, "low": 1}
_GRAPH_DIR = config.DATA_DIR / "graph"


def _modeled_edges() -> list[dict]:
    """All resolved edges, deduped to the best-sourced (source, target, relation)."""
    best: dict[tuple, dict] = {}
    for src in TICKERS:
        for e in relationships.get_edges(src, resolved_only=True):
            tgt = e["target_ticker"]
            if tgt not in BY_TICKER:
                continue
            key = (src, tgt, e["relation"])
            score = _CONF_RANK.get(e.get("confidence"), 0)
            if key not in best or score > _CONF_RANK.get(best[key].get("confidence"), 0):
                best[key] = {
                    "source": src,
                    "target": tgt,
                    "relation": e["relation"],
                    "confidence": e.get("confidence"),
                    "method": e.get("extraction_method"),
                    "evidence": e.get("evidence", ""),
                }
    return list(best.values())


def _load_correlations() -> dict:
    try:
        if config.CORRELATIONS_FILE.exists():
            return json.loads(config.CORRELATIONS_FILE.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        pass
    return {}


def _edges_with_corr() -> list[dict]:
    """Modeled edges annotated with the signals-layer return correlation."""
    edges = _modeled_edges()
    pair_corr = _load_correlations().get("pair_corr", {})
    for e in edges:
        e["corr"] = pair_corr.get("|".join(sorted((e["source"], e["target"]))))
    return edges


def _pagerank(node_ids: list[str], edges: list[dict], damping=0.85, iters=200, tol=1e-10) -> dict:
    """Weighted PageRank on the symmetrized graph -> systemic-importance score.

    Edges are treated as undirected (the filer-perspective direction is
    inconsistent across our sources) and weighted by |return correlation| so
    importance flows along economically real links. A node scores high when it
    is strongly connected to other strongly-connected nodes.
    """
    import numpy as np

    idx = {n: i for i, n in enumerate(node_ids)}
    n = len(node_ids)
    if not n:
        return {}
    w = np.zeros((n, n))
    for e in edges:
        s, t = idx.get(e["source"]), idx.get(e["target"])
        if s is None or t is None:
            continue
        weight = abs(e.get("corr") or 0.0) or 0.2  # default for edges lacking a correlation
        w[s, t] += weight
        w[t, s] += weight
    col = w.sum(axis=0)
    col[col == 0] = 1.0
    m = w / col
    r = np.full(n, 1.0 / n)
    for _ in range(iters):
        rn = (1 - damping) / n + damping * (m @ r)
        if float(np.abs(rn - r).sum()) < tol:
            r = rn
            break
        r = rn
    r = r / r.sum()
    return {node_ids[i]: round(float(r[i]), 5) for i in range(n)}


def compute_centrality(edges: list[dict] | None = None) -> dict:
    """Systemic-importance (weighted PageRank) per ticker."""
    return _pagerank(list(TICKERS), edges if edges is not None else _edges_with_corr())


def build() -> dict:
    edges = _edges_with_corr()
    node_corr = _load_correlations().get("node_neighbor_corr", {})
    centrality = compute_centrality(edges)

    degree: dict[str, int] = {}
    for e in edges:
        degree[e["source"]] = degree.get(e["source"], 0) + 1
        degree[e["target"]] = degree.get(e["target"], 0) + 1

    nodes = []
    for t in TICKERS:
        q = quant.load(t) or {}
        s = sentiment.load(t) or {}
        company = BY_TICKER[t]
        metrics = q.get("metrics", {})

        def mv(name: str):
            return (metrics.get(name) or {}).get("value")

        nodes.append({
            "id": t,
            "label": q.get("name") or company.name,
            "group": company.group,
            "sector": q.get("sector"),
            "industry": q.get("industry"),
            "market_cap": q.get("market_cap"),
            "employees": q.get("employees"),
            "sentiment_score": s.get("score"),
            "sentiment_label": s.get("label"),
            "sentiment_summary": s.get("summary"),
            "degree": degree.get(t, 0),
            "neighbor_corr": node_corr.get(t),
            "centrality": centrality.get(t),
            "metrics": {
                "pe_ttm": mv("pe_ttm"),
                "ev_ebitda": mv("ev_ebitda"),
                "revenue_growth_yoy": mv("revenue_growth_yoy"),
                "gross_margin": mv("gross_margin"),
                "operating_margin": mv("operating_margin"),
            },
        })

    return {"nodes": nodes, "edges": edges}


def run() -> dict:
    _GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    graph = _clean(build())
    out = _GRAPH_DIR / "graph.json"
    # allow_nan=False guards against any NaN slipping back in (would be invalid JSON).
    out.write_text(json.dumps(graph, indent=2, allow_nan=False), encoding="utf-8")
    print(f"[graph] {len(graph['nodes'])} nodes, {len(graph['edges'])} edges -> {out}")
    return graph
