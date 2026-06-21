"""Vault rendering: assemble quant + sentiment + relationships into markdown notes.

Renders FROM the data layers (quant JSON, sentiment JSON, relationships DB) so
notes are disposable and regenerable. Output is Obsidian-flavoured markdown with
YAML frontmatter (Dataview-friendly) and [[wikilinks]] (graph-view-friendly).
"""
from __future__ import annotations

import datetime as dt
import math
import re

from . import config, graph_export, quant, relationships, sentiment
from .universe import BY_TICKER, SECTORS, TICKERS

# ── Formatting helpers ───────────────────────────────────────────────────────


def _num(x) -> float | None:
    """Coerce to a finite float, or None for missing/NaN/inf/non-numeric."""
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return None if (math.isnan(f) or math.isinf(f)) else f


def _money(x) -> str:
    f = _num(x)
    if f is None:
        return "N/A"
    for div, suf in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if abs(f) >= div:
            return f"${f / div:.2f}{suf}"
    return f"${f:,.0f}"


def _pct(x) -> str:
    f = _num(x)
    return "N/A" if f is None else f"{f * 100:.1f}%"


def _ratio(x) -> str:
    f = _num(x)
    return "N/A" if f is None else f"{f:.2f}"


def _de(x) -> str:  # yfinance debt/equity is a percent figure (180.5 -> 1.81x)
    f = _num(x)
    return "N/A" if f is None else f"{f / 100:.2f}"


def _int(x) -> str:
    f = _num(x)
    return "N/A" if f is None else f"{int(f):,}"


def _ordinal(p) -> str:
    f = _num(p)
    if f is None:
        return "—"
    p = int(round(f))
    suffix = "th" if 11 <= p % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(p % 10, "th")
    return f"{p}{suffix}"


# metric -> (label, value formatter)
_FMT = {
    "pe_ttm": ("P/E (TTM)", _ratio), "ps_ttm": ("P/S", _ratio), "pb": ("P/B", _ratio),
    "ev_ebitda": ("EV/EBITDA", _ratio), "peg": ("PEG", _ratio),
    "revenue_growth_yoy": ("Revenue Growth (YoY)", _pct),
    "revenue_cagr_3yr": ("Revenue CAGR (3yr)", _pct),
    "eps_cagr_3yr": ("EPS CAGR (3yr)", _pct),
    "gross_margin": ("Gross Margin", _pct), "operating_margin": ("Operating Margin", _pct),
    "net_margin": ("Net Margin", _pct), "roe": ("ROE", _pct), "roa": ("ROA", _pct),
    "beta": ("Beta", _ratio), "debt_to_equity": ("Debt/Equity", _de),
    "current_ratio": ("Current Ratio", _ratio), "volatility_30d": ("Volatility (30d annualized)", _pct),
}


def _table(note: dict, metrics: list[str]) -> str:
    rows = ["| Metric | Value | Sector Avg | Percentile |", "|---|---|---|---|"]
    for m in metrics:
        label, fmt = _FMT[m]
        cell = note["metrics"].get(m, {})
        rows.append(
            f"| {label} | {fmt(cell.get('value'))} | {fmt(cell.get('sector_avg'))} "
            f"| {_ordinal(cell.get('percentile'))} |"
        )
    return "\n".join(rows)


def _bullets(note: dict, metrics: list[str]) -> str:
    out = []
    for m in metrics:
        label, fmt = _FMT[m]
        out.append(f"- **{label}:** {fmt(note['metrics'].get(m, {}).get('value'))}")
    return "\n".join(out)


def _label_for_score(score: float) -> str:
    return "Bullish" if score >= 0.25 else "Bearish" if score <= -0.25 else "Neutral"


# ── Relationship sections ────────────────────────────────────────────────────


def _edge_line(e: dict) -> str:
    conf = e.get("confidence", "")
    note = f" *(conf: {conf})*" if conf else ""
    if e.get("target_ticker") and e["target_ticker"] in BY_TICKER:
        ev = f" — {e['evidence']}" if e.get("evidence") else ""
        return f"- [[{e['target_ticker']}]] — {e['target_name']}{note}{ev}"
    return f"- {e['target_name']} *(external / not modeled)*{note}"


_CONF_RANK = {"high": 3, "medium": 2, "low": 1}
_METHOD_RANK = {"manual_override": 3, "llm_extraction": 2, "finnhub_peers": 1}


def _dedupe_edges(edges: list[dict]) -> list[dict]:
    """Collapse edges to one per target, keeping the best-sourced version.

    A target can arrive from several methods (Finnhub peer + filing mention);
    prefer the highest confidence, then the strongest method, then one with
    evidence text. Resolved (wikilinkable) edges sort ahead of external stubs.
    """
    best: dict[str, tuple[tuple, dict]] = {}
    for e in edges:
        # Normalize name keys so "Marvell Technology, Inc" and "...Inc." collapse.
        name_key = re.sub(r"[^A-Z0-9]", "", (e.get("target_name") or "").upper())
        key = e["target_ticker"] or f"NAME:{name_key}"
        score = (
            _CONF_RANK.get(e.get("confidence"), 0),
            _METHOD_RANK.get(e.get("extraction_method"), 0),
            1 if e.get("evidence") else 0,
        )
        if key not in best or score > best[key][0]:
            best[key] = (score, e)
    deduped = [v[1] for v in best.values()]
    deduped.sort(key=lambda e: (e.get("target_ticker") is None, -_CONF_RANK.get(e.get("confidence"), 0),
                                e.get("target_name") or ""))
    return deduped


def _relationship_block(ticker: str) -> tuple[str, int]:
    sections = [
        ("Suppliers (companies this filer buys from)", "supplier"),
        ("Customers (companies that buy from this filer)", "customer"),
        ("Competitors", "competitor"),
    ]
    parts: list[str] = []
    total = 0
    for title, rel in sections:
        edges = _dedupe_edges(relationships.get_edges(ticker, relation=rel))
        total += len(edges)
        parts.append(f"### {title}")
        parts.append("\n".join(_edge_line(e) for e in edges) if edges else "- *(none extracted)*")
    return "\n\n".join(parts), total


# ── Note assembly ────────────────────────────────────────────────────────────


def render_ticker(ticker: str) -> bool:
    q = quant.load(ticker)
    if not q:
        print(f"  [vault] skip {ticker}: no quant data")
        return False
    s = sentiment.load(ticker) or {}
    company = BY_TICKER.get(ticker)
    today = dt.date.today().isoformat()

    score = float(s.get("score", 0.0))
    label = s.get("label") or _label_for_score(score)
    sector = q.get("sector") or "Unknown"
    group = company.group if company else "company"

    rel_block, rel_count = _relationship_block(ticker)
    connected = relationships.connected_market_cap(ticker, quant.load)

    tags = ["sp500", sector.lower().replace(" ", "-"), group]

    frontmatter = "\n".join([
        "---",
        f"ticker: {ticker}",
        f"name: {q.get('name')}",
        f"sector: {sector}",
        f"industry: {q.get('industry') or 'Unknown'}",
        f"market_cap: {q.get('market_cap') or 'null'}",
        f"last_updated: {today}",
        f"sentiment_score: {score:.2f}",
        f"sentiment_label: {label}",
        f"relationship_count: {rel_count}",
        f"tags: [{', '.join(tags)}]",
        "---",
    ])

    sent_section = (
        f"**Score: {score:+.2f} ({label})** — as of {s.get('as_of', today)}\n"
        f"Generated from {s.get('article_count', 0)} news articles "
        f"(see [[{ticker}_news_log]])\n\n"
        f"> {s.get('summary', 'No sentiment computed yet.')}"
    )

    body = f"""{frontmatter}

# {q.get('name')} ({ticker})

## Overview
- **Market Cap:** {_money(q.get('market_cap'))}
- **Enterprise Value:** {_money(q.get('enterprise_value'))}
- **Sector / Industry:** {sector} / {q.get('industry') or 'Unknown'}
- **Employees:** {_int(q.get('employees'))}

## Quant Analysis
### Valuation
{_table(q, ['pe_ttm', 'ps_ttm', 'pb', 'ev_ebitda', 'peg'])}

### Growth
{_bullets(q, ['revenue_growth_yoy', 'revenue_cagr_3yr', 'eps_cagr_3yr'])}

### Profitability
{_bullets(q, ['gross_margin', 'operating_margin', 'net_margin', 'roe', 'roa'])}

### Risk
{_bullets(q, ['beta', 'debt_to_equity', 'current_ratio', 'volatility_30d'])}

## Sentiment Analysis
{sent_section}

## Relationships
{rel_block}

### Exposure Web
- **Modeled relationships:** {rel_count}
- **Connected node value:** sum of linked, modeled market caps = {_money(connected)}
"""

    (config.VAULT_DIR / f"{ticker}.md").write_text(body, encoding="utf-8")
    _render_news_log(ticker, s)
    return True


def _render_news_log(ticker: str, s: dict) -> None:
    articles = s.get("articles", [])
    lines = [
        f"# {ticker} — News Log",
        f"_As of {s.get('as_of', dt.date.today().isoformat())} · "
        f"sentiment {float(s.get('score', 0.0)):+.2f} ({s.get('label', 'Neutral')})_",
        "",
    ]
    if not articles:
        lines.append("*(no recent articles)*")
    for a in articles:
        date = (a.get("datetime") or "")[:10]
        head = a.get("headline", "")
        url = a.get("url", "")
        title = f"[{head}]({url})" if url else head
        lines.append(f"### {title}")
        lines.append(f"*{date} · {a.get('source', '')}*")
        if a.get("summary"):
            lines.append(f"\n{a['summary']}")
        lines.append("")
    (config.VAULT_DIR / f"{ticker}_news_log.md").write_text("\n".join(lines), encoding="utf-8")


def render_dashboard() -> None:
    """A vault map-of-content with Dataview queries + a static rendered summary.

    Dataview blocks are live in Obsidian (with the plugin); the static tables
    below them make the dashboard useful even without it, and reflect the vault
    at render time.
    """
    centrality = graph_export.compute_centrality()
    rows = []
    for t in TICKERS:
        q = quant.load(t)
        if not q:
            continue
        s = sentiment.load(t) or {}
        rows.append({
            "ticker": t,
            "name": q.get("name"),
            "sector": q.get("sector") or "Unknown",
            "group": BY_TICKER[t].group,
            "market_cap": q.get("market_cap") or 0,
            "score": float(s.get("score", 0.0)),
            "label": s.get("label", "Neutral"),
            "connected": relationships.connected_market_cap(t, quant.load),
            "degree": len(relationships.get_edges(t, resolved_only=True)),
            "centrality": centrality.get(t, 0.0),
        })

    by_score = sorted(rows, key=lambda r: r["score"], reverse=True)
    by_conn = sorted(rows, key=lambda r: r["connected"], reverse=True)
    by_cent = sorted(rows, key=lambda r: r["centrality"], reverse=True)
    bearish = [r for r in rows if r["score"] < 0.4]

    def _mini(rs, cols):
        head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols)
        body = "\n".join(
            "| " + f"[[{r['ticker']}]] | {r['label']} ({r['score']:+.2f}) | "
            f"{_money(r['market_cap'])} | {_money(r['connected'])} | {r['degree']} |"
            for r in rs
        )
        return head + "\n" + body

    cols = ["Ticker", "Sentiment", "Market Cap", "Connected Value", "Edges"]
    sector_counts = {g: sum(1 for r in rows if r["group"] == g) for g in SECTORS}

    body = f"""# 📊 Vault Dashboard

_Generated {dt.date.today().isoformat()} · {len(rows)} companies · """ + \
        " · ".join(f"{g}: {n}" for g, n in sector_counts.items()) + """_

This note is the entry point to the vault. Open **Graph view** to see the supply
web. The queries below are live with the **Dataview** plugin; static tables follow.

## All companies by sentiment (Dataview)
```dataview
TABLE sentiment_label AS Sentiment, sentiment_score AS Score, market_cap AS "Market Cap", relationship_count AS Edges
FROM #sp500
SORT sentiment_score DESC
```

## Watchlist — sentiment below 0.4 (Dataview)
```dataview
TABLE sentiment_score AS Score, sentiment_label AS Sentiment, sector AS Sector
FROM #sp500
WHERE sentiment_score < 0.4
SORT sentiment_score ASC
```

---

## Sentiment leaders (static)
""" + _mini(by_score[:8], cols) + """

## Most connected nodes (static — by linked market-cap exposure)
""" + _mini(by_conn[:8], cols) + """

## 🧭 Most systemically central (weighted PageRank)
| Ticker | Centrality | Cluster | Edges |
|---|---|---|---|
""" + "\n".join(
        f"| [[{r['ticker']}]] | {r['centrality']:.4f} | {r['group'].replace('_', '/')} | {r['degree']} |"
        for r in by_cent[:8]) + f"""

## Watchlist — sentiment < 0.4 (static · {len(bearish)} names)
""" + _mini(sorted(bearish, key=lambda r: r["score"])[:12], cols) + "\n"

    (config.VAULT_DIR / "_Dashboard.md").write_text(body, encoding="utf-8")
    print(f"[vault] wrote dashboard -> {config.VAULT_DIR / '_Dashboard.md'}")


def run(tickers: list[str]) -> None:
    print(f"[vault] rendering {len(tickers)} notes…")
    n = sum(render_ticker(t) for t in tickers)
    render_dashboard()
    print(f"[vault] wrote {n} notes (+ news logs) to {config.VAULT_DIR}")
