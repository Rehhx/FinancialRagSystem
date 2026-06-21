"""Graph-metric question routing — answer rankings from the structured graph.

Some analyst questions are *rankings/aggregations over the whole universe*, not
semantic lookups:

    "Which company is the most systemically central in the chip supply chain?"
    "Which memory and storage makers have the most bullish sentiment?"
    "Cheapest names by P/E?"

Vector search answers these badly: there is no single note chunk that *states*
"NVDA is the most central" — the answer is a computation over every node's
metric. So we detect this question class and answer it from the structured
graph (``data/graph/graph.json``), returning the ranked nodes as LangChain
Documents so the existing ``rag.query()`` / eval path works unchanged (the LLM
still writes the prose; the eval still reads ``metadata['ticker']``).

Routing is conservative — it fires only when the question contains *both* a
metric keyword (centrality / sentiment / P/E / …) *and* a superlative/ranking
cue ("most", "highest", "cheapest", …). Plain relationship questions ("who
supplies NVIDIA?") never match and fall through to vector retrieval.
"""
from __future__ import annotations

import json
import operator
import re

from langchain_core.documents import Document

from . import config

# ── Metric registry ──────────────────────────────────────────────────────────
# Each metric: the node field to rank on, the default direction, the trigger
# keywords, and words that flip the direction. Order = match priority.

_METRICS: list[dict] = [
    {
        "key": "centrality",
        "field": "centrality",
        "ascending": False,
        "label": "systemic centrality (weighted PageRank)",
        "any": ["systemically central", "systemic", "central", "centrality",
                "importance", "important", "influential", "interconnected",
                "linchpin", "backbone", "keystone", "most connected hub"],
        "flip": [],
        "fmt": lambda v: f"{v:.4f}",
    },
    {
        "key": "degree",
        "field": "degree",
        "ascending": False,
        "label": "relationship degree (number of modeled links)",
        "any": ["most connected", "most relationships", "most links",
                "most connections", "most partners", "most edges",
                "best connected", "well connected", "well-connected",
                "most ties", "highest degree"],
        "flip": [],
        "fmt": lambda v: f"{int(v)} links",
    },
    {
        "key": "sentiment",
        "field": "sentiment_score",
        "ascending": False,
        "label": "news sentiment",
        "any": ["sentiment", "bullish", "bearish", "optimistic", "pessimistic",
                "most positive", "most negative"],
        "flip": ["bearish", "pessimistic", "most negative", "least bullish",
                 "worst sentiment", "lowest sentiment"],
        "fmt": lambda v: f"{v:+.2f}",
    },
    {
        "key": "pe",
        "field": ("metrics", "pe_ttm"),
        "ascending": True,  # "cheapest" is the natural default
        "label": "P/E (TTM)",
        "any": ["p/e", "pe ratio", "price to earnings", "price-to-earnings",
                "price/earnings", "cheapest", "valuation", "expensive",
                "priciest", "richly valued"],
        "flip": ["highest", "expensive", "priciest", "richest", "richly valued",
                 "most expensive", "highest p/e"],
        "fmt": lambda v: f"{v:.1f}x",
    },
    {
        "key": "market_cap",
        "field": "market_cap",
        "ascending": False,
        "label": "market cap",
        "any": ["market cap", "market capitalization", "largest company",
                "biggest company", "most valuable", "mega cap", "megacap",
                "largest by market", "biggest by market"],
        "flip": ["smallest", "least valuable", "lowest market cap"],
        "fmt": lambda v: _money(v),
    },
    {
        "key": "revenue_growth",
        "field": ("metrics", "revenue_growth_yoy"),
        "ascending": False,
        "label": "revenue growth (YoY)",
        "any": ["fastest growing", "fastest-growing", "revenue growth",
                "sales growth", "top line growth", "top-line growth",
                "growing fastest"],
        "flip": ["slowest", "slowest growing", "shrinking"],
        "fmt": lambda v: f"{v * 100:+.0f}%",
    },
    {
        "key": "gross_margin",
        "field": ("metrics", "gross_margin"),
        "ascending": False,
        "label": "gross margin",
        "any": ["gross margin", "highest margin", "best margin", "fattest margin"],
        "flip": ["lowest margin", "worst margin", "thinnest margin"],
        "fmt": lambda v: f"{v * 100:.0f}%",
    },
    {
        "key": "operating_margin",
        "field": ("metrics", "operating_margin"),
        "ascending": False,
        "label": "operating margin",
        "any": ["operating margin", "operating efficiency"],
        "flip": ["lowest operating margin"],
        "fmt": lambda v: f"{v * 100:.0f}%",
    },
]

# Superlative / ranking cues — at least one must be present for a question to
# route here (so descriptive mentions like "what is NVDA's sentiment?" don't).
_RANK_CUES = [
    "most", "highest", "lowest", "largest", "biggest", "smallest", "top",
    "best", "worst", "cheapest", "expensive", "priciest", "fastest", "slowest",
    "least", "greatest", "strongest", "weakest", "leader", "leading",
    "dominant", "rank", "ranked", "ranking", "which company", "which companies",
    "who has", "name the", "highest-", "lowest-",
]

# ── Scope buckets ─────────────────────────────────────────────────────────────
# yfinance "industry" is too coarse to separate (e.g.) memory/storage makers
# from the broad "Semiconductors"/"Computer Hardware" labels, so we map domain
# terms to the modeled tickers. Pilot-scale and explicit on purpose; extend as
# the universe grows. Unknown scope terms fall back to the cluster groups below.
_SCOPE_BUCKETS: dict[str, set[str]] = {
    "memory": {"MU"},
    "dram": {"MU"},
    "nand": {"MU", "WDC"},
    "storage": {"WDC", "STX", "MU"},
    "hard drive": {"WDC", "STX"},
    "hard disk": {"WDC", "STX"},
    "gpu": {"NVDA", "AMD"},
    "graphics": {"NVDA", "AMD"},
    "foundry": {"GFS", "INTC"},
    "rf ": {"QCOM", "SWKS", "QRVO"},
    "wireless chip": {"QCOM", "SWKS", "QRVO"},
    "hyperscaler": {"MSFT", "AMZN", "GOOGL", "META", "ORCL"},
}

# Cluster-group aliases (maps a word in the question to a universe ``group``).
_GROUP_ALIASES: dict[str, str] = {
    "automaker": "automotive", "automakers": "automotive", "carmaker": "automotive",
    "auto ": "automotive", "vehicle": "automotive",
    "chipmaker": "semiconductor", "chip ": "semiconductor", "chips": "semiconductor",
    "semiconductor": "semiconductor", "semis": "semiconductor",
    "hardware": "hardware",
    "cloud": "cloud_software", "software": "cloud_software",
    "telecom": "communications", "communications": "communications", "carrier": "communications",
}


# ── Aggregations (arithmetic over a set) ─────────────────────────────────────
# "average P/E of NVIDIA's suppliers", "total market cap of the hyperscalers",
# "how many competitors does AMD have". LLMs are unreliable at averaging a list,
# so the reducer computes the number in Python and hands it over grounded.


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0


# key -> (trigger phrases, reducer over a list of floats)
_AGGS: dict[str, tuple[list[str], object]] = {
    "count": (["how many", "number of", "count of", "count the"], len),
    "average": (["average", "mean ", " avg", "typical"], lambda xs: sum(xs) / len(xs)),
    "median": (["median"], _median),
    "total": (["total", "sum of", "combined", "aggregate", "altogether", "summed"], sum),
}

# Relationship words -> edge relation (None = any neighbor). Drives the
# "<company>'s suppliers / customers / competitors / peers" set selector.
_RELATION_WORDS: dict[str, str | None] = {
    "supplier": "supplier", "suppliers": "supplier", "supplies": "supplier",
    "vendor": "supplier", "vendors": "supplier",
    "customer": "customer", "customers": "customer", "buyer": "customer", "buyers": "customer",
    "competitor": "competitor", "competitors": "competitor", "rival": "competitor",
    "rivals": "competitor", "competes": "competitor",
    "peer": None, "peers": None, "neighbor": None, "neighbors": None,
    "partner": None, "partners": None, "counterpart": None, "counterparties": None,
    "supply chain": None, "ecosystem": None, "related companies": None,
}


def _money(v) -> str:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "—"
    for unit, scale in (("T", 1e12), ("B", 1e9), ("M", 1e6)):
        if abs(v) >= scale:
            return f"${v / scale:.2f}{unit}"
    return f"${v:,.0f}"


def _value(node: dict, field):
    if isinstance(field, tuple):
        cur = node
        for key in field:
            cur = (cur or {}).get(key) if isinstance(cur, dict) else None
        return cur
    return node.get(field)


# ── Detection ─────────────────────────────────────────────────────────────────


def detect(question: str) -> dict | None:
    """Return the metric spec a question is asking to rank by, or ``None``.

    Requires both a metric keyword *and* a ranking cue, so it only fires on
    genuine "which company is the most/highest/cheapest …" questions.
    """
    q = " " + question.lower().strip() + " "
    if not any(cue in q for cue in _RANK_CUES):
        return None
    qm = " " + _strip_predicate_text(question).lower().strip() + " "   # exclude filter clauses
    for spec in _METRICS:
        if any(kw in qm for kw in spec["any"]):
            ascending = spec["ascending"]
            if any(fw in qm for fw in spec["flip"]):
                ascending = not ascending
            return {**spec, "ascending": ascending}
    return None


def _scope(nodes: list[dict], question: str) -> list[dict]:
    """Restrict candidates to the sub-industry / cluster the question names.

    Falls back to the full universe when no scope term is recognized.
    """
    q = " " + question.lower() + " "
    tickers: set[str] = set()
    for kw, tics in _SCOPE_BUCKETS.items():
        if kw in q:
            tickers |= tics
    groups: set[str] = {grp for kw, grp in _GROUP_ALIASES.items() if kw in q}
    if groups:
        tickers |= {n["id"] for n in nodes if n.get("group") in groups}
    if not tickers:
        return nodes
    scoped = [n for n in nodes if n["id"] in tickers]
    return scoped or nodes


def _named_tickers(question: str) -> set[str]:
    """Tickers/company names mentioned in the question (no neighbor expansion)."""
    from .universe import BY_TICKER, TICKERS

    text = " " + question.upper() + " "
    found: set[str] = set()
    for t in TICKERS:                       # tickers >= 3 chars (avoid T/F/GM false positives)
        if len(t) >= 3 and re.search(rf"\b{re.escape(t)}\b", text):
            found.add(t)
    for t, c in BY_TICKER.items():          # company-name mentions (catches short tickers too)
        core = c.name.split(",")[0].split(" ")[0].upper()
        if len(core) >= 4 and f" {core}" in text:
            found.add(t)
    return found


def _anchor_pos(question: str, focus: set[str]) -> int | None:
    """Index of the earliest focus mention (ticker or company name), or None."""
    from .universe import BY_TICKER

    q = question.lower()
    positions = []
    for t in focus:
        core = BY_TICKER[t].name.split(",")[0].split(" ")[0].lower() if t in BY_TICKER else t.lower()
        for token in (t.lower(), core):
            i = q.find(token)
            if i >= 0:
                positions.append(i)
    return min(positions) if positions else None


def parse_chain(question: str) -> tuple[set[str], list[str | None]]:
    """Parse a (possibly multi-hop) relationship phrase.

    "suppliers of NVIDIA's customers" -> (focus={NVDA}, relations=[customer, supplier])
    "NVIDIA's competitors"            -> (focus={NVDA}, relations=[competitor])
    "average P/E of semiconductors"   -> (set(), [])   # no relation word

    Relations are ordered **nearest-to-the-anchor first**, so they apply from the
    company outward regardless of surface form ("X's Y", "Y of X", or nested).
    Pure — no DB access.
    """
    q = " " + question.lower() + " "
    occ: list[tuple[int, str | None]] = []
    for word, rel in _RELATION_WORDS.items():
        i = q.find(word)
        if i >= 0:
            occ.append((i, rel))
    if not occ:
        return set(), []
    focus = _named_tickers(question)
    anchor = _anchor_pos(question, focus)
    if anchor is None:
        anchor = len(question)      # scope-anchored: scope usually trails → rightmost first
    occ.sort(key=lambda pr: abs(pr[0] - anchor))
    relations: list[str | None] = []
    seen: set[str] = set()
    for _pos, rel in occ:
        key = rel if rel is not None else "__any__"
        if key not in seen:
            seen.add(key)
            relations.append(rel)
    return focus, relations


def _neighbors(focus: set[str], relation: str | None) -> set[str]:
    """Modeled neighbors of a set of tickers, optionally filtered by relation."""
    from .relationships import get_edges

    out: set[str] = set()
    for t in focus:
        for e in get_edges(t, resolved_only=True):
            if relation is not None and e.get("relation") != relation:
                continue
            tgt = e.get("target_ticker")
            if tgt:
                out.add(tgt)
    return out


def _traverse(base: set[str], relations: list[str | None]) -> set[str]:
    """Walk the relationship graph: apply each hop in turn (multi-hop set algebra)."""
    current = set(base)
    for rel in relations:
        current = _neighbors(current, rel)
        if not current:
            break
    return current


def _describe_chain(base_desc: str, relations: list[str | None]) -> str:
    desc = base_desc
    for rel in relations:
        label = (rel or "neighbor") + "s"
        desc = f"{label} of {desc}"
    return desc


# ── Threshold / predicate filters (the WHERE clause) ─────────────────────────
# "suppliers with sentiment > 0.5", "competitors with P/E under 20", "names with
# market cap over $1T". A predicate filters the selected set *before* it's listed,
# ranked, or aggregated — so it composes with everything above.

_OPS = {"gt": operator.gt, "ge": operator.ge, "lt": operator.lt, "le": operator.le}
_OPSYM = {"gt": ">", "ge": "≥", "lt": "<", "le": "≤"}

# Comparator phrases -> op code. Longer/more-specific phrases first (regex
# alternation is order-sensitive; "no more than" must precede "more than", ">="
# must precede ">").
_OP_PHRASES: list[tuple[str, str]] = [
    ("no less than", "ge"), ("no more than", "le"),
    ("greater than or equal to", "ge"), ("less than or equal to", "le"),
    ("at least", "ge"), ("at most", "le"),
    (">=", "ge"), ("<=", "le"), ("=>", "ge"), ("=<", "le"),
    ("greater than", "gt"), ("more than", "gt"), ("higher than", "gt"),
    ("larger than", "gt"), ("bigger than", "gt"),
    ("less than", "lt"), ("lower than", "lt"), ("smaller than", "lt"),
    ("fewer than", "lt"),
    ("above", "gt"), ("over", "gt"), ("exceeding", "gt"), ("exceeds", "gt"),
    ("north of", "gt"),
    ("under", "lt"), ("below", "lt"), ("south of", "lt"),
    (">", "gt"), ("<", "lt"),
]
_OP_LOOKUP = {p: op for p, op in _OP_PHRASES}

_OP_ALT = "|".join((r"\b" + re.escape(p) + r"\b") if p[0].isalpha() else re.escape(p)
                   for p, _ in _OP_PHRASES)
# <comparator> <number> <unit?>  — unit words before single letters so "trillion"
# isn't grabbed as bare "t".
_PRED_RX = re.compile(
    rf"(?P<op>{_OP_ALT})\s*\$?\s*(?P<num>[0-9][0-9,]*(?:\.[0-9]+)?)\s*"
    r"(?P<unit>trillion|billion|million|%|x|t|b|m)?", re.I)

# Metric a predicate can filter on: node field, value scale, label, formatter,
# and the keywords that name it. ``kind`` drives threshold parsing.
_FILTER_METRICS: list[dict] = [
    {"key": "sentiment", "field": "sentiment_score", "kind": "plain", "label": "sentiment",
     "fmt": lambda v: f"{v:+.2f}", "kws": ["sentiment", "bullishness"]},
    {"key": "pe", "field": ("metrics", "pe_ttm"), "kind": "plain", "label": "P/E",
     "fmt": lambda v: f"{v:.0f}x",
     "kws": ["p/e", "pe ratio", "p/e ratio", "price to earnings", "price/earnings", "earnings multiple"]},
    {"key": "market_cap", "field": "market_cap", "kind": "money", "label": "market cap",
     "fmt": lambda v: _money(v), "kws": ["market cap", "market capitalization", "mkt cap", "market value"]},
    {"key": "centrality", "field": "centrality", "kind": "plain", "label": "centrality",
     "fmt": lambda v: f"{v:.4f}", "kws": ["centrality", "systemic importance"]},
    {"key": "degree", "field": "degree", "kind": "degree", "label": "degree",
     "fmt": lambda v: f"{int(v)} links", "kws": ["degree", "connections", "links"]},
    {"key": "revenue_growth", "field": ("metrics", "revenue_growth_yoy"), "kind": "pct",
     "label": "revenue growth", "fmt": lambda v: f"{v * 100:.0f}%",
     "kws": ["revenue growth", "sales growth", "top line growth", "growth"]},
    {"key": "gross_margin", "field": ("metrics", "gross_margin"), "kind": "pct",
     "label": "gross margin", "fmt": lambda v: f"{v * 100:.0f}%", "kws": ["gross margin", "margin"]},
    {"key": "operating_margin", "field": ("metrics", "operating_margin"), "kind": "pct",
     "label": "operating margin", "fmt": lambda v: f"{v * 100:.0f}%", "kws": ["operating margin"]},
]


def _scale(unit: str | None) -> float:
    return {"t": 1e12, "trillion": 1e12, "b": 1e9, "billion": 1e9,
            "m": 1e6, "million": 1e6}.get((unit or "").lower(), 1.0)


def _threshold(kind: str, num: str, unit: str | None) -> float:
    val = float(num.replace(",", ""))
    if kind == "money":
        return val * _scale(unit)
    if kind == "pct":   # "40%" or bare "40" -> 0.40 ; "0.4" stays 0.4
        return val / 100.0 if (unit == "%" or val > 1) else val
    return val          # plain (sentiment / P/E / centrality), degree compared as float


def _metric_positions(q: str) -> list[tuple[int, dict]]:
    out: list[tuple[int, dict]] = []
    for m in _FILTER_METRICS:
        for kw in m["kws"]:
            i = q.find(kw)
            if i >= 0:
                out.append((i, m))
    return out


def parse_predicates(question: str) -> list[dict]:
    """Parse threshold clauses ("P/E under 20", "sentiment > 0.5") into filters.

    Each comparator binds to the nearest metric keyword to its left ("<metric>
    <comparator> <value>"). Pure — no DB/graph access. Returns a list of
    {key, field, op, op_sym, threshold, label, fmt}.
    """
    q = question.lower()
    positions = _metric_positions(q)
    if not positions:
        return []
    preds: list[dict] = []
    for mt in _PRED_RX.finditer(q):
        cpos = mt.start()
        left = [(p, m) for p, m in positions if p <= cpos]
        _, metric = max(left, key=lambda pm: pm[0]) if left else min(positions, key=lambda pm: pm[0])
        op = _OP_LOOKUP[mt.group("op").lower()]
        preds.append({
            "key": metric["key"], "field": metric["field"], "op": op, "op_sym": _OPSYM[op],
            "threshold": _threshold(metric["kind"], mt.group("num"), mt.group("unit")),
            "label": metric["label"], "fmt": metric["fmt"],
        })
    return preds


def _strip_predicate_text(question: str) -> str:
    """Remove predicate clauses ("… with sentiment above 0.4") so the rank/aggregate
    *metric* detector doesn't latch onto the filter's metric keyword. Each clause
    is dropped from its bound metric keyword through the comparator+value."""
    ql = question.lower()
    positions = _metric_positions(ql)
    if not positions:
        return question
    spans: list[tuple[int, int]] = []
    for mt in _PRED_RX.finditer(ql):
        left = [(p, m) for p, m in positions if p <= mt.start()]
        start = max(left, key=lambda pm: pm[0])[0] if left else mt.start()
        spans.append((start, mt.end()))
    out = question
    for s, e in sorted(spans, reverse=True):
        out = out[:s] + " " + out[e:]
    return out


def _apply_filters(subset: list[dict], preds: list[dict]) -> list[dict]:
    out = []
    for n in subset:
        keep = True
        for p in preds:
            v = _value(n, p["field"])
            if v is None or not _OPS[p["op"]](float(v), p["threshold"]):
                keep = False
                break
        if keep:
            out.append(n)
    return out


def _describe_filters(preds: list[dict]) -> str:
    if not preds:
        return ""
    parts = [f"{p['label']} {p['op_sym']} {p['fmt'](p['threshold'])}" for p in preds]
    return " with " + " and ".join(parts)


# ── Set selection (base ∘ filters) ───────────────────────────────────────────


def _base_set(nodes: list[dict], question: str) -> tuple[list[dict], str]:
    """The candidate set before predicate filtering: a relationship neighborhood
    (single- or multi-hop, "suppliers of NVIDIA's customers"), else a
    sub-industry/cluster scope, else the whole universe."""
    focus, relations = parse_chain(question)
    if relations:
        if focus:                       # ticker-anchored traversal
            members = _traverse(focus, relations)
            base_desc = "/".join(sorted(focus))
        else:                           # scope-anchored: "suppliers of <scope>"
            scoped = _scope(nodes, question)
            base = {n["id"] for n in scoped} if len(scoped) < len(nodes) else set()
            members = _traverse(base, relations) if base else set()
            base_desc = "the selected group"
        subset = [n for n in nodes if n["id"] in members]
        if subset:
            return subset, _describe_chain(base_desc, relations)
    scoped = _scope(nodes, question)
    if len(scoped) < len(nodes):
        return scoped, "the selected group"
    return nodes, "the universe"


def _select_set(nodes: list[dict], question: str) -> tuple[list[dict], str]:
    """Candidate set with any threshold predicates applied (base ∘ WHERE)."""
    subset, desc = _base_set(nodes, question)
    preds = parse_predicates(question)
    if preds:
        subset = _apply_filters(subset, preds)
        desc = desc + _describe_filters(preds)
    return subset, desc


def rank(nodes: list[dict], question: str, k: int = 6) -> tuple[dict, list[tuple[dict, float]]] | None:
    """Detect the metric, select the candidate set, and return the top-k ranked.

    Returns ``(spec, [(node, value), …])`` or ``None`` if not a metric question.
    Pure over ``nodes`` unless the question names a relationship neighborhood
    (then it reads the relationships DB to resolve members).
    """
    spec = detect(question)
    if spec is None:
        return None
    field = spec["field"]
    subset, _desc = _select_set(nodes, question)
    scored = [(n, _value(n, field)) for n in subset]
    scored = [(n, float(v)) for n, v in scored if v is not None]
    if not scored:
        return None
    scored.sort(key=lambda nv: nv[1], reverse=not spec["ascending"])
    return spec, scored[:k]


# ── Aggregate detection + reduction ──────────────────────────────────────────


def detect_aggregate(question: str) -> dict | None:
    """Return the aggregate spec a question asks for, or ``None``.

    Shape: ``{"agg": "average"|"median"|"total"|"count", "fn": reducer,
    "metric": <metric spec or None>}``. ``count`` needs no metric (it just sizes
    the selected set).
    """
    q = " " + question.lower().strip() + " "
    agg = next((key for key, (cues, _) in _AGGS.items() if any(c in q for c in cues)), None)
    if agg is None:
        return None
    fn = _AGGS[agg][1]
    if agg == "count":
        return {"agg": "count", "fn": fn, "metric": None}
    qm = " " + _strip_predicate_text(question).lower().strip() + " "   # exclude filter clauses
    metric = next((spec for spec in _METRICS if any(kw in qm for kw in spec["any"])), None)
    if metric is None:
        return None
    return {"agg": agg, "fn": fn, "metric": metric}


def aggregate(nodes: list[dict], question: str) -> dict | None:
    """Compute an aggregate over the selected set. Returns a result dict or
    ``None``. Pure over ``nodes`` except for the relationships lookup in
    ``_select_set`` when the question names a neighborhood.
    """
    spec = detect_aggregate(question)
    if spec is None:
        return None
    subset, desc = _select_set(nodes, question)
    if not subset:
        return None
    if spec["agg"] == "count":
        return {"agg": "count", "metric": None, "desc": desc,
                "value": len(subset), "members": [(n, None) for n in subset]}
    field = spec["metric"]["field"]
    members = [(n, float(_value(n, field))) for n in subset if _value(n, field) is not None]
    if not members:
        return None
    value = spec["fn"]([v for _, v in members])
    return {"agg": spec["agg"], "metric": spec["metric"], "desc": desc,
            "value": value, "members": members}


# ── Documents (the seam back into rag.query / eval) ──────────────────────────


def _load_nodes() -> list[dict]:
    gf = config.DATA_DIR / "graph" / "graph.json"
    if gf.exists():
        return json.loads(gf.read_text(encoding="utf-8")).get("nodes", [])
    from . import graph_export
    return graph_export.build().get("nodes", [])


def _aggregate_documents(question: str, nodes: list[dict]) -> list[Document] | None:
    """Documents for an arithmetic-aggregate question (the computed value as a
    summary Document the LLM must quote, plus the member rows for grounding)."""
    res = aggregate(nodes, question)
    if res is None:
        return None
    members = res["members"]
    if res["agg"] == "count":
        metric_label = "members"
        value_str = str(res["value"])
        member_line = ", ".join(sorted(n["id"] for n, _ in members))
        section = f"Graph · count {res['desc']}"
    else:
        metric = res["metric"]
        fmt = metric["fmt"]
        metric_label = metric["label"]
        value_str = fmt(res["value"])
        member_line = ", ".join(f"{n['id']} {fmt(v)}" for n, v in members)
        section = f"Graph · {res['agg']} {metric['key']}"

    summary = (
        f"Aggregate — {res['agg']} {metric_label} across {res['desc']} "
        f"(n={len(members)}): {value_str}.\n"
        f"Computed over: {member_line}."
    )
    docs: list[Document] = [Document(
        page_content=summary,
        metadata={"ticker": None, "section": section, "sector": "Unknown",
                  "sentiment_label": "Neutral"},
    )]
    for n, v in members[:15]:   # member rows ground the number; whole set drives it
        line = (
            f"{n['id']} — {n.get('label') or n['id']}\n"
            f"Member of {res['desc']}. "
            + (f"{res['metric']['label']}: {res['metric']['fmt'](v)}. " if v is not None else "")
            + f"Cluster: {n.get('group')}. Market cap: {_money(n.get('market_cap'))}. "
            f"Sentiment: {('%+.2f' % n['sentiment_score']) if n.get('sentiment_score') is not None else 'n/a'} "
            f"({n.get('sentiment_label') or 'n/a'})."
        )
        docs.append(Document(
            page_content=line,
            metadata={"ticker": n["id"], "section": "Graph · member",
                      "sector": n.get("sector") or "Unknown",
                      "sentiment_label": n.get("sentiment_label") or "Neutral"},
        ))
    return docs


_LIST_CUES = [" which ", " what ", " list ", " show ", " find ", " name ",
              " companies ", " stocks ", " names ", " screen ", " any "]


def _is_screen(nodes: list[dict], question: str) -> bool:
    """A filtered-list query targets a set (relationship/scope) or explicitly
    screens companies — guards against hijacking a plain vector question that
    merely contains a number."""
    if parse_chain(question)[1]:
        return True
    if len(_scope(nodes, question)) < len(nodes):
        return True
    q = " " + question.lower() + " "
    return any(c in q for c in _LIST_CUES)


def _filtered_list_documents(question: str, nodes: list[dict]) -> list[Document] | None:
    """Documents for a pure threshold screen ("suppliers with sentiment > 0.5")
    — no ranking/aggregate metric, just the matching members."""
    preds = parse_predicates(question)
    if not preds or not _is_screen(nodes, question):
        return None
    subset, desc = _select_set(nodes, question)
    ids = sorted(n["id"] for n in subset)
    summary = (f"Filter — {desc}: {len(ids)} match.\n"
               f"Members: {', '.join(ids) if ids else 'none'}.")
    docs: list[Document] = [Document(
        page_content=summary,
        metadata={"ticker": None, "section": "Graph · filter", "sector": "Unknown",
                  "sentiment_label": "Neutral"},
    )]
    for n in subset[:20]:
        vals = " · ".join(f"{p['label']} {p['fmt'](float(_value(n, p['field'])))}"
                          for p in preds if _value(n, p["field"]) is not None)
        docs.append(Document(
            page_content=(
                f"{n['id']} — {n.get('label') or n['id']}\n"
                f"Matches {desc}. {vals}. "
                f"Cluster: {n.get('group')}. Market cap: {_money(n.get('market_cap'))}. "
                f"Sentiment: {('%+.2f' % n['sentiment_score']) if n.get('sentiment_score') is not None else 'n/a'}."
            ),
            metadata={"ticker": n["id"], "section": "Graph · member",
                      "sector": n.get("sector") or "Unknown",
                      "sentiment_label": n.get("sentiment_label") or "Neutral"},
        ))
    return docs


def as_documents(question: str, k: int = 6) -> list[Document] | None:
    """Graph-layer Documents for a metric question, else ``None``.

    Tries the arithmetic-aggregate path first ("average P/E of NVIDIA's
    suppliers"), then ranking ("most central company"), then a pure threshold
    screen ("suppliers with sentiment > 0.5"). All three apply any predicate
    filter via ``_select_set``. ``rag.retrieve`` calls this first; when it returns
    documents they replace vector search.
    """
    nodes = _load_nodes()
    if not nodes:
        return None
    agg = _aggregate_documents(question, nodes)
    if agg is not None:
        return agg
    result = rank(nodes, question, k)
    if result is None:
        return _filtered_list_documents(question, nodes)
    spec, ranked = result
    total = len(_select_set(nodes, question)[0])
    fmt = spec["fmt"]
    docs: list[Document] = []
    for i, (n, v) in enumerate(ranked, start=1):
        cen = n.get("centrality")
        sent = n.get("sentiment_score")
        content = (
            f"{n['id']} — {n.get('label') or n['id']}\n"
            f"Rank #{i} of {total} by {spec['label']}: {fmt(v)}.\n"
            f"Cluster: {n.get('group')} · sector: {n.get('sector') or 'n/a'}. "
            f"Sentiment: {('%+.2f' % sent) if sent is not None else 'n/a'} "
            f"({n.get('sentiment_label') or 'n/a'}). "
            f"Market cap: {_money(n.get('market_cap'))}. "
            f"Relationship degree: {n.get('degree', 0)} links. "
            f"Centrality: {('%.4f' % cen) if cen is not None else 'n/a'}."
        )
        docs.append(Document(
            page_content=content,
            metadata={
                "ticker": n["id"],
                "section": f"Graph · {spec['key']}",
                "sector": n.get("sector") or "Unknown",
                "sentiment_label": n.get("sentiment_label") or "Neutral",
            },
        ))
    return docs
