"""RAG evaluation harness (Lean Phase 1).

You can't improve retrieval you can't measure. This runs the curated golden set
(`eval/golden_questions.json`) through the retriever and reports:

    recall@k   fraction of expected tickers actually retrieved
    MRR        1/rank of the first relevant ticker (ranking quality)
    hit-rate   fraction of questions with >=1 expected ticker retrieved

Retrieval metrics use no LLM (only cheap embeddings) so the suite is fast and
free to run on every change — gate retrieval tweaks (rerankers, embeddings,
chunking) on these numbers. `--judge` additionally LLM-grades answer
faithfulness with Claude (costs one Claude call per question).

    python -m sp500_vault.pipeline eval
    python -m sp500_vault.pipeline eval --k 10 --judge
"""
from __future__ import annotations

import datetime as dt
import json
import math
import operator
import statistics

from . import config, graph_qa, llm, rag, tracing
from .tracing import observe

_REF_OPS = {"gt": operator.gt, "ge": operator.ge, "lt": operator.lt, "le": operator.le}

_GOLDEN = config.BASE_DIR / "eval" / "golden_questions.json"
_REPORT = config.BASE_DIR / "eval" / "eval_report.json"

# Independent reducers — recompute aggregates a second way so a regression in
# graph_qa's reducer (not just the set selection) is caught.
_REF_AGG = {
    "average": lambda xs: sum(xs) / len(xs),
    "median": statistics.median,
    "total": sum,
}


def _retrieved_tickers(docs) -> list[str]:
    out, seen = [], set()
    for d in docs:
        t = d.metadata.get("ticker")
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _recall_mrr(expected: list[str], retrieved: list[str]) -> tuple[float, float, list[str]]:
    exp = [t.upper() for t in expected]
    got = [t.upper() for t in retrieved]
    hits = [t for t in exp if t in got]
    recall = len(hits) / len(exp) if exp else 0.0
    rr = 0.0
    for i, t in enumerate(got):
        if t in exp:
            rr = 1.0 / (i + 1)
            break
    return recall, rr, hits


def _predicate_holds(node: dict, pred: dict) -> bool:
    field = pred["field"]
    f = tuple(field.split(".")) if "." in field else field
    v = graph_qa._value(node, f)
    return v is not None and _REF_OPS[pred["op"]](float(v), float(pred["value"]))


def _check_graph_query(gq: dict, nodes: list) -> dict:
    """Deterministically verify a router query: it fires, selects the exact
    expected set, and (for aggregates) computes a value consistent with an
    independent reducer over that set."""
    q = gq["question"]
    fires = graph_qa.as_documents(q) is not None      # detection didn't regress

    # Filter queries: recompute the expected set independently from the live base
    # set + the stated predicate — drift-proof (both sides see the same nodes), so
    # it guards parse_predicates/_apply_filters without hardcoding membership.
    if gq.get("type") == "filter":
        base, _ = graph_qa._base_set(nodes, q)
        expected_ids = {n["id"] for n in base if _predicate_holds(n, gq["predicate"])}
        got_ids = {n["id"] for n in graph_qa._select_set(nodes, q)[0]}
        ok = fires and got_ids == expected_ids
        bits = []
        if not fires:
            bits.append("router did not fire")
        if got_ids != expected_ids:
            bits.append(f"filtered {sorted(got_ids)} != independent {sorted(expected_ids)}")
        return {"question": q, "type": "filter", "ok": ok, "detail": "; ".join(bits)}

    expected = {t.upper() for t in gq.get("expected_set", [])}
    subset, _desc = graph_qa._select_set(nodes, q)    # set algebra / scope / multi-hop
    got = {n["id"] for n in subset}
    set_ok = got == expected
    value_ok = True
    if gq.get("type") == "aggregate":
        res = graph_qa.aggregate(nodes, q)
        if res is None:
            value_ok = False
        elif res["agg"] != "count":
            field = res["metric"]["field"]
            vals = [float(graph_qa._value(n, field)) for n in subset
                    if graph_qa._value(n, field) is not None]
            ref = _REF_AGG[res["agg"]](vals)
            value_ok = bool(vals) and math.isclose(ref, res["value"], rel_tol=1e-6, abs_tol=1e-9)
        else:
            value_ok = res["value"] == len(subset)
    ok = fires and set_ok and value_ok
    bits = []
    if not fires:
        bits.append("router did not fire")
    if not set_ok:
        bits.append(f"set {sorted(got)} != expected {sorted(expected)}")
    if not value_ok:
        bits.append("value inconsistent with independent reducer")
    return {"question": q, "type": gq.get("type"), "ok": ok, "detail": "; ".join(bits)}


@observe(name="eval-graph-query")
def _scored_graph_check(gq: dict, nodes: list, run_id: str) -> dict:
    """Run a graph-query guard and emit a pass/fail boolean score to Langfuse."""
    r = _check_graph_query(gq, nodes)
    tracing.update_trace(input=gq["question"], session_id=run_id, tags=["eval", "graph-query"],
                         metadata={"type": r.get("type"), "detail": r.get("detail")})
    tracing.score("graph_query_pass", bool(r["ok"]), "BOOLEAN", comment=r.get("detail") or None)
    return r


def _eval_graph_queries(data: dict, run_id: str) -> dict:
    gqs = data.get("graph_queries", [])
    if not gqs:
        return {}
    nodes = graph_qa._load_nodes()
    rows = [_scored_graph_check(gq, nodes, run_id) for gq in gqs]
    passed = sum(1 for r in rows if r["ok"])
    print(f"\n[eval] graph-query regression guards — {passed}/{len(rows)} exact "
          f"(set algebra + aggregate value consistency):")
    for r in rows:
        flag = "✓" if r["ok"] else "✗"
        print(f"  {flag} {r['type'] or '?':9} | {r['question'][:58]}"
              + (f"  — {r['detail']}" if r["detail"] else ""))
    return {"n": len(rows), "passed": passed, "checks": rows}


@observe(name="eval-question")
def _score_question(q: dict, k: int, judge: bool, run_id: str) -> dict:
    """Evaluate one golden question and emit its scores to Langfuse (recall@k,
    reciprocal rank, hit, and — when judged — answer faithfulness)."""
    tracing.update_trace(input=q["question"], session_id=run_id, tags=["eval"],
                         metadata={"expected": q["expected_tickers"], "k": k})
    docs = rag.retrieve(q["question"], k=k)
    got = _retrieved_tickers(docs)
    recall, rr, hits = _recall_mrr(q["expected_tickers"], got)
    row = {
        "question": q["question"],
        "expected": q["expected_tickers"],
        "retrieved": got[:8],
        "recall": round(recall, 3),
        "rr": round(rr, 3),
        "hit": bool(hits),
    }
    tracing.score("recall_at_k", recall, "NUMERIC")
    tracing.score("reciprocal_rank", rr, "NUMERIC")
    tracing.score("hit", bool(hits), "BOOLEAN")
    if judge:
        answer = llm.rag_answer(q["question"], rag._contexts(docs))
        row["faithfulness"] = llm.grade_faithfulness(q["question"], answer, rag._contexts(docs))["score"]
        tracing.score("faithfulness", row["faithfulness"], "NUMERIC")
        tracing.update_trace(output=answer)
    return row


def run(k: int = 8, judge: bool = False) -> dict:
    data = json.loads(_GOLDEN.read_text(encoding="utf-8"))
    questions = data["questions"]
    run_id = "eval-" + dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    print(f"[eval] running {len(questions)} golden questions (k={k}"
          f"{', + LLM faithfulness judge' if judge else ''})"
          f"{'  [tracing -> Langfuse]' if tracing.ENABLED else ''}…\n")

    rows = []
    for q in questions:
        row = _score_question(q, k, judge, run_id)
        rows.append(row)
        flag = "✓" if row["recall"] == 1 else ("·" if row["hit"] else "✗")
        print(f"  {flag} recall {row['recall']:.2f}  mrr {row['rr']:.2f}  | {q['question'][:54]}")

    n = len(rows)
    agg = {
        "k": k, "n": n,
        "recall_at_k": round(sum(r["recall"] for r in rows) / n, 3),
        "mrr": round(sum(r["rr"] for r in rows) / n, 3),
        "hit_rate": round(sum(1 for r in rows if r["hit"]) / n, 3),
    }
    if judge:
        fs = [r["faithfulness"] for r in rows if r.get("faithfulness") is not None]
        agg["faithfulness"] = round(sum(fs) / len(fs), 3) if fs else None

    graph_q = _eval_graph_queries(data, run_id)

    # Run-level aggregates → session scores, so each eval run is one Langfuse
    # session you can trend over time.
    tracing.score_session(run_id, "recall_at_k", agg["recall_at_k"], "NUMERIC")
    tracing.score_session(run_id, "mrr", agg["mrr"], "NUMERIC")
    tracing.score_session(run_id, "hit_rate", agg["hit_rate"], "NUMERIC")
    if judge and agg.get("faithfulness") is not None:
        tracing.score_session(run_id, "faithfulness", agg["faithfulness"], "NUMERIC")
    if graph_q:
        tracing.score_session(run_id, "graph_query_pass_rate",
                              round(graph_q["passed"] / graph_q["n"], 3), "NUMERIC")
    tracing.flush()   # short-lived run — make sure scores are sent

    report = {"as_of": dt.date.today().isoformat(), "run_id": run_id, "aggregate": agg,
              "questions": rows, "graph_queries": graph_q}
    _REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    gq_line = (f"  graph-queries={graph_q['passed']}/{graph_q['n']}" if graph_q else "")
    print(f"\n[eval] recall@{k}={agg['recall_at_k']}  MRR={agg['mrr']}  hit-rate={agg['hit_rate']}"
          + (f"  faithfulness={agg['faithfulness']}" if judge else "")
          + gq_line
          + f"\n[eval] report -> {_REPORT}")
    return report
