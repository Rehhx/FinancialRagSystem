"""Reranking — reorder a candidate pool by query relevance before answering.

Vector search finds the right *neighborhood*; a cross-encoder reranker reads each
(query, chunk) pair jointly and is far better at fine-grained relevance — the
single biggest retrieval-quality lever after data. Retrieval fetches a wide pool
(~24+), the reranker scores every candidate, and we keep the top-k. Pluggable:

    cross_encoder  BAAI/bge-reranker-v2-m3 via sentence-transformers (free, local)
    llm            listwise rerank via Claude or OpenAI (RERANK_PROVIDER); one call/query
    none           pass-through (vector order)

Selected via ``config.RERANKER``. Any backend failure falls back to vector order,
so retrieval never hard-fails.
"""
from __future__ import annotations

import json
from functools import lru_cache

from . import config

_warned = {"shown": False}


# ── cross-encoder (bge) ──────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def _cross_encoder():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(config.RERANKER_MODEL)


def _rerank_cross_encoder(query: str, docs: list, top_n: int) -> list:
    scores = _cross_encoder().predict([(query, d.page_content) for d in docs])
    order = sorted(range(len(docs)), key=lambda i: float(scores[i]), reverse=True)
    return [docs[i] for i in order[:top_n]]


# ── LLM listwise rerank (Claude or OpenAI, per ANSWER_PROVIDER) ───────────────

_RANK_SCHEMA = {
    "type": "object",
    "properties": {"ranking": {"type": "array", "items": {"type": "integer"}}},
    "required": ["ranking"],
    "additionalProperties": False,
}

_RANK_SYSTEM = ("You are a search reranker. Order candidates by how directly they help "
                "answer the query. Respond with JSON {\"ranking\": [indices best-first]}.")


def _rerank_llm(query: str, docs: list, top_n: int) -> list:
    from . import llm

    listed = "\n".join(
        f"[{i}] {d.metadata.get('ticker')} · {d.metadata.get('section')}: {d.page_content[:280]}"
        for i, d in enumerate(docs))
    user = (f"Query: {query}\n\nCandidate chunks:\n{listed}\n\n"
            f"Return the indices of the most relevant chunks, best first, as a JSON array.")
    if config.RERANK_PROVIDER.lower() == "openai":
        resp = llm.openai_client().chat.completions.create(
            model=config.OPENAI_ANSWER_MODEL, max_tokens=512,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": _RANK_SYSTEM},
                      {"role": "user", "content": user}])
        idx = json.loads(resp.choices[0].message.content or "{}").get("ranking", [])
    else:
        msg = llm.anthropic_client().messages.create(
            model=config.ANTHROPIC_MODEL, max_tokens=512, system=_RANK_SYSTEM,
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": _RANK_SCHEMA}},
            messages=[{"role": "user", "content": user}])
        idx = json.loads(llm._first_text(msg)).get("ranking", [])
    seen, out = set(), []
    for i in idx:
        if isinstance(i, int) and 0 <= i < len(docs) and i not in seen:
            seen.add(i)
            out.append(docs[i])
    for i, d in enumerate(docs):            # append anything the model dropped (protect recall)
        if i not in seen:
            out.append(d)
    return out[:top_n]


# ── dispatch ─────────────────────────────────────────────────────────────────


def rerank(query: str, docs: list, top_n: int) -> list:
    backend = config.RERANKER
    if backend == "none" or len(docs) <= 1:
        return docs[:top_n]
    try:
        if backend == "cross_encoder":
            return _rerank_cross_encoder(query, docs, top_n)
        if backend == "llm":
            return _rerank_llm(query, docs, top_n)
    except Exception as e:  # noqa: BLE001 - never hard-fail retrieval
        if not _warned["shown"]:
            print(f"[rerank] backend '{backend}' unavailable ({e}); using vector order")
            _warned["shown"] = True
    return docs[:top_n]
