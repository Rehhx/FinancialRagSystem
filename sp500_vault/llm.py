"""LLM access layer: Claude (extraction, sentiment, RAG answers) + OpenAI embeddings.

Everything that touches an LLM goes through here so prompts, models, and
structured-output schemas live in one place. Claude calls use structured
outputs (``output_config.format``) for the extraction/scoring tasks that must
return machine-parseable JSON.
"""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

import anthropic
from openai import OpenAI

from . import config, tracing
from .tracing import observe

# ── Clients (lazy singletons) ────────────────────────────────────────────────


@lru_cache(maxsize=1)
def anthropic_client() -> anthropic.Anthropic:
    config.require("ANTHROPIC_API_KEY")
    tracing.instrument_anthropic()   # auto-traces every messages.create / stream when enabled
    return anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)


@lru_cache(maxsize=1)
def openai_client() -> OpenAI:
    config.require("OPENAI_API_KEY")
    # The Langfuse drop-in (when tracing is on) auto-traces chat + embeddings.
    return tracing.openai_class(OpenAI)(api_key=config.OPENAI_API_KEY)


def _first_text(message: anthropic.types.Message) -> str:
    return next((b.text for b in message.content if b.type == "text"), "")


# ── Relationship extraction ──────────────────────────────────────────────────

_RELATIONSHIP_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "relationships": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "company_name": {"type": "string"},
                    "ticker": {
                        "type": "string",
                        "description": "US stock ticker if the company is publicly traded and you are confident; otherwise an empty string.",
                    },
                    "relation": {
                        "type": "string",
                        "enum": ["supplier", "customer", "competitor"],
                    },
                    "confidence": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                    },
                    "evidence": {
                        "type": "string",
                        "description": "Short quote or paraphrase from the filing supporting this edge.",
                    },
                },
                "required": ["company_name", "ticker", "relation", "confidence", "evidence"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["relationships"],
    "additionalProperties": False,
}

_RELATIONSHIP_SYSTEM = (
    "You are a financial analyst extracting supply-chain and competitive relationships "
    "from SEC 10-K filings. Be precise about relationship *type* — do not conflate a "
    "competitor named in a 'Competition' section with a supply-chain partner.\n"
    "- supplier: a company that sells goods/services TO the filer.\n"
    "- customer: a company that BUYS the filer's products.\n"
    "- competitor: a company that competes with the filer.\n"
    "Only include named companies (not generic categories). Prefer publicly traded "
    "companies. Provide a ticker only when you are confident; otherwise leave it empty. "
    "Do not invent relationships that are not supported by the text."
)


@observe(name="relationship-extract")
def extract_relationships(source_ticker: str, source_name: str, business_text: str) -> list[dict]:
    """Extract supplier/customer/competitor edges from a 10-K Business section."""
    if not business_text.strip():
        return []
    # Explicit, compact trace input — don't dump the 45K-char filing as the input.
    tracing.update_span(input={"ticker": source_ticker, "name": source_name,
                               "business_chars": len(business_text)},
                        metadata={"feature": "relationship-extraction"})
    user = (
        f"Filer: {source_name} ({source_ticker}).\n"
        f"Below is the 'Business' section (Item 1) of its latest 10-K. Extract every "
        f"named company mentioned as a supplier, customer, or competitor.\n\n"
        f"=== FILING TEXT ===\n{business_text}"
    )
    msg = anthropic_client().messages.create(
        model=config.ANTHROPIC_MODEL,
        max_tokens=8000,
        system=_RELATIONSHIP_SYSTEM,
        output_config={"effort": config.ANTHROPIC_EFFORT,
                       "format": {"type": "json_schema", "schema": _RELATIONSHIP_SCHEMA}},
        messages=[{"role": "user", "content": user}],
    )
    data = json.loads(_first_text(msg))
    return data.get("relationships", [])


# ── Sentiment scoring ────────────────────────────────────────────────────────

_SENTIMENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "score": {
            "type": "number",
            "description": "Sentiment from -1.0 (very bearish) to 1.0 (very bullish).",
        },
        "label": {"type": "string", "enum": ["Bullish", "Neutral", "Bearish"]},
        "summary": {"type": "string", "description": "Two-sentence rationale."},
    },
    "required": ["score", "label", "summary"],
    "additionalProperties": False,
}

_SENTIMENT_SYSTEM = (
    "You score equity news sentiment for a single company. Weight financial and "
    "operational news (earnings, guidance, demand, margins, legal/regulatory) much more "
    "heavily than generic mentions. Return a score in [-1, 1], a label, and a concise "
    "two-sentence summary of what is driving it. Judge the company's OWN business — do "
    "not inherit sentiment from its customers or suppliers."
)


@observe(name="sentiment-score")
def score_sentiment(ticker: str, name: str, headlines: list[str]) -> dict:
    """Score sentiment for a company from recent headlines/snippets."""
    if not headlines:
        return {"score": 0.0, "label": "Neutral", "summary": "No recent news available."}
    tracing.update_span(input={"ticker": ticker, "name": name, "headlines": len(headlines)},
                        metadata={"feature": "sentiment"})
    joined = "\n".join(f"- {h}" for h in headlines)
    user = f"Company: {name} ({ticker}).\nRecent news:\n{joined}"
    msg = anthropic_client().messages.create(
        model=config.ANTHROPIC_MODEL,
        max_tokens=1024,
        system=_SENTIMENT_SYSTEM,
        output_config={"effort": config.ANTHROPIC_EFFORT,
                       "format": {"type": "json_schema", "schema": _SENTIMENT_SCHEMA}},
        messages=[{"role": "user", "content": user}],
    )
    out = json.loads(_first_text(msg))
    # Clamp defensively.
    out["score"] = max(-1.0, min(1.0, float(out.get("score", 0.0))))
    return out


# ── 8-K material-event summary ───────────────────────────────────────────────

_8K_SUMMARY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {
            "type": "string",
            "description": "One specific sentence (<= 30 words) stating what materially "
                           "happened. Empty string if the body has no material detail.",
        },
    },
    "required": ["summary"],
    "additionalProperties": False,
}

_8K_SUMMARY_SYSTEM = (
    "You summarize the material event in an SEC 8-K filing in ONE specific sentence "
    "(<= 30 words). Name the concrete fact: for a 5.02, the executive and their role and "
    "whether they departed or were appointed; for a 1.01/2.01, the counterparty/target and "
    "any dollar amount; for a 2.06, the size and nature of the impairment; for a 4.02, what "
    "financials are non-reliable. Omit legal boilerplate, forward-looking disclaimers, and "
    "exhibit lists. If the body is only an exhibit pointer or press-release reference with no "
    "substance, return an empty string."
)


@observe(name="filing-8k-summary")
def summarize_8k(ticker: str, item_labels: str, body_text: str) -> str:
    """One-line 'what happened' summary of a high-signal 8-K from its body text."""
    if not body_text.strip():
        return ""
    tracing.update_span(input={"ticker": ticker, "items": item_labels, "body_chars": len(body_text)},
                        metadata={"feature": "8k-summary"})
    user = (f"Filer: {ticker}. 8-K item(s): {item_labels}.\n\n"
            f"=== 8-K BODY ===\n{body_text}")
    msg = anthropic_client().messages.create(
        model=config.ANTHROPIC_MODEL,
        max_tokens=400,
        system=_8K_SUMMARY_SYSTEM,
        output_config={"effort": config.ANTHROPIC_EFFORT,
                       "format": {"type": "json_schema", "schema": _8K_SUMMARY_SCHEMA}},
        messages=[{"role": "user", "content": user}],
    )
    return json.loads(_first_text(msg)).get("summary", "").strip()


# ── RAG answer ───────────────────────────────────────────────────────────────

_RAG_SYSTEM = (
    "You answer questions about an S&P 500 (pilot: semiconductor + hardware) research "
    "vault. Each company has a note with quant metrics, an LLM sentiment score, "
    "explicit supplier/customer/competitor relationships rendered as [[wikilinks]], a "
    "'Recent News' section of dated headlines, and a 'Material Events' section of recent "
    "8-K filings (earnings releases, executive changes, acquisitions, etc.). "
    "Use the relationship structure to reason about connected exposure (e.g. who is "
    "exposed to a given company's guidance), and cite recent headlines (with their "
    "dates) when the question is about news or catalysts. Ground every claim in the "
    "provided context and cite the ticker/section it came from. If the context is "
    "insufficient, say so.\n"
    "CITE SOURCES: cite the [TICKER · Section] each claim came from. When a context "
    "line carries a 'SOURCE: <url>' (news headlines, 8-K filings), cite it as a "
    "markdown link — e.g. [Reuters](url) or [SEC 8-K](url) — so the reader can click "
    "through to the original.\n"
    "IMPORTANT: when the context includes a pre-computed 'Graph · ...' summary (an "
    "aggregate value, ranking, count, or filtered set), report that value and that "
    "membership EXACTLY as given — do not recompute, re-sum, or drop/add members "
    "yourself. Those numbers are computed deterministically; your job is to present "
    "them, not to redo the arithmetic."
)


def _history_block(history: list[dict] | None) -> str:
    if not history:
        return ""
    convo = "\n".join(f"{(h.get('role') or 'user').upper()}: {h.get('content', '')}"
                      for h in history[-6:])
    return f"Conversation so far:\n{convo}\n\n"


def _rag_user_prompt(question: str, contexts: list[str], history: list[dict] | None = None) -> str:
    context_block = "\n\n---\n\n".join(contexts)
    return (f"{_history_block(history)}Context from the vault:\n\n{context_block}\n\n"
            f"=== QUESTION ===\n{question}")


def _rag_answer_claude(question: str, contexts: list[str], history=None) -> str:
    with anthropic_client().messages.stream(
        model=config.ANTHROPIC_MODEL,
        max_tokens=2000,
        system=_RAG_SYSTEM,
        messages=[{"role": "user", "content": _rag_user_prompt(question, contexts, history)}],
    ) as stream:
        message = stream.get_final_message()
    return _first_text(message)


def _rag_answer_openai(question: str, contexts: list[str], history=None) -> str:
    resp = openai_client().chat.completions.create(
        model=config.OPENAI_ANSWER_MODEL,
        max_tokens=2000,
        name="rag-answer",          # readable generation name in Langfuse
        messages=[
            {"role": "system", "content": _RAG_SYSTEM},
            {"role": "user", "content": _rag_user_prompt(question, contexts, history)},
        ],
    )
    return (resp.choices[0].message.content or "").strip()


# ── Conversational follow-ups ────────────────────────────────────────────────

_CONDENSE_SYSTEM = (
    "You rewrite a user's follow-up question into a standalone question for a search "
    "engine. Resolve pronouns and ellipsis using the conversation (e.g. 'what about "
    "its suppliers?' after a question about NVIDIA → 'Who are NVIDIA's suppliers?'). "
    "Keep any named companies/tickers/metrics. Return ONLY the rewritten question."
)


def condense_question(history: list[dict] | None, question: str) -> str:
    """Rewrite a follow-up into a standalone question using the conversation, so
    retrieval works on the resolved intent (not the bare 'what about them?')."""
    if not history:
        return question
    convo = "\n".join(f"{(h.get('role') or 'user').upper()}: {h.get('content', '')}"
                      for h in history[-6:])
    user = f"Conversation:\n{convo}\n\nFollow-up: {question}\n\nStandalone question:"
    try:
        if config.ANSWER_PROVIDER.lower() == "openai":
            resp = openai_client().chat.completions.create(
                model=config.OPENAI_ANSWER_MODEL, max_tokens=120, name="condense-question",
                messages=[{"role": "system", "content": _CONDENSE_SYSTEM},
                          {"role": "user", "content": user}])
            out = (resp.choices[0].message.content or "").strip()
        else:
            msg = anthropic_client().messages.create(
                model=config.ANTHROPIC_MODEL, max_tokens=150, system=_CONDENSE_SYSTEM,
                messages=[{"role": "user", "content": user}])
            out = _first_text(msg).strip()
    except Exception:  # noqa: BLE001 - fall back to the raw question
        return question
    return out or question


def rag_answer(question: str, contexts: list[str], history: list[dict] | None = None) -> str:
    """Answer a natural-language question grounded in retrieved note chunks.

    Provider is configurable (``ANSWER_PROVIDER``): the answer LLM only narrates
    pre-computed/retrieved context, so Claude or OpenAI both work — choose by cost.
    ``history`` (prior turns) makes follow-up answers conversational.
    """
    if config.ANSWER_PROVIDER.lower() == "openai":
        return _rag_answer_openai(question, contexts, history)
    return _rag_answer_claude(question, contexts, history)


# ── Answer faithfulness judge (eval) ─────────────────────────────────────────

_FAITH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "score": {"type": "number", "description": "0.0 (unsupported/hallucinated) to 1.0 (fully grounded)."},
        "verdict": {"type": "string", "enum": ["grounded", "partial", "unsupported"]},
    },
    "required": ["score", "verdict"],
    "additionalProperties": False,
}


@observe(name="faithfulness-judge")
def grade_faithfulness(question: str, answer: str, contexts: list[str]) -> dict:
    """LLM-as-judge: how well is the answer grounded ONLY in the retrieved context?"""
    tracing.update_span(input={"question": question}, metadata={"feature": "eval-judge"})
    joined = "\n\n---\n\n".join(contexts)
    user = (
        f"Question:\n{question}\n\nRetrieved context:\n{joined}\n\n"
        f"Answer to grade:\n{answer}\n\n"
        "Score how well every claim in the answer is supported by the context above."
    )
    msg = anthropic_client().messages.create(
        model=config.ANTHROPIC_MODEL,
        max_tokens=512,
        system=("You are a strict RAG evaluator. Score how well the answer is grounded ONLY in "
                "the provided context. Penalize any claim not supported by the context, even if "
                "it is true in reality. Return a score in [0,1] and a verdict."),
        output_config={"effort": config.ANTHROPIC_EFFORT,
                       "format": {"type": "json_schema", "schema": _FAITH_SCHEMA}},
        messages=[{"role": "user", "content": user}],
    )
    out = json.loads(_first_text(msg))
    out["score"] = max(0.0, min(1.0, float(out.get("score", 0.0))))
    return out


# ── Embeddings ───────────────────────────────────────────────────────────────


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts with OpenAI embeddings."""
    if not texts:
        return []
    resp = openai_client().embeddings.create(model=config.OPENAI_EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]
