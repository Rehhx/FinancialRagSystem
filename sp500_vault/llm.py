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

from . import config

# ── Clients (lazy singletons) ────────────────────────────────────────────────


@lru_cache(maxsize=1)
def anthropic_client() -> anthropic.Anthropic:
    config.require("ANTHROPIC_API_KEY")
    return anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)


@lru_cache(maxsize=1)
def openai_client() -> OpenAI:
    config.require("OPENAI_API_KEY")
    return OpenAI(api_key=config.OPENAI_API_KEY)


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


def extract_relationships(source_ticker: str, source_name: str, business_text: str) -> list[dict]:
    """Extract supplier/customer/competitor edges from a 10-K Business section."""
    if not business_text.strip():
        return []
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


def score_sentiment(ticker: str, name: str, headlines: list[str]) -> dict:
    """Score sentiment for a company from recent headlines/snippets."""
    if not headlines:
        return {"score": 0.0, "label": "Neutral", "summary": "No recent news available."}
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


# ── RAG answer ───────────────────────────────────────────────────────────────

_RAG_SYSTEM = (
    "You answer questions about an S&P 500 (pilot: semiconductor + hardware) research "
    "vault. Each company has a note with quant metrics, an LLM sentiment score, and "
    "explicit supplier/customer/competitor relationships rendered as [[wikilinks]]. "
    "Use the relationship structure to reason about connected exposure (e.g. who is "
    "exposed to a given company's guidance). Ground every claim in the provided context "
    "and cite the ticker/section it came from. If the context is insufficient, say so."
)


def rag_answer(question: str, contexts: list[str]) -> str:
    """Answer a natural-language question grounded in retrieved note chunks."""
    context_block = "\n\n---\n\n".join(contexts)
    user = (
        f"Context from the vault:\n\n{context_block}\n\n"
        f"=== QUESTION ===\n{question}"
    )
    with anthropic_client().messages.stream(
        model=config.ANTHROPIC_MODEL,
        max_tokens=2000,
        system=_RAG_SYSTEM,
        messages=[{"role": "user", "content": user}],
    ) as stream:
        message = stream.get_final_message()
    return _first_text(message)


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


def grade_faithfulness(question: str, answer: str, contexts: list[str]) -> dict:
    """LLM-as-judge: how well is the answer grounded ONLY in the retrieved context?"""
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
