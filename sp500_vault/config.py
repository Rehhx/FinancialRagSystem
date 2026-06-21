"""Central configuration: env loading, paths, model selection.

All secrets come from the project-root ``.env`` (never hard-coded). Model
choices and tunables can be overridden via environment variables so the same
code runs cheaply in dev and at full fidelity in a real refresh.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

DATA_DIR = BASE_DIR / "data"
QUANT_DIR = DATA_DIR / "quant"
SENTIMENT_DIR = DATA_DIR / "sentiment"
EDGAR_DIR = DATA_DIR / "edgar"
CHROMA_DIR = DATA_DIR / "chroma"
SIGNALS_DIR = DATA_DIR / "signals"
FILINGS_DIR = DATA_DIR / "filings"
VAULT_DIR = BASE_DIR / "vault"

RELATIONSHIPS_DB = DATA_DIR / "relationships.db"
MANUAL_OVERRIDES_CSV = BASE_DIR / "relationships_manual_overrides.csv"
CORRELATIONS_FILE = SIGNALS_DIR / "correlations.json"

for _d in (DATA_DIR, QUANT_DIR, SENTIMENT_DIR, EDGAR_DIR, SIGNALS_DIR, FILINGS_DIR, VAULT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Secrets ──────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET", "")
# Extra news providers (free tiers) — broaden headline coverage for the sentiment
# layer. Each is optional; a missing key just skips that provider gracefully.
ALPHA_API_KEY = os.getenv("ALPHA_API_KEY", "")          # Alpha Vantage NEWS_SENTIMENT
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")            # NewsAPI.org /everything
MARKETAUX_API_KEY = os.getenv("MARKETAUX_API_KEY", "")  # Marketaux /news/all
FMP_API_KEY = os.getenv("FMP_API_KEY", "")              # Financial Modeling Prep fundamentals

# ── Models / tunables ────────────────────────────────────────────────────────
# Default to the most capable Claude model; override with ANTHROPIC_MODEL.
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-8")
# Effort applies on Opus 4.6+/Sonnet 4.6 — "low" keeps extraction/sentiment cheap.
ANTHROPIC_EFFORT = os.getenv("ANTHROPIC_EFFORT", "low")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

# Which LLM writes the final RAG answer prose: "claude" or "openai". The answer
# layer only *narrates* (graph numbers are computed in Python, semantic answers
# summarize retrieved chunks), so either works — pick by cost/credits. Extraction
# and sentiment always use Claude (structured-output schemas).
ANSWER_PROVIDER = os.getenv("ANSWER_PROVIDER", "claude")
OPENAI_ANSWER_MODEL = os.getenv("OPENAI_ANSWER_MODEL", "gpt-4o")

# Fundamentals source: "auto" (FMP when FMP_API_KEY is set, else yfinance),
# "fmp" (force FMP, fall back to yfinance on failure), or "yfinance".
FUNDAMENTALS_SOURCE = os.getenv("FUNDAMENTALS_SOURCE", "auto")

# RAG reranker: "llm" (listwise), "cross_encoder" (local; generic MiniLM hurt —
# needs a finance-tuned model), or "none". Eval (recall@8): none 0.72 ·
# cross_encoder/MiniLM 0.62 · llm-openai 0.78 · llm-claude 0.83 (Claude reranks
# sharper, MRR ~1.0). The "llm" backend's model is set by RERANK_PROVIDER below —
# independent of ANSWER_PROVIDER, so you can rerank on Claude and narrate on OpenAI.
RERANKER = os.getenv("RERANKER", "llm")
RERANK_PROVIDER = os.getenv("RERANK_PROVIDER", "claude")  # "claude" | "openai"
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# Hybrid retrieval: fuse a BM25 keyword ranking with the dense vectors to lift
# recall on exact-token matches. Eval verdict on this corpus: it *hurts* (recall@8
# 0.83 -> 0.73 reorder / 0.78 union) — dense + graph-expand + LLM-rerank already
# covers the relationship/semantic questions, and BM25 only injects noise the
# reranker over-trusts. Kept as a toggle (off) for re-evaluation as the corpus grows.
HYBRID_RETRIEVAL = os.getenv("HYBRID_RETRIEVAL", "false").lower() in ("1", "true", "yes")

# SEC requires a descriptive User-Agent with contact info on every request.
SEC_USER_AGENT = os.getenv(
    "SEC_USER_AGENT",
    "sp500-rag-vault research-bot (contact: " + os.getenv("SEC_CONTACT_EMAIL", "research@example.com") + ")",
)

# How far back to pull news for the sentiment pass.
SENTIMENT_LOOKBACK_DAYS = int(os.getenv("SENTIMENT_LOOKBACK_DAYS", "14"))
SENTIMENT_MAX_ARTICLES = int(os.getenv("SENTIMENT_MAX_ARTICLES", "10"))
# Which news providers to aggregate, in priority order (comma-separated). Unknown
# names or keyed providers with no key are skipped. Free RSS feeds (googlenews,
# yahoo) need no key and have no daily quota, so they lead as the reliable backbone.
NEWS_PROVIDERS = os.getenv("NEWS_PROVIDERS", "googlenews,yahoo,finnhub,marketaux,newsapi,alphavantage")
# Per-provider request cap (free tiers are small; keep pulls modest).
NEWS_PER_PROVIDER = int(os.getenv("NEWS_PER_PROVIDER", "10"))

# Cap the 10-K "Business" section text we send to Claude (chars) to bound cost.
EDGAR_BUSINESS_MAX_CHARS = int(os.getenv("EDGAR_BUSINESS_MAX_CHARS", "45000"))

# 8-K material-event ingestion: how far back to look and how many to keep per ticker.
EDGAR_8K_LOOKBACK_DAYS = int(os.getenv("EDGAR_8K_LOOKBACK_DAYS", "90"))
EDGAR_8K_MAX = int(os.getenv("EDGAR_8K_MAX", "8"))


def require(*keys: str) -> None:
    """Raise a clear error if a required secret is missing."""
    missing = [k for k in keys if not globals().get(k)]
    if missing:
        raise RuntimeError(
            f"Missing required config: {', '.join(missing)}. "
            f"Add them to {BASE_DIR / '.env'}."
        )
