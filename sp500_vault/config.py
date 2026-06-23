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

# ── Langfuse tracing (optional) ──────────────────────────────────────────────
# When both keys are present, LLM calls are traced to Langfuse (OpenAI via the
# drop-in client, Anthropic via the OTel instrumentor). Absent keys → tracing
# no-ops and the app runs unchanged. The CLI/skill uses LANGFUSE_BASE_URL while
# the SDK reads LANGFUSE_HOST, so accept either and mirror it into the env the
# SDK expects (must happen before langfuse is imported in tracing.py).
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST = (os.getenv("LANGFUSE_HOST") or os.getenv("LANGFUSE_BASE_URL")
                 or "https://us.cloud.langfuse.com")
if LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY:
    os.environ.setdefault("LANGFUSE_HOST", LANGFUSE_HOST)
TRACING_ENABLED = bool(LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY)

# Online eval: grade answer faithfulness on every LIVE rag query and attach it as a
# Langfuse score. Off by default — it costs one judge LLM call per query. The eval
# harness (`pipeline eval --judge`) scores faithfulness offline on the golden set
# regardless of this flag.
RAG_SCORE_FAITHFULNESS = os.getenv("RAG_SCORE_FAITHFULNESS", "false").lower() in ("1", "true", "yes")

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

# RAG reranker: "none" (MMR order + coverage), "llm" (listwise Claude/OpenAI), or
# "cross_encoder" (local MiniLM). Eval verdict EVOLVED: once relation-aware
# COVERAGE_RERANK landed, plain MMR + coverage (none) **beats** the LLM reranker —
# recall@8 0.95 / MRR 1.0 (deterministic, free) vs llm 0.944 / 0.856 (stochastic,
# a Claude call per query). The reranker was approximating the relationship-aware
# selection that coverage now does directly and better, so "none" is the default.
# History: cross_encoder/MiniLM 0.62 and BM25 hybrid both also eval-rejected. The
# "llm" backend's model is RERANK_PROVIDER below (kept pluggable for re-evaluation).
RERANKER = os.getenv("RERANKER", "none")
RERANK_PROVIDER = os.getenv("RERANK_PROVIDER", "claude")  # "claude" | "openai"
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# Hybrid retrieval: fuse a BM25 keyword ranking with the dense vectors to lift
# recall on exact-token matches. Eval verdict on this corpus: it *hurts* (recall@8
# 0.83 -> 0.73 reorder / 0.78 union) — dense + graph-expand + LLM-rerank already
# covers the relationship/semantic questions, and BM25 only injects noise the
# reranker over-trusts. Kept as a toggle (off) for re-evaluation as the corpus grows.
HYBRID_RETRIEVAL = os.getenv("HYBRID_RETRIEVAL", "false").lower() in ("1", "true", "yes")

# Coverage-aware reranking: guarantee the entities the question is about survive
# the top-k truncation — the named subject ("what does Tesla use?" was dropping
# TSLA) and the asked-for relation's neighbors resolved in BOTH directions ("who
# supplies Dell?" kept Dell's competitors but dropped its chip suppliers; "who buys
# from Broadcom?" needs the *customers* recorded in the customers' own filings).
# Injection is ordered by candidate-pool relevance so hubs don't flood. Eval-gated:
# recall@8 0.80 -> 0.95 and MRR to 1.0 on the golden set (the single biggest lever).
COVERAGE_RERANK = os.getenv("COVERAGE_RERANK", "true").lower() in ("1", "true", "yes")

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

# 8-K body-text LLM summaries: for the highest-signal item types (material
# agreements, acquisitions, exec changes, impairments) we fetch the actual filing
# body and have Claude write a one-line "what happened" summary. Item codes like
# 2.02 (earnings — just numbers) or 9.01 (exhibits) carry no narrative worth
# summarizing, so they are excluded. Summaries are cached by accession (one LLM
# call per filing, ever), so this is a small, bounded, one-time cost.
EDGAR_8K_SUMMARIZE = os.getenv("EDGAR_8K_SUMMARIZE", "true").lower() in ("1", "true", "yes")
EDGAR_8K_SUMMARY_ITEMS = os.getenv(
    "EDGAR_8K_SUMMARY_ITEMS", "1.01,1.02,1.03,2.01,2.05,2.06,4.02,5.01,5.02")
EDGAR_8K_BODY_MAX_CHARS = int(os.getenv("EDGAR_8K_BODY_MAX_CHARS", "12000"))

# ── Signal engine (external trading-engine API) ──────────────────────────────
# `engine.py` blends the vault's three validated strategies into a per-ticker
# LONG/SHORT/FLAT verdict, each weighted by its measured Information Coefficient
# (Grinold–Kahn signal combination; Fundamental Law IR = IC·√breadth). Designed
# to be polled by an *external* trading engine over HTTP (GET /signal/{ticker}).
TRADE_HORIZON = int(os.getenv("TRADE_HORIZON", "5"))            # default forward horizon (trading days)
SUPMOM_K = int(os.getenv("SUPMOM_K", "5"))                      # supplier-momentum lookback (days)
CONVICTION_TAU = float(os.getenv("CONVICTION_TAU", "0.5"))     # |conviction z| needed to act (else FLAT)
IC_FLOOR = float(os.getenv("IC_FLOOR", "0.0"))                 # min |IC| for a component to contribute
EVENT_LOOKBACK_DAYS = int(os.getenv("EVENT_LOOKBACK_DAYS", "7"))  # an 8-K's drift is "live" this many days
EVENT_MIN_T = float(os.getenv("EVENT_MIN_T", "2.0"))          # only trade event drift with |t| ≥ this
EVENT_IC = float(os.getenv("EVENT_IC", "0.10"))              # trust weight for a significant event drift
# Sentiment lead-lag IC: prefer the measured rank-IC at the horizon (claude panel,
# then provider); fall back to this when neither has scorable obs yet.
SENTIMENT_IC_FALLBACK = float(os.getenv("SENTIMENT_IC_FALLBACK", "0.05"))


def high_signal_items() -> set[str]:
    """The 8-K item codes worth an LLM body summary (from EDGAR_8K_SUMMARY_ITEMS)."""
    return {c.strip() for c in EDGAR_8K_SUMMARY_ITEMS.split(",") if c.strip()}


def require(*keys: str) -> None:
    """Raise a clear error if a required secret is missing."""
    missing = [k for k in keys if not globals().get(k)]
    if missing:
        raise RuntimeError(
            f"Missing required config: {', '.join(missing)}. "
            f"Add them to {BASE_DIR / '.env'}."
        )
