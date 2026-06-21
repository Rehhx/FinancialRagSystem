# Résumé blurbs — S&P 500 Financial RAG Vault

Pick the version that fits your space. All numbers are real (from the eval harness,
backtest, and test suite). Swap in your dates/role line as needed.

**Links:** GitHub `github.com/Rehhx/FinancialRagSystem` · Demo `youtu.be/mjyl6hkgMoY`

---

## One-line summary (top of project section)

**Financial Knowledge-Graph RAG** — a retrieval system over S&P 500 companies that
links them by SEC-filing-derived supply-chain relationships and answers
cross-company questions in natural language, with the quantitative answers computed
deterministically (not by the LLM). *Python · FastAPI · LangChain · ChromaDB ·
Claude + OpenAI · D3.js · SEC EDGAR.*

---

## Full bullets (5–7, for a dedicated project section)

- Built a **financial knowledge-graph RAG** over S&P 500 companies (Python, FastAPI,
  LangChain, ChromaDB, Claude + OpenAI embeddings) that links companies by
  supplier/customer/competitor relationships extracted from **SEC 10-K filings** and
  answers cross-company questions in natural language.
- Designed a **deterministic graph query engine** that routes ranking, aggregate,
  multi-hop, and threshold-filter questions away from the LLM and computes them in
  Python (e.g. *"average P/E of the competitors of NVIDIA's suppliers"*) — making
  every numeric answer **auditable and regression-testable**, with the LLM only
  narrating pre-computed results.
- Built an **evaluation harness** (recall@k, MRR, hit-rate + 10 deterministic
  graph-query regression guards) and **gated every retrieval change on it** —
  rejected BM25 hybrid retrieval and a larger embedding model that *hurt* recall, and
  validated an LLM reranker that lifted **recall@8 from 0.72 → 0.86–0.90**.
- Engineered a **multi-source data pipeline**: SEC EDGAR 10-K relationship
  extraction via Claude structured outputs, a **real-time 8-K material-event poller**
  that reacts within minutes of a filing hitting the wire, 6 news providers (free RSS
  + APIs) merged with priority-interleaved dedup, and FMP→yfinance fundamentals with
  automatic per-ticker fallback.
- Implemented a supply-chain **lead-lag backtest** (market-neutral **Sharpe ≈ 1.2**,
  positive Information Coefficient across the parameter grid) and a signals-validation
  layer showing linked companies co-move at **0.33 vs 0.15** for unlinked (**+0.17
  lift**) — i.e. the relationship graph tracks real market behavior.
- Shipped a **conversational RAG** (multi-turn follow-ups with query condensing,
  inline clickable source citations) and a dependency-free **D3.js graph explorer**
  (weighted-PageRank centrality sizing, sector cluster hulls, backtest equity
  curve/IC heatmap, portfolio supply-chain exposure overlay).
- Made the LLM layer **provider-pluggable** (Claude/OpenAI, decoupled answer vs.
  rerank) and **cost-engineered** the system to ~$11–40/mo via incremental
  content-hash indexing, narrating on one model while reranking on another, and
  skipping the LLM entirely for computed answers; **57 unit tests**, eval-gated
  decisions throughout.

## Condensed bullets (3, for a packed résumé)

- Built a **financial knowledge-graph RAG** over S&P 500 companies (Python, FastAPI,
  LangChain, ChromaDB, Claude/OpenAI) linking them by SEC-filing supply-chain
  relationships; a **deterministic graph engine** computes ranking/aggregate/multi-hop
  answers in Python so results are auditable, with the LLM only narrating them.
- Drove retrieval quality with an **eval harness** (recall@k/MRR + 10 regression
  guards), rejecting changes that hurt recall and shipping an LLM reranker that took
  **recall@8 0.72 → 0.86–0.90**; ingested SEC EDGAR (10-K + **real-time 8-K poller**),
  6 news sources, and FMP/yfinance fundamentals.
- Implemented a supply-chain **lead-lag backtest** (market-neutral **Sharpe ≈ 1.2**)
  and a **D3.js** graph UI (centrality, cluster hulls, portfolio exposure); 57 tests,
  provider-pluggable LLMs, ~$11–40/mo run cost.

## One-liner (skills/projects list)

- **Financial Knowledge-Graph RAG (S&P 500):** SEC-filing supply-chain graph +
  deterministic query engine (LLM narrates, Python computes) + eval-gated retrieval
  (recall@8 0.86–0.90) + lead-lag backtest (Sharpe ≈ 1.2). Python, FastAPI, LangChain,
  ChromaDB, Claude/OpenAI, D3.js.

---

## Skills demonstrated (for a skills section / keyword coverage)

**ML / RAG:** retrieval-augmented generation, vector search (ChromaDB), embeddings,
LLM reranking, hybrid retrieval, evaluation (recall@k / MRR / faithfulness),
prompt/structured-output engineering, LLM cost optimization.
**Data / Quant:** SEC EDGAR (10-K/8-K) parsing, financial fundamentals, return
correlations, weighted PageRank / network centrality, lead-lag backtesting,
Information Coefficient, Sharpe.
**Engineering:** Python, FastAPI, REST APIs, D3.js, pandas/numpy, multi-provider
data ingestion with graceful fallback, incremental indexing, scheduling/automation,
pytest, real-time polling.

## Interview talking points (not for the résumé — for you)

- *"Is it a ChatGPT wrapper?"* No — graph/aggregate answers are computed in Python;
  the LLM only writes the prose. That's why they're auditable and regression-tested.
- *"How did you make engineering decisions?"* Eval-gated. The harness **rejected**
  three plausible upgrades (CPU cross-encoder, BM25 hybrid, larger embeddings) that
  measurably hurt recall, and justified the ones I kept with numbers.
- *"Does it find real signal?"* Linked companies co-move at 0.33 vs 0.15 unlinked;
  the supplier-momentum lead-lag has positive IC and ~1.2 market-neutral Sharpe.
