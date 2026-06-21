# RAG System — Roadmap & Procurement Plan

**Owner:** Quant Research Engineering
**Status of the pilot:** production-shaped, running on free/low-cost data. This
document lists what to **buy / provision** to take it from a strong pilot to a
best-in-class, institutional RAG over the S&P 500 supply-chain graph.

---

## 1. Where we are today

| Layer | Pilot implementation | Limitation |
|---|---|---|
| Relationship graph | LLM extraction from 10-Ks (Claude) + Finnhub peers + manual CSV | LLM extraction has false positives; coverage is filing-dependent |
| Fundamentals | yfinance (free) | Unreliable, rate-limited, no point-in-time / restatement history |
| News & sentiment | Finnhub free + Claude scoring | Shallow news depth, ~10 articles, no intraday, no historical sentiment series |
| Prices | Alpaca **IEX** (free) | Partial tape (IEX ~2–3% of volume), no full SIP, limited history |
| Embeddings | OpenAI `text-embedding-3-small` | General-purpose, not finance-tuned |
| Retrieval | Chroma (local) + MMR + **graph-aware expansion** | Single-node, dense-only (no hybrid/BM25), no reranker |
| Answering | Claude Opus 4.8 | Good; no answer-faithfulness eval gate |
| Orchestration | Windows Task Scheduler → `scheduler tick` daily | Single workstation, not HA, no alerting |
| Eval | None (manual spot-checks) | **No quantitative retrieval/answer quality measurement** |

**Already shipped this iteration:** incremental (hash-based) re-indexing so daily
updates re-embed only changed chunks; MMR retrieval; graph-aware retrieval
(named tickers + their neighbors pulled into context); a daily scheduled job.

---

## 2. What "best on the market" requires — gap analysis

The single biggest differentiator for a **supply-chain** RAG is **data quality on
the edges** (who actually supplies whom, with revenue %). After that: **retrieval
quality** (hybrid + reranking + finance embeddings), and **evaluation** (you can't
improve what you don't measure). Everything below is ranked by ROI for a quant desk.

---

## 3. Procurement list (what to get)

> Costs are rough monthly order-of-magnitude for budgeting only; negotiate enterprise terms.
> Priority: **P0** = do first / unblocks everything, **P1** = high ROI, **P2** = scale/hardening.

### 3.1 Data feeds — *the highest-leverage spend*

| Item | Vendor options | Priority | Est. cost/mo | Why |
|---|---|---|---|---|
| **Supply-chain relationships** (named customers/suppliers, revenue %) | FactSet *Supply Chain Relationships*, Bloomberg **SPLC**, S&P Global **Panjiva** / Capital IQ, Refinitiv | **P0** | $$$ (5–6 figs/yr) | Replaces error-prone LLM extraction with audited, revenue-weighted edges. This is the core asset. |
| **Point-in-time fundamentals** | S&P Capital IQ, FactSet, FMP (budget), Polygon | **P0** | $–$$$ | yfinance is not investable. Need PIT, restatements, GAAP/non-GAAP. |
| **Institutional news + sentiment** | RavenPack, Bloomberg, Alexandria, Benzinga Pro | **P1** | $$–$$$ | Deep, historical, entity-tagged news → real sentiment time series (see §4). |
| **Full-tape market data** | Polygon.io, Databento, Alpaca **SIP** upgrade | **P1** | $–$$ | Replace IEX partial tape; full history for the signals/backtest layers. |
| **Filings / transcripts** | AlphaSense, S&P, Bloomberg transcripts | **P2** | $$ | Earnings-call Q&A is rich relationship/guidance signal. |

### 3.2 Models & retrieval

| Item | Vendor options | Priority | Est. cost/mo | Why |
|---|---|---|---|---|
| **Reranker (cross-encoder)** | Cohere **Rerank 3**, Voyage **rerank-2**, or self-host `bge-reranker-v2-m3` | **P0** | $ (API) or GPU | Biggest single retrieval-quality lever after data. Rerank top-50 → top-8. |
| **Finance-domain embeddings** | Voyage **voyage-3-large / voyage-finance-2**, Cohere embed v3, or OpenAI `text-embedding-3-large` | **P1** | $ | Domain embeddings materially lift recall on financial text. |
| **LLM answer budget** | Anthropic Claude (current) | P1 | $$ (scales w/ volume) | Keep Claude; budget for higher QPS + enable prompt caching for the system/graph context. |

### 3.3 Vector store & infrastructure

| Item | Vendor options | Priority | Est. cost/mo | Why |
|---|---|---|---|---|
| **Production vector DB w/ hybrid search** | **Qdrant**, pgvector (Postgres), Weaviate, Pinecone | **P1** | $–$$ | Chroma is a great pilot; production needs hybrid (BM25 + dense), filtering at scale, HA, backups. |
| **Orchestration** | **Prefect** / Dagster / Airflow (managed) | **P1** | $–$$ | Replace Windows Task Scheduler: retries, alerting, backfills, lineage. |
| **Compute** | 1 small cloud VM (always-on) + optional GPU (rerank/embeddings self-host) | **P1** | $–$$ | Move off the workstation; GPU only if self-hosting rerankers. |
| **Cache** | Redis + semantic cache (GPTCache) | **P2** | $ | Cut repeat-query latency and LLM cost. |

### 3.4 Evaluation & observability — *non-negotiable for a quant firm*

| Item | Vendor options | Priority | Est. cost/mo | Why |
|---|---|---|---|---|
| **Golden eval set** | Analyst-curated Q&A (100–300 questions w/ expected sources) | **P0** | analyst time | You cannot tune retrieval without ground truth. Build this first. |
| **RAG eval framework** | **RAGAS**, TruLens, Arize Phoenix | **P0** | free–$ | Track recall@k, MRR, context precision, **faithfulness/answer-correctness**. Gate releases on it. |
| **Tracing / observability** | **Langfuse** (OSS), LangSmith, Arize | **P1** | free–$$ | Per-query latency/cost, retrieval traces, drift, regression alerts. |
| **Answer guardrails** | Citation-verification pass + refusal on low-grounding | **P1** | LLM cost | Hallucination control; require every claim to map to a retrieved chunk. |

### 3.5 People & process

| Item | Priority | Why |
|---|---|---|
| **Data engineer** (feeds, PIT correctness, vendor mapping) | P0 | The feeds above need integration + entity resolution (ticker/CIK/ISIN mapping). |
| **ML/RAG engineer** (eval harness, rerank, retrieval tuning) | P1 | Owns the quality loop. |
| **Compliance / data licensing review** | P1 | Market-data + news licensing for a fund has redistribution/usage constraints. |

---

## 4. Specific high-ROI upgrades enabled by the above

1. **Sentiment *time series* → real sentiment lead-lag backtest.** We already log
   `data/sentiment/history.csv` daily, but with **RavenPack-style historical
   sentiment** we can backfill years and test the plan's original hypothesis:
   *does aggregate supplier sentiment lead the customer's return?* (Current backtest
   uses supplier price momentum as a proxy.)
2. **Revenue-weighted edges.** With FactSet/Bloomberg SPLC we get *% of revenue*
   per relationship → weight graph edges and exposure overlay by economic
   magnitude, not just existence (the plan's "hard problem #4").
3. **Hybrid retrieval + reranker** → measurable recall/precision lift on the eval
   set; fewer "context-insufficient" answers.
4. **Full SIP history** → cleaner correlations and a longer, more credible backtest.

---

## 5. Phased plan

**Phase 1 — Measure & harden (P0, ~2–4 wks)**
Build the golden eval set; wire RAGAS + Langfuse; add a reranker (Cohere API);
stand up Qdrant or pgvector. *No new data spend yet — prove the quality loop.*

**Phase 2 — Buy the data that matters (P0/P1)**
License supply-chain relationships + PIT fundamentals; replace LLM-extracted edges
with audited ones (keep LLM extraction as a fallback/augment). Upgrade market data.

**Phase 3 — Scale & automate (P1/P2)**
Move orchestration to Prefect on a cloud VM; add semantic cache; finance-domain
embeddings; institutional news/sentiment + historical backfill for the sentiment
backtest.

---

## 6. Rough budget envelope

| Tier | Monthly | What it buys |
|---|---|---|
| **Lean** | low 4 figures | Reranker API + fundamentals (FMP) + Polygon market data + OSS eval/observability + 1 cloud VM |
| **Serious** | low–mid 5 figures | + institutional news/sentiment (RavenPack) + managed vector DB + finance embeddings |
| **Institutional** | 6 figures+ | + FactSet/Bloomberg supply-chain & fundamentals, full transcripts, enterprise support |

**Recommendation:** start **Lean** *plus* the analyst eval set (Phase 1) — it
de-risks every later purchase by letting us prove each upgrade moves the metrics
before committing to the 6-figure data contracts.

---

## 7. Vendor links — cheapest/free first → most expensive

> Within each row, options are ordered **free → paid**. ✅ = what the pilot uses today.

### Evaluation & observability (start here — all free/OSS)
- RAGAS (OSS) — https://github.com/explodinggradients/ragas
- Arize Phoenix (OSS, local tracing/eval) — https://github.com/Arize-ai/phoenix
- TruLens (OSS) — https://github.com/truera/trulens
- Langfuse (OSS self-host + generous free cloud tier) — https://langfuse.com
- LangSmith (free dev tier → paid) — https://www.langchain.com/langsmith

### Vector database (hybrid search)
- Chroma ✅ (OSS, local) — https://www.trychroma.com
- pgvector (OSS, Postgres) — https://github.com/pgvector/pgvector
- Qdrant (OSS + free 1GB cloud → paid) — https://qdrant.tech
- Weaviate (OSS + free sandbox → paid) — https://weaviate.io
- Pinecone (free starter → paid) — https://www.pinecone.io

### Embeddings
- OpenAI `text-embedding-3-small` ✅ / `-3-large` (cheap) — https://platform.openai.com/docs/guides/embeddings
- Jina embeddings (free tier) — https://jina.ai/embeddings
- Voyage AI incl. `voyage-finance-2` (free tier → paid) — https://www.voyageai.com
- Cohere Embed v3 (free trial → paid) — https://cohere.com/embeddings

### Reranker (biggest retrieval-quality lever after data)
- `bge-reranker-v2-m3` (free OSS model, self-host) — https://huggingface.co/BAAI/bge-reranker-v2-m3
- Jina Reranker (free tier → paid) — https://jina.ai/reranker
- Voyage `rerank-2` (free tier → paid) — https://docs.voyageai.com/docs/reranker
- Cohere Rerank 3 (free trial → paid) — https://cohere.com/rerank

### Market data (prices)
- Alpaca IEX ✅ (free) / SIP upgrade (paid) — https://alpaca.markets/data
- yfinance (free, unofficial) — https://github.com/ranaroussi/yfinance
- Tiingo (free tier → cheap paid) — https://www.tiingo.com
- Polygon.io (free tier → paid full SIP) — https://polygon.io
- Databento (paid, institutional) — https://databento.com

### Fundamentals
- yfinance ✅ (free) — https://github.com/ranaroussi/yfinance
- Alpha Vantage (free tier) — https://www.alphavantage.co
- SimFin (free/cheap) — https://www.simfin.com
- Financial Modeling Prep / FMP (free tier → paid) — https://site.financialmodelingprep.com
- S&P Capital IQ / FactSet (enterprise) — https://www.spglobal.com/marketintelligence · https://www.factset.com

### News & sentiment
- Finnhub ✅ (free tier → paid) — https://finnhub.io
- NewsAPI (free dev tier → paid) — https://newsapi.org
- Marketaux (free tier → paid) — https://www.marketaux.com
- Benzinga (paid) — https://www.benzinga.com/apis
- RavenPack / Bigdata.com (enterprise) — https://www.ravenpack.com

### Supply-chain relationship data (the core asset to license)
- SEC EDGAR ✅ (free, full-text filings) — https://www.sec.gov/edgar/searchedgar/companysearch
- Finnhub peers ✅ (free) — https://finnhub.io/docs/api/company-peers
- S&P Global Panjiva (enterprise) — https://panjiva.com
- FactSet Supply Chain Relationships (enterprise) — https://www.factset.com/marketplace/catalog/product/factset-supply-chain-relationships
- Bloomberg SPLC (enterprise terminal) — https://www.bloomberg.com/professional/products/data/

### Orchestration
- Windows Task Scheduler / cron ✅ (free) — built-in
- Prefect (OSS + free cloud tier → paid) — https://www.prefect.io
- Dagster (OSS + cloud) — https://dagster.io
- Apache Airflow (OSS) — https://airflow.apache.org

### LLM (answer generation)
- Anthropic Claude ✅ (pay-as-you-go) — https://www.anthropic.com/api
- OpenAI (pay-as-you-go) — https://platform.openai.com

**Lean starting cart (all free / free-tier):** RAGAS + Phoenix (eval) · keep Chroma ·
add a reranker · keep OpenAI embeddings · keep Alpaca/Finnhub/yfinance.
**First task: build the eval set** (done — see `eval/golden_questions.json` +
`pipeline eval`).

### Lean progress log (eval-gated)

| Change | recall@8 | MRR | Decision |
|---|---|---|---|
| Baseline (MMR + graph-aware retrieval) | 0.67 | 0.90 | — |
| + cross-encoder rerank (MiniLM, CPU) | 0.62 | 0.62 | **Rejected** — generic CE hurts this corpus |
| + **LLM rerank (Claude listwise)** | **0.76–0.83** | 0.86 | **Shipped** as the default (`RERANKER=llm`) |
| + **graph-metric router** (`graph_qa`) | **0.88** | **0.93** | **Shipped** — closed the ranking-question failure class |

The eval harness paid for itself immediately: it **prevented shipping the CPU
cross-encoder** (which degraded quality) and **justified the LLM reranker** with
numbers. The reranker is pluggable (`config.RERANKER` = `llm` | `cross_encoder` |
`none`). Next reranker upgrade to evaluate: a **finance-tuned cross-encoder on GPU**
(`bge-reranker-v2-m3` / `bge-reranker-large`) or a **hosted reranker API** (Cohere
Rerank 3 / Voyage rerank-2) — faster and cheaper per query than an LLM rerank at
matching quality. Gate the switch on `pipeline eval`.

**Graph-metric routing (shipped).** The eval surfaced a structural gap: *ranking*
questions ("most systemically central?", "most bullish memory makers?") scored
recall **0.00** — vector search can't answer a computation that has no single
supporting chunk. `graph_qa` now detects this class (metric keyword + superlative
cue) and answers it from the structured `graph.json` — ranking on centrality,
degree, sentiment, P/E, market cap, growth or margin, with an optional
sub-industry scope (e.g. "memory and storage" → MU/WDC/STX). Both questions now
score **1.00** and the suite rose to recall@8 **0.88** / MRR **0.93** / hit-rate
**1.0**. Hybrid retrieval done the cheap way: send each question to the layer that
can actually answer it.

**Arithmetic aggregates (shipped).** The router now also answers *"average P/E of
NVIDIA's suppliers"*, *"total market cap of the hyperscalers"*, *"how many
competitors does AMD have"* — composing a **set selector** (relationship
neighborhood `<company>'s suppliers/customers/competitors/peers`, or a
sub-industry/cluster scope) with a **reducer** (`average` / `median` / `total` /
`count`) over any node metric. The value is computed in Python — not by the LLM,
which can't reliably average a list — and handed over as a grounded summary plus
the member rows, so every answer is auditable to the underlying numbers (and null
metrics are dropped, not invented). This is the "graph filter ∘ metric reducer"
composition flagged above.

**Multi-hop set algebra (shipped).** The set selector now walks the relationship
graph outward from a named company (`graph_qa.parse_chain` + `_traverse`):
*"competitors of NVIDIA's suppliers"* = competitors(suppliers(NVDA)),
*"suppliers of Apple's competitors"* = suppliers(competitors(AAPL)). Hops apply
nearest-the-company first, so the surface form is irrelevant. Composed with the
reducer, *"average P/E of the competitors of NVIDIA's suppliers"* returns one
grounded number — graph-database traversal semantics without standing up a graph
database.

**Regression-guarded (shipped).** The golden set now carries a `graph_queries`
section: deterministic guards that assert each router query fires, selects the
*exact* member set (multi-hop/scope), and — for aggregates — computes a value
consistent with an independent reducer (drift-proof vs live market data).
`pipeline eval` reports `graph-queries=10/10` alongside recall@k, so a refactor
that breaks `parse_chain`, the scope buckets, or a reducer fails loudly instead of
silently degrading.

**Threshold/predicate filters (shipped) — the query algebra is complete.** The set
selector now takes a WHERE clause (`graph_qa.parse_predicates`): *"suppliers with
sentiment > 0.5"*, *"competitors with P/E under 20"*, *"names with market cap over
$1T"*, compound `… and …`. The predicate filters the set before listing, ranking,
or aggregating, so the full pipeline is **set selection · multi-hop traversal ·
filter · rank/aggregate** — a small auditable query language over the graph, routed
to from English. Filter clauses are stripped before rank/aggregate metric
detection so the filter metric can't hijack it (caught in testing: *"average P/E …
with sentiment > 0.4"* was averaging sentiment). Guarded in the golden set with a
drift-proof check: the filtered set is recomputed independently from the live base
set + the stated predicate, so `parse_predicates`/`_apply_filters` regressions fail
loudly without hardcoding membership that market data would drift. With this the
"graph filter ∘ metric reducer" track is feature-complete; remaining roadmap work
is the institutional-grade data/observability items above (hosted reranker,
Phoenix tracing).

**FMP fundamentals feed (shipped — first cut of the P0 fundamentals item).** The
quant layer now reads **Financial Modeling Prep** (cleaner TTM ratios/margins from
filings) instead of scraped yfinance, controlled by `FUNDAMENTALS_SOURCE`
(`auto` → FMP when `FMP_API_KEY` is set, else yfinance; or force `fmp`/`yfinance`).
FMP is primary with **automatic yfinance fallback** per ticker, and field *scales*
are normalized to one convention (notably debt/equity, which downstream treats as a
percent) so a mixed run stays internally consistent. Each quant note records its
`data_source`, and `pipeline stats` shows the source split. The FMP→dict mapping is
a pure function (`market._map_fmp`) with unit tests, so it's guarded without
needing a key or network. This is the budget tier of point-in-time fundamentals;
true PIT/restatement-aware data (Capital IQ / FactSet) remains the institutional
upgrade, and full-tape prices (Polygon/Databento) is the matching P1.
