# S&P 500 Financial RAG Vault — Project Plan

## 1. The Idea, In One Paragraph

Build an Obsidian vault where every S&P 500 company gets its own markdown note. Each note holds quant-level financial analysis (valuation, growth, profitability, risk metrics) plus a live sentiment score pulled from recent news via an LLM. Notes are wikilinked to each other based on real supplier/customer relationships, so opening the graph view shows the actual economic web of the index — who sells to whom, and how exposed each company is to the others. A RAG layer sits on top so you can ask natural-language questions across the whole vault ("which companies are most exposed to NVIDIA guidance cuts?") and get answers grounded in the notes themselves.

This is built incrementally — start with ~20-30 stocks in one sector, prove the pipeline end-to-end, then scale node count and relationship depth over time.

---

## 2. Core Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌────────────────────┐
│  Data Ingestion  │ --> │  Processing Layer │ --> │  Obsidian Vault     │
│  (APIs + filings)│     │  (quant + NLP)    │     │  (markdown + links) │
└─────────────────┘     └──────────────────┘     └────────────────────┘
                                                            │
                                                            v
                                                  ┌────────────────────┐
                                                  │  RAG Layer          │
                                                  │  (ChromaDB + Claude)│
                                                  └────────────────────┘
```

Four layers, each independently buildable and testable:

1. **Ingestion** — pull raw data (prices, fundamentals, filings, news)
2. **Processing** — turn raw data into the quant metrics, relationship edges, and sentiment scores
3. **Vault generation** — render everything into markdown notes with proper Obsidian linking syntax
4. **RAG** — index the vault so you can query it conversationally

You already have the RAG muscle memory from Merlin's Apprentice (FastAPI + ChromaDB + OpenAI embeddings) — this reuses that exact pattern, just pointed at a different corpus.

---

## 3. Node Anatomy — What's In Each Stock's Note

Each note is one markdown file, named by ticker (e.g. `AAPL.md`). Suggested structure:

```markdown
---
ticker: AAPL
sector: Technology
industry: Consumer Electronics
market_cap: 3200000000000
last_updated: 2026-06-18
sentiment_score: 0.72
sentiment_label: Bullish
tags: [sp500, technology, mega-cap]
---

# Apple Inc. (AAPL)

## Overview
- **Market Cap:** $3.2T
- **Sector / Industry:** Technology / Consumer Electronics
- **Employees:** 164,000

## Quant Analysis
### Valuation
| Metric | Value | Sector Avg | Percentile |
|---|---|---|---|
| P/E (TTM) | 31.2 | 24.1 | 78th |
| P/S | 8.1 | 5.3 | 85th |
| EV/EBITDA | 22.4 | 17.8 | 80th |

### Growth
- Revenue CAGR (3yr): 6.2%
- EPS CAGR (3yr): 11.4%

### Profitability
- Gross Margin: 46.2%
- Operating Margin: 31.5%
- ROE: 147.9%

### Risk
- Beta: 1.24
- Debt/Equity: 1.8
- Volatility (30d annualized): 22%

## Sentiment Analysis
**Score: 0.72 (Bullish)** — as of 2026-06-18
Generated from last 10 news articles (see [[AAPL_news_log]])
> Summary: Coverage skews positive on services growth and AI feature rollout;
> some concern over China unit sales softness.

## Relationships
### Customers (companies that use AAPL's products)
- [[FOXCONN_PROXY]] *(N/A — private, see supply chain note)*
- Enterprise customers — diffuse, not modeled at company level

### Suppliers (companies AAPL buys from)
- [[QCOM]] — modem chips, ~$X est. annual spend
- [[SWKS]] — RF components
- [[TSM_ADR]] — chip fabrication (if modeled as ADR/public proxy)
- [[CRUS]] — audio chips

### Competitors (optional secondary edge type)
- [[GOOGL]], [[MSFT]], [[SSNLF_proxy]]

### Exposure Web
- **Revenue concentration risk:** ~19% of revenue from Greater China
- **Connected node value:** sum of linked supplier/customer market caps = $X.XT

## Sources
- 10-K filed 2025-11-01: [[AAPL_10K_2025]]
- Latest earnings call: [[AAPL_Q2_2026_call]]
```

**Why this structure works in Obsidian specifically:**
- YAML frontmatter → lets you use Dataview plugin later to build sortable tables across the whole vault (e.g. "show me all nodes with sentiment < 0.4")
- `[[wikilinks]]` → this is what builds the graph view automatically, zero extra tooling needed
- Separate `_news_log` and `_10K` notes → keeps the main node clean and skimmable, full detail one click away

---

## 4. The Relationship Graph (The Hard Part)

This is the most valuable and most labor-intensive part. Three-tier sourcing strategy, cheapest/most reliable first:

### Tier 1: Structured free/cheap data
- **13F-style supply chain datasets** — some free tools expose "named customers/suppliers" extracted from filings (e.g. certain financial data APIs surface this under "key customers" fields)
- **revenue concentration disclosures** — public companies must disclose customers representing >10% of revenue in 10-Ks; this is a goldmine because it's *required* disclosure, not inference
- Start here. It's free, structured, and legally required to be accurate.

### Tier 2: LLM extraction from filings
- Pull 10-K "Customers," "Suppliers," and "Competition" sections (Item 1, Business) via SEC EDGAR (free, full-text search API)
- Feed the relevant section to Claude with a structured extraction prompt: *"Extract every named public company mentioned as a customer, supplier, or competitor. Return JSON: company name, ticker if identifiable, relationship type, confidence."*
- This is where you'll get the bulk of your edges. Budget for false positives — competitors get named differently than suppliers, and not every company in a 10-K is a true supply-chain partner.

### Tier 3: Inference / manual backfill
- For high-profile relationships not explicitly disclosed (e.g. everyone "supplies" TSMC's foundry ecosystem), allow manual edge addition
- Keep a `relationships_manual_overrides.csv` so manual edges are tracked separately from extracted ones — important for auditability later

### Edge metadata to store (not just "linked," but *how*)
```json
{
  "source": "AAPL",
  "target": "QCOM",
  "relation": "supplier",
  "confidence": "high",
  "source_doc": "AAPL_10K_2025",
  "extraction_method": "llm_extraction",
  "estimated_revenue_pct": null
}
```
Store this in a `relationships.json` or in a lightweight SQLite/Postgres table — the markdown vault renders *from* this, it isn't the source of truth for graph structure. Markdown files get regenerated; the relationship database doesn't.

**Practical scoping note:** don't try to extract supplier/customer edges for all 500 tickers on day one. Pick one sector (e.g. semiconductors + their direct customers in tech hardware) as your pilot — it has dense, well-documented relationships and will stress-test your extraction pipeline fast.

---

## 5. Quant Analysis Layer

Pull from a market data API (yfinance is free and fine for v1; upgrade to a paid feed like Polygon or FMP if you hit rate limits or need cleaner fundamentals). Per-node metrics, grouped the way the note template above shows:

| Category | Metrics |
|---|---|
| **Valuation** | P/E, P/S, P/B, EV/EBITDA, PEG |
| **Growth** | Revenue CAGR, EPS CAGR, YoY revenue growth |
| **Profitability** | Gross/Operating/Net margin, ROE, ROIC |
| **Leverage/Risk** | Debt/Equity, Current Ratio, Beta, 30d volatility |
| **Size** | Market cap, Enterprise Value — this is your "node size" for graph viz |

Compute sector percentiles for each metric (you'll want this for relative valuation context — "expensive vs. its own history" is a different signal than "expensive vs. its sector," and the table format above shows both).

This part is straightforward pandas work — closest in spirit to your diabetes risk pipeline or the MLB feature engineering, just swapped onto fundamentals data instead of game logs.

---

## 6. Sentiment Layer

Per node, per refresh cycle:

1. **Pull news** — News API, or Claude with web search tool, scoped to the company name + ticker, last N days
2. **Score via Claude** — structured prompt: *"Given these headlines/snippets, return a sentiment score from -1 to 1 and a 2-sentence summary. Weight financial/operational news higher than generic mentions."*
3. **Store both the score AND the raw articles** — score goes in frontmatter (for Dataview sorting), raw articles go in the linked `_news_log` note (for traceability — you want to be able to click through and see *why* a score moved)
4. **Replicate for supplier/customer nodes too**, as you specified — a chip supplier's sentiment should reflect news about *its* business, not get inherited from the companies it supplies. Each node runs its own independent sentiment pass.

This reuses your Kalshi/Polymarket sentiment-adjacent work (you've already built API auth + structured prediction pipelines) — the new piece here is just the LLM-as-scorer pattern instead of XGBoost-as-classifier.

**Refresh cadence:** sentiment is the most perishable data here. Quant fundamentals update quarterly (on earnings), relationships update annually (on 10-K filing), but sentiment should refresh daily or weekly depending on API budget. Design the pipeline so each layer refreshes on its own schedule — don't couple them.

---

## 7. RAG Layer (Querying the Vault)

Once notes exist, index them exactly like the GoodLeap pipeline:

1. **Chunk** each note (frontmatter + Overview as one chunk, Quant Analysis as another, Sentiment as another, Relationships as another — chunk by section, not by fixed token count, since sections have different semantic density)
2. **Embed** with OpenAI embeddings (or Claude's embedding-compatible flow) into ChromaDB
3. **Metadata filter fields**: ticker, sector, relation-type, sentiment_label — lets you pre-filter before semantic search (e.g. "only search nodes tagged supplier of AAPL")
4. **Query via Claude**, RAG-style: retrieve top-k relevant chunks, feed to Claude with a system prompt that knows about the graph structure, so it can answer relationship-aware questions like *"if NVDA misses earnings, which nodes in this vault have the highest connected exposure?"*

This is a near 1:1 reuse of your Merlin's Apprentice architecture (FastAPI + ChromaDB + OpenAI embeddings + PostgreSQL for relationship metadata). You could legitimately stand up a FastAPI service that (a) regenerates the vault on a schedule and (b) exposes a `/query` endpoint backed by the same RAG pattern.

---

## 8. Suggested Build Order (Incremental, Side-Project Pace)

| Phase | Goal | Output |
|---|---|---|
| **0. Pilot** | 1 sector (~20-30 tickers, e.g. semiconductors) | Hand-validate relationship extraction works before scaling |
| **1. Quant pipeline** | Automate fundamentals → markdown note generation for pilot set | Script: ticker list in, `.md` files out |
| **2. Relationship extraction** | EDGAR pull + Claude extraction prompt + manual override CSV | `relationships.json` for pilot sector |
| **3. Vault wiring** | Render `[[wikilinks]]` from relationships.json into notes | Open in Obsidian, confirm graph view looks right |
| **4. Sentiment layer** | News pull + Claude scoring, written into frontmatter + news log notes | Sentiment updates without touching quant/relationship data |
| **5. RAG indexing** | Chunk + embed pilot vault into ChromaDB | Can ask cross-node questions about the pilot sector |
| **6. Scale out** | Repeat phases 1-5 sector by sector across S&P 500 | Full vault, built incrementally, sector by sector |
| **7. Automation** | Scheduler (cron / Airflow-lite) for refresh cadences per layer | Self-updating vault |

Phases 0-5 on one sector is a genuinely complete, demoable project on its own — full S&P 500 coverage is just "run phases 1-5 again" for each remaining sector once the pipeline is proven.

---

## 9. Tech Stack Summary

| Layer | Tool |
|---|---|
| Market/fundamentals data | yfinance (free, v1) → Polygon/FMP (paid, later) |
| Filings | SEC EDGAR full-text search API (free) |
| Relationship extraction | Claude API, structured JSON extraction prompts |
| News | News API or Claude w/ web search |
| Sentiment scoring | Claude API, structured scoring prompts |
| Relationship storage | SQLite (pilot) → PostgreSQL (scaled) |
| Vault rendering | Python script: data → Jinja2 markdown templates → `.md` files |
| Vault viewer | Obsidian (graph view, Dataview plugin for cross-note tables) |
| RAG embeddings | OpenAI embeddings or Claude-compatible embedding model |
| RAG vector store | ChromaDB |
| RAG query interface | FastAPI endpoint, same shape as Merlin's Apprentice |
| Orchestration (later) | Simple cron jobs per layer; Airflow only if this grows past side-project scale |

---

## 10. Known Hard Problems (Don't Skip These)

- **False positives in relationship extraction.** A 10-K mentioning "competitors include X, Y, Z" is not the same as a supply relationship — your extraction prompt needs to force the model to classify relationship *type*, not just detect co-occurrence.
- **Private suppliers/customers.** Plenty of real relationships involve private companies (Foxconn, many raw material suppliers). You said it must be public-to-public, so these get dropped or represented as a labeled "private — not modeled" stub node, not silently omitted (silent omission makes the graph look more complete than it is).
- **Sentiment score volatility/staleness mismatch.** A node's sentiment can shift daily while its quant fundamentals are quarter-old — make sure the note clearly timestamps *each section independently*, not just one "last updated" field, or the vault will mislead you about what's current.
- **Edge weight = relationship strength, not just existence.** "10% of revenue" and "mentioned once as a minor vendor" shouldn't render as visually identical links. If you want this in Obsidian's graph view, you're limited (Obsidian doesn't natively support weighted-edge visualization) — Dataview + custom CSS can fake it via node size/color, but full weighted-graph rendering may eventually want the custom-app route you said you'd consider down the line.

---

## 11. Where This Could Go (V2+ Ideas, Not Now)

- Web app graph visualization (React + D3/recharts) once the Obsidian version proves the data model — you'd reuse the relationship database directly, just swap the rendering layer
- Backtesting signal: does aggregate supplier-sentiment predict the customer's stock movement with any lead time? (Natural extension of your existing quant trading system work)
- Portfolio overlay: given a held position, surface its full connected exposure web automatically
