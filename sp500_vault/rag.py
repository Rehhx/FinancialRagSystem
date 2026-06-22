"""RAG layer (LangChain + OpenAI embeddings + Chroma) with incremental indexing.

Built for **daily updates**: re-embedding all chunks every day is wasteful when
only the sentiment section of each note changed. So indexing is incremental — we
keep a content-hash manifest and only embed chunks whose text actually changed
(new/updated), delete chunks that disappeared, and skip everything unchanged.
On a typical daily refresh only ~1 chunk per note (the sentiment section) is
re-embedded instead of the whole vault.

Stack:
    chunking      section-based (frontmatter+overview / quant / sentiment / …)
    embeddings    OpenAI (`langchain_openai.OpenAIEmbeddings`)
    vector store  Chroma (`langchain_chroma.Chroma`, persistent)
    answer        Claude (`llm.rag_answer`, Anthropic SDK)
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from . import config, graph_qa, llm, rerank

_COLLECTION = "sp500_vault"
_MANIFEST = config.CHROMA_DIR / "rag_manifest.json"

_vectorstore_cache: Chroma | None = None


def _vectorstore() -> Chroma:
    global _vectorstore_cache
    if _vectorstore_cache is None:
        config.require("OPENAI_API_KEY")
        embeddings = OpenAIEmbeddings(model=config.OPENAI_EMBED_MODEL, api_key=config.OPENAI_API_KEY)
        _vectorstore_cache = Chroma(
            collection_name=_COLLECTION,
            persist_directory=str(config.CHROMA_DIR),
            embedding_function=embeddings,
        )
    return _vectorstore_cache


# ── Markdown -> chunks (section-based; unchanged, unit-tested) ────────────────


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    if not text.startswith("---"):
        return {}, text
    _, fm, body = text.split("---", 2)
    meta: dict[str, str] = {}
    for line in fm.strip().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    return meta, body


def _chunk_note(text: str) -> tuple[dict, list[tuple[str, str]]]:
    """Return (frontmatter, [(section_title, section_text)])."""
    meta, body = _parse_frontmatter(text)
    parts = re.split(r"\n## ", body)
    chunks: list[tuple[str, str]] = []
    head = parts[0].strip()
    if head:
        chunks.append(("Header", head))
    for p in parts[1:]:
        title = p.splitlines()[0].strip()
        chunks.append((title, "## " + p.strip()))
    return meta, chunks


def _news_digest(ticker: str, articles: list[dict], limit: int = 12) -> str:
    """A compact, retrievable digest of a ticker's recent headlines (with URLs so
    answers can cite the original source)."""
    lines = []
    for a in articles[:limit]:
        when = (a.get("datetime") or "")[:10]
        src = a.get("source") or a.get("provider") or ""
        head = (a.get("headline") or "").strip()
        if not head:
            continue
        line = f"- [{when}] {head}" + (f" ({src})" if src else "")
        summ = (a.get("summary") or "").strip()
        if summ:
            line += f" — {summ[:200]}"
        if a.get("url"):
            line += f" SOURCE: {a['url']}"
        lines.append(line)
    return f"{ticker} — Recent News ({len(lines)} headlines)\n" + "\n".join(lines)


def _filings_digest(ticker: str, events: list[dict], limit: int = 10) -> str:
    """A retrievable digest of a ticker's recent 8-K material events (with the SEC
    filing URL so answers can link the source)."""
    lines = []
    for e in events[:limit]:
        labels = "; ".join(it.get("label", "") for it in e.get("items", [])) or "8-K filing"
        line = f"- [{e.get('filing_date')}] 8-K — {labels}"
        if e.get("summary"):
            line += f": {e['summary']}"          # one-line LLM summary of high-signal events
        if e.get("url"):
            line += f" SOURCE: {e['url']}"
        lines.append(line)
    return (f"{ticker} — Recent SEC Filings / Material Events ({len(lines)} 8-Ks)\n"
            + "\n".join(lines))


def _documents() -> tuple[list[Document], list[str]]:
    """Build LangChain Documents (+ stable ids) from the vault notes, plus a
    per-ticker recent-news chunk so the RAG can answer news-cycle questions
    (headlines are fetched for free and only embedded when they change)."""
    from . import filings, sentiment

    docs: list[Document] = []
    ids: list[str] = []
    for path in sorted(config.VAULT_DIR.glob("*.md")):
        if path.stem.endswith("_news_log") or path.stem.startswith("_"):
            continue
        meta, chunks = _chunk_note(path.read_text(encoding="utf-8"))
        ticker = meta.get("ticker", path.stem)
        sector = meta.get("sector", "Unknown")
        sentiment_label = meta.get("sentiment_label", "Neutral")
        seen: dict[str, int] = {}
        for title, body in chunks:
            n = seen.get(title, 0)
            seen[title] = n + 1
            uid = f"{ticker}:{title}" + (f"#{n}" if n else "")
            docs.append(Document(
                page_content=f"{ticker} — {title}\n{body}",
                metadata={"ticker": ticker, "sector": sector,
                          "sentiment_label": sentiment_label, "section": title},
            ))
            ids.append(uid)

        # Recent-news chunk from the stored articles (one per ticker).
        articles = (sentiment.load(ticker) or {}).get("articles") or []
        if articles:
            docs.append(Document(
                page_content=_news_digest(ticker, articles),
                metadata={"ticker": ticker, "sector": sector,
                          "sentiment_label": sentiment_label, "section": "News"},
            ))
            ids.append(f"{ticker}:News")

        # Recent 8-K material-events chunk (one per ticker).
        events = (filings.load(ticker) or {}).get("events") or []
        if events:
            docs.append(Document(
                page_content=_filings_digest(ticker, events),
                metadata={"ticker": ticker, "sector": sector,
                          "sentiment_label": sentiment_label, "section": "Material Events"},
            ))
            ids.append(f"{ticker}:Filings")
    return docs, ids


# ── Incremental indexing ─────────────────────────────────────────────────────


def _load_manifest() -> dict:
    return json.loads(_MANIFEST.read_text(encoding="utf-8")) if _MANIFEST.exists() else {}


def _save_manifest(m: dict) -> None:
    _MANIFEST.write_text(json.dumps(m), encoding="utf-8")


def index_vault(force: bool = False) -> dict:
    """Incrementally sync the vector store with the vault. Returns a change summary."""
    docs, ids = _documents()
    if not ids:
        print("[rag] no notes to index — run the vault layer first")
        return {}

    current = {uid: hashlib.sha256(d.page_content.encode("utf-8")).hexdigest()
               for uid, d in zip(ids, docs)}
    manifest = {} if force else _load_manifest()

    changed = [i for i, uid in enumerate(ids) if manifest.get(uid) != current[uid]]
    removed = [uid for uid in manifest if uid not in current]

    vs = _vectorstore()
    if force:
        # Rebuild from scratch — drop the whole collection.
        try:
            vs.delete_collection()
        except Exception:  # noqa: BLE001
            pass
        global _vectorstore_cache
        _vectorstore_cache = None
        vs = _vectorstore()
        changed = list(range(len(ids)))
        removed = []

    if removed:
        vs.delete(ids=removed)
    if changed:
        cids = [ids[i] for i in changed]
        if not force:
            vs.delete(ids=cids)  # drop old versions before re-adding
        # add_documents embeds via OpenAI in one batched call
        for j in range(0, len(changed), 200):
            batch = changed[j:j + 200]
            vs.add_documents([docs[i] for i in batch], ids=[ids[i] for i in batch])

    _save_manifest(current)
    global _bm25_cache
    _bm25_cache = None      # corpus changed — rebuild the keyword index lazily
    res = {"embedded": len(changed), "deleted": len(removed),
           "unchanged": len(ids) - len(changed), "total": len(ids)}
    print(f"[rag] index: {res['embedded']} embedded, {res['deleted']} removed, "
          f"{res['unchanged']} unchanged ({res['total']} chunks) -> {config.CHROMA_DIR}")
    return res


# ── Query ────────────────────────────────────────────────────────────────────


def _build_where(ticker=None, sector=None, sentiment=None) -> dict | None:
    clauses = []
    if ticker:
        clauses.append({"ticker": ticker.upper()})
    if sector:
        clauses.append({"sector": sector})
    if sentiment:
        clauses.append({"sentiment_label": sentiment})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


# ── Hybrid retrieval: BM25 (sparse) fused with dense vectors ──────────────────
# Dense embeddings match meaning but miss exact tokens (a ticker, "P/E", a product
# name); BM25 catches those. We fuse the two rankings with Reciprocal Rank Fusion
# so a chunk strong in *either* signal surfaces — which lifts recall. Pure-Python,
# no extra dependency; the corpus is small (~350 chunks).

_TOKEN_RE = re.compile(r"[a-z0-9.]+")


def _tok(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


class _BM25:
    """Okapi BM25 over an in-memory corpus, via an inverted index."""

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.N = len(corpus)
        self.dl = [len(d) for d in corpus]
        self.avgdl = (sum(self.dl) / self.N) if self.N else 0.0
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        df: dict[str, int] = defaultdict(int)
        for i, doc in enumerate(corpus):
            for w, f in Counter(doc).items():
                self.postings[w].append((i, f))
                df[w] += 1
        self.idf = {w: math.log(1 + (self.N - d + 0.5) / (d + 0.5)) for w, d in df.items()}

    def top(self, query: list[str], n: int) -> list[int]:
        scores: dict[int, float] = defaultdict(float)
        for w in query:
            idf = self.idf.get(w)
            if idf is None:
                continue
            for i, f in self.postings[w]:
                denom = f + self.k1 * (1 - self.b + self.b * self.dl[i] / (self.avgdl or 1))
                scores[i] += idf * f * (self.k1 + 1) / denom
        return sorted(scores, key=lambda i: scores[i], reverse=True)[:n]


# (count, bm25, texts, metadatas) — rebuilt when the collection size changes.
_bm25_cache: tuple[int, _BM25, list[str], list[dict]] | None = None


def _bm25_corpus():
    global _bm25_cache
    vs = _vectorstore()
    try:
        n = vs._collection.count()
    except Exception:  # noqa: BLE001
        n = -1
    if _bm25_cache is not None and _bm25_cache[0] == n:
        return _bm25_cache[1], _bm25_cache[2], _bm25_cache[3]
    got = vs.get(include=["documents", "metadatas"])
    texts = got.get("documents") or []
    metas = got.get("metadatas") or []
    bm25 = _BM25([_tok(t) for t in texts])
    _bm25_cache = (n, bm25, texts, metas)
    return bm25, texts, metas


def _bm25_docs(question: str, n: int) -> list[Document]:
    bm25, texts, metas = _bm25_corpus()
    out = []
    for i in bm25.top(_tok(question), n):
        out.append(Document(page_content=texts[i], metadata=metas[i] if i < len(metas) else {}))
    return out


def _doc_key(d: Document):
    return (d.metadata.get("ticker"), d.metadata.get("section"))


def _rrf_fuse(dense: list[Document], sparse: list[Document], c: int = 60) -> list[Document]:
    """Reciprocal Rank Fusion of two ranked document lists (keyed by ticker+section)."""
    pool: dict = {}
    for d in (*dense, *sparse):
        pool.setdefault(_doc_key(d), d)
    score: dict = defaultdict(float)
    for ranking in (dense, sparse):
        for rank, d in enumerate(ranking):
            score[_doc_key(d)] += 1.0 / (c + rank + 1)
    return [pool[k] for k in sorted(score, key=lambda k: score[k], reverse=True)]


def _graph_expand(question: str) -> set[str]:
    """Tickers named in the question, expanded to their modeled neighbors.

    This is what makes the RAG relationship-aware: ask about NVIDIA and we also
    pull its suppliers'/customers'/competitors' notes into context, so the model
    can reason about connected exposure instead of only NVIDIA's own chunk.
    """
    from .relationships import get_edges
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
    expanded = set(found)
    for t in found:
        for e in get_edges(t, resolved_only=True):
            if e.get("target_ticker"):
                expanded.add(e["target_ticker"])
    return expanded


def retrieve(question: str, k: int = 6, ticker=None, sector=None, sentiment=None) -> list:
    """Retrieve context docs (no answer LLM) — used by query() and the eval harness.

    Routing: a *ranking* question ("most central", "most bullish sentiment",
    "cheapest by P/E") is answered from the structured graph, not vector search —
    no single chunk states "NVDA is the most central", so we compute it over every
    node's metric (``graph_qa``). Everything else fetches a wide candidate pool
    (MMR + graph-aware expansion), then reranks and keeps the top-k.
    """
    # 0) Graph-metric router — only for open questions (an explicit metadata
    #    filter signals a targeted lookup, not a universe-wide ranking).
    if not (ticker or sector or sentiment):
        gdocs = graph_qa.as_documents(question, k)
        if gdocs is not None:
            return gdocs

    vs = _vectorstore()
    where = _build_where(ticker, sector, sentiment)
    reranking = config.RERANKER != "none"
    pool_k = max(24, k * 4) if reranking else k

    # 1) Diversified semantic retrieval — MMR drops near-duplicate chunks.
    try:
        docs = vs.max_marginal_relevance_search(question, k=pool_k, fetch_k=max(pool_k * 2, 40), filter=where)
    except Exception:  # noqa: BLE001 - fall back if MMR unavailable
        docs = vs.similarity_search(question, k=pool_k, filter=where)

    # 1b) Hybrid: add BM25 (sparse) candidates the dense vectors missed, so
    #     exact-token matches (a ticker, "P/E", a product name) still reach the
    #     reranker. Union (dense-priority) — never displaces a dense candidate
    #     before reranking. Open questions only (a metadata filter already narrows).
    if config.HYBRID_RETRIEVAL and not where:
        have = {_doc_key(d) for d in docs}
        docs = docs + [d for d in _bm25_docs(question, pool_k) if _doc_key(d) not in have]
    seen = {(d.metadata.get("ticker"), d.metadata.get("section")) for d in docs}

    # 2) Graph-aware expansion — pull chunks for named tickers + their neighbors
    #    (skipped when the caller already pinned a metadata filter).
    if not where:
        focus = _graph_expand(question)
        if focus:
            present = {d.metadata.get("ticker") for d in docs}
            missing = [t for t in focus if t not in present]
            if missing:
                extra = vs.similarity_search(
                    question, k=2 * len(missing) + 4, filter={"ticker": {"$in": missing}})
                covered: set[str] = set()
                for d in extra:                 # one best chunk per missing ticker -> max coverage
                    t = d.metadata.get("ticker")
                    if t in covered:
                        continue
                    key = (t, d.metadata.get("section"))
                    if key not in seen:
                        seen.add(key)
                        covered.add(t)
                        docs.append(d)

    # 3) Rerank the candidate pool and keep the top-k (cross-encoder >> vector order).
    if reranking:
        docs = rerank.rerank(question, docs, k)
    return docs[:k]


def _contexts(docs) -> list[str]:
    out = []
    for d in docs:
        t, s = d.metadata.get("ticker"), d.metadata.get("section")
        tag = f"{t} · {s}" if t else s   # synthetic summary rows have no ticker
        out.append(f"[{tag}]\n{d.page_content}")
    return out


def query(question: str, k: int = 6, ticker=None, sector=None, sentiment=None,
          history: list[dict] | None = None) -> dict:
    # Follow-ups: rewrite to a standalone question so retrieval works on the
    # resolved intent ("what about its suppliers?" -> "Who supplies NVIDIA?").
    search_q = llm.condense_question(history, question) if history else question
    docs = retrieve(search_q, k, ticker, sector, sentiment)
    if not docs:
        return {"answer": "No indexed context matched that query.", "sources": []}
    answer = llm.rag_answer(question, _contexts(docs), history=history)
    # Skip synthetic context with no ticker (e.g. the aggregate summary row).
    sources = [{"ticker": d.metadata.get("ticker"), "section": d.metadata.get("section")}
               for d in docs if d.metadata.get("ticker")]
    out = {"answer": answer, "sources": sources}
    if history and search_q != question:
        out["resolved_question"] = search_q   # surfaced so the UI can show what was searched
    return out


def count() -> int:
    try:
        return _vectorstore()._collection.count()
    except Exception:  # noqa: BLE001
        return 0
