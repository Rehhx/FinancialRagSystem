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
import re

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
    """A compact, retrievable digest of a ticker's recent headlines."""
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
        lines.append(line)
    return f"{ticker} — Recent News ({len(lines)} headlines)\n" + "\n".join(lines)


def _documents() -> tuple[list[Document], list[str]]:
    """Build LangChain Documents (+ stable ids) from the vault notes, plus a
    per-ticker recent-news chunk so the RAG can answer news-cycle questions
    (headlines are fetched for free and only embedded when they change)."""
    from . import sentiment

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


def query(question: str, k: int = 6, ticker=None, sector=None, sentiment=None) -> dict:
    docs = retrieve(question, k, ticker, sector, sentiment)
    if not docs:
        return {"answer": "No indexed context matched that query.", "sources": []}
    answer = llm.rag_answer(question, _contexts(docs))
    # Skip synthetic context with no ticker (e.g. the aggregate summary row).
    sources = [{"ticker": d.metadata.get("ticker"), "section": d.metadata.get("section")}
               for d in docs if d.metadata.get("ticker")]
    return {"answer": answer, "sources": sources}


def count() -> int:
    try:
        return _vectorstore()._collection.count()
    except Exception:  # noqa: BLE001
        return 0
