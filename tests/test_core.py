"""Unit tests for the pure logic in the pipeline.

These deliberately avoid network and the data caches — they exercise the
deterministic functions (formatting, parsing, graph math, resolution) so a
refactor that breaks them fails fast. Run with: pytest -q
"""
from __future__ import annotations

import math

from sp500_vault import backtest, graph_export, graph_qa, rag, relationships, signals, vault_render
from sp500_vault.universe import resolve_ticker, tickers_for_group


# ── universe resolution ──────────────────────────────────────────────────────

def test_resolve_ticker_exact_and_name():
    assert resolve_ticker("NVDA") == "NVDA"
    assert resolve_ticker("nvda") == "NVDA"
    assert resolve_ticker("Alphabet Inc.") == "GOOGL"
    assert resolve_ticker("Amazon, Inc.") == "AMZN"


def test_resolve_ticker_misses():
    assert resolve_ticker(None) is None
    assert resolve_ticker("") is None
    assert resolve_ticker("Definitely Not A Public Co") is None


def test_tickers_for_group():
    assert "TSLA" in tickers_for_group("automotive")
    assert "NVDA" in tickers_for_group("semiconductor")
    assert tickers_for_group("nope") == []


# ── number formatting ────────────────────────────────────────────────────────

def test_money():
    assert vault_render._money(5.1e12) == "$5.10T"
    assert vault_render._money(2.5e9) == "$2.50B"      # always 2 decimals
    assert vault_render._money(None) == "N/A"
    assert vault_render._money(float("nan")) == "N/A"


def test_pct_ratio_de_int():
    assert vault_render._pct(0.462) == "46.2%"
    assert vault_render._ratio(1.234) == "1.23"
    assert vault_render._de(180.0) == "1.80"          # yfinance percent -> ratio
    assert vault_render._int(42000.0) == "42,000"
    assert vault_render._int(float("nan")) == "N/A"


def test_ordinal():
    assert vault_render._ordinal(1) == "1st"
    assert vault_render._ordinal(2) == "2nd"
    assert vault_render._ordinal(11) == "11th"
    assert vault_render._ordinal(22) == "22nd"
    assert vault_render._ordinal(None) == "—"


def test_num_guards_nan_inf():
    assert vault_render._num(float("nan")) is None
    assert vault_render._num(float("inf")) is None
    assert vault_render._num("x") is None
    assert vault_render._num("3.5") == 3.5


# ── FMP fundamentals mapping ─────────────────────────────────────────────────

def test_fmp_num_coercions():
    from sp500_vault.data_sources import market
    assert market._fnum("12.5") == 12.5
    assert market._fnum(None) is None
    assert market._fnum("n/a") is None
    assert market._fnum(float("nan")) is None
    assert market._iint("29600") == 29600


def test_map_fmp_fields_and_scales():
    from sp500_vault.data_sources import market
    # FMP "stable" API field names (symbol= query param; PE/PS/PB/PEG/margins/DE in
    # ratios-ttm, ROE/ROA/EV-EBITDA in key-metrics-ttm).
    profile = [{"companyName": "NVIDIA Corporation", "sector": "Technology",
                "industry": "Semiconductors", "marketCap": 3.0e12,
                "fullTimeEmployees": "29600", "beta": 1.7}]
    km = [{"evToEBITDATTM": 55.0, "enterpriseValueTTM": 3.1e12,
           "currentRatioTTM": 4.2, "returnOnEquityTTM": 0.91,
           "returnOnAssetsTTM": 0.45, "marketCap": 3.0e12}]
    ratios = [{"priceToEarningsRatioTTM": 60.0, "priceToSalesRatioTTM": 35.0,
               "priceToBookRatioTTM": 50.0, "priceToEarningsGrowthRatioTTM": 1.2,
               "grossProfitMarginTTM": 0.73, "operatingProfitMarginTTM": 0.54,
               "netProfitMarginTTM": 0.49, "debtToEquityRatioTTM": 0.45}]
    income = [{"revenue": 60.9e9, "epsDiluted": 11.9},   # newest first
              {"revenue": 26.9e9, "epsDiluted": 3.3},
              {"revenue": 27.0e9, "epsDiluted": 3.85},
              {"revenue": 16.7e9, "epsDiluted": 6.9}]
    d = market._map_fmp("NVDA", profile, km, ratios, income, None)
    assert d["data_source"] == "fmp"
    assert d["name"] == "NVIDIA Corporation" and d["employees"] == 29600
    assert d["market_cap"] == 3.0e12 and d["pe_ttm"] == 60.0
    assert d["gross_margin"] == 0.73 and d["ev_ebitda"] == 55.0
    # FMP D/E ratio 0.45 -> percent 45.0 (yfinance convention `_de` then shows 0.45x)
    assert d["debt_to_equity"] == 45.0
    assert vault_render._de(d["debt_to_equity"]) == "0.45"
    # YoY uses the two latest years: (60.9-26.9)/26.9
    assert abs(d["revenue_growth_yoy"] - (60.9e9 - 26.9e9) / 26.9e9) < 1e-9
    assert d["revenue_cagr_3yr"] is not None and d["revenue_cagr_3yr"] > 0


def test_map_fmp_empty_profile():
    from sp500_vault.data_sources import market
    assert market._map_fmp("X", [], [], [], [], None) == {}


def test_fmp_fundamentals_no_key(monkeypatch):
    from sp500_vault import config
    from sp500_vault.data_sources import market
    monkeypatch.setattr(config, "FMP_API_KEY", "")
    assert market._fmp_get("profile", "NVDA") is None
    assert market._fmp_fundamentals("NVDA") == {}   # no key -> no network -> {}


# ── news: RSS parsing, interleave, RAG digest ────────────────────────────────

_RSS = """<?xml version="1.0"?>
<rss version="2.0"><channel>
<item><title>NVDA soars on AI demand</title><link>http://x/1</link>
<pubDate>Fri, 20 Jun 2026 13:00:00 GMT</pubDate>
<description>&lt;p&gt;Big &lt;b&gt;day&lt;/b&gt;&lt;/p&gt;</description>
<source url="http://m">Morningstar</source></item>
<item><title>Chip demand rises</title><link>http://x/2</link>
<pubDate>Thu, 19 Jun 2026 09:00:00 GMT</pubDate><description>demand up</description></item>
</channel></rss>"""


def test_strip_html():
    from sp500_vault.data_sources import news
    assert news._strip_html("<p>hi <b>there</b></p>") == "hi there"


def test_parse_rss():
    from sp500_vault.data_sources import news
    items = news._parse_rss(_RSS, "googlenews", 10)
    assert len(items) == 2
    assert items[0]["headline"] == "NVDA soars on AI demand"
    assert items[0]["source"] == "Morningstar"
    assert items[0]["datetime"].startswith("2026-06-20")
    assert items[0]["summary"] == "Big day"        # HTML entities + tags stripped
    assert items[0]["provider"] == "googlenews"


def test_interleave_round_robin():
    from sp500_vault.data_sources import news
    a = [{"headline": "a1"}, {"headline": "a2"}]
    b = [{"headline": "b1"}]
    merged = news._interleave([a, b])
    assert [m["headline"] for m in merged] == ["a1", "b1", "a2"]   # priority-first, then round-robin


def test_news_digest():
    arts = [{"datetime": "2026-06-20T10:00:00", "headline": "NVDA up", "source": "Barrons",
             "summary": "rally"},
            {"datetime": "2026-06-19T08:00:00", "headline": "Guidance raised", "provider": "yahoo"}]
    d = rag._news_digest("NVDA", arts)
    assert "Recent News" in d and "NVDA up" in d and "2026-06-20" in d
    assert "(Barrons)" in d and "rally" in d


# ── 8-K material events (EDGAR submissions parsing) ──────────────────────────

_RECENT = {
    "form": ["8-K", "10-Q", "8-K", "8-K"],
    "filingDate": ["2026-06-18", "2026-06-01", "2026-05-20", "2020-01-01"],
    "reportDate": ["2026-06-18", "", "2026-05-19", ""],
    "accessionNumber": ["0001-26-1", "x", "0001-26-3", "old"],
    "items": ["8.01,9.01", "", "2.02,9.01", "5.02"],
    "primaryDocDescription": ["8-K", "10-Q", "8-K", "8-K"],
}


def test_item_label():
    from sp500_vault.data_sources import edgar
    assert "earnings" in edgar._item_label("2.02").lower()
    assert edgar._item_label("5.02") == "Departure / Appointment of Directors or Officers"
    assert edgar._item_label("9.99") == "Item 9.99"        # unknown -> graceful fallback


def test_parse_8k_filings():
    import datetime as _dt
    from sp500_vault.data_sources import edgar
    today = _dt.date(2026, 6, 20)
    events = edgar._parse_8k_filings(_RECENT, lookback_days=90, limit=8, today=today)
    assert len(events) == 2                                  # 10-Q skipped, 2020 8-K out of window
    assert events[0]["filing_date"] == "2026-06-18"
    # 9.01 (exhibits) dropped as administrative; 8.01 kept
    assert [it["code"] for it in events[0]["items"]] == ["8.01"]
    assert events[1]["items"][0]["code"] == "2.02"


def test_parse_8k_filings_limit():
    import datetime as _dt
    from sp500_vault.data_sources import edgar
    events = edgar._parse_8k_filings(_RECENT, lookback_days=90, limit=1, today=_dt.date(2026, 6, 20))
    assert len(events) == 1


# ── 8-K body-text LLM summaries ──────────────────────────────────────────────

def test_parse_8k_captures_primary_doc():
    import datetime as _dt
    from sp500_vault.data_sources import edgar
    recent = dict(_RECENT, primaryDocument=["nvda-8k.htm", "q.htm", "b.htm", "old.htm"])
    events = edgar._parse_8k_filings(recent, lookback_days=90, limit=8, today=_dt.date(2026, 6, 20))
    assert events[0]["primary_doc"] == "nvda-8k.htm"      # carried through for the body fetch


def test_extract_8k_body_slices_to_first_item():
    from sp500_vault.data_sources import edgar
    raw = ("UNITED STATES SECURITIES AND EXCHANGE COMMISSION   FORM 8-K   "
           "Registrant address boilerplate here.   Item 5.02 Departure of Directors. "
           "The CFO resigned effective today.")
    body = edgar._extract_8k_body(raw, max_chars=500)
    assert body.startswith("Item 5.02")                  # cover-page boilerplate dropped
    assert "CFO resigned" in body


def test_extract_8k_body_caps_length():
    from sp500_vault.data_sources import edgar
    body = edgar._extract_8k_body("Item 1.01 " + "x " * 5000, max_chars=100)
    assert len(body) == 100


def test_high_signal_items_and_filter():
    from sp500_vault import config, filings
    wanted = config.high_signal_items()
    assert {"1.01", "2.01", "5.02"} <= wanted and "2.02" not in wanted and "9.01" not in wanted
    exec_change = {"items": [{"code": "5.02", "label": "x"}]}
    earnings = {"items": [{"code": "2.02", "label": "x"}]}
    assert filings._is_high_signal(exec_change, wanted)
    assert not filings._is_high_signal(earnings, wanted)


def test_attach_summaries_uses_cache_without_llm(monkeypatch):
    """A cached accession is reused verbatim — no body fetch, no LLM call."""
    from sp500_vault import filings
    from sp500_vault.data_sources import edgar

    def _boom(*a, **k):
        raise AssertionError("should not fetch/summarize a cached filing")
    monkeypatch.setattr(edgar, "fetch_8k_body", _boom)

    events = [{"accession": "0001-26-9", "doc_url": "http://x/8k.htm",
               "items": [{"code": "5.02", "label": "Departure"}]}]
    cache = {"0001-26-9": {"summary": "The CEO stepped down."}}
    added = filings._attach_summaries("NVDA", events, cache)
    assert added == 0 and events[0]["summary"] == "The CEO stepped down."


_ATOM = """<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
<entry>
<title>8-K - NVIDIA CORP (0001045810) (Filer)</title>
<summary type="html"> &lt;b&gt;AccNo:&lt;/b&gt; 0001045810-26-000099 &lt;br&gt;Item 2.02: Results &lt;br&gt;Item 9.01: Exhibits</summary>
<updated>2026-06-21T16:31:00-04:00</updated>
<id>urn:tag:sec.gov,2008:accession-number=0001045810-26-000099</id>
</entry>
<entry>
<title>8-K - SOME RANDOM CO (0000999999) (Filer)</title>
<summary type="html">Item 8.01: Other Events</summary>
<updated>2026-06-21T16:30:00-04:00</updated>
<id>urn:tag:sec.gov,2008:accession-number=0000999999-26-000001</id>
</entry>
</feed>"""


def test_edgar_live_parse_entries():
    from sp500_vault import edgar_live
    entries = edgar_live._parse_entries(_ATOM)
    assert len(entries) == 2
    assert entries[0]["cik"] == "1045810"                  # leading zeros stripped
    assert entries[0]["accession"] == "0001045810-26-000099"
    assert entries[1]["cik"] == "999999"


def test_edgar_live_summary_items_drops_exhibits():
    from sp500_vault import edgar_live
    label = edgar_live._summary_items("Item 2.02: Results <br>Item 9.01: Exhibits")
    assert "earnings" in label.lower() and "9.01" not in label   # 9.01 dropped as administrative


# ── archive + event-driven backtest ──────────────────────────────────────────

def test_event_forward_return():
    import pandas as pd
    from sp500_vault import event_backtest as eb
    idx = pd.date_range("2026-06-01", periods=10, freq="D")
    close = pd.Series([100.0, 101, 102, 103, 104, 105, 106, 107, 108, 109], index=idx)
    # event on 2026-06-02 (pos 1, p0=101); +3 trading days -> pos 4, p1=104
    assert abs(eb._forward_return(close, "2026-06-02", 3) - (104 / 101 - 1)) < 1e-9
    # not enough room past the end -> None
    assert eb._forward_return(close, "2026-06-09", 3) is None


def test_sentiment_rank_ic():
    from sp500_vault import sentiment_backtest as sb
    # perfectly monotonic -> IC = +1; reversed -> -1; too few obs -> None
    assert abs(sb._rank_ic([1, 2, 3, 4, 5], [10, 20, 30, 40, 50]) - 1.0) < 1e-9
    assert abs(sb._rank_ic([1, 2, 3, 4, 5], [50, 40, 30, 20, 10]) + 1.0) < 1e-9
    assert sb._rank_ic([1, 2], [1, 2]) is None
    assert sb._rank_ic([1, 1, 1], [1, 2, 3]) is None        # no variance in x -> None


def test_normalize_history_panel():
    import pandas as pd
    from sp500_vault import sentiment_backtest as sb
    raw = pd.DataFrame({
        "ticker": ["NVDA", "AMD", "BAD"],
        "date": ["2026-06-20", "2026-06-21", "not-a-date"],
        "score": ["0.35", "-0.4", "0.1"],
        "label": ["Bullish", "Bearish", "Neutral"],
    })
    panel = sb._normalize_history(raw)
    assert list(panel.columns) == ["ticker", "date", "sentiment", "volume"]
    assert len(panel) == 2                                   # unparseable date dropped
    assert panel.iloc[0]["sentiment"] == 0.35 and (panel["volume"] == 1.0).all()


def test_archive_news_key_dedup():
    from sp500_vault import archive
    a = {"url": "https://x.com/a?utm=1", "headline": "NVDA up"}
    b = {"url": "https://x.com/a/", "headline": "totally different"}
    assert archive._news_key(a) == archive._news_key(b)        # query + trailing slash stripped
    assert archive._news_key({"headline": "Same H"}) == archive._news_key({"headline": "same h"})


def test_archive_append_filings_dedups(tmp_path, monkeypatch):
    from sp500_vault import archive
    monkeypatch.setattr(archive, "ARCHIVE_DIR", tmp_path)
    monkeypatch.setattr(archive, "_FILINGS", tmp_path / "filings.csv")
    ev = [{"accession": "A-1", "filing_date": "2026-06-01", "items": [{"code": "2.02"}]}]
    assert archive.append_filings("NVDA", ev) == 1
    assert archive.append_filings("NVDA", ev) == 0             # same accession -> deduped
    assert archive.append_filings("NVDA", [{"accession": "A-2", "filing_date": "2026-06-02",
                                            "items": [{"code": "5.02"}]}]) == 1
    df = archive.load_filings()
    assert len(df) == 2 and set(df["accession"]) == {"A-1", "A-2"}


def test_filings_digest():
    events = [{"filing_date": "2026-06-18", "url": "http://sec/1",
               "items": [{"code": "2.02", "label": "Results of Operations (earnings)"}]},
              {"filing_date": "2026-05-08", "items": [{"code": "5.02", "label": "Departure of Officers"}]}]
    d = rag._filings_digest("NVDA", events)
    assert "Material Events" in d and "2026-06-18" in d
    assert "earnings" in d and "Departure" in d
    assert "SOURCE: http://sec/1" in d                  # citation URL embedded


def test_news_digest_includes_source_url():
    arts = [{"datetime": "2026-06-20T10:00:00", "headline": "NVDA up", "source": "Barrons",
             "url": "http://news/1", "summary": "rally"}]
    assert "SOURCE: http://news/1" in rag._news_digest("NVDA", arts)


# ── hybrid retrieval primitives (BM25 + RRF) ─────────────────────────────────

def test_bm25_ranks_keyword_match():
    corpus = [rag._tok(t) for t in [
        "NVIDIA supplies GPUs and AI accelerators",
        "Apple designs iPhones and Macs",
        "Micron makes memory chips used by NVIDIA"]]
    bm = rag._BM25(corpus)
    assert bm.top(rag._tok("memory chips"), 1)[0] == 2     # only doc 2 has 'memory chips'


def test_rrf_fuse_blends_and_dedupes():
    from langchain_core.documents import Document
    d1 = Document(page_content="a", metadata={"ticker": "A", "section": "x"})
    d2 = Document(page_content="b", metadata={"ticker": "B", "section": "x"})
    d3 = Document(page_content="c", metadata={"ticker": "C", "section": "x"})
    fused = rag._rrf_fuse([d1, d2], [d3, d1])              # d1 in both lists -> ranks first
    assert fused[0].metadata["ticker"] == "A"
    assert {d.metadata["ticker"] for d in fused} == {"A", "B", "C"}


def test_condense_no_history_is_identity():
    from sp500_vault import llm
    assert llm.condense_question(None, "Who supplies NVIDIA?") == "Who supplies NVIDIA?"
    assert llm.condense_question([], "x") == "x"           # empty history -> no LLM call


# ── graph math ───────────────────────────────────────────────────────────────

def test_clean_sanitizes_nan():
    dirty = {"a": float("nan"), "b": [1.0, float("inf")], "c": "x", "d": 2}
    assert graph_export._clean(dirty) == {"a": None, "b": [1.0, None], "c": "x", "d": 2}


def test_pagerank_hub_scores_highest():
    nodes = ["A", "B", "C"]
    edges = [
        {"source": "A", "target": "B", "corr": 1.0},
        {"source": "B", "target": "C", "corr": 1.0},
    ]
    pr = graph_export._pagerank(nodes, edges)
    assert math.isclose(sum(pr.values()), 1.0, abs_tol=1e-3)   # values rounded to 5dp
    assert pr["B"] > pr["A"] and pr["B"] > pr["C"]   # the hub is most central


def test_pagerank_empty():
    assert graph_export._pagerank([], []) == {}


# ── signals + relationships helpers ──────────────────────────────────────────

def test_pair_key_is_undirected():
    assert signals._pair_key("B", "A") == "A|B"
    assert signals._pair_key("A", "B") == "A|B"


def test_make_edge_shape():
    e = relationships._make_edge("AAPL", "Qualcomm", "QCOM", "supplier", "high", "manual_override")
    assert e["source"] == "AAPL"
    assert e["target_ticker"] == "QCOM"
    assert e["target_key"] == "QCOM"
    assert e["relation"] == "supplier"
    # Unresolved target falls back to name-based key.
    e2 = relationships._make_edge("AAPL", "Foxconn", None, "supplier", "high", "llm_extraction")
    assert e2["target_ticker"] is None
    assert e2["target_key"] == "FOXCONN"


# ── markdown parsing (RAG chunking) ──────────────────────────────────────────

NOTE = """---
ticker: NVDA
sector: Technology
sentiment_label: Neutral
---

# NVIDIA (NVDA)

## Overview
- Market cap: big

## Relationships
- [[MU]]
"""


def test_parse_frontmatter():
    meta, body = rag._parse_frontmatter(NOTE)
    assert meta["ticker"] == "NVDA"
    assert meta["sector"] == "Technology"
    assert "# NVIDIA" in body


def test_chunk_note_sections():
    meta, chunks = rag._chunk_note(NOTE)
    titles = [t for t, _ in chunks]
    assert titles[0] == "Header"          # pre-heading block
    assert "Overview" in titles
    assert "Relationships" in titles
    # No duplicate section titles would collide on ids
    assert len(titles) == len(set(titles))


# ── backtest math ────────────────────────────────────────────────────────────

def test_backtest_ic_perfect_correlation():
    import numpy as np
    import pandas as pd

    idx = pd.date_range("2020-01-01", periods=50)
    sig = pd.DataFrame({"A": np.arange(50.0)}, index=idx)
    tgt = pd.DataFrame({"A": np.arange(50.0) * 2 + 1.0}, index=idx)   # perfectly correlated
    ic, n = backtest._ic(sig, tgt)
    assert n == 50 and ic is not None and ic > 0.99


def test_backtest_ic_too_few_obs():
    import pandas as pd

    idx = pd.date_range("2020-01-01", periods=5)
    sig = pd.DataFrame({"A": [1.0, 2, 3, 4, 5]}, index=idx)
    ic, n = backtest._ic(sig, sig)
    assert ic is None and n < 30


def test_backtest_sparkline():
    assert len(backtest._sparkline([1, 2, 3, 4, 5, 6, 7, 8])) == 8
    assert backtest._sparkline([]) == ""


# ── graph-metric routing ─────────────────────────────────────────────────────

# Minimal synthetic graph nodes (the fields graph_qa ranks/scopes on).
_NODES = [
    {"id": "NVDA", "label": "NVIDIA", "group": "semiconductor", "sector": "Technology",
     "market_cap": 3.0e12, "sentiment_score": 0.55, "sentiment_label": "Bullish",
     "degree": 23, "centrality": 0.045, "metrics": {"pe_ttm": 60.0}},
    {"id": "QCOM", "label": "Qualcomm", "group": "semiconductor", "sector": "Technology",
     "market_cap": 1.8e11, "sentiment_score": 0.20, "sentiment_label": "Neutral",
     "degree": 12, "centrality": 0.044, "metrics": {"pe_ttm": 18.0}},
    {"id": "MU", "label": "Micron", "group": "semiconductor", "sector": "Technology",
     "market_cap": 1.2e11, "sentiment_score": 0.72, "sentiment_label": "Bullish",
     "degree": 6, "centrality": 0.020, "metrics": {"pe_ttm": 14.0}},
    {"id": "WDC", "label": "Western Digital", "group": "hardware", "sector": "Technology",
     "market_cap": 2.0e10, "sentiment_score": 0.70, "sentiment_label": "Bullish",
     "degree": 4, "centrality": 0.010, "metrics": {"pe_ttm": 12.0}},
    {"id": "STX", "label": "Seagate", "group": "hardware", "sector": "Technology",
     "market_cap": 1.8e10, "sentiment_score": 0.68, "sentiment_label": "Bullish",
     "degree": 3, "centrality": 0.009, "metrics": {"pe_ttm": 11.0}},
    {"id": "NOW", "label": "ServiceNow", "group": "cloud_software", "sector": "Technology",
     "market_cap": 1.6e11, "sentiment_score": 0.69, "sentiment_label": "Bullish",
     "degree": 2, "centrality": 0.008, "metrics": {"pe_ttm": 90.0}},
]


def test_detect_centrality_and_sentiment():
    assert graph_qa.detect("Which company is the most systemically central?")["key"] == "centrality"
    assert graph_qa.detect("Which makers have the most bullish sentiment?")["key"] == "sentiment"
    # cheapest -> P/E, ascending (low is "cheap")
    pe = graph_qa.detect("Cheapest names by P/E?")
    assert pe["key"] == "pe" and pe["ascending"] is True


def test_detect_requires_rank_cue_and_ignores_relationships():
    # Mentions sentiment but isn't a ranking question -> no route.
    assert graph_qa.detect("What is NVDA's sentiment?") is None
    # Pure relationship questions never route to the graph-metric path.
    assert graph_qa.detect("Which companies supply memory chips to NVIDIA?") is None
    assert graph_qa.detect("Who are NVIDIA's main competitors?") is None


def test_sentiment_flip_to_bearish_is_ascending():
    assert graph_qa.detect("Which company has the worst sentiment?")["ascending"] is True


def test_rank_centrality_top_is_nvda():
    spec, ranked = graph_qa.rank(_NODES, "most systemically central company?", k=3)
    assert spec["key"] == "centrality"
    assert ranked[0][0]["id"] == "NVDA"        # highest centrality first


def test_rank_sentiment_scoped_to_memory_storage():
    # "memory and storage" scopes to MU/WDC/STX even though NOW/NVDA are bullish too.
    spec, ranked = graph_qa.rank(_NODES, "memory and storage makers with the most bullish sentiment?", k=8)
    ids = {n["id"] for n, _ in ranked}
    assert ids == {"MU", "WDC", "STX"}
    assert ranked[0][0]["id"] == "MU"          # 0.72 ranks first


def test_rank_returns_none_for_non_metric():
    assert graph_qa.rank(_NODES, "Who supplies NVIDIA?", k=5) is None


# ── arithmetic aggregates ────────────────────────────────────────────────────

def test_detect_aggregate_average_and_total():
    a = graph_qa.detect_aggregate("What is the average P/E of semiconductor companies?")
    assert a["agg"] == "average" and a["metric"]["key"] == "pe"
    t = graph_qa.detect_aggregate("Total market cap of the hyperscalers?")
    assert t["agg"] == "total" and t["metric"]["key"] == "market_cap"
    c = graph_qa.detect_aggregate("How many storage makers are there?")
    assert c["agg"] == "count" and c["metric"] is None


def test_detect_aggregate_misses():
    # Ranking question, not an aggregate.
    assert graph_qa.detect_aggregate("Which company has the highest P/E?") is None
    # Aggregate cue but no recognizable metric.
    assert graph_qa.detect_aggregate("What is the average vibe?") is None


def test_aggregate_average_pe_scoped():
    # "memory and storage" scopes to MU(14)/WDC(12)/STX(11) -> mean 12.33.
    res = graph_qa.aggregate(_NODES, "average P/E of memory and storage makers?")
    assert res["agg"] == "average" and len(res["members"]) == 3
    assert math.isclose(res["value"], (14.0 + 12.0 + 11.0) / 3, abs_tol=1e-6)


def test_aggregate_total_market_cap():
    res = graph_qa.aggregate(_NODES, "combined market cap of memory and storage makers?")
    assert res["agg"] == "total"
    assert math.isclose(res["value"], 1.2e11 + 2.0e10 + 1.8e10, rel_tol=1e-9)


def test_aggregate_count_scoped():
    res = graph_qa.aggregate(_NODES, "how many storage makers are there?")
    assert res["agg"] == "count" and res["value"] == 3   # WDC, STX, MU


def test_aggregate_returns_none_for_non_aggregate():
    assert graph_qa.aggregate(_NODES, "Which company is most central?") is None


# ── multi-hop set algebra (relation chain parsing) ───────────────────────────

def test_parse_chain_single_hop():
    focus, rels = graph_qa.parse_chain("NVIDIA's suppliers")
    assert focus == {"NVDA"} and rels == ["supplier"]


def test_parse_chain_multi_hop_applies_inner_first():
    # "suppliers of NVIDIA's customers" -> walk NVDA -> customers -> suppliers.
    focus, rels = graph_qa.parse_chain("suppliers of NVIDIA's customers")
    assert focus == {"NVDA"}
    assert rels == ["customer", "supplier"]   # nearest-to-focus (customer) first


def test_parse_chain_order_independent_of_surface_form():
    # Same semantics whether written "of … of X" or possessive.
    _, a = graph_qa.parse_chain("competitors of customers of AMD")
    _, b = graph_qa.parse_chain("AMD's customers' competitors")
    assert a == ["customer", "competitor"] == b


def test_parse_chain_no_relation():
    assert graph_qa.parse_chain("average P/E of semiconductor companies") == (set(), [])


def test_describe_chain_reads_outward():
    assert graph_qa._describe_chain("NVDA", ["customer", "supplier"]) == \
        "suppliers of customers of NVDA"


# ── threshold / predicate filters ────────────────────────────────────────────

def test_parse_predicates_metrics_and_scales():
    (p,) = graph_qa.parse_predicates("suppliers with sentiment > 0.5")
    assert (p["key"], p["op"], p["threshold"]) == ("sentiment", "gt", 0.5)
    (p,) = graph_qa.parse_predicates("competitors with P/E under 20")
    assert (p["key"], p["op"], p["threshold"]) == ("pe", "lt", 20.0)
    (p,) = graph_qa.parse_predicates("names with market cap over $1T")
    assert (p["key"], p["op"], p["threshold"]) == ("market_cap", "gt", 1e12)
    (p,) = graph_qa.parse_predicates("companies with gross margin above 40%")
    assert p["key"] == "gross_margin" and abs(p["threshold"] - 0.40) < 1e-9   # 40% -> 0.40
    (p,) = graph_qa.parse_predicates("names with degree of at least 10")
    assert (p["key"], p["op"], p["threshold"]) == ("degree", "ge", 10.0)


def test_parse_predicates_compound_and_none():
    ps = graph_qa.parse_predicates("suppliers with sentiment above 0.5 and P/E under 30")
    assert {p["key"] for p in ps} == {"sentiment", "pe"}
    assert graph_qa.parse_predicates("Who supplies NVIDIA?") == []   # no comparator+number


def test_apply_filters():
    got = {n["id"] for n in graph_qa._apply_filters(
        _NODES, [{"field": ("metrics", "pe_ttm"), "op": "lt", "threshold": 15.0}])}
    assert got == {"MU", "WDC", "STX"}              # P/E 14/12/11 < 15; NVDA 60, QCOM 18 excluded
    # null metric is excluded, never coerced
    got2 = {n["id"] for n in graph_qa._apply_filters(
        _NODES, [{"field": "sentiment_score", "op": "gt", "threshold": 0.6}])}
    assert got2 == {"MU", "WDC", "STX", "NOW"}      # >0.6: 0.72/0.70/0.68/0.69


def test_strip_predicate_text_protects_metric_detection():
    # The filter's metric ("sentiment") must not hijack the aggregate metric ("P/E").
    stripped = graph_qa._strip_predicate_text("average P/E of competitors with sentiment above 0.4")
    assert "sentiment" not in stripped.lower() and "p/e" in stripped.lower()
    agg = graph_qa.detect_aggregate("average P/E of competitors with sentiment above 0.4")
    assert agg["metric"]["key"] == "pe"


# ── eval metrics ─────────────────────────────────────────────────────────────

def test_recall_mrr():
    from sp500_vault.evaluation import _recall_mrr
    r, rr, hits = _recall_mrr(["NVDA", "MU"], ["NVDA", "MU", "AMD"])
    assert r == 1.0 and rr == 1.0 and set(hits) == {"NVDA", "MU"}
    r, rr, _ = _recall_mrr(["MU", "NVDA"], ["AMD", "MU", "INTC"])   # first hit at rank 2
    assert r == 0.5 and rr == 0.5
    r, rr, hits = _recall_mrr(["MU"], ["AMD", "INTC"])             # miss
    assert r == 0.0 and rr == 0.0 and hits == []
