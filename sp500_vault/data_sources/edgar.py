"""SEC EDGAR access: locate a company's latest 10-K and extract Item 1 (Business).

The Business section is where filers name suppliers, customers, and competitors —
the highest-signal free, legally-required text for relationship extraction.

SEC asks for a descriptive User-Agent with contact info and rate-limits to ~10
req/s; we stay well under that with small sleeps.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import warnings

import requests
from bs4 import BeautifulSoup
from bs4 import XMLParsedAsHTMLWarning

from .. import config

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
_ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"
_HEADERS = {"User-Agent": config.SEC_USER_AGENT, "Accept-Encoding": "gzip, deflate"}
_CACHE = config.EDGAR_DIR


def _get(url: str) -> requests.Response:
    resp = requests.get(url, headers=_HEADERS, timeout=30)
    resp.raise_for_status()
    time.sleep(0.2)  # be polite to SEC
    return resp


def _ticker_to_cik_map() -> dict[str, str]:
    cache = _CACHE / "company_tickers.json"
    if not cache.exists():
        cache.write_text(_get(_TICKERS_URL).text, encoding="utf-8")
    raw = json.loads(cache.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for entry in raw.values():
        out[entry["ticker"].upper()] = str(entry["cik_str"]).zfill(10)
    return out


def get_cik(ticker: str) -> str | None:
    return _ticker_to_cik_map().get(ticker.upper())


def _latest_10k(cik: str) -> dict | None:
    data = _get(_SUBMISSIONS_URL.format(cik=cik)).json()
    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    for i, form in enumerate(forms):
        if form == "10-K":
            return {
                "accession": recent["accessionNumber"][i].replace("-", ""),
                "primary_doc": recent["primaryDocument"][i],
                "filing_date": recent["filingDate"][i],
            }
    return None


def _extract_business_section(text: str) -> str:
    """Slice out 'Item 1. Business' (up to 'Item 1A. Risk Factors').

    A 10-K names these headings twice — once in the table of contents (where the
    text between them is just a page number) and once for the real section body.
    We pick the *longest* span between a Business heading and the next Risk
    Factors heading, which reliably selects the body over the TOC entry.
    """
    norm = re.sub(r"\s+", " ", text)
    starts = [m.start() for m in re.finditer(r"Item\s+1\.?\s*Business", norm, re.IGNORECASE)]
    ends = [m.start() for m in re.finditer(r"Item\s+1A\.?\s*Risk\s+Factors", norm, re.IGNORECASE)]
    if not starts:
        return norm[: config.EDGAR_BUSINESS_MAX_CHARS]

    best: tuple[int, int] | None = None  # (length, start)
    for s in starts:
        e = next((x for x in ends if x > s), len(norm))
        span = e - s
        if best is None or span > best[0]:
            best = (span, s)
    s = best[1]
    e = next((x for x in ends if x > s), len(norm))
    return norm[s:e][: config.EDGAR_BUSINESS_MAX_CHARS]


def fetch_business_section(ticker: str) -> dict | None:
    """Return {accession, filing_date, text} for the latest 10-K Business section."""
    cik = get_cik(ticker)
    if not cik:
        return None
    filing = _latest_10k(cik)
    if not filing:
        return None

    cache = _CACHE / f"{ticker}_10K_business.txt"
    meta = {"accession": filing["accession"], "filing_date": filing["filing_date"]}
    if cache.exists():
        meta["text"] = cache.read_text(encoding="utf-8")
        return meta

    url = _ARCHIVE_URL.format(
        cik=int(cik), acc=filing["accession"], doc=filing["primary_doc"]
    )
    html = _get(url).text
    soup = BeautifulSoup(html, "lxml")
    section = _extract_business_section(soup.get_text(" "))
    cache.write_text(section, encoding="utf-8")
    meta["text"] = section
    return meta
