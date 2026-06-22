"""SEC EDGAR access: locate a company's latest 10-K and extract Item 1 (Business).

The Business section is where filers name suppliers, customers, and competitors —
the highest-signal free, legally-required text for relationship extraction.

SEC asks for a descriptive User-Agent with contact info and rate-limits to ~10
req/s; we stay well under that with small sleeps.
"""
from __future__ import annotations

import datetime as dt
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


# ── 8-K material-event ingestion ─────────────────────────────────────────────
# 8-K item codes ARE the material-event taxonomy — mapping them to labels gives a
# free, structured catalyst signal straight from the submissions index (no need to
# fetch each document).

_8K_ITEMS = {
    "1.01": "Entry into a Material Definitive Agreement",
    "1.02": "Termination of a Material Definitive Agreement",
    "1.03": "Bankruptcy or Receivership",
    "2.01": "Completion of Acquisition or Disposition of Assets",
    "2.02": "Results of Operations and Financial Condition (earnings)",
    "2.03": "Creation of a Direct Financial Obligation",
    "2.04": "Triggering Event Accelerating a Financial Obligation",
    "2.05": "Costs Associated with Exit or Disposal Activities",
    "2.06": "Material Impairments",
    "3.01": "Notice of Delisting or Failure to Satisfy a Listing Rule",
    "3.02": "Unregistered Sales of Equity Securities",
    "3.03": "Material Modification to Rights of Security Holders",
    "4.01": "Change in Registrant's Certifying Accountant",
    "4.02": "Non-Reliance on Previously Issued Financials",
    "5.01": "Changes in Control of Registrant",
    "5.02": "Departure / Appointment of Directors or Officers",
    "5.03": "Amendments to Articles of Incorporation or Bylaws",
    "5.07": "Submission of Matters to a Vote of Security Holders",
    "7.01": "Regulation FD Disclosure",
    "8.01": "Other Events",
    "9.01": "Financial Statements and Exhibits",
}


def _item_label(code: str) -> str:
    return _8K_ITEMS.get(code.strip(), f"Item {code.strip()}")


def _parse_8k_filings(recent: dict, lookback_days: int, limit: int,
                      today: dt.date | None = None) -> list[dict]:
    """Pure: extract recent 8-K events from a submissions ``filings.recent`` block.

    Arrays are newest-first, so we keep order and stop at ``limit``. Item codes
    map to human labels (9.01 'Exhibits' is dropped from the label list — it's
    administrative, not a material event).
    """
    today = today or dt.date.today()
    cutoff = today - dt.timedelta(days=lookback_days)
    forms = recent.get("form", [])
    n = len(forms)

    def col(name):
        c = recent.get(name, [])
        return c if len(c) == n else c + [""] * (n - len(c))

    fdates, rdates, accs, items, descs, pdocs = (
        col("filingDate"), col("reportDate"), col("accessionNumber"),
        col("items"), col("primaryDocDescription"), col("primaryDocument"))

    out: list[dict] = []
    for i, form in enumerate(forms):
        if form != "8-K":
            continue
        try:
            d = dt.date.fromisoformat(fdates[i])
        except (TypeError, ValueError):
            continue
        if d < cutoff:
            continue
        codes = [c.strip() for c in (items[i] or "").split(",") if c.strip()]
        material = [{"code": c, "label": _item_label(c)} for c in codes if c != "9.01"]
        out.append({
            "filing_date": fdates[i],
            "report_date": rdates[i] or fdates[i],
            "items": material or [{"code": c, "label": _item_label(c)} for c in codes],
            "accession": accs[i],
            "primary_doc": pdocs[i] or "",
            "description": descs[i] or "",
        })
        if len(out) >= limit:
            break
    return out


def fetch_recent_8k(ticker: str, lookback_days: int | None = None,
                    limit: int | None = None) -> list[dict]:
    """Recent 8-K material events for a ticker (one submissions call, no doc fetch)."""
    lookback_days = lookback_days or config.EDGAR_8K_LOOKBACK_DAYS
    limit = limit or config.EDGAR_8K_MAX
    cik = get_cik(ticker)
    if not cik:
        return []
    try:
        data = _get(_SUBMISSIONS_URL.format(cik=cik)).json()
    except Exception:  # noqa: BLE001
        return []
    events = _parse_8k_filings(data.get("filings", {}).get("recent", {}), lookback_days, limit)
    for e in events:
        folder = (f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
                  f"{e['accession'].replace('-', '')}/")
        e["url"] = folder
        if e.get("primary_doc"):
            e["doc_url"] = folder + e["primary_doc"]   # the actual 8-K body document
    return events


# ── 8-K body text (for one-line LLM summaries of high-signal events) ──────────


def _extract_8k_body(text: str, max_chars: int | None = None) -> str:
    """Pure: collapse an 8-K document to readable text, starting at the first
    'Item X.XX' heading (drops cover-page boilerplate) and capped for cost."""
    max_chars = max_chars or config.EDGAR_8K_BODY_MAX_CHARS
    norm = re.sub(r"\s+", " ", text).strip()
    m = re.search(r"Item\s+\d\.\d\d", norm)
    if m:
        norm = norm[m.start():]
    return norm[:max_chars]


def fetch_8k_body(doc_url: str, max_chars: int | None = None) -> str:
    """Fetch and clean the body text of one 8-K primary document."""
    if not doc_url:
        return ""
    html = _get(doc_url).text
    soup = BeautifulSoup(html, "lxml")
    return _extract_8k_body(soup.get_text(" "), max_chars)


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
