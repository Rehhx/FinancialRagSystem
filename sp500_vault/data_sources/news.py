"""Company news aggregation across multiple free-tier providers.

Feeds the sentiment layer. We pull from several sources and merge them so a thin
day on one feed is covered by another — more headlines means a better-grounded
Claude sentiment score. Providers (toggle/order via ``config.NEWS_PROVIDERS``):

    finnhub        company_news (primary; generous free tier)
    marketaux      /news/all, entity-filtered to the ticker (+ provider sentiment)
    newsapi        /everything, queried by company name (broad press coverage)
    alphavantage   NEWS_SENTIMENT (+ provider sentiment; small daily quota)

Every provider is wrapped so a missing key, a quota error, or a network blip
returns ``[]`` instead of breaking the pass. Articles are normalized to a common
shape, de-duplicated (by URL, then headline), sorted newest-first, and capped.
"""
from __future__ import annotations

import datetime as dt
import re
from functools import lru_cache

import finnhub
import requests

from .. import config
from ..universe import BY_TICKER

_TIMEOUT = 10


# ── Finnhub ──────────────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def _client() -> finnhub.Client:
    config.require("FINNHUB_API_KEY")
    return finnhub.Client(api_key=config.FINNHUB_API_KEY)


def _finnhub_news(ticker: str, start: dt.date, today: dt.date, limit: int) -> list[dict]:
    if not config.FINNHUB_API_KEY:
        return []
    try:
        raw = _client().company_news(ticker, _from=start.isoformat(), to=today.isoformat())
    except Exception:  # noqa: BLE001
        return []
    out = []
    for a in raw[:limit]:
        ts = a.get("datetime", 0)
        out.append(_article(
            headline=a.get("headline", ""),
            summary=a.get("summary", ""),
            source=a.get("source", ""),
            url=a.get("url", ""),
            when=dt.datetime.utcfromtimestamp(ts).isoformat() if ts else "",
            provider="finnhub",
        ))
    return out


# ── Marketaux ────────────────────────────────────────────────────────────────


def _marketaux_news(ticker: str, start: dt.date, limit: int) -> list[dict]:
    if not config.MARKETAUX_API_KEY:
        return []
    try:
        r = requests.get(
            "https://api.marketaux.com/v1/news/all",
            params={
                "symbols": ticker,
                "filter_entities": "true",
                "language": "en",
                "published_after": start.isoformat() + "T00:00",
                "limit": min(limit, 100),
                "api_token": config.MARKETAUX_API_KEY,
            },
            timeout=_TIMEOUT,
        )
        data = r.json()
    except Exception:  # noqa: BLE001
        return []
    out = []
    for a in (data.get("data") or [])[:limit]:
        # entity sentiment for the requested ticker, if present
        ent_sent = None
        for e in a.get("entities", []):
            if (e.get("symbol") or "").upper() == ticker.upper() and e.get("sentiment_score") is not None:
                ent_sent = e["sentiment_score"]
                break
        out.append(_article(
            headline=a.get("title", ""),
            summary=a.get("description") or a.get("snippet") or "",
            source=a.get("source", ""),
            url=a.get("url", ""),
            when=(a.get("published_at") or "").replace("Z", ""),
            provider="marketaux",
            provider_sentiment=ent_sent,
        ))
    return out


# ── NewsAPI.org ──────────────────────────────────────────────────────────────


def _clean_name(name: str) -> str:
    """Core company name for a news query (drop suffixes that add noise)."""
    core = name.split(",")[0]
    for suffix in (" Incorporated", " Inc.", " Corporation", " Corp.", " Company",
                   " Holdings", " Technologies", " plc", " N.V.", " PLC", " Ltd."):
        core = core.replace(suffix, "")
    return core.strip() or name


def _newsapi_news(ticker: str, start: dt.date, today: dt.date, limit: int) -> list[dict]:
    if not config.NEWS_API_KEY:
        return []
    name = BY_TICKER[ticker].name if ticker in BY_TICKER else ticker
    try:
        r = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                # qInTitle requires the company name in the headline — far less
                # off-topic noise than a full-body match when merging 4 feeds.
                "qInTitle": f'"{_clean_name(name)}"',
                "from": start.isoformat(),
                "to": today.isoformat(),
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": min(limit, 100),
                "apiKey": config.NEWS_API_KEY,
            },
            timeout=_TIMEOUT,
        )
        data = r.json()
    except Exception:  # noqa: BLE001
        return []
    if data.get("status") != "ok":
        return []
    out = []
    for a in (data.get("articles") or [])[:limit]:
        out.append(_article(
            headline=a.get("title", ""),
            summary=a.get("description") or "",
            source=(a.get("source") or {}).get("name", ""),
            url=a.get("url", ""),
            when=(a.get("publishedAt") or "").replace("Z", ""),
            provider="newsapi",
        ))
    return out


# ── Alpha Vantage NEWS_SENTIMENT ─────────────────────────────────────────────


def _alphavantage_news(ticker: str, start: dt.date, limit: int) -> list[dict]:
    if not config.ALPHA_API_KEY:
        return []
    try:
        r = requests.get(
            "https://www.alphavantage.co/query",
            params={
                "function": "NEWS_SENTIMENT",
                "tickers": ticker,
                "time_from": start.strftime("%Y%m%dT0000"),
                "sort": "LATEST",
                "limit": min(limit, 50),
                "apikey": config.ALPHA_API_KEY,
            },
            timeout=_TIMEOUT,
        )
        data = r.json()
    except Exception:  # noqa: BLE001
        return []
    feed = data.get("feed")
    if not feed:  # quota/throttle responses carry "Note"/"Information" and no feed
        return []
    out = []
    for a in feed[:limit]:
        # ticker-specific sentiment if available, else overall
        tsent = None
        for ts in a.get("ticker_sentiment", []):
            if (ts.get("ticker") or "").upper() == ticker.upper():
                try:
                    tsent = float(ts.get("ticker_sentiment_score"))
                except (TypeError, ValueError):
                    tsent = None
                break
        out.append(_article(
            headline=a.get("title", ""),
            summary=a.get("summary", ""),
            source=a.get("source", ""),
            url=a.get("url", ""),
            when=_parse_av_time(a.get("time_published", "")),
            provider="alphavantage",
            provider_sentiment=tsent,
        ))
    return out


def _parse_av_time(s: str) -> str:
    """Alpha Vantage stamps are 'YYYYMMDDTHHMMSS' — normalize to ISO."""
    try:
        return dt.datetime.strptime(s, "%Y%m%dT%H%M%S").isoformat()
    except (ValueError, TypeError):
        return ""


# ── Free RSS feeds (no key, no quota — the news-cycle backbone) ───────────────

_RSS_HEADERS = {"User-Agent": "Mozilla/5.0 (sp500-rag-vault research-bot)"}


def _strip_html(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text or "")).strip()


def _rfc822(s: str) -> str:
    from email.utils import parsedate_to_datetime
    try:
        return parsedate_to_datetime(s).isoformat()
    except Exception:  # noqa: BLE001
        return ""


def _parse_rss(text: str, provider: str, limit: int) -> list[dict]:
    """Parse an RSS 2.0 feed into normalized articles (stdlib XML, no new dep)."""
    import xml.etree.ElementTree as ET
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return []
    out: list[dict] = []
    for item in root.findall(".//item"):
        title = (item.findtext("title") or "").strip()
        if not title:
            continue
        src_el = item.find("source")
        source = (src_el.text or "").strip() if src_el is not None and src_el.text else ""
        out.append(_article(
            headline=title,
            summary=_strip_html(item.findtext("description") or ""),
            source=source,
            url=(item.findtext("link") or "").strip(),
            when=_rfc822(item.findtext("pubDate") or ""),
            provider=provider,
        ))
        if len(out) >= limit:
            break
    return out


def _yahoo_rss(ticker: str, limit: int) -> list[dict]:
    try:
        r = requests.get(
            "https://feeds.finance.yahoo.com/rss/2.0/headline",
            params={"s": ticker, "region": "US", "lang": "en-US"},
            headers=_RSS_HEADERS, timeout=_TIMEOUT)
        return _parse_rss(r.text, "yahoo", limit) if r.status_code == 200 else []
    except Exception:  # noqa: BLE001
        return []


def _googlenews_rss(ticker: str, lookback_days: int, limit: int) -> list[dict]:
    """Google News search RSS, scoped to the company name + recency window."""
    name = BY_TICKER[ticker].name if ticker in BY_TICKER else ticker
    query = f"{_clean_name(name)} stock when:{max(1, lookback_days)}d"
    try:
        r = requests.get(
            "https://news.google.com/rss/search",
            params={"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"},
            headers=_RSS_HEADERS, timeout=_TIMEOUT)
        return _parse_rss(r.text, "googlenews", limit) if r.status_code == 200 else []
    except Exception:  # noqa: BLE001
        return []


# ── Aggregation ──────────────────────────────────────────────────────────────

_PROVIDERS = {
    # Free, no key, no daily quota — listed first so they're the reliable backbone.
    "googlenews": lambda t, s, today, lim: _googlenews_rss(t, (today - s).days, lim),
    "yahoo": lambda t, s, today, lim: _yahoo_rss(t, lim),
    # Keyed APIs (free tiers with daily caps).
    "finnhub": lambda t, s, today, lim: _finnhub_news(t, s, today, lim),
    "marketaux": lambda t, s, today, lim: _marketaux_news(t, s, lim),
    "newsapi": lambda t, s, today, lim: _newsapi_news(t, s, today, lim),
    "alphavantage": lambda t, s, today, lim: _alphavantage_news(t, s, lim),
}

# Providers that need no API key (always available when listed in NEWS_PROVIDERS).
_KEYLESS = {"googlenews", "yahoo"}


def _article(headline, summary, source, url, when, provider, provider_sentiment=None) -> dict:
    return {
        "headline": headline or "",
        "summary": summary or "",
        "source": source or "",
        "url": url or "",
        "datetime": when or "",
        "provider": provider,
        "provider_sentiment": provider_sentiment,
    }


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def _dedup(articles: list[dict]) -> list[dict]:
    """Drop duplicate stories (same URL, or near-identical headline)."""
    seen: set[str] = set()
    out: list[dict] = []
    for a in articles:
        url = (a.get("url") or "").split("?")[0].rstrip("/").lower()
        key = url or _norm(a.get("headline"))[:80]
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(a)
    return out


def active_providers() -> list[str]:
    """Configured providers (in priority order): keyless RSS always on; keyed
    APIs only when their key is set."""
    keyed = {
        "finnhub": config.FINNHUB_API_KEY,
        "marketaux": config.MARKETAUX_API_KEY,
        "newsapi": config.NEWS_API_KEY,
        "alphavantage": config.ALPHA_API_KEY,
    }
    out = []
    for name in (p.strip().lower() for p in config.NEWS_PROVIDERS.split(",")):
        if name in _KEYLESS or (name in _PROVIDERS and keyed.get(name)):
            out.append(name)
    return out


def _interleave(per_provider: list[list[dict]]) -> list[dict]:
    """Round-robin merge across providers (each already newest-first) so every
    source is represented and the priority-ordered ones lead — instead of a
    global recency sort that lets one fresh-but-generic feed crowd out the rest."""
    merged: list[dict] = []
    depth = max((len(a) for a in per_provider), default=0)
    for i in range(depth):
        for arts in per_provider:
            if i < len(arts):
                merged.append(arts[i])
    return merged


def fetch_news(ticker: str, lookback_days: int | None = None, limit: int | None = None) -> list[dict]:
    """Recent company news merged across all configured providers.

    Returns a list of {headline, summary, source, url, datetime, provider,
    provider_sentiment}, de-duplicated and interleaved by provider priority
    (free RSS leads). Capped at ``limit`` (default ``config.SENTIMENT_MAX_ARTICLES``).
    """
    lookback_days = lookback_days or config.SENTIMENT_LOOKBACK_DAYS
    limit = limit or config.SENTIMENT_MAX_ARTICLES
    today = dt.date.today()
    start = today - dt.timedelta(days=lookback_days)

    per_provider: list[list[dict]] = []
    for name in active_providers():
        try:
            arts = _PROVIDERS[name](ticker, start, today, config.NEWS_PER_PROVIDER)
        except Exception:  # noqa: BLE001 - never let one provider sink the pass
            arts = []
        arts.sort(key=lambda x: x["datetime"], reverse=True)   # newest-first within a provider
        per_provider.append(arts)

    return _dedup(_interleave(per_provider))[:limit]


def fetch_peers(ticker: str) -> list[str]:
    """Finnhub peer companies (tickers) — a cheap Tier-1 relationship seed."""
    if not config.FINNHUB_API_KEY:
        return []
    try:
        peers = _client().company_peers(ticker)
        return [p for p in peers if p and p != ticker]
    except Exception:  # noqa: BLE001
        return []
