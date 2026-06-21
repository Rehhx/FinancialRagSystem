"""Real-time 8-K poller — react to material events within minutes of filing.

SEC EDGAR's ``getcurrent`` feed lists filings in near-real-time (acceptance to
feed is seconds-to-minutes, and it updates continuously). Polling it on a short
interval and matching the entry CIKs against our universe lets the vault react to
a material 8-K — refresh that ticker's filings and re-embed its `Material Events`
RAG chunk — **minutes after it hits the wire, for $0** (no API key, no quota).

The feed entry even carries the item codes inline (``Item 5.07: …``), so we know
the event *type* (earnings / exec change / acquisition) straight from the poll.

Run it:
    python -m sp500_vault.edgar_live tick                 # one poll (cron-friendly)
    python -m sp500_vault.edgar_live watch --interval 120 # daemon, every 2 min
    python -m sp500_vault.edgar_live status               # what it would match now

Complements ``scheduler.py`` (the daily full refresh) — this is the intraday
fast path for the single most time-sensitive layer.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import time
import xml.etree.ElementTree as ET

import requests

from . import config, filings, rag
from .data_sources import edgar
from .universe import TICKERS

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

_GETCURRENT = ("https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent"
               "&type=8-K&company=&dateb=&owner=include&count=100&output=atom")
_HEADERS = {"User-Agent": config.SEC_USER_AGENT, "Accept-Encoding": "gzip, deflate"}
_SEEN_FILE = config.DATA_DIR / "edgar_live_seen.json"
_NS = "{http://www.w3.org/2005/Atom}"

_CIK_MAP: dict[str, str] | None = None


def _cik_to_ticker() -> dict[str, str]:
    """Map zero-stripped CIK -> ticker for our universe (cached on the module)."""
    global _CIK_MAP
    if _CIK_MAP is None:
        _CIK_MAP = {}
        for t in TICKERS:
            cik = edgar.get_cik(t)
            if cik:
                _CIK_MAP[str(int(cik))] = t   # strip leading zeros for feed matching
    return _CIK_MAP


# ── feed parsing (pure) ───────────────────────────────────────────────────────


def _parse_entries(xml_text: str) -> list[dict]:
    """Parse the getcurrent atom feed into {cik, accession, title, summary, updated}."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []
    out: list[dict] = []
    for e in root.findall(f"{_NS}entry"):
        title = e.findtext(f"{_NS}title") or ""
        idtext = e.findtext(f"{_NS}id") or ""
        m_cik = re.search(r"\((\d{4,10})\)", title)        # "8-K - NAME (0001045810) (Filer)"
        m_acc = re.search(r"accession-number=([\d-]+)", idtext)
        if not m_cik or not m_acc:
            continue
        out.append({
            "cik": str(int(m_cik.group(1))),
            "accession": m_acc.group(1),
            "title": title,
            "summary": e.findtext(f"{_NS}summary") or "",
            "updated": e.findtext(f"{_NS}updated") or "",
        })
    return out


def _summary_items(summary: str) -> str:
    """Human label for the 8-K's item codes, pulled straight from the feed summary."""
    codes = re.findall(r"Item (\d\.\d\d)", summary or "")
    material = [edgar._item_label(c) for c in codes if c != "9.01"]
    labels = material or [edgar._item_label(c) for c in codes]
    return "; ".join(labels) if labels else "8-K"


# ── seen-set (dedup across polls/restarts) ────────────────────────────────────


def _load_seen() -> set[str]:
    if _SEEN_FILE.exists():
        try:
            return set(json.loads(_SEEN_FILE.read_text(encoding="utf-8")))
        except Exception:  # noqa: BLE001
            return set()
    return set()


def _save_seen(seen: set[str]) -> None:
    _SEEN_FILE.write_text(json.dumps(sorted(seen)[-3000:]), encoding="utf-8")  # bound the file


def _seed_seen() -> None:
    """Mark already-stored 8-K accessions as seen so startup doesn't re-react to
    filings the daily layer already ingested."""
    seen = _load_seen()
    before = len(seen)
    for p in config.FILINGS_DIR.glob("*.json"):
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        for ev in rec.get("events", []):
            if ev.get("accession"):
                seen.add(ev["accession"])
    if len(seen) > before:
        _save_seen(seen)
        print(f"[live] seeded {len(seen) - before} known accessions (won't re-react)")


# ── poll + react ──────────────────────────────────────────────────────────────


def poll_once() -> list[dict]:
    """Fetch the feed and return new 8-K hits for our universe (excludes seen)."""
    cmap = _cik_to_ticker()
    try:
        r = requests.get(_GETCURRENT, headers=_HEADERS, timeout=20)
    except Exception as e:  # noqa: BLE001
        print(f"[live] poll error: {e}")
        return []
    time.sleep(0.2)  # be polite to SEC
    if r.status_code != 200:
        print(f"[live] getcurrent HTTP {r.status_code}")
        return []
    seen = _load_seen()
    hits = []
    for e in _parse_entries(r.text):
        t = cmap.get(e["cik"])
        if t and e["accession"] not in seen:
            hits.append({**e, "ticker": t})
    return hits


def tick() -> list[dict]:
    """One poll: react to any new universe 8-K, then re-embed changed chunks."""
    hits = poll_once()
    if not hits:
        print(f"[live] {dt.datetime.now():%H:%M:%S} no new universe 8-K")
        return []

    seen = _load_seen()
    by_ticker: dict[str, list[dict]] = {}
    for h in hits:
        by_ticker.setdefault(h["ticker"], []).append(h)

    for t, ths in by_ticker.items():
        for h in ths:
            print(f"[live] 🔔 NEW 8-K {t}: {_summary_items(h['summary'])}  "
                  f"acc={h['accession']} ({h['updated'][:19]})")
        filings.run_for_ticker(t)            # re-pull this ticker's 8-Ks from EDGAR
        for h in ths:
            seen.add(h["accession"])

    _save_seen(seen)
    rag.index_vault()                        # incremental — re-embeds the changed Material Events chunks
    print(f"[live] reacted to {len(by_ticker)} ticker(s); vault is current")
    return hits


def watch(interval: int = 120) -> None:
    print(f"[live] watching EDGAR getcurrent every {interval}s for "
          f"{len(_cik_to_ticker())} universe CIKs — Ctrl-C to stop")
    _seed_seen()
    while True:
        try:
            tick()
        except Exception as e:  # noqa: BLE001 - never let one poll kill the daemon
            print(f"[live] tick error: {e}")
        time.sleep(interval)


def status() -> None:
    cmap = _cik_to_ticker()
    seen = _load_seen()
    print(f"[live] universe CIKs mapped: {len(cmap)}/{len(TICKERS)}")
    print(f"[live] accessions seen (won't re-react): {len(seen)}")
    hits = poll_once()
    print(f"[live] new universe 8-Ks in the current feed window: {len(hits)}")
    for h in hits[:10]:
        print(f"        {h['ticker']}: {_summary_items(h['summary'])} ({h['updated'][:19]})")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="sp500_vault.edgar_live", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("tick", help="Poll once and react (cron-friendly)")
    wp = sub.add_parser("watch", help="Run as a polling daemon")
    wp.add_argument("--interval", type=int, default=120, help="Seconds between polls")
    sub.add_parser("status", help="Show mapping + what the current feed would match")

    args = p.parse_args(argv)
    if args.cmd == "tick":
        tick()
    elif args.cmd == "watch":
        watch(args.interval)
    elif args.cmd == "status":
        status()


if __name__ == "__main__":
    main()
