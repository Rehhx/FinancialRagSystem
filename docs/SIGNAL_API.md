# RAG Vault — Signal API integration

Drop this file into your trading-engine repo. It documents how to configure and
call the **S&P 500 RAG Vault** signal service to get a per-ticker
**LONG / SHORT / FLAT** verdict (and optionally blend in your own signal).

The vault blends three *validated* strategies — sentiment lead-lag, supplier
lead-lag, and 8-K event drift — each weighted by its **measured Information
Coefficient** (Grinold–Kahn signal combination; Fundamental Law IR = IC·√breadth),
z-scores the blend into a **conviction**, and only flags a trade when
`|conviction| ≥ τ` **and** the name has a validated edge. Everything else is
`flat` — that's the "or not."

> Reads are served from a **daily snapshot**, so calls are fast and make no
> market-data or LLM calls per request. The vault side must run
> `python -m sp500_vault.pipeline trades` once a day to refresh it.

---

## 1. Configure

The service is plain HTTP, no auth (see [Security](#7-security)). Point your repo
at it with one env var.

```bash
# .env in your trading-engine repo
SIGNAL_API_URL=http://127.0.0.1:8000     # where the RAG-vault FastAPI service runs
SIGNAL_API_TIMEOUT=5                      # seconds
SIGNAL_HORIZON=5                          # default forward horizon (trading days)
SIGNAL_TAU=0.5                            # conviction (σ) needed to act
```

Start the vault service (on the vault repo / host):

```bash
uvicorn sp500_vault.api:app --host 0.0.0.0 --port 8000
```

Smoke-test from your repo:

```bash
curl "$SIGNAL_API_URL/health"            # {"status":"ok","universe":50}
curl "$SIGNAL_API_URL/signal/NVDA"
```

---

## 2. Endpoints

| Method & path | Purpose |
|---|---|
| `GET /signal/{ticker}` | One verdict for one ticker. |
| `GET /signals?tickers=NVDA,AMD` | Batch verdicts for specific names. |
| `GET /signals` | The whole universe, ranked by conviction (LONG → SHORT). |
| `GET /health` | Liveness + universe size. |

### Query params (all optional)

| Param | Applies to | Default | Meaning |
|---|---|---|---|
| `horizon` | both | `5` | Forward holding period in trading days. Re-weights every component from the same snapshot. |
| `tau` | both | `0.5` | Conviction (σ) needed to act; below it → `flat`. |
| `overlay` | `/signal/{ticker}` | — | **Your own** normalized signal (positive = long, negative = short). |
| `overlay_weight` | `/signal/{ticker}` | `1.0` | How much to trust your overlay vs the vault, `0`–`1`. Combined as `(1−w)·vault + w·overlay`. |

---

## 3. Response schema

```jsonc
// GET /signal/NVDA
{
  "ticker": "NVDA",
  "as_of": "2026-06-22",          // snapshot date (UTC)
  "horizon_days": 5,
  "direction": "long",            // "long" | "short" | "flat"  <- act on this
  "conviction": 0.60,             // signed, in cross-sectional σ units
  "strength": 0.54,               // tanh(|conviction|), 0..1 — use for position sizing
  "confidence": "high",           // "none" | "low" | "medium" | "high"
  "breadth": 2,                   // # of strategies that fired (0..3)
  "agreement": 1.0,               // fraction of fired strategies agreeing with the net sign
  "percentile": 66,               // cross-sectional rank of conviction (0..100)
  "coverage": true,               // false => not in the vault universe / no data
  "price_source": "alpaca",
  "recommendation": "LONG NVDA — conviction +0.60σ (5d, high confidence); sentiment · supplier_leadlag",
  "components": {                 // only the strategies that fired are present
    "sentiment": {
      "signal": 0.35,             // raw Claude sentiment score (-1..1)
      "label": "Bullish",
      "z": 0.20,                  // cross-sectional z-score of the raw signal
      "ic": 0.31,                 // signed Information Coefficient (the weight)
      "ic_source": "provider rank-IC",
      "contribution": 0.0607,     // z * ic — what this strategy added to the blend
      "reason": "Claude sentiment Bullish (+0.35); 5d IC +0.310 (provider rank-IC)"
    },
    "supplier_leadlag": {
      "signal": 0.2107,           // suppliers' trailing-k cumulative return
      "suppliers": 1,             // # of modeled suppliers with price data
      "z": 3.37,
      "ic": 0.039,
      "ic_source": "k5→h5",
      "contribution": 0.1315,
      "reason": "Suppliers' 5d momentum +21.07% (n=1); lead-lag IC +0.039"
    },
    "event_drift": {              // present only if a recent significant 8-K
      "edge": 0.018,              // summed market-adjusted drift (returns)
      "z": 1.40,
      "ic": 0.10,                 // EVENT_IC trust weight
      "events": ["8-K 2.02 Results of Operations +1.80% 5d (t +2.4)"],
      "contribution": 0.014,
      "reason": "8-K 2.02 Results of Operations +1.80% 5d (t +2.4)"
    }
  }
}
```

### With an overlay (`?overlay=-3&overlay_weight=0.6`)

Adds these fields and **re-derives** `direction`/`conviction`/`strength` from the
combined score:

```jsonc
{
  "direction": "short",          // recomputed from the combined score
  "conviction": -1.56,           // (1-0.6)*0.60 + 0.6*(-3)
  "base_direction": "long",      // the vault-only verdict, preserved
  "base_conviction": 0.60,
  "overlay": { "signal": -3.0, "weight": 0.6 },
  "overlay_applied": true,
  // ... components unchanged ...
}
```

### Batch (`GET /signals?tickers=NVDA,AMD,TSLA`)

```jsonc
{ "horizon_days": 5, "signals": [ { /* verdict */ }, { /* verdict */ }, ... ] }
```

### Not covered / snapshot missing

```jsonc
{ "ticker": "ZZZZ", "coverage": false, "direction": "flat", "confidence": "none",
  "reason": "ticker not in the vault universe / no data" }
```

---

## 4. How to act on it

The vault answers **direction + how much to trust it**. Suggested gate for your
engine:

```text
skip if   coverage == false            # name not modeled by the vault
skip if   direction == "flat"          # no edge, or below the τ threshold
skip if   confidence in {"none","low"} # only one strategy fired — optional, stricter
trade     direction ("long"/"short")
size  ∝   strength                     # 0..1; or |conviction|
```

Field meanings that drive the decision:

- **`direction`** — the verdict. `long` when `conviction ≥ τ`, `short` when
  `conviction ≤ −τ`, else `flat`. Gated on `breadth ≥ 1`.
- **`conviction`** — the blended signal, z-scored **across the universe**. It's a
  *relative* read (is this name's edge strong vs its peers today), not an absolute
  price target. Signed: + = long, − = short.
- **`strength`** — `tanh(|conviction|)` in `0..1`. Convenient sizing multiplier.
- **`breadth`** — how many of the 3 strategies fired. `confidence` is derived:
  `none` (0), `low` (1), `medium` (≥2, mixed signs), `high` (≥2, all agree).
- **`agreement`** — fraction of fired strategies whose sign matches the net. `1.0`
  means every active strategy points the same way.
- **`components[*].contribution`** — exactly how much each strategy moved the
  blend (`z × ic`). Lets you **cherry-pick**: e.g. trust only `event_drift`, or
  drop `sentiment` and re-blend on your side.

---

## 5. Python client (drop-in)

```python
"""rag_vault_client.py — thin client for the RAG Vault signal API."""
from __future__ import annotations

import os
import requests


class RagVaultSignals:
    def __init__(self, base_url: str | None = None, timeout: float | None = None):
        self.base = (base_url or os.getenv("SIGNAL_API_URL", "http://127.0.0.1:8000")).rstrip("/")
        self.timeout = timeout or float(os.getenv("SIGNAL_API_TIMEOUT", "5"))
        self.horizon = int(os.getenv("SIGNAL_HORIZON", "5"))
        self.tau = float(os.getenv("SIGNAL_TAU", "0.5"))

    def signal(self, ticker: str, *, horizon: int | None = None, tau: float | None = None,
               overlay: float | None = None, overlay_weight: float = 1.0) -> dict:
        """One LONG/SHORT/FLAT verdict. `overlay` blends in your own signal."""
        params = {"horizon": horizon or self.horizon, "tau": self.tau if tau is None else tau}
        if overlay is not None:
            params |= {"overlay": overlay, "overlay_weight": overlay_weight}
        r = requests.get(f"{self.base}/signal/{ticker.upper()}", params=params, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def signals(self, tickers: list[str] | None = None, *, horizon: int | None = None) -> list[dict]:
        """Batch verdicts; omit `tickers` for the whole ranked universe."""
        params = {"horizon": horizon or self.horizon}
        if tickers:
            params["tickers"] = ",".join(t.upper() for t in tickers)
        r = requests.get(f"{self.base}/signals", params=params, timeout=self.timeout)
        r.raise_for_status()
        return r.json()["signals"]

    def decide(self, ticker: str, *, min_confidence: str = "medium", **kw) -> dict | None:
        """Return a {ticker, side, size} order, or None to skip — your gate, tweak freely."""
        v = self.signal(ticker, **kw)
        rank = {"none": 0, "low": 1, "medium": 2, "high": 3}
        if not v.get("coverage") or v["direction"] == "flat":
            return None
        if rank.get(v.get("confidence", "none"), 0) < rank[min_confidence]:
            return None
        return {"ticker": v["ticker"], "side": v["direction"], "size": v["strength"],
                "conviction": v["conviction"], "why": v["recommendation"]}


if __name__ == "__main__":
    api = RagVaultSignals()
    print(api.decide("NVDA"))                         # {'ticker':'NVDA','side':'long', ...} or None
    print(api.signal("NVDA", overlay=0.8, overlay_weight=0.5)["direction"])
    for v in api.signals(["NVDA", "AMD", "TSLA"]):
        print(v["ticker"], v["direction"], round(v["conviction"], 2), v["confidence"])
```

---

## 6. Operational notes

- **Freshness:** verdicts are only as fresh as the snapshot. Schedule
  `python -m sp500_vault.pipeline trades` daily on the vault host (it's already in
  `pipeline all`). `as_of` on every response tells you the snapshot date.
- **Universe:** the vault models a 50-name pilot universe. `coverage=false` means
  the ticker isn't modeled — treat as "no opinion," not "flat with conviction."
- **Horizon is free:** the same daily snapshot serves any `horizon`; it just
  re-weights the components, so you can poll 1/3/5-day reads without a rebuild.
- **Sentiment IC fallback:** the sentiment weight uses the dense Claude panel's
  rank-IC when available, else the news-provider panel, else a small fallback. The
  `ic_source` field tells you which fired.
- **Determinism:** no LLM/market calls at request time → identical inputs give
  identical outputs within a snapshot day. Safe to cache aggressively.

---

## 7. Security

The endpoints are **unauthenticated** today. Before exposing them beyond
localhost:

- keep the service on a private network / VPN, or put a reverse proxy
  (nginx/Caddy) with an API key or mTLS in front, and
- do **not** expose the vault's other write-ish endpoints (`/regenerate`,
  `/query`) publicly if you only need signals.

If you want a bearer-token check added on `/signal*` directly, ask the vault side
to add it — it's a small FastAPI dependency.
