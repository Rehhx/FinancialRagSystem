"""FastAPI service: RAG query endpoint + the graph-explorer web UI.

Run:  uvicorn sp500_vault.api:app --reload
Then open http://127.0.0.1:8000/ for the interactive graph + ask-the-vault UI.
API:  POST /query  {"question": "...", "k": 6, "sector": "Technology"}
      GET  /graph.json   (node/edge graph for the visualization)
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from . import backtest, config, graph_export, rag, tracing
from .universe import PILOT_UNIVERSE, TICKERS

app = FastAPI(title="S&P 500 RAG Vault", version="0.3.0")


@app.on_event("shutdown")
def _flush_traces() -> None:
    """Flush any buffered Langfuse events when the server stops."""
    tracing.flush()

_WEB_DIR = Path(__file__).resolve().parent / "web"
_GRAPH_FILE = config.DATA_DIR / "graph" / "graph.json"
_BACKTEST_FILE = config.SIGNALS_DIR / "backtest.json"


class Turn(BaseModel):
    role: str
    content: str


class QueryRequest(BaseModel):
    question: str
    k: int = 6
    ticker: str | None = None
    sector: str | None = None
    sentiment: str | None = None
    history: list[Turn] | None = None   # prior turns, for conversational follow-ups


class Source(BaseModel):
    ticker: str | None
    section: str | None


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
    resolved_question: str | None = None


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Serve the graph-explorer single-page app."""
    return (_WEB_DIR / "graph.html").read_text(encoding="utf-8")


@app.get("/graph.json")
def graph() -> JSONResponse:
    """Current node/edge graph. Uses the last export, rebuilding if absent."""
    if _GRAPH_FILE.exists():
        return JSONResponse(json.loads(_GRAPH_FILE.read_text(encoding="utf-8")))
    return JSONResponse(graph_export.run())


@app.get("/backtest.json")
def backtest_data() -> JSONResponse:
    """Lead-lag backtest results (IC grid + long/short equity curve)."""
    if _BACKTEST_FILE.exists():
        return JSONResponse(json.loads(_BACKTEST_FILE.read_text(encoding="utf-8")))
    return JSONResponse(backtest.run(list(TICKERS)))


@app.post("/regenerate")
def regenerate() -> JSONResponse:
    """Re-export graph.json from the current vault data and return it.

    Fast and free — rebuilds the graph from the quant/sentiment/relationship
    caches (no market-data or LLM calls). Use it after editing the manual
    overrides CSV or re-running a data layer from the CLI.
    """
    return JSONResponse(graph_export.run())


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "universe": len(PILOT_UNIVERSE)}


@app.get("/universe")
def universe() -> list[dict]:
    return [{"ticker": c.ticker, "name": c.name, "group": c.group} for c in PILOT_UNIVERSE]


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    history = [t.model_dump() for t in req.history] if req.history else None
    result = rag.query(
        req.question, k=req.k, ticker=req.ticker,
        sector=req.sector, sentiment=req.sentiment, history=history,
    )
    return QueryResponse(**result)
