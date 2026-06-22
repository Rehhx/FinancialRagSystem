"""Optional Langfuse tracing — clean no-op when not configured.

Enabled only when ``LANGFUSE_PUBLIC_KEY`` + ``LANGFUSE_SECRET_KEY`` are set (see
``config``). When enabled, LLM calls are captured automatically — *prefer
framework integrations over manual instrumentation* (Langfuse best practice):

  * **OpenAI** — ``llm.openai_client()`` is built from the Langfuse drop-in class,
    so ``chat.completions`` and ``embeddings`` are traced with model name, token
    usage, and I/O for free.
  * **Anthropic** — ``AnthropicInstrumentor`` patches the SDK, so every
    ``messages.create`` / ``messages.stream`` is captured the same way.

``observe`` then decorates the orchestration entry points (e.g. ``rag.query``,
the batch scorers) to create descriptively-named parent traces that the
auto-captured generations nest under. ``update_span`` sets a *clean* input/output
explicitly so giant filing text or config args don't become the trace input.

When disabled — or if ``langfuse`` / the instrumentor isn't installed — every
helper degrades to a passthrough so the app behaves exactly as before. Tracing
must never change application behavior or fail a request.
"""
from __future__ import annotations

from typing import Any, Callable

from . import config

ENABLED = config.TRACING_ENABLED

_client = None
_anthropic_instrumented = False


def _noop_observe(*d_args, **d_kwargs):
    """Stand-in for langfuse.observe supporting both @observe and @observe(name=...)."""
    if len(d_args) == 1 and callable(d_args[0]) and not d_kwargs:
        return d_args[0]

    def deco(fn: Callable) -> Callable:
        return fn
    return deco


if ENABLED:
    try:
        from langfuse import get_client, observe as _observe  # noqa: F401
        _client = get_client()                 # reads LANGFUSE_* from the env config set up
        observe = _observe
    except Exception as exc:                    # noqa: BLE001 - degrade, never crash the app
        print(f"[tracing] Langfuse unavailable ({exc}); tracing disabled")
        ENABLED = False
        observe = _noop_observe
else:
    observe = _noop_observe


def instrument_anthropic() -> None:
    """Patch the Anthropic SDK once so all messages calls are auto-traced."""
    global _anthropic_instrumented
    if not ENABLED or _anthropic_instrumented:
        return
    try:
        from opentelemetry.instrumentation.anthropic import AnthropicInstrumentor
        AnthropicInstrumentor().instrument()
    except Exception as exc:                    # noqa: BLE001
        print(f"[tracing] Anthropic auto-instrumentation unavailable ({exc})")
    _anthropic_instrumented = True              # don't retry on every client build


def openai_class(default_cls):
    """The Langfuse-wrapped OpenAI class when enabled, else the plain one."""
    if not ENABLED:
        return default_cls
    try:
        from langfuse.openai import OpenAI as _LangfuseOpenAI
        return _LangfuseOpenAI
    except Exception:                           # noqa: BLE001
        return default_cls


def update_span(**kwargs: Any) -> None:
    """Set input/output/metadata on the current observation (best-effort)."""
    if ENABLED and _client is not None:
        try:
            _client.update_current_span(**kwargs)
        except Exception:                       # noqa: BLE001
            pass


def update_trace(**kwargs: Any) -> None:
    """Set trace-level attributes (name, session_id, user_id, tags, input/output)."""
    if ENABLED and _client is not None:
        try:
            _client.update_current_trace(**kwargs)
        except Exception:                       # noqa: BLE001
            pass


def score(name: str, value, data_type: str | None = None, comment: str | None = None) -> None:
    """Attach a score to the *current* trace (e.g. recall/faithfulness on an
    eval-question trace). Best-effort; no-op when tracing is off."""
    if not (ENABLED and _client is not None):
        return
    kw: dict[str, Any] = {"name": name, "value": value}
    if data_type:
        kw["data_type"] = data_type
    if comment:
        kw["comment"] = comment
    try:
        _client.score_current_trace(**kw)
    except Exception:                           # noqa: BLE001
        pass


def score_session(session_id: str, name: str, value, data_type: str | None = None,
                  comment: str | None = None) -> None:
    """Attach a run-level aggregate score to a whole session (e.g. an eval run's
    mean recall@k). Best-effort; no-op when tracing is off."""
    if not (ENABLED and _client is not None):
        return
    kw: dict[str, Any] = {"name": name, "value": value, "session_id": session_id}
    if data_type:
        kw["data_type"] = data_type
    if comment:
        kw["comment"] = comment
    try:
        _client.create_score(**kw)
    except Exception:                           # noqa: BLE001
        pass


def flush() -> None:
    """Send any buffered events — call before a short-lived process exits."""
    if ENABLED and _client is not None:
        try:
            _client.flush()
        except Exception:                       # noqa: BLE001
            pass


def status() -> str:
    if ENABLED:
        return f"enabled -> {config.LANGFUSE_HOST}"
    return "disabled (set LANGFUSE_PUBLIC_KEY + LANGFUSE_SECRET_KEY to enable)"
