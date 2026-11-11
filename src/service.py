"""Synchronous adapter between Streamlit and the async resolve graph.

Streamlit reruns its script synchronously, while the graph is async. Every
public function here owns one ``asyncio.run`` call, one ``Runtime`` and one
graph compilation, then closes the runtime before returning. The checkpointer
is supplied by the caller and lives in ``st.session_state`` for the session.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Callable
from contextlib import nullcontext
from typing import Any, cast

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.types import Command

from src.config import get_settings
from src.llm import llm_failure_message
from src.resolver import Runtime, build_resolve_graph
from src.tracing import Tracer, build_tracer

logger = logging.getLogger(__name__)

RuntimeFactory = Callable[[], Runtime]

# Kept here so a resume with neither an index nor a cancel fails loudly.
_RESUME_ERROR = "resume_search needs index or cancel=True."

# Sentinel: the caller did not choose, so the service decides whether to trace.
_DEFAULT_TRACER = object()


def start_search(
    query: str,
    artist: str | None = None,
    *,
    checkpointer: BaseCheckpointSaver,
    runtime_factory: RuntimeFactory | None = None,
    tracer: Tracer | None | object = _DEFAULT_TRACER,
) -> dict[str, Any]:
    """Run a fresh search on a new thread and return a JSON-compatible result.

    ``runtime_factory`` lets tests inject a fake ``Runtime``; when omitted, the
    function builds one from ``get_settings()``. When ``runtime_factory`` is
    supplied and ``tracer`` is omitted, tracing stays off so tests never touch
    the network. Pass ``tracer=None`` explicitly to disable tracing. The
    runtime is always closed before returning, because it is bound to the event
    loop that just ended.
    """
    return asyncio.run(
        _start_search(
            query,
            artist,
            checkpointer=checkpointer,
            runtime_factory=runtime_factory,
            tracer=tracer,
        )
    )


def resume_search(
    thread_id: str,
    index: int | None = None,
    cancel: bool = False,
    *,
    checkpointer: BaseCheckpointSaver,
    runtime_factory: RuntimeFactory | None = None,
    tracer: Tracer | None | object = _DEFAULT_TRACER,
) -> dict[str, Any]:
    """Resume a paused thread with a chosen candidate or a cancellation.

    ``index`` or ``cancel=True`` is required; ``cancel=True`` takes precedence
    when both are given. The same ``checkpointer`` used by ``start_search`` must
    be passed so the graph can find the interrupted state.
    """
    if cancel:
        resume: dict[str, Any] = {"cancel": True}
    elif index is not None:
        resume = {"index": index}
    else:
        raise ValueError(_RESUME_ERROR)
    return asyncio.run(
        _resume_search(
            thread_id,
            resume,
            checkpointer=checkpointer,
            runtime_factory=runtime_factory,
            tracer=tracer,
        )
    )


async def _start_search(
    query: str,
    artist: str | None,
    *,
    checkpointer: BaseCheckpointSaver,
    runtime_factory: RuntimeFactory | None,
    tracer: Tracer | None | object,
) -> dict[str, Any]:
    payload = {"query": query, "artist": artist}
    return await _invoke(
        payload,
        thread_id=str(uuid.uuid4()),
        checkpointer=checkpointer,
        runtime_factory=runtime_factory,
        resume=None,
        tracer=tracer,
    )


async def _resume_search(
    thread_id: str,
    resume: dict[str, Any],
    *,
    checkpointer: BaseCheckpointSaver,
    runtime_factory: RuntimeFactory | None,
    tracer: Tracer | None | object,
) -> dict[str, Any]:
    return await _invoke(
        None,
        thread_id=thread_id,
        checkpointer=checkpointer,
        runtime_factory=runtime_factory,
        resume=resume,
        tracer=tracer,
    )


async def _invoke(
    payload: dict[str, Any] | None,
    *,
    thread_id: str,
    checkpointer: BaseCheckpointSaver,
    runtime_factory: RuntimeFactory | None,
    resume: dict[str, Any] | None,
    tracer: Tracer | None | object,
) -> dict[str, Any]:
    """Compile the graph, run one call and translate the state into a payload."""
    runtime: Runtime | None = None
    effective_tracer: Tracer | None = None
    try:
        if runtime_factory is None:
            settings = get_settings()
            effective_tracer = (
                build_tracer(settings)
                if tracer is _DEFAULT_TRACER
                else cast(Tracer | None, tracer)
            )
            runtime = Runtime(settings, tracer=effective_tracer)
        else:
            runtime = runtime_factory()
            if tracer is _DEFAULT_TRACER:
                effective_tracer = getattr(runtime, "tracer", None)
            else:
                effective_tracer = cast(Tracer | None, tracer)
                runtime.tracer = effective_tracer

        graph = build_resolve_graph(runtime, checkpointer)
        config = {"configurable": {"thread_id": thread_id}}
        with _root_span(effective_tracer) as root:
            if resume is None:
                result = await graph.ainvoke(payload, config)
            else:
                result = await graph.ainvoke(Command(resume=resume), config)
            outcome = _payload(thread_id, result)
            _record_root_metrics(root, outcome)
        return outcome
    except Exception as exc:  # never let a traceback or a key reach the UI
        return {
            "thread_id": thread_id,
            "status": "error",
            "error": llm_failure_message(exc),
        }
    finally:
        close = getattr(runtime, "aclose", None) if runtime is not None else None
        if close is not None:
            try:
                await close()
            except Exception:  # closing must not hide a successful result
                logger.warning("Failed to close the runtime", exc_info=True)
        if effective_tracer is not None:
            flush = getattr(effective_tracer, "flush", None)
            if flush is not None:
                try:
                    flush()
                except Exception as exc:  # tracing must never break a search
                    logger.warning(
                        "Failed to flush the tracer (%s)", type(exc).__name__
                    )


def _root_span(tracer: Tracer | None):
    """Return a root span when tracing is enabled, or a no-op context."""
    return tracer.node("graph") if tracer is not None else nullcontext()


def _record_root_metrics(observation: Any, payload: dict[str, Any]) -> None:
    """Record the final status on the root span using structured fields only."""
    record = getattr(observation, "metrics", None)
    if record is None:
        return
    try:
        record(
            status=payload.get("status"),
            report_present=bool(payload.get("report")),
        )
    except Exception as exc:
        logger.warning("Failed to record graph metrics (%s)", type(exc).__name__)


def _payload(thread_id: str, result: dict[str, Any]) -> dict[str, Any]:
    """Map the graph state to the small dict the UI consumes."""
    interrupted = result.get("__interrupt__")
    if interrupted:
        value = getattr(interrupted[0], "value", None) or {}
        payload: dict[str, Any] = {
            "thread_id": thread_id,
            "status": "needs_input",
            "candidates": value.get("candidates") or [],
            "query": value.get("query"),
            "artist": value.get("artist"),
        }
    elif result.get("error"):
        payload = {
            "thread_id": thread_id,
            "status": "error",
            "error": result["error"],
            "resolution": result.get("resolution"),
        }
    elif result.get("report"):
        payload = {
            "thread_id": thread_id,
            "status": "done",
            "report": result["report"],
            "resolution": result.get("resolution"),
        }
    else:
        payload = {
            "thread_id": thread_id,
            "status": "done",
            "resolution": result.get("resolution"),
        }

    notice = result.get("notice")
    if notice is not None:
        payload["notice"] = notice
    return payload
