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
from src.http import Http
from src.llm import llm_failure_message
from src.resolver import Runtime, build_resolve_graph
from src.spotify import client_from_settings
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
            if outcome.get("status") == "done" and outcome.get("report"):
                await add_spotify_links(
                    outcome["report"],
                    getattr(runtime, "settings", None),
                    getattr(runtime, "http", None),
                )
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


# Spotify links are optional decoration: at most twenty sample searches per report.
_MAX_SAMPLE_SEARCHES = 20


def _walk_samples(nodes: list[dict[str, Any]]):
    """Yield every sample node depth first, parents before their children."""
    for node in nodes:
        yield node
        yield from _walk_samples(node.get("samples") or [])


async def _add_sample_track_links(
    nodes: list[dict[str, Any]], client: Any, budget: int
) -> int:
    """Spend the budget on sample titles first, as in the previous step."""
    for node in _walk_samples(nodes):
        if budget <= 0 or client.blocked:
            return budget
        node["spotify_url"] = await client.find_track_url(
            node.get("title"), node.get("artist")
        )
        budget -= 1
    return budget


async def _add_sample_artist_links(
    nodes: list[dict[str, Any]], client: Any, budget: int
) -> int:
    """Use the remaining budget on artists; an empty artist spends nothing."""
    for node in _walk_samples(nodes):
        if budget <= 0 or client.blocked:
            return budget
        artist = str(node.get("artist") or "").strip()
        if not artist:
            continue
        node["spotify_artist_url"] = await client.find_artist_url(artist)
        budget -= 1
    return budget


async def add_spotify_links(
    report: dict[str, Any], settings: Any, http: Any | None = None
) -> None:
    """Attach Spotify search links to a report, outside the graph.

    With missing credentials or search failures the fields stay ``None`` and
    the report renders exactly as before. The function never raises: links are
    optional, and a Spotify outage must not break a finished report.
    """
    if not report:
        return
    client_id = getattr(settings, "spotify_client_id", None)
    client_secret = getattr(settings, "spotify_client_secret", None)
    if not (client_id and client_secret):
        return

    owned_http: Http | None = None
    try:
        if http is None:
            owned_http = Http(
                settings.cache_dir,
                ttl=settings.cache_ttl_seconds,
                user_agent=settings.user_agent,
            )
            http = owned_http
        client = client_from_settings(settings, http)
        if client is None:
            return

        track = report.get("track") or {}
        report["spotify_track_url"] = await client.find_track_url(
            track.get("title"), track.get("artist")
        )
        report["spotify_artist_url"] = await client.find_artist_url(
            track.get("artist")
        )
        report["spotify_album_url"] = await client.find_album_url(
            track.get("album"), track.get("artist")
        )
        remaining = await _add_sample_track_links(
            report.get("samples") or [], client, _MAX_SAMPLE_SEARCHES
        )
        await _add_sample_artist_links(
            report.get("samples") or [], client, remaining
        )
    except Exception as exc:  # links are decoration, never a reason to fail
        logger.warning("Spotify link enrichment failed (%s)", type(exc).__name__)
    finally:
        if owned_http is not None:
            try:
                await owned_http.aclose()
            except Exception:  # a close failure must not replace the report
                logger.warning("Failed to close the Spotify HTTP client", exc_info=True)
