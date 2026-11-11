"""Metadata-only Langfuse tracing for the resolve graph.

A traced operation only ever receives a node name or a model id. The tracer
records timings, outcomes, exception types, retry attempts, a closed set of
structured counters and flags, and public catalog fields (title, artist,
album, year, MBID, genres). It never gets the graph state, prompts, model
messages, song text or encyclopedia text. Counters such as
``primary_text_chars`` are lengths: the text itself is never passed here.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from typing import Any, Protocol

from langfuse import Langfuse

from src.config import Settings

logger = logging.getLogger(__name__)

# Structured metadata is an allowlist by construction: the graph may only send
# the fields below, with an integer, boolean, known label or public catalog
# string as their value.
MetricValue = int | bool | str

ALLOWED_METRICS = frozenset(
    {
        "resolution",
        "status",
        "title_present",
        "artist_present",
        "retry_used",
        "query_chars",
        "candidate_count",
        "genre_count",
        "primary_text_chars",
        "encyclopedia_chars",
        "mb_sample_count",
        "missing_source_count",
        "summary_present",
        "topic_present",
        "artist_view_present",
        "sample_count",
        "top_level_sample_count",
        "total_sample_count",
        "cap_reached",
        "source_count",
        "note_count",
        "has_summary",
        "error_present",
        "report_present",
        "track_title",
        "track_artist",
        "track_primary_artist",
        "track_album",
        "track_year",
        "track_duration_ms",
        "track_disambiguation",
        "track_release_type",
        "track_release_group_id",
        "track_release_count",
        "track_is_compilation",
        "track_mbid",
        "genres",
    }
)

# Public catalog fields may contain text, unlike the counters above. They are
# still an allowlist: the raw user query is deliberately absent.
PUBLIC_TEXT_METRICS = frozenset(
    {
        "track_title",
        "track_artist",
        "track_primary_artist",
        "track_album",
        "track_disambiguation",
        "track_release_type",
        "track_release_group_id",
        "track_mbid",
        "genres",
    }
)

TEXT_VALUE_LIMIT = 200  # Titles and names are short; this blocks pasted text.

# Labels are internal enumerations; free-form strings are never accepted.
ALLOWED_LABELS = frozenset(
    {
        "ok",
        "auto",
        "user",
        "needs_input",
        "cancelled",
        "not_found",
        "llm_error",
        "internal_error",
        "done",
        "error",
    }
)


def _clean_metrics(values: dict[str, Any]) -> dict[str, MetricValue]:
    """Keep only allowlisted fields with a safe, non-negative scalar value."""
    clean: dict[str, MetricValue] = {}
    for key, value in values.items():
        if key not in ALLOWED_METRICS or value is None:
            continue
        if isinstance(value, bool):
            clean[key] = value
        elif isinstance(value, int):
            if value >= 0:
                clean[key] = value
        elif key in {"resolution", "status"} and isinstance(value, str):
            if value in ALLOWED_LABELS:
                clean[key] = value
        elif key in PUBLIC_TEXT_METRICS and isinstance(value, str):
            text = " ".join(value.split())[:TEXT_VALUE_LIMIT]
            if text:
                clean[key] = text
    return clean


class ObservationHandle(Protocol):
    """Mutable handle yielded to a traced operation."""

    attempts: int

    def attempt(self) -> None:
        """Count one attempt of an LLM call."""

    def metrics(self, **values: MetricValue) -> None:
        """Record structured, non-textual metadata for the observation."""


class Tracer(Protocol):
    """Minimal tracing surface used by the graph and the service."""

    def node(self, name: str) -> AbstractContextManager[Any]:
        """Trace one graph node."""

    def llm_call(self, model: str) -> AbstractContextManager[ObservationHandle]:
        """Trace one LLM call, including its retry attempts."""

    def flush(self) -> None:
        """Send buffered observations without blocking for long."""


class _NullObservation:
    """No-op handle used when tracing is disabled or fails to start."""

    def __init__(self) -> None:
        self.attempts = 0

    def attempt(self) -> None:
        self.attempts += 1

    def metrics(self, **values: Any) -> None:
        """Drop metrics: nothing is sent when tracing is unavailable."""


class _Observation:
    """Thin wrapper that keeps observation failures away from the graph."""

    def __init__(self, span: Any) -> None:
        self._span = span
        self._metrics: dict[str, MetricValue] = {}
        self.attempts = 0

    def attempt(self) -> None:
        self.attempts += 1

    def metrics(self, **values: Any) -> None:
        clean = _clean_metrics(values)
        provided = sum(1 for value in values.values() if value is not None)
        if len(clean) < provided:
            logger.warning(
                "Langfuse dropped %d invalid metric fields", provided - len(clean)
            )
        self._metrics.update(clean)

    def structured_metrics(self) -> dict[str, MetricValue]:
        return dict(self._metrics)

    def update(self, **kwargs: Any) -> None:
        try:
            self._span.update(**kwargs)
        except Exception as exc:
            logger.warning(
                "Langfuse observation update failed (%s)", type(exc).__name__
            )


class LangfuseTracer:
    """Tracer backed by a Langfuse client.

    Every public method is failure-tolerant: an SDK error is logged with its
    type only, then ignored, so a broken tracer can never interrupt a search.
    """

    def __init__(self, client: Langfuse) -> None:
        self._client = client

    def node(self, name: str) -> AbstractContextManager[Any]:
        """Return a span context manager for a graph node."""
        return self._observation(name=name, as_type="span")

    def llm_call(self, model: str) -> AbstractContextManager[ObservationHandle]:
        """Return a generation context manager for an LLM call."""
        return self._observation(name=f"llm:{model}", as_type="generation", model=model)

    def flush(self) -> None:
        """Flush buffered observations, ignoring Langfuse and network failures."""
        try:
            self._client.flush()
        except Exception as exc:
            logger.warning("Langfuse flush failed (%s)", type(exc).__name__)

    @contextmanager
    def _observation(
        self, *, name: str, as_type: str, model: str | None = None
    ) -> Iterator[_Observation]:
        try:
            manager = self._client.start_as_current_observation(
                name=name, as_type=as_type, model=model
            )
            span = manager.__enter__()
        except Exception as exc:
            logger.warning(
                "Langfuse could not start observation %s (%s)",
                name,
                type(exc).__name__,
            )
            yield _NullObservation()
            return

        observation = _Observation(span)
        operation_error: BaseException | None = None
        try:
            yield observation
        except BaseException as exc:  # the graph must see its own exception
            operation_error = exc
            self._finish(observation, error=exc)
        else:
            self._finish(observation)
        finally:
            try:
                manager.__exit__(
                    type(operation_error) if operation_error is not None else None,
                    operation_error,
                    (
                        operation_error.__traceback__
                        if operation_error is not None
                        else None
                    ),
                )
            except Exception as exc:
                logger.warning(
                    "Langfuse could not close observation %s (%s)",
                    name,
                    type(exc).__name__,
                )
            if operation_error is not None:
                raise operation_error

    def _finish(
        self, observation: _Observation, *, error: BaseException | None = None
    ) -> None:
        metadata: dict[str, Any] = {"outcome": "ok"}
        metadata.update(observation.structured_metrics())
        if error is None:
            if observation.attempts:
                metadata["attempts"] = observation.attempts
            observation.update(metadata=metadata)
            return

        error_type = type(error).__name__
        interrupted = error_type == "GraphInterrupt"
        metadata["outcome"] = "interrupted" if interrupted else "error"
        metadata["error_type"] = error_type
        if observation.attempts:
            metadata["attempts"] = observation.attempts
        observation.update(
            metadata=metadata,
            level="WARNING" if interrupted else "ERROR",
            status_message=error_type,
        )


def build_tracer(settings: Settings) -> Tracer | None:
    """Build the Langfuse tracer, or ``None`` when tracing is disabled.

    The client is created only when both keys are set. Construction failures
    are logged by exception type and never raised to the caller.
    """
    public_key = (getattr(settings, "langfuse_public_key", None) or "").strip()
    secret_key = (getattr(settings, "langfuse_secret_key", None) or "").strip()
    enabled = getattr(settings, "tracing_enabled", None)
    if enabled is None:
        enabled = bool(public_key and secret_key)
    if not enabled:
        return None

    try:
        client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            base_url=getattr(
                settings, "langfuse_host", "https://cloud.langfuse.com"
            ),
            timeout=3,
            flush_at=16,
            flush_interval=1.0,
        )
    except Exception as exc:
        logger.warning("Langfuse tracing could not start (%s)", type(exc).__name__)
        return None
    return LangfuseTracer(client)
