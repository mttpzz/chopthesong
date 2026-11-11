"""LangGraph that extracts title/artist, searches MusicBrainz and resolves ambiguity."""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from src.config import Settings
from src.http import Http
from src.llm import (
    INTERNAL_MESSAGE,
    build_llm,
    invoke_with_retry,
    llm_failure_message,
)
from src.matching import canonicalize, is_confident, normalize_text, score_candidate
from src.musicbrainz import MusicBrainzClient, parse_recording
from src.report_checks import find_evidence, is_attributed, leaks
from src.schemas import (
    Candidate,
    LyricsSummary,
    QueryExtraction,
    SampleNode,
    TrackRef,
    TrackReport,
    WikiSampleList,
)
from src.tools import SOURCE_ERRORS, build_tools

if TYPE_CHECKING:
    from src.tracing import Tracer

logger = logging.getLogger(__name__)

MAX_CANDIDATES = 8  # More choices than this would overwhelm the user.
Sleep = Callable[[float], Awaitable[None]]

SOURCE_NAMES = ("Deezer", "LRCLIB", "Wikipedia", "MusicBrainz samples")

LYRICS_CHARS = 4000
WIKI_CHARS = 9000

# An empty structured output is retried; a summary discarded by the leak check is not.
SUMMARY_ATTEMPTS = 3

# Safety cap on the whole sample tree; it is not user-configurable.
max_sample_nodes: int = 40


class ResolveState(TypedDict, total=False):
    query: str
    artist: str | None
    candidates: list[dict[str, Any]]
    chosen: dict[str, Any] | None
    # needs_input | auto | user | cancelled | not_found | llm_error | internal_error
    resolution: str
    error: str | None
    notice: str | None
    genres: list[str]
    lyrics: str | None
    wiki_text: str | None
    mb_samples: list[dict[str, Any]]
    missing_sources: list[str]
    lyrics_summary: dict[str, Any] | None
    summary_status: str | None
    samples: list[dict[str, Any]]
    report: dict[str, Any] | None


class Runtime:
    """Long-lived clients shared by the graph and the app process."""

    def __init__(self, settings: Settings, tracer: Tracer | None = None) -> None:
        self.settings = settings
        self.tracer = tracer
        self.http = Http(
            settings.cache_dir,
            ttl=settings.cache_ttl_seconds,
            user_agent=settings.user_agent,
        )
        self.musicbrainz = MusicBrainzClient(
            self.http, min_interval=settings.musicbrainz_min_interval
        )
        self.llm = build_llm(settings)

    async def aclose(self) -> None:
        """Close the shared HTTP client and the model's async Groq client.

        ``ChatGroq`` creates its own ``groq.AsyncGroq`` client, which owns an
        ``httpx.AsyncClient``. LangChain exposes the completions resource as
        ``async_client``; the root Groq client is reachable through its private
        ``_client`` attribute and must be closed explicitly.
        """
        try:
            await self.http.aclose()
        finally:
            resource = getattr(self.llm, "async_client", None)
            client = getattr(resource, "_client", None)
            close = getattr(client, "close", None)
            if callable(close):
                await close()


def _get_tracer(runtime: Runtime) -> Tracer | None:
    """Return the optional tracer, tolerating test doubles without the field."""
    return getattr(runtime, "tracer", None)


def _model_name(runtime: Runtime) -> str:
    """Return the configured model id, or a neutral label for test doubles."""
    settings = getattr(runtime, "settings", None)
    return str(getattr(settings, "groq_model", "unknown"))


def _count(value: Any) -> int:
    """Return the length of a sequence, or zero for missing values."""
    try:
        return len(value or ())
    except TypeError:
        return 0


def _nested_count(items: Any) -> int:
    """Count a sample tree without reading anything but its shape."""
    total = 0
    for item in items or []:
        total += 1
        if isinstance(item, dict):
            total += _nested_count(item.get("samples"))
    return total


def _resolution_label(result: dict[str, Any]) -> str:
    """Return the node's own resolution label, defaulting to ``ok``."""
    value = result.get("resolution")
    return value if isinstance(value, str) and value else "ok"


def _node_metrics(name: str, state: ResolveState, result: Any) -> dict[str, Any]:
    """Translate one node result into structured, non-textual metadata.

    Only fixed keys are read, and only their presence, count or length is
    returned. No value taken from the state is ever forwarded as text.
    """
    if not isinstance(result, dict):
        return {}

    if name == "extract":
        return {
            "title_present": bool(result.get("query")),
            "artist_present": bool(result.get("artist")),
            "retry_used": bool(result.get("notice")),
            "query_chars": len(state.get("query") or ""),
            "resolution": _resolution_label(result),
        }
    if name == "search":
        return {
            "candidate_count": _count(result.get("candidates")),
            "resolution": _resolution_label(result),
        }
    if name in {"decide", "ask"}:
        return {
            "candidate_count": _count(state.get("candidates")),
            "resolution": _resolution_label(result),
        }
    if name == "gather":
        return {
            "genre_count": _count(result.get("genres")),
            "primary_text_chars": len(result.get("lyrics") or ""),
            "encyclopedia_chars": len(result.get("wiki_text") or ""),
            "mb_sample_count": _count(result.get("mb_samples")),
            "missing_source_count": _count(result.get("missing_sources")),
        }
    if name == "summarize":
        summary = result.get("lyrics_summary")
        summary = summary if isinstance(summary, dict) else {}
        return {
            "primary_text_chars": len(state.get("lyrics") or ""),
            "encyclopedia_chars": len(state.get("wiki_text") or ""),
            "summary_present": bool(summary),
            "topic_present": bool(summary.get("topic")),
            "artist_view_present": bool(summary.get("artist_view")),
            "retry_used": bool(result.get("notice")),
            "resolution": _resolution_label(result),
        }
    if name == "verify_samples":
        return {
            "mb_sample_count": _count(state.get("mb_samples")),
            "sample_count": _count(result.get("samples")),
            "retry_used": bool(result.get("notice")),
            "resolution": _resolution_label(result),
        }
    if name == "expand":
        total = _nested_count(result.get("samples"))
        return {
            "top_level_sample_count": _count(state.get("samples")),
            "total_sample_count": total,
            "cap_reached": total >= max_sample_nodes,
        }
    if name == "build_report":
        report = result.get("report")
        report = report if isinstance(report, dict) else {}
        track = report.get("track")
        track = track if isinstance(track, dict) else {}
        genres = report.get("genres")
        genre_names = (
            ", ".join(item for item in genres if isinstance(item, str))
            if isinstance(genres, list)
            else ""
        )
        return {
            "sample_count": _nested_count(report.get("samples")),
            "source_count": _count(report.get("sources")),
            "note_count": _count(report.get("notes")),
            "has_summary": bool(report.get("lyrics_summary")),
            "track_title": track.get("title"),
            "track_artist": track.get("artist"),
            "track_primary_artist": track.get("primary_artist"),
            "track_album": track.get("album"),
            "track_year": track.get("year"),
            "track_duration_ms": track.get("duration_ms"),
            "track_disambiguation": track.get("disambiguation"),
            "track_release_type": track.get("release_primary_type"),
            "track_release_group_id": track.get("release_group_id"),
            "track_release_count": track.get("release_count"),
            "track_is_compilation": track.get("is_compilation"),
            "track_mbid": track.get("mbid"),
            "genres": genre_names,
        }
    if name == "handle_error":
        return {"error_present": bool(result.get("error"))}
    return {}


def _record_node_metrics(
    observation: Any, name: str, state: ResolveState, result: Any
) -> None:
    """Attach metrics without ever letting a broken handle reach the graph."""
    try:
        metrics = _node_metrics(name, state, result)
        record = getattr(observation, "metrics", None)
        if metrics and record is not None:
            record(**metrics)
    except Exception as exc:
        logger.warning(
            "Could not record %s metrics (%s)", name, type(exc).__name__
        )


def _instrument_node(
    name: str,
    node: Callable[[ResolveState], Any],
    tracer: Tracer | None,
) -> Callable[[ResolveState], Any]:
    """Wrap a graph node in a span without changing what the node does."""
    if tracer is None:
        return node

    if inspect.iscoroutinefunction(node):

        async def traced_async(state: ResolveState) -> Any:
            with tracer.node(name) as observation:
                result = await node(state)
                _record_node_metrics(observation, name, state, result)
                return result

        return traced_async

    def traced_sync(state: ResolveState) -> Any:
        with tracer.node(name) as observation:
            result = node(state)
            _record_node_metrics(observation, name, state, result)
            return result

    return traced_sync


async def _invoke_llm(
    runtime: Runtime,
    call: Callable[[], Awaitable[Any]],
    *,
    on_wait: Callable[[str], None] | None = None,
    sleep: Sleep = asyncio.sleep,
) -> Any:
    """Run ``invoke_with_retry`` and trace the model, attempts and outcome."""
    tracer = _get_tracer(runtime)
    if tracer is None:
        return await invoke_with_retry(call, on_wait=on_wait, sleep=sleep)

    with tracer.llm_call(_model_name(runtime)) as observation:

        async def counted() -> Any:
            observation.attempt()
            return await call()

        return await invoke_with_retry(counted, on_wait=on_wait, sleep=sleep)


EXTRACT_SYSTEM_PROMPT = (
    "Extract the song title and artist from the user's request. "
    "Do not invent: return an empty title when the title is unclear and a null artist "
    "when no artist is named."
)

SUMMARIZE_SYSTEM_PROMPT = (
    "You are a music critic. Write in English, clearly and concretely. Rules: "
    "topic = the central subject in 3-8 words, not the song title; "
    "summary = at most 120 words about history, subject and tone; "
    "artist_view = one sentence only when the encyclopedic text explicitly attributes it "
    "to the artist, otherwise null. "
    "Never copy lyrics or phrases: always paraphrase and never use more than four "
    "consecutive words from the lyrics. Do not invent anything."
)

WIKI_SAMPLES_SYSTEM_PROMPT = (
    "You are a music researcher. List ONLY the samples or interpolations used by the "
    "track that the provided text states explicitly. Ignore covers, remixes and mere "
    "references or tributes. Do not invent. Always return the structured object with a "
    "'samples' list; when there is nothing, return an empty 'samples' list, never a bare "
    "array."
)


def _extract(runtime: Runtime, *, sleep: Sleep = asyncio.sleep):
    """Build the node that asks the LLM for title and artist, retrying short 429s."""

    async def node(state: ResolveState) -> dict[str, Any]:
        original = (state.get("query") or "").strip()
        user_artist = (state.get("artist") or "").strip() or None
        if not original:
            return {
                "query": "",
                "artist": user_artist,
                "error": None,
                "resolution": "",
                "notice": None,
            }

        notice: str | None = None

        def on_wait(message: str) -> None:
            nonlocal notice
            notice = message

        try:
            extractor = runtime.llm.with_structured_output(QueryExtraction)

            async def call() -> Any:
                return await extractor.ainvoke(
                    [
                        {"role": "system", "content": EXTRACT_SYSTEM_PROMPT},
                        {"role": "user", "content": original},
                    ]
                )

            extracted = QueryExtraction.model_validate(
                await _invoke_llm(runtime, call, on_wait=on_wait, sleep=sleep)
            )
        except Exception as exc:  # provider and parsing failures must not crash the graph
            return {
                "error": llm_failure_message(exc),
                "resolution": "llm_error",
                "notice": notice,
            }

        title = extracted.title.strip() or original
        artist = user_artist or (extracted.artist or "").strip() or None
        return {
            "query": title,
            "artist": artist,
            "error": None,
            "resolution": "",
            "notice": notice,
        }

    return node


def _search(runtime: Runtime):
    """Build the node that queries MusicBrainz and ranks the results."""

    async def node(state: ResolveState) -> dict[str, Any]:
        query = (state.get("query") or "").strip()
        artist = (state.get("artist") or "").strip() or None
        if not query:
            return {"candidates": [], "chosen": None, "resolution": "not_found"}

        recordings = await runtime.musicbrainz.search_recordings(query, artist=artist, limit=25)
        if not recordings and artist:
            recordings = await runtime.musicbrainz.search_recordings(query, limit=25)

        seen: set[str] = set()
        candidates: list[Candidate] = []
        for recording in recordings:
            track = parse_recording(recording)
            if not track.mbid or track.mbid in seen:
                continue
            seen.add(track.mbid)
            candidates.append(score_candidate(track, query_title=query, query_artist=artist))

        ranked = canonicalize(candidates)[:MAX_CANDIDATES]
        return {
            "candidates": [candidate.model_dump(mode="json") for candidate in ranked],
            "chosen": None,
            "resolution": "" if ranked else "not_found",
        }

    return node


def _decide(state: ResolveState) -> dict[str, Any]:
    """Auto-pick the best candidate, or flag that the user must choose."""
    candidates = [Candidate.model_validate(item) for item in state.get("candidates") or []]
    if not candidates:
        return {"chosen": None, "resolution": "not_found"}
    if is_confident(candidates):
        return {"chosen": candidates[0].track.model_dump(mode="json"), "resolution": "auto"}
    return {"chosen": None, "resolution": "needs_input"}


def _ask(state: ResolveState) -> dict[str, Any]:
    """Pause the graph and let the user pick one of the candidates."""
    candidates = state.get("candidates") or []
    answer = interrupt(
        {
            "query": state.get("query"),
            "artist": state.get("artist"),
            "candidates": candidates,
        }
    )
    index = answer.get("index") if isinstance(answer, dict) else None
    if not isinstance(index, int) or not 0 <= index < len(candidates):
        return {"chosen": None, "resolution": "cancelled"}
    return {"chosen": candidates[index]["track"], "resolution": "user"}


def _source_result(name: str, value: Any, *, default: Any) -> Any:
    """Return a source's value, or ``default`` when the tool raised.

    Expected source failures are already handled inside the tool, so an exception here
    is a bug: log it with its traceback, but keep the rest of the report.
    """
    if not isinstance(value, Exception):
        return value
    if not isinstance(value, SOURCE_ERRORS):
        logger.error("Unexpected error in source %s", name, exc_info=value)
    return default


def _gather(runtime: Runtime):
    """Build the node that fetches every external source in parallel."""

    tools = build_tools(runtime)

    async def node(state: ResolveState) -> dict[str, Any]:
        chosen = state.get("chosen")
        if not chosen:
            return {
                "genres": [],
                "lyrics": None,
                "wiki_text": None,
                "mb_samples": [],
                "missing_sources": list(SOURCE_NAMES),
            }

        track = TrackRef.model_validate(chosen)
        genres, lyrics, wiki_text, mb_samples = await asyncio.gather(
            tools["get_genres"].ainvoke(
                {"title": track.title, "artist": track.artist}
            ),
            tools["get_lyrics"].ainvoke(
                {
                    "title": track.title,
                    "artist": track.artist,
                    "duration_ms": track.duration_ms,
                }
            ),
            tools["get_wiki_text"].ainvoke(
                {"title": track.title, "artist": track.artist}
            ),
            tools["get_mb_samples"].ainvoke({"mbid": track.mbid or ""}),
            return_exceptions=True,
        )

        # A tool swallows source errors itself. Anything that reaches here is a bug:
        # it is logged, and the source counts as missing so the other sources survive.
        genres = _source_result("Deezer", genres, default=[])
        lyrics = _source_result("LRCLIB", lyrics, default=None)
        wiki_text = _source_result("Wikipedia", wiki_text, default=None)
        mb_samples = _source_result("MusicBrainz samples", mb_samples, default=[])

        missing: list[str] = []
        if not genres:
            missing.append("Deezer")
        if not lyrics:
            missing.append("LRCLIB")
        if not wiki_text:
            missing.append("Wikipedia")
        if not mb_samples:
            missing.append("MusicBrainz samples")

        return {
            "genres": genres or [],
            "lyrics": lyrics or None,
            "wiki_text": wiki_text or None,
            "mb_samples": mb_samples or [],
            "missing_sources": missing,
        }

    return node


def _summarize(runtime: Runtime, *, sleep: Sleep = asyncio.sleep):
    """Build the node that asks the LLM for a lyrics summary."""

    async def node(state: ResolveState) -> dict[str, Any]:
        chosen = state.get("chosen")
        lyrics = state.get("lyrics") or ""
        wiki_text = state.get("wiki_text") or ""
        if not chosen or (not lyrics and not wiki_text):
            return {"lyrics_summary": None, "summary_status": "not_attempted"}

        track = TrackRef.model_validate(chosen)
        notice: str | None = None

        def on_wait(message: str) -> None:
            nonlocal notice
            notice = message

        try:
            summarizer = runtime.llm.with_structured_output(LyricsSummary)

            async def call() -> Any:
                return await summarizer.ainvoke(
                    [
                        {"role": "system", "content": SUMMARIZE_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": _summarize_prompt(track, lyrics, wiki_text),
                        },
                    ]
                )

            summary = None
            for _ in range(SUMMARY_ATTEMPTS):
                summary = LyricsSummary.model_validate(
                    await _invoke_llm(runtime, call, on_wait=on_wait, sleep=sleep)
                )
                if _clean(summary.topic) and _clean(summary.summary):
                    break
        except Exception as exc:  # provider and parsing failures must not crash the graph
            return {
                "error": llm_failure_message(exc),
                "resolution": "llm_error",
                "notice": notice,
            }

        assert summary is not None
        topic = _clean(summary.topic)
        body = _clean(summary.summary)
        artist_view = _clean(summary.artist_view) or None

        leaked = False
        if leaks(topic, lyrics):
            topic = ""
            leaked = True
        if leaks(body, lyrics):
            body = ""
            leaked = True
        if artist_view and (
            leaks(artist_view, lyrics) or not is_attributed(artist_view, wiki_text)
        ):
            artist_view = None

        if leaked:
            summary_status = "leak_check"
        elif not topic or not body:
            summary_status = "empty"
        else:
            summary_status = "ok"

        return {
            "lyrics_summary": LyricsSummary(
                topic=topic, summary=body, artist_view=artist_view
            ).model_dump(),
            "summary_status": summary_status,
            "notice": notice,
        }

    return node


def _verify_samples(runtime: Runtime, *, sleep: Sleep = asyncio.sleep):
    """Build the node that keeps only samples backed by a source."""

    async def node(state: ResolveState) -> dict[str, Any]:
        chosen = state.get("chosen")
        if not chosen:
            return {"samples": []}

        track = TrackRef.model_validate(chosen)
        lyrics = state.get("lyrics") or ""
        wiki_text = state.get("wiki_text") or ""

        nodes = [_mb_node(item) for item in state.get("mb_samples") or []]
        nodes = [sample for sample in nodes if sample.title]

        notice: str | None = None
        if wiki_text:

            def on_wait(message: str) -> None:
                nonlocal notice
                notice = message

            try:
                # json_schema handles an empty list; function calling rejects [] as a non-call.
                extractor = runtime.llm.with_structured_output(
                    WikiSampleList, method="json_schema"
                )

                async def call() -> Any:
                    return await extractor.ainvoke(
                        [
                            {"role": "system", "content": WIKI_SAMPLES_SYSTEM_PROMPT},
                            {
                                "role": "user",
                                "content": _wiki_samples_prompt(track, wiki_text),
                            },
                        ]
                    )

                found = WikiSampleList.model_validate(
                    await _invoke_llm(runtime, call, on_wait=on_wait, sleep=sleep)
                )
            except Exception as exc:  # provider and parsing failures must not crash the graph
                return {
                    "error": llm_failure_message(exc),
                    "resolution": "llm_error",
                    "notice": notice,
                }

            for candidate in found.samples:
                title = _clean(candidate.title)
                if not title:
                    continue
                evidence = find_evidence(wiki_text, title)
                if evidence is None or leaks(evidence, lyrics):
                    continue
                artist = _clean(candidate.artist) or None
                if artist and not is_attributed(artist, wiki_text):
                    artist = None
                nodes.append(
                    SampleNode(
                        title=title,
                        artist=artist,
                        relation="sample",
                        source="Wikipedia",
                        confidence="medium",
                    )
                )

        nodes = _dedupe_first_level(nodes, track)
        result: dict[str, Any] = {
            "samples": [sample.model_dump(mode="json") for sample in nodes]
        }
        if wiki_text:
            result["notice"] = notice
        return result

    return node


def _expand(runtime: Runtime):
    """Build the node that expands the sample tree through MusicBrainz only."""

    tools = build_tools(runtime)

    async def node(state: ResolveState) -> dict[str, Any]:
        top = [SampleNode.model_validate(item) for item in state.get("samples") or []]
        missing = list(state.get("missing_sources") or [])
        if not top:
            if "MusicBrainz samples" not in missing:
                missing.append("MusicBrainz samples")
            return {"samples": [], "missing_sources": missing}

        chosen = state.get("chosen") or {}
        seen = {chosen["mbid"]} if chosen.get("mbid") else set()
        remaining_nodes = max_sample_nodes - len(top)
        depth = _report_depth(runtime)

        async def resolve_sample(sample: SampleNode) -> SampleNode:
            """Resolve an unidentified Wikipedia sample through MusicBrainz."""
            if sample.source != "Wikipedia" or sample.mbid:
                return sample
            try:
                recordings = await runtime.musicbrainz.search_recordings(
                    sample.title, artist=sample.artist, limit=15
                )
                candidates: list[Candidate] = []
                for recording in recordings or []:
                    track = parse_recording(recording)
                    if not track.mbid:
                        continue
                    candidates.append(
                        score_candidate(
                            track,
                            query_title=sample.title,
                            query_artist=sample.artist,
                        )
                    )
                ranked = canonicalize(candidates)
            except Exception as exc:  # a dead search must not break the tree
                if not isinstance(exc, SOURCE_ERRORS):
                    logger.error(
                        "Unexpected error resolving a Wikipedia sample", exc_info=exc
                    )
                return sample

            if not is_confident(ranked):
                return sample
            best = ranked[0].track
            sample.mbid = best.mbid
            sample.year = best.year
            sample.album = best.album
            return sample

        async def expand_sample(sample: SampleNode, level: int) -> None:
            """Add one MusicBrainz level to a sample, respecting cycles and the cap."""
            nonlocal remaining_nodes
            if level > depth or not sample.mbid or sample.mbid in seen:
                return
            seen.add(sample.mbid)
            if remaining_nodes <= 0:
                return

            children = await _sample_children(tools, sample.mbid)
            for child in children:
                if remaining_nodes <= 0:
                    break
                sample.samples.append(child)
                remaining_nodes -= 1
                await expand_sample(child, level + 1)

        if depth > 0:
            for sample in top:
                await resolve_sample(sample)
            for sample in top:
                await expand_sample(sample, 1)

        return {"samples": [sample.model_dump(mode="json") for sample in top]}

    return node


def _mb_node(item: dict[str, Any]) -> SampleNode:
    """Turn one gathered MusicBrainz sample into a high-confidence report node."""
    return SampleNode(
        title=_clean(item.get("title")) or "Unknown",
        artist=_clean(item.get("artist")) or None,
        year=item.get("year"),
        album=_clean(item.get("album")) or None,
        mbid=item.get("mbid"),
        relation="sample",
        source="MusicBrainz",
        confidence="high",
    )


def _dedupe_first_level(nodes: list[SampleNode], track: TrackRef) -> list[SampleNode]:
    """Drop the chosen track and duplicates, keeping the first (MusicBrainz) node."""
    chosen_title = normalize_text(track.title)
    seen: set[str] = set()
    result: list[SampleNode] = []
    for node in nodes:
        key = normalize_text(node.title)
        if not key or key == chosen_title or key in seen:
            continue
        if track.mbid and node.mbid == track.mbid:
            continue
        seen.add(key)
        result.append(node)
    return result


async def _sample_children(tools: dict[str, Any], mbid: str) -> list[SampleNode]:
    """Fetch one MusicBrainz sample level; expected source failures stay silent."""
    try:
        items = await tools["get_mb_samples"].ainvoke({"mbid": mbid})
        return [_mb_node(item) for item in items or []]
    except Exception as exc:  # a dead source must not break the tree
        if not isinstance(exc, SOURCE_ERRORS):
            logger.error(
                "Unexpected error expanding a MusicBrainz sample", exc_info=exc
            )
        return []


def _report_depth(runtime: Runtime) -> int:
    """Read the validated expansion depth, with a safe default for test doubles."""
    settings = getattr(runtime, "settings", None)
    return max(0, min(int(getattr(settings, "report_depth", 2)), 3))


def _iter_samples(nodes: list[SampleNode]):
    """Yield every node of the sample tree, depth first."""
    for node in nodes:
        yield node
        yield from _iter_samples(node.samples)


# Only the two report confidences exist; an unknown value loses to both.
_CONFIDENCE_RANK = {"high": 2, "medium": 1}


def _sample_identity(node: SampleNode) -> tuple[str, str]:
    """Normalized title and artist used when at least one node lacks an MBID."""
    return normalize_text(node.title), normalize_text(node.artist or "")


def _same_sample(left: SampleNode, right: SampleNode) -> bool:
    """Apply the pairwise identity rule from step 5b."""
    if left.mbid and right.mbid:
        return left.mbid == right.mbid
    return _sample_identity(left) == _sample_identity(right)


def _collect_samples(
    nodes: list[SampleNode],
    depth: int,
    prefix: tuple[int, ...],
    collected: list[tuple[SampleNode, int, tuple[int, ...]]],
) -> None:
    """Record every node with its depth and path in pre-order (visit order)."""
    for index, node in enumerate(nodes):
        path = prefix + (index,)
        collected.append((node, depth, path))
        _collect_samples(node.samples, depth + 1, path, collected)


def _dedupe_samples(nodes: list[SampleNode]) -> list[SampleNode]:
    """Keep one occurrence per track, preferring the strongest source.

    The identity rule is transitive: a node without an MBID can bridge two
    different MBIDs, so those occurrences share one group. The winner is the
    strongest confidence, then the shallowest node, then the first in visit
    order. A losing node takes its whole subtree with it.
    """
    collected: list[tuple[SampleNode, int, tuple[int, ...]]] = []
    _collect_samples(nodes, 1, (), collected)
    if not collected:
        return []

    parents = list(range(len(collected)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for left in range(len(collected)):
        for right in range(left + 1, len(collected)):
            if _same_sample(collected[left][0], collected[right][0]):
                union(left, right)

    groups: dict[int, list[int]] = {}
    for index in range(len(collected)):
        groups.setdefault(find(index), []).append(index)

    winners: set[tuple[int, ...]] = set()
    for members in groups.values():
        best = min(
            members,
            key=lambda index: (
                -_CONFIDENCE_RANK.get(collected[index][0].confidence, 0),
                collected[index][1],
                index,
            ),
        )
        winners.add(collected[best][2])

    def rebuild(current: list[SampleNode], prefix: tuple[int, ...]) -> list[SampleNode]:
        kept: list[SampleNode] = []
        for position, node in enumerate(current):
            path = prefix + (position,)
            if path not in winners:
                continue
            children = rebuild(node.samples, path)
            kept.append(node.model_copy(update={"samples": children}))
        return kept

    return rebuild(nodes, ())


def _build_report():
    """Build the node that assembles the final report from the graph state."""

    async def node(state: ResolveState) -> dict[str, Any]:
        if state.get("error"):
            return {}
        if not state.get("chosen") or state.get("resolution") not in {"auto", "user"}:
            return {}

        try:
            track = TrackRef.model_validate(state["chosen"])
            samples = [
                SampleNode.model_validate(item) for item in state.get("samples") or []
            ]
            # Step 5b: the report keeps one occurrence per track. Sources and
            # the nested-samples note must reflect the deduplicated tree.
            samples = _dedupe_samples(samples)
            summary_value = state.get("lyrics_summary")
            summary = (
                LyricsSummary.model_validate(summary_value) if summary_value else None
            )
            summary_status = state.get("summary_status")
            missing = list(state.get("missing_sources") or [])
            flat = list(_iter_samples(samples))
            has_mb_sample = any(sample.source == "MusicBrainz" for sample in flat)

            sources = ["MusicBrainz"]
            for name in SOURCE_NAMES:
                if name == "MusicBrainz samples":
                    if has_mb_sample:
                        sources.append(name)
                elif name not in missing:
                    sources.append(name)

            notes = [f"No data from {name}." for name in missing if name not in sources]
            if summary_status == "leak_check":
                notes.append(
                    "Summary unavailable: the summary copied lyrics and was discarded."
                )
            elif summary_status == "empty":
                notes.append(
                    "Summary unavailable: the model returned an empty summary."
                )
            elif summary_status == "not_attempted":
                notes.append(
                    "Summary not attempted: no lyrics or Wikipedia text."
                )
            elif (
                summary_status is None
                and summary is not None
                and not summary.topic
                and not summary.summary
            ):
                # Backward-compatible path for states built before summary_status.
                notes.append(
                    "Summary unavailable: the lyrics or the Wikipedia text did not "
                    "pass the checks."
                )
            if any(sample.samples for sample in flat):
                notes.append(
                    "Samples beyond the first level come from MusicBrainz only."
                )

            report = TrackReport(
                track=track,
                genres=list(state.get("genres") or []),
                lyrics_summary=summary,
                summary_status=summary_status,
                samples=samples,
                sources=sources,
                notes=notes,
            )
            return {"report": report.model_dump(mode="json")}
        except Exception as exc:  # malformed state is a bug, not a source failure
            logger.error("Unexpected error while building the report", exc_info=exc)
            return {"error": INTERNAL_MESSAGE, "resolution": "internal_error"}

    return node


def _summarize_prompt(track: TrackRef, lyrics: str, wiki_text: str) -> str:
    """Build the user prompt from the capped lyrics and Wikipedia extract."""
    parts = [f'Song: "{track.title}" by {track.artist}.']
    if lyrics:
        parts.append(f"LYRICS:\n<lyrics>\n{lyrics[:LYRICS_CHARS]}\n</lyrics>")
    if wiki_text:
        parts.append(
            f"ENCYCLOPEDIC DESCRIPTION:\n<wiki>\n{wiki_text[:WIKI_CHARS]}\n</wiki>"
        )
    return "\n\n".join(parts)


def _wiki_samples_prompt(track: TrackRef, wiki_text: str) -> str:
    """Build the user prompt for Wikipedia sample extraction."""
    return (
        f'Song: "{track.title}" by {track.artist}.\n\n'
        f"TEXT:\n<wiki>\n{wiki_text[:WIKI_CHARS]}\n</wiki>\n\n"
        "List only the samples or interpolations explicitly stated above. "
        "Return the structured object, with an empty samples list when none are mentioned."
    )


def _clean(value: Any) -> str:
    """Collapse whitespace and turn ``None`` into an empty string."""
    return " ".join(str(value or "").split())


def _failure_route(state: ResolveState) -> str:
    """Send LLM failures and internal bugs to handle_error, everything else on."""
    resolution = state.get("resolution")
    return "error" if resolution in {"llm_error", "internal_error"} else "ok"


def _handle_error(state: ResolveState) -> dict[str, Any]:
    """End the graph with only the user-facing error message."""
    return {"error": state.get("error")}


def _route(state: ResolveState) -> str:
    """Ask for a choice, gather data for a chosen track, or finish."""
    resolution = state.get("resolution")
    if resolution == "needs_input":
        return "ask"
    if resolution in {"auto", "user"}:
        return "gather"
    return "done"


def build_resolve_graph(
    runtime: Runtime,
    checkpointer: BaseCheckpointSaver | None = None,
    *,
    sleep: Sleep = asyncio.sleep,
):
    """Compile the resolve graph, optionally with a checkpointer for HITL.

    ``sleep`` is injectable so tests can exercise retries without waiting.
    """
    graph = StateGraph(ResolveState)
    tracer = _get_tracer(runtime)
    graph.add_node(
        "extract", _instrument_node("extract", _extract(runtime, sleep=sleep), tracer)
    )
    graph.add_node(
        "handle_error", _instrument_node("handle_error", _handle_error, tracer)
    )
    graph.add_node("search", _instrument_node("search", _search(runtime), tracer))
    graph.add_node("decide", _instrument_node("decide", _decide, tracer))
    graph.add_node("ask", _instrument_node("ask", _ask, tracer))
    graph.add_node("gather", _instrument_node("gather", _gather(runtime), tracer))
    graph.add_node(
        "summarize",
        _instrument_node("summarize", _summarize(runtime, sleep=sleep), tracer),
    )
    graph.add_node(
        "verify_samples",
        _instrument_node(
            "verify_samples", _verify_samples(runtime, sleep=sleep), tracer
        ),
    )
    graph.add_node("expand", _instrument_node("expand", _expand(runtime), tracer))
    graph.add_node(
        "build_report", _instrument_node("build_report", _build_report(), tracer)
    )

    graph.add_edge(START, "extract")
    graph.add_conditional_edges(
        "extract", _failure_route, {"error": "handle_error", "ok": "search"}
    )
    graph.add_edge("handle_error", END)
    graph.add_edge("search", "decide")
    graph.add_conditional_edges(
        "decide", _route, {"ask": "ask", "gather": "gather", "done": END}
    )
    graph.add_conditional_edges("ask", _route, {"gather": "gather", "done": END})
    graph.add_edge("gather", "summarize")
    graph.add_conditional_edges(
        "summarize", _failure_route, {"error": "handle_error", "ok": "verify_samples"}
    )
    graph.add_conditional_edges(
        "verify_samples", _failure_route, {"error": "handle_error", "ok": "expand"}
    )
    graph.add_edge("expand", "build_report")
    graph.add_conditional_edges(
        "build_report", _failure_route, {"error": "handle_error", "ok": END}
    )
    return graph.compile(checkpointer=checkpointer)
