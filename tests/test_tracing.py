from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver

from src import tracing
from src.config import Settings
from src.llm import UNAVAILABLE_MESSAGE, LlmUnavailable
from src.schemas import LyricsSummary, QueryExtraction, WikiSampleList
from src.service import start_search
from src.tracing import LangfuseTracer, build_tracer

NAS = "11111111-1111-1111-1111-111111111111"
SAMPLE = "33333333-3333-3333-3333-333333333333"
RELEASE_GROUP = "22222222-2222-2222-2222-222222222222"

LYRICS_MARKER = "LYRICS_MARKER_123"
WIKI_MARKER = "WIKI_MARKER_456"


class FakeSpan:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.updates: list[dict[str, Any]] = []

    def update(self, **kwargs: Any) -> None:
        self.updates.append(kwargs)


class FakeManager:
    def __init__(self, span: FakeSpan) -> None:
        self.span = span

    def __enter__(self) -> FakeSpan:
        return self.span

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        return False


class FakeLangfuse:
    """In-memory stand-in for the Langfuse client; never touches the network."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.spans: list[FakeSpan] = []
        self.flushes = 0
        self.fail_flush = False
        self.fail_start = False
        self.fail_update = False

    def start_as_current_observation(self, **kwargs: Any) -> FakeManager:
        if self.fail_start:
            raise RuntimeError("client is down")
        span = FakeSpan(**kwargs)
        if self.fail_update:
            def broken_update(**update: Any) -> None:
                raise RuntimeError("update is down")

            span.update = broken_update  # type: ignore[method-assign]
        self.spans.append(span)
        return FakeManager(span)

    def flush(self) -> None:
        self.flushes += 1
        if self.fail_flush:
            raise RuntimeError("flush is down")

    def records(self) -> list[dict[str, Any]]:
        return [
            {"start": span.kwargs, "updates": span.updates} for span in self.spans
        ]


def settings(**overrides: Any) -> SimpleNamespace:
    values: dict[str, Any] = {
        "langfuse_public_key": None,
        "langfuse_secret_key": None,
        "langfuse_host": "https://cloud.langfuse.com",
        "groq_model": "test-model",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def recording(mbid: str, title: str, artist: str, year: str = "2002-11-19") -> dict:
    return {
        "id": mbid,
        "score": 100,
        "title": title,
        "length": 234000,
        "disambiguation": "single version",
        "artist-credit": [{"name": artist, "joinphrase": ""}],
        "releases": [
            {
                "title": "Album",
                "date": year,
                "release-group": {
                    "id": RELEASE_GROUP,
                    "primary-type": "Album",
                },
            }
        ],
    }


class FakeMusicBrainz:
    def __init__(self, recordings: list[dict]) -> None:
        self.recordings = recordings

    async def search_recordings(
        self, title: str, *, artist: str | None = None, limit: int = 15
    ) -> list[dict]:
        if artist:
            return [
                item
                for item in self.recordings
                if artist.lower() in item["artist-credit"][0]["name"].lower()
            ]
        return self.recordings

    async def lookup_recording(self, mbid: str, *, inc: str = "") -> dict:
        return {
            "relations": [
                {
                    "type": "samples material",
                    "direction": "forward",
                    "recording": {
                        "id": SAMPLE,
                        "title": "Sample Song",
                        "artist-credit": [{"name": "Sample Artist", "joinphrase": ""}],
                        "releases": [{"title": "Sample Album", "date": "1990-01-01"}],
                    },
                }
            ]
        }


class FakeHttp:
    async def get_json(self, url: str, params: dict | None = None) -> Any:
        if "api.deezer.com" in url:
            if "/album/" in url:
                return {"genres": {"data": [{"name": "Hip-Hop"}]}}
            return {"data": [{"artist": {"name": "Nas"}, "album": {"id": 7}}]}
        if "lrclib.net" in url:
            return [
                {
                    "trackName": "The Message",
                    "artistName": "Nas",
                    "duration": 234,
                    "plainLyrics": LYRICS_MARKER,
                    "instrumental": False,
                }
            ]
        if "wikipedia.org" in url:
            if params and params.get("list") == "search":
                return {"query": {"search": [{"title": "The Message"}]}}
            return {"query": {"pages": {"1": {"extract": WIKI_MARKER}}}}
        raise AssertionError(f"unexpected URL: {url}")


class FakeLlm:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.schema: type = QueryExtraction

    def with_structured_output(
        self, schema: type, *, method: str | None = None
    ) -> FakeLlm:
        self.schema = schema
        return self

    async def ainvoke(self, messages: list) -> Any:
        if self.fail:
            raise LlmUnavailable(UNAVAILABLE_MESSAGE)
        last = messages[-1]
        content = last["content"] if isinstance(last, dict) else last.content
        if self.schema is QueryExtraction:
            return QueryExtraction(title=content, artist=None)
        if self.schema is LyricsSummary:
            return LyricsSummary(
                topic="A song about life",
                summary="A short summary.",
                artist_view=None,
            )
        assert self.schema is WikiSampleList
        return WikiSampleList(samples=[])


class FakeRuntime:
    def __init__(self, recordings: list[dict], *, fail_llm: bool = False) -> None:
        self.musicbrainz = FakeMusicBrainz(recordings)
        self.llm = FakeLlm(fail=fail_llm)
        self.http = FakeHttp()
        self.settings = SimpleNamespace(report_depth=1, groq_model="test-model")
        self.tracer = None
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


def factory(recordings: list[dict], *, fail_llm: bool = False):
    def make() -> FakeRuntime:
        return FakeRuntime(recordings, fail_llm=fail_llm)

    return make


def test_build_tracer_is_none_without_keys() -> None:
    assert build_tracer(settings()) is None
    assert build_tracer(settings(langfuse_public_key="pk")) is None
    assert build_tracer(settings(langfuse_secret_key="sk")) is None


def test_build_tracer_uses_the_settings_host(
    monkeypatch: Any,
) -> None:
    created: list[FakeLangfuse] = []

    def fake_client(**kwargs: Any) -> FakeLangfuse:
        client = FakeLangfuse(**kwargs)
        created.append(client)
        return client

    monkeypatch.setattr(tracing, "Langfuse", fake_client)

    tracer = build_tracer(
        settings(
            langfuse_public_key="pk-test",
            langfuse_secret_key="sk-test",
            langfuse_host="https://example.invalid",
        )
    )

    assert tracer is not None
    assert created[0].kwargs["base_url"] == "https://example.invalid"
    assert created[0].kwargs["public_key"] == "pk-test"
    assert created[0].kwargs["secret_key"] == "sk-test"


def test_tracing_enabled_requires_both_keys() -> None:
    assert (
        Settings(
            _env_file=None,
            langfuse_public_key="pk-test",
            langfuse_secret_key="sk-test",
        ).tracing_enabled
        is True
    )
    assert (
        Settings(
            _env_file=None,
            langfuse_public_key="pk-test",
            langfuse_secret_key="",
        ).tracing_enabled
        is False
    )
    assert (
        Settings(
            _env_file=None,
            langfuse_public_key=None,
            langfuse_secret_key=None,
        ).tracing_enabled
        is False
    )


def test_full_search_traces_metadata_without_lyrics_or_wiki_text() -> None:
    client = FakeLangfuse()
    tracer = LangfuseTracer(client)

    result = start_search(
        "The Message",
        "Nas",
        checkpointer=InMemorySaver(),
        runtime_factory=factory([recording(NAS, "The Message", "Nas")]),
        tracer=tracer,
    )

    assert result["status"] == "done"
    assert result["report"]["track"]["mbid"] == NAS
    assert client.flushes == 1

    names = [span.kwargs["name"] for span in client.spans]
    assert "graph" in names
    for node in (
        "extract",
        "search",
        "decide",
        "gather",
        "summarize",
        "verify_samples",
        "expand",
        "build_report",
    ):
        assert node in names
    assert any(name.startswith("llm:") for name in names)

    traced_payload = json.dumps(client.records())
    assert LYRICS_MARKER not in traced_payload
    assert WIKI_MARKER not in traced_payload

    metadata_keys = {
        key
        for span in client.spans
        for update in span.updates
        for key in (update.get("metadata") or {})
    }
    assert "lyrics" not in metadata_keys
    assert "wiki_text" not in metadata_keys
    assert "evidence" not in metadata_keys
    assert "query" not in metadata_keys


def _merged_metadata(client: FakeLangfuse) -> dict[str, dict[str, Any]]:
    """Return the last merged metadata for each observation name."""
    by_name: dict[str, dict[str, Any]] = {}
    for span in client.spans:
        merged: dict[str, Any] = {}
        for update in span.updates:
            merged.update(update.get("metadata") or {})
        by_name[span.kwargs["name"]] = merged
    return by_name


def test_node_metrics_are_structured_and_public_only() -> None:
    client = FakeLangfuse()
    tracer = LangfuseTracer(client)

    result = start_search(
        "The Message",
        "Nas",
        checkpointer=InMemorySaver(),
        runtime_factory=factory([recording(NAS, "The Message", "Nas")]),
        tracer=tracer,
    )

    assert result["status"] == "done"
    metadata = _merged_metadata(client)

    assert metadata["graph"]["status"] == "done"
    assert metadata["graph"]["report_present"] is True
    assert metadata["extract"]["query_chars"] == len("The Message")
    assert metadata["search"]["candidate_count"] == 1
    assert metadata["gather"]["genre_count"] == 1
    assert metadata["gather"]["primary_text_chars"] == len(LYRICS_MARKER)
    assert metadata["gather"]["encyclopedia_chars"] == len(WIKI_MARKER)
    assert metadata["verify_samples"]["mb_sample_count"] == 1
    assert metadata["build_report"]["sample_count"] == 1
    assert metadata["build_report"]["source_count"] >= 1
    assert metadata["build_report"]["track_title"] == "The Message"
    assert metadata["build_report"]["track_artist"] == "Nas"
    assert metadata["build_report"]["track_primary_artist"] == "Nas"
    assert metadata["build_report"]["track_album"] == "Album"
    assert metadata["build_report"]["track_year"] == 2002
    assert metadata["build_report"]["track_duration_ms"] == 234000
    assert metadata["build_report"]["track_disambiguation"] == "single version"
    assert metadata["build_report"]["track_release_type"] == "Album"
    assert metadata["build_report"]["track_release_group_id"] == RELEASE_GROUP
    assert metadata["build_report"]["track_release_count"] == 1
    assert metadata["build_report"]["track_is_compilation"] is False
    assert metadata["build_report"]["track_mbid"] == NAS
    assert metadata["build_report"]["genres"] == "Hip-Hop"

    values = [
        value
        for span in client.spans
        for update in span.updates
        for value in (update.get("metadata") or {}).values()
    ]
    assert values
    assert all(isinstance(value, (bool, int, str)) for value in values)
    assert LYRICS_MARKER not in json.dumps(client.records())
    assert WIKI_MARKER not in json.dumps(client.records())


def test_tracer_drops_unlisted_metrics_and_sanitizes_public_text() -> None:
    client = FakeLangfuse()
    tracer = LangfuseTracer(client)

    with tracer.node("probe") as observation:
        observation.metrics(
            candidate_count=2,
            title_present=True,
            resolution="auto",
            status="done",
            retry_used="yes",
            sample_count=-1,
            track_title="  The   Message ",
            track_album="Album\nDeluxe",
            track_year=1975,
            track_duration_ms=123000,
            track_release_count=3,
            track_is_compilation=True,
            track_disambiguation=None,
            genres="rock, pop",
            lyrics=LYRICS_MARKER,
            wiki_text=WIKI_MARKER,
            evidence=WIKI_MARKER,
            query=LYRICS_MARKER,
            secret=LYRICS_MARKER,
        )

    metadata = client.spans[0].updates[-1]["metadata"]
    assert metadata["candidate_count"] == 2
    assert metadata["title_present"] is True
    assert metadata["resolution"] == "auto"
    assert metadata["status"] == "done"
    assert metadata["track_title"] == "The Message"
    assert metadata["track_album"] == "Album Deluxe"
    assert metadata["track_year"] == 1975
    assert metadata["track_duration_ms"] == 123000
    assert metadata["track_release_count"] == 3
    assert metadata["track_is_compilation"] is True
    assert "track_disambiguation" not in metadata
    assert metadata["genres"] == "rock, pop"
    assert "retry_used" not in metadata
    assert "sample_count" not in metadata
    assert "lyrics" not in metadata
    assert "wiki_text" not in metadata
    assert "evidence" not in metadata
    assert "query" not in metadata
    assert "secret" not in metadata
    assert LYRICS_MARKER not in json.dumps(client.records())
    assert WIKI_MARKER not in json.dumps(client.records())


def test_public_text_metrics_are_capped() -> None:
    client = FakeLangfuse()
    tracer = LangfuseTracer(client)

    with tracer.node("probe") as observation:
        observation.metrics(track_title="x" * 500)

    assert client.spans[0].updates[-1]["metadata"]["track_title"] == "x" * 200


def test_llm_failure_is_traced_by_type_only() -> None:
    client = FakeLangfuse()
    tracer = LangfuseTracer(client)

    result = start_search(
        "The Message",
        "Nas",
        checkpointer=InMemorySaver(),
        runtime_factory=factory(
            [recording(NAS, "The Message", "Nas")], fail_llm=True
        ),
        tracer=tracer,
    )

    assert result["status"] == "error"
    assert result["error"] == UNAVAILABLE_MESSAGE

    errors = [
        update["metadata"]
        for span in client.spans
        for update in span.updates
        if update.get("metadata", {}).get("outcome") == "error"
    ]
    assert errors
    assert errors[0]["error_type"] == "LlmUnavailable"
    assert UNAVAILABLE_MESSAGE not in json.dumps(client.records())


def test_flush_error_does_not_break_the_search() -> None:
    client = FakeLangfuse()
    client.fail_flush = True
    tracer = LangfuseTracer(client)

    result = start_search(
        "The Message",
        "Nas",
        checkpointer=InMemorySaver(),
        runtime_factory=factory([recording(NAS, "The Message", "Nas")]),
        tracer=tracer,
    )

    assert result["status"] == "done"
    assert client.flushes == 1


def test_tracer_start_and_update_errors_do_not_break_the_search() -> None:
    client = FakeLangfuse()
    client.fail_start = True
    tracer = LangfuseTracer(client)

    result = start_search(
        "The Message",
        "Nas",
        checkpointer=InMemorySaver(),
        runtime_factory=factory([recording(NAS, "The Message", "Nas")]),
        tracer=tracer,
    )

    assert result["status"] == "done"
    assert client.spans == []

    client.fail_start = False
    client.fail_update = True
    tracer = LangfuseTracer(client)

    result = start_search(
        "The Message",
        "Nas",
        checkpointer=InMemorySaver(),
        runtime_factory=factory([recording(NAS, "The Message", "Nas")]),
        tracer=tracer,
    )

    assert result["status"] == "done"
