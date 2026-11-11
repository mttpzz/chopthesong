from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver

from src.llm import UNAVAILABLE_MESSAGE, LlmUnavailable
from src.schemas import LyricsSummary, QueryExtraction, WikiSampleList
from src.service import resume_search, start_search

NAS = "11111111-1111-1111-1111-111111111111"
FLASH = "22222222-2222-2222-2222-222222222222"
SAMPLE = "33333333-3333-3333-3333-333333333333"

SECRET_LYRICS = "SECRET LYRICS MUST NOT LEAK"
SECRET_WIKI = "SECRET WIKI MUST NOT LEAK"


def recording(mbid: str, title: str, artist: str, year: str = "2002-11-19") -> dict:
    return {
        "id": mbid,
        "score": 100,
        "title": title,
        "length": 234000,
        "artist-credit": [{"name": artist, "joinphrase": ""}],
        "releases": [{"title": "Album", "date": year}],
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
    """Deezer, LRCLIB and Wikipedia responses with recognizable secret text."""

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
                    "plainLyrics": SECRET_LYRICS,
                    "instrumental": False,
                }
            ]
        if "wikipedia.org" in url:
            if params and params.get("list") == "search":
                return {"query": {"search": [{"title": "The Message"}]}}
            return {"query": {"pages": {"1": {"extract": SECRET_WIKI}}}}
        raise AssertionError(f"unexpected URL: {url}")


class FakeLlm:
    """Structured-output fake that can fail on the first call."""

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
    """Duck-typed Runtime: no network, but the same attributes the graph uses."""

    def __init__(self, recordings: list[dict], *, fail_llm: bool = False) -> None:
        self.musicbrainz = FakeMusicBrainz(recordings)
        self.llm = FakeLlm(fail=fail_llm)
        self.http = FakeHttp()
        self.settings = SimpleNamespace(report_depth=2)
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


def factory(recordings: list[dict], *, fail_llm: bool = False):
    def make() -> FakeRuntime:
        return FakeRuntime(recordings, fail_llm=fail_llm)

    return make


def ambiguous() -> list[dict]:
    return [
        recording(NAS, "The Message", "Nas", "1996-07-02"),
        recording(FLASH, "The Message", "Grandmaster Flash", "1982-07-01"),
    ]


def keys(value: Any):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from keys(child)


def test_start_search_with_auto_match_returns_a_report() -> None:
    result = start_search(
        "The Message",
        "Nas",
        checkpointer=InMemorySaver(),
        runtime_factory=factory([recording(NAS, "The Message", "Nas")]),
    )

    assert result["status"] == "done"
    assert result["resolution"] == "auto"
    assert result["report"]["track"]["mbid"] == NAS
    assert result["thread_id"]


def test_start_search_with_versions_returns_candidates() -> None:
    result = start_search(
        "The Message",
        None,
        checkpointer=InMemorySaver(),
        runtime_factory=factory(ambiguous()),
    )

    assert result["status"] == "needs_input"
    assert result["candidates"]
    assert result["query"] == "The Message"
    assert result["thread_id"]


def test_resume_with_an_index_uses_the_same_checkpointer() -> None:
    saver = InMemorySaver()
    runtime = factory(ambiguous())
    started = start_search("The Message", None, checkpointer=saver, runtime_factory=runtime)
    assert started["status"] == "needs_input"

    resumed = resume_search(
        started["thread_id"], index=0, checkpointer=saver, runtime_factory=runtime
    )

    assert resumed["status"] == "done"
    assert resumed["resolution"] == "user"
    assert resumed["report"]["track"]["mbid"] in {NAS, FLASH}
    assert resumed["thread_id"] == started["thread_id"]


def test_resume_with_cancel_finishes_without_a_report() -> None:
    saver = InMemorySaver()
    runtime = factory(ambiguous())
    started = start_search("The Message", None, checkpointer=saver, runtime_factory=runtime)

    resumed = resume_search(
        started["thread_id"], cancel=True, checkpointer=saver, runtime_factory=runtime
    )

    assert resumed["status"] == "done"
    assert resumed["resolution"] == "cancelled"
    assert "report" not in resumed


def test_llm_error_returns_the_retry_message_without_a_report() -> None:
    result = start_search(
        "The Message",
        "Nas",
        checkpointer=InMemorySaver(),
        runtime_factory=factory([recording(NAS, "The Message", "Nas")], fail_llm=True),
    )

    assert result["status"] == "error"
    assert result["error"] == UNAVAILABLE_MESSAGE
    assert "report" not in result


def test_a_new_checkpointer_does_not_find_the_previous_thread() -> None:
    runtime = factory(ambiguous())
    started = start_search(
        "The Message", None, checkpointer=InMemorySaver(), runtime_factory=runtime
    )
    assert started["status"] == "needs_input"

    other = resume_search(
        started["thread_id"],
        index=0,
        checkpointer=InMemorySaver(),
        runtime_factory=runtime,
    )

    assert "report" not in other
    assert other["thread_id"] == started["thread_id"]


def test_result_is_json_serializable_and_never_contains_raw_text() -> None:
    result = start_search(
        "The Message",
        "Nas",
        checkpointer=InMemorySaver(),
        runtime_factory=factory([recording(NAS, "The Message", "Nas")]),
    )

    payload = json.dumps(result)

    assert SECRET_LYRICS not in payload
    assert SECRET_WIKI not in payload
    all_keys = set(keys(result))
    assert "lyrics" not in all_keys
    assert "wiki_text" not in all_keys
