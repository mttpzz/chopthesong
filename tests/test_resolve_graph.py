from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from src import resolver
from src.llm import INTERNAL_MESSAGE
from src.resolver import build_resolve_graph
from src.schemas import LyricsSummary, QueryExtraction, WikiSampleList

NAS = "11111111-1111-1111-1111-111111111111"
FLASH = "22222222-2222-2222-2222-222222222222"
SAMPLE = "33333333-3333-3333-3333-333333333333"


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
    """Deezer, LRCLIB and Wikipedia responses; one source can be made to fail."""

    def __init__(self, *, deezer_down: bool = False) -> None:
        self.deezer_down = deezer_down

    async def get_json(self, url: str, params: dict | None = None) -> Any:
        if "api.deezer.com" in url:
            if self.deezer_down:
                raise RuntimeError("Deezer is down")
            if "/album/" in url:
                return {"genres": {"data": [{"name": "Hip-Hop"}]}}
            return {"data": [{"artist": {"name": "Nas"}, "album": {"id": 7}}]}
        if "lrclib.net" in url:
            return [
                {
                    "trackName": "The Message",
                    "artistName": "Nas",
                    "duration": 234,
                    "plainLyrics": "These are the lyrics",
                    "instrumental": False,
                }
            ]
        if "wikipedia.org" in url:
            if params and params.get("list") == "search":
                return {"query": {"search": [{"title": "The Message"}]}}
            return {"query": {"pages": {"1": {"extract": "The Message is a song by Nas."}}}}
        raise AssertionError(f"unexpected URL: {url}")


class FakeLlm:
    """Structured-output fake for extraction, the summary and Wikipedia samples."""

    def __init__(self) -> None:
        self.schema: type = QueryExtraction
        self.calls: list[str] = []

    def with_structured_output(self, schema: type, *, method: str | None = None) -> FakeLlm:
        self.schema = schema
        return self

    async def ainvoke(self, messages: list) -> Any:
        last = messages[-1]
        content = last["content"] if isinstance(last, dict) else last.content
        self.calls.append(self.schema.__name__)
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
    def __init__(self, recordings: list[dict], http: FakeHttp | None = None) -> None:
        self.musicbrainz = FakeMusicBrainz(recordings)
        self.llm = FakeLlm()
        self.http = http or FakeHttp()
        self.settings = SimpleNamespace(report_depth=2)


async def test_unambiguous_track_is_resolved_without_interrupt() -> None:
    graph = build_resolve_graph(
        FakeRuntime([recording(NAS, "The Message", "Nas")]), InMemorySaver()
    )

    result = await graph.ainvoke(
        {"query": "The Message", "artist": "Nas"}, {"configurable": {"thread_id": "auto"}}
    )

    # Step 3a: a clear match now goes through gather instead of ending the graph.
    assert "__interrupt__" not in result
    assert result["resolution"] == "auto"
    assert result["chosen"]["mbid"] == NAS


async def test_ambiguous_track_interrupts_and_resumes_with_choice() -> None:
    graph = build_resolve_graph(
        FakeRuntime(
            [
                recording(NAS, "The Message", "Nas", "1996-07-02"),
                recording(FLASH, "The Message", "Grandmaster Flash", "1982-07-01"),
            ]
        ),
        InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "ambiguous"}}

    result = await graph.ainvoke({"query": "The Message"}, config)

    assert "__interrupt__" in result
    candidates = result["__interrupt__"][0].value["candidates"]
    assert len(candidates) == 2

    flash_index = next(
        index for index, item in enumerate(candidates) if item["track"]["mbid"] == FLASH
    )
    result = await graph.ainvoke(Command(resume={"index": flash_index}), config)

    # Step 3a: the chosen track now goes through gather instead of ending.
    assert result["resolution"] == "user"
    assert result["chosen"]["mbid"] == FLASH


async def test_cancel_ends_without_choice() -> None:
    graph = build_resolve_graph(
        FakeRuntime(
            [
                recording(NAS, "The Message", "Nas"),
                recording(FLASH, "The Message", "Grandmaster Flash"),
            ]
        ),
        InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "cancel"}}

    await graph.ainvoke({"query": "The Message"}, config)
    result = await graph.ainvoke(Command(resume={"cancel": True}), config)

    assert result["resolution"] == "cancelled"
    assert result["chosen"] is None
    assert "genres" not in result  # cancellation must not reach gather


async def test_not_found_ends_without_interrupt() -> None:
    graph = build_resolve_graph(FakeRuntime([]), InMemorySaver())

    result = await graph.ainvoke(
        {"query": "Nothing at all"}, {"configurable": {"thread_id": "empty"}}
    )

    assert "__interrupt__" not in result
    assert result["resolution"] == "not_found"
    assert result["chosen"] is None
    assert "genres" not in result


async def test_auto_match_gathers_every_source() -> None:
    graph = build_resolve_graph(
        FakeRuntime([recording(NAS, "The Message", "Nas")]), InMemorySaver()
    )

    result = await graph.ainvoke(
        {"query": "The Message", "artist": "Nas"},
        {"configurable": {"thread_id": "gather-auto"}},
    )

    assert result["resolution"] == "auto"
    assert result["genres"] == ["Hip-Hop"]
    assert result["lyrics"] == "These are the lyrics"
    assert result["wiki_text"] == "The Message is a song by Nas."
    assert result["mb_samples"][0]["mbid"] == SAMPLE
    assert result["missing_sources"] == []
    # Step 3b: gather now continues through summarize and verify_samples.
    assert result["lyrics_summary"]["topic"] == "A song about life"
    assert [sample["source"] for sample in result["samples"]] == ["MusicBrainz"]
    # Step 4: verify_samples now continues through expand, which adds the next level.
    assert result["samples"][0]["samples"][0]["mbid"] == SAMPLE
    # Step 5: expand now continues through build_report.
    assert result["report"]["track"]["mbid"] == NAS
    # Step 5b: the nested node repeats the parent MBID, so the report dedupe
    # removes it together with the now-unused nested-samples note.
    assert result["report"]["samples"][0]["samples"] == []
    assert result["report"]["sources"] == [
        "MusicBrainz",
        "Deezer",
        "LRCLIB",
        "Wikipedia",
        "MusicBrainz samples",
    ]
    assert result["report"]["notes"] == []


async def test_failing_source_is_reported_and_others_still_run() -> None:
    graph = build_resolve_graph(
        FakeRuntime(
            [recording(NAS, "The Message", "Nas")], http=FakeHttp(deezer_down=True)
        ),
        InMemorySaver(),
    )

    result = await graph.ainvoke(
        {"query": "The Message", "artist": "Nas"},
        {"configurable": {"thread_id": "gather-deezer-down"}},
    )

    assert result["genres"] == []
    assert result["missing_sources"] == ["Deezer"]
    assert result["lyrics"] is not None
    assert result["wiki_text"] is not None
    assert result["mb_samples"]


async def test_user_choice_goes_through_gather() -> None:
    graph = build_resolve_graph(
        FakeRuntime(
            [
                recording(NAS, "The Message", "Nas"),
                recording(FLASH, "The Message", "Grandmaster Flash"),
            ]
        ),
        InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "gather-user"}}

    result = await graph.ainvoke({"query": "The Message"}, config)
    candidates = result["__interrupt__"][0].value["candidates"]
    flash_index = next(
        index for index, item in enumerate(candidates) if item["track"]["mbid"] == FLASH
    )

    result = await graph.ainvoke(Command(resume={"index": flash_index}), config)

    assert result["resolution"] == "user"
    assert result["chosen"]["mbid"] == FLASH
    assert result["genres"] == ["Hip-Hop"]
    assert result["missing_sources"] == []
    assert result["report"]["track"]["mbid"] == FLASH


async def test_internal_report_error_goes_through_handle_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failing_report():
        async def node(state: dict) -> dict:
            return {"error": INTERNAL_MESSAGE, "resolution": "internal_error"}

        return node

    handled: list[str | None] = []

    def spy_handle(state: dict) -> dict:
        handled.append(state.get("error"))
        return {"error": state.get("error")}

    monkeypatch.setattr(resolver, "_build_report", failing_report)
    monkeypatch.setattr(resolver, "_handle_error", spy_handle)
    graph = build_resolve_graph(
        FakeRuntime([recording(NAS, "The Message", "Nas")]), InMemorySaver()
    )

    result = await graph.ainvoke(
        {"query": "The Message", "artist": "Nas"},
        {"configurable": {"thread_id": "internal-error"}},
    )

    assert handled == [INTERNAL_MESSAGE]
    assert result["resolution"] == "internal_error"
    assert result["error"] == INTERNAL_MESSAGE
    assert "report" not in result
