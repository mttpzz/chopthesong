from __future__ import annotations

from typing import Any

import groq
import httpx
from langgraph.checkpoint.memory import InMemorySaver

from src.resolver import _verify_samples, build_resolve_graph
from src.schemas import (
    LyricsSummary,
    QueryExtraction,
    TrackRef,
    WikiSample,
    WikiSampleList,
)

CHOSEN = "11111111-1111-1111-1111-111111111111"
SAMPLE = "22222222-2222-2222-2222-222222222222"
TRACK = TrackRef(mbid=CHOSEN, title="The Message", artist="Nas")
LYRICS = "alpha beta gamma delta epsilon zeta eta theta"
WIKI = (
    "The Message is a song by Nas. "
    "It samples the track Sample Song by Sample Artist. "
    "The production also interpolates a melody from Cafe del Mar. "
    "The album was released in 2002."
)
MB_SAMPLE = {
    "title": "Sample Song",
    "artist": "Sample Artist",
    "year": 1990,
    "album": "Sample Album",
    "mbid": SAMPLE,
}


class FakeSleep:
    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)


def rate_limited(retry_after: str | None = None) -> groq.RateLimitError:
    headers = {"retry-after": retry_after} if retry_after is not None else {}
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    return groq.RateLimitError(
        "429 SECRET-TOKEN-123",
        response=httpx.Response(429, headers=headers, request=request),
        body=None,
    )


class FakeWikiLlm:
    def __init__(
        self,
        result: WikiSampleList | None = None,
        errors: list[Exception] | None = None,
    ) -> None:
        self.result = result
        self.errors = list(errors or [])
        self.calls = 0

    def with_structured_output(
        self, schema: type[WikiSampleList], *, method: str | None = None
    ) -> FakeWikiLlm:
        assert schema is WikiSampleList
        assert method == "json_schema"
        return self

    async def ainvoke(self, messages: list[Any]) -> WikiSampleList:
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        assert self.result is not None
        return self.result


class FakeRuntime:
    def __init__(self, llm: FakeWikiLlm) -> None:
        self.llm = llm


def state(
    *,
    mb_samples: list[dict] | None = None,
    wiki_text: str | None = WIKI,
    chosen: dict | None = None,
    lyrics: str | None = LYRICS,
) -> dict[str, Any]:
    return {
        "chosen": TRACK.model_dump() if chosen is None else chosen,
        "lyrics": lyrics,
        "wiki_text": wiki_text,
        "mb_samples": list(mb_samples or []),
    }


async def test_musicbrainz_sample_is_high_confidence() -> None:
    llm = FakeWikiLlm(WikiSampleList(samples=[]))
    node = _verify_samples(FakeRuntime(llm), sleep=FakeSleep())

    result = await node(state(mb_samples=[MB_SAMPLE], wiki_text=None))

    assert result["samples"] == [
        {
            "title": "Sample Song",
            "artist": "Sample Artist",
            "year": 1990,
            "album": "Sample Album",
            "mbid": SAMPLE,
            "relation": "sample",
            "source": "MusicBrainz",
            "confidence": "high",
            "samples": [],
        }
    ]
    assert llm.calls == 0


async def test_wikipedia_sample_needs_evidence_but_does_not_keep_it() -> None:
    llm = FakeWikiLlm(
        WikiSampleList(
            samples=[
                WikiSample(title="Sample Song", artist="Sample Artist"),
                WikiSample(title="Cafe del Mar"),
            ]
        )
    )
    node = _verify_samples(FakeRuntime(llm), sleep=FakeSleep())

    result = await node(state(mb_samples=[]))

    assert [sample["source"] for sample in result["samples"]] == [
        "Wikipedia",
        "Wikipedia",
    ]
    assert [sample["confidence"] for sample in result["samples"]] == [
        "medium",
        "medium",
    ]
    assert all("evidence" not in sample for sample in result["samples"])


async def test_wikipedia_hallucination_is_dropped() -> None:
    llm = FakeWikiLlm(
        WikiSampleList(samples=[WikiSample(title="Invented Track", artist="Nobody")])
    )
    node = _verify_samples(FakeRuntime(llm), sleep=FakeSleep())

    result = await node(state(mb_samples=[]))

    assert result["samples"] == []


async def test_duplicate_keeps_the_musicbrainz_version() -> None:
    llm = FakeWikiLlm(
        WikiSampleList(samples=[WikiSample(title="sample song", artist="Sample Artist")])
    )
    node = _verify_samples(FakeRuntime(llm), sleep=FakeSleep())

    result = await node(state(mb_samples=[MB_SAMPLE]))

    assert len(result["samples"]) == 1
    assert result["samples"][0]["source"] == "MusicBrainz"
    assert result["samples"][0]["confidence"] == "high"


async def test_chosen_track_is_removed_from_samples() -> None:
    llm = FakeWikiLlm(
        WikiSampleList(
            samples=[
                WikiSample(title="The Message"),
                WikiSample(title="Sample Song"),
            ]
        )
    )
    node = _verify_samples(FakeRuntime(llm), sleep=FakeSleep())
    wiki_text = (
        "The Message samples the drum break from Sample Song. "
        "The Message is the title track."
    )

    result = await node(
        state(
            mb_samples=[{"title": "The Message", "mbid": "another-id"}],
            wiki_text=wiki_text,
        )
    )

    assert [sample["title"] for sample in result["samples"]] == ["Sample Song"]


async def test_llm_error_writes_no_samples() -> None:
    llm = FakeWikiLlm(errors=[rate_limited("7200")])
    node = _verify_samples(FakeRuntime(llm), sleep=FakeSleep())

    result = await node(state(mb_samples=[MB_SAMPLE]))

    assert result["resolution"] == "llm_error"
    assert "2 hours" in result["error"]
    assert "samples" not in result


class GraphLlm:
    """Extraction and summary succeed; the Wikipedia sample call is rate-limited."""

    def __init__(self) -> None:
        self.schema: type | None = None

    def with_structured_output(self, schema: type, *, method: str | None = None) -> GraphLlm:
        self.schema = schema
        return self

    async def ainvoke(self, messages: list[Any]) -> Any:
        if self.schema is QueryExtraction:
            content = messages[-1]["content"]
            return QueryExtraction(title=content, artist=None)
        if self.schema is LyricsSummary:
            return LyricsSummary(topic="A theme", summary="A short summary.")
        assert self.schema is WikiSampleList
        raise rate_limited("7200")


class GraphMusicBrainz:
    async def search_recordings(
        self, title: str, *, artist: str | None = None, limit: int = 15
    ) -> list[dict]:
        return [
            {
                "id": CHOSEN,
                "score": 100,
                "title": "The Message",
                "length": 234000,
                "artist-credit": [{"name": "Nas", "joinphrase": ""}],
                "releases": [{"title": "Album", "date": "2002-11-19"}],
            }
        ]

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


class GraphHttp:
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
                    "plainLyrics": LYRICS,
                    "instrumental": False,
                }
            ]
        if "wikipedia.org" in url:
            if params and params.get("list") == "search":
                return {"query": {"search": [{"title": "The Message"}]}}
            return {"query": {"pages": {"1": {"extract": WIKI}}}}
        raise AssertionError(f"unexpected URL: {url}")


class GraphRuntime:
    def __init__(self) -> None:
        self.llm = GraphLlm()
        self.musicbrainz = GraphMusicBrainz()
        self.http = GraphHttp()


async def test_verify_samples_error_goes_to_handle_error() -> None:
    graph = build_resolve_graph(GraphRuntime(), InMemorySaver(), sleep=FakeSleep())

    result = await graph.ainvoke(
        {"query": "The Message", "artist": "Nas"},
        {"configurable": {"thread_id": "verify-error"}},
    )

    assert result["resolution"] == "llm_error"
    assert "2 hours" in result["error"]
    assert "samples" not in result
    # Step 5: an LLM failure must not reach build_report.
    assert "report" not in result
    # gather already ran, so the raw sample data is in state but was not published.
    assert result["mb_samples"]
