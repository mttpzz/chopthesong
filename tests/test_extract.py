from __future__ import annotations

from typing import Any

import groq
import httpx
import respx
from langgraph.checkpoint.memory import InMemorySaver

from src.llm import RATE_LIMITED_MESSAGE, UNEXPECTED_MESSAGE, WAIT_NOTICE
from src.resolver import _extract, build_resolve_graph
from src.schemas import QueryExtraction


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


class FakeStructuredLlm:
    def __init__(
        self,
        result: QueryExtraction | None = None,
        error: Exception | None = None,
        errors: list[Exception] | None = None,
    ) -> None:
        self.result = result
        self.error = error
        self.errors = list(errors or [])
        self.calls = 0

    def with_structured_output(self, schema: type[QueryExtraction]) -> FakeStructuredLlm:
        assert schema is QueryExtraction
        return self

    async def ainvoke(self, messages: Any) -> QueryExtraction:
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        if self.error is not None:
            raise self.error
        assert self.result is not None
        return self.result


class FakeRuntime:
    def __init__(self, llm: FakeStructuredLlm, musicbrainz: Any = None) -> None:
        self.llm = llm
        self.musicbrainz = musicbrainz


class ExplodingMusicBrainz:
    async def search_recordings(self, *args: Any, **kwargs: Any) -> list[dict]:
        raise AssertionError("MusicBrainz must not be called when the LLM fails")


async def test_extract_sets_title_and_artist() -> None:
    llm = FakeStructuredLlm(QueryExtraction(title="Bohemian Rhapsody", artist="Queen"))
    node = _extract(FakeRuntime(llm))

    result = await node({"query": "Bohemian Rhapsody dei Queen"})

    assert result["query"] == "Bohemian Rhapsody"
    assert result["artist"] == "Queen"
    assert result["resolution"] == ""
    assert result["error"] is None


async def test_user_artist_wins_over_extraction() -> None:
    llm = FakeStructuredLlm(QueryExtraction(title="The Message", artist="Someone Else"))
    node = _extract(FakeRuntime(llm))

    result = await node({"query": "quella canzone", "artist": "Nas"})

    assert result["artist"] == "Nas"


async def test_empty_title_falls_back_to_the_original_query() -> None:
    llm = FakeStructuredLlm(QueryExtraction(title="", artist=None))
    node = _extract(FakeRuntime(llm))

    result = await node({"query": "Bohemian Rhapsody dei Queen"})

    assert result["query"] == "Bohemian Rhapsody dei Queen"
    assert result["artist"] is None


async def test_empty_query_skips_the_llm() -> None:
    llm = FakeStructuredLlm(QueryExtraction(title="should not be used"))
    node = _extract(FakeRuntime(llm))

    result = await node({"query": "   "})

    assert result["query"] == ""
    assert llm.calls == 0


async def test_unexpected_error_becomes_a_generic_llm_error() -> None:
    llm = FakeStructuredLlm(error=ValueError("SECRET-TOKEN-123 parser detail"))
    node = _extract(FakeRuntime(llm))

    result = await node({"query": "Bohemian Rhapsody dei Queen"})

    assert result["resolution"] == "llm_error"
    assert result["error"] == UNEXPECTED_MESSAGE


@respx.mock
async def test_llm_error_goes_to_handle_error_without_musicbrainz() -> None:
    musicbrainz_route = respx.route(host="musicbrainz.org")
    llm = FakeStructuredLlm(error=rate_limited())
    graph = build_resolve_graph(
        FakeRuntime(llm, ExplodingMusicBrainz()),
        InMemorySaver(),
        sleep=FakeSleep(),
    )

    result = await graph.ainvoke(
        {"query": "Bohemian Rhapsody dei Queen"},
        {"configurable": {"thread_id": "llm-error"}},
    )

    assert result["resolution"] == "llm_error"
    assert result["error"] == RATE_LIMITED_MESSAGE
    assert result.get("chosen") is None
    assert "__interrupt__" not in result
    assert llm.calls == 3
    assert not musicbrainz_route.called


async def test_short_rate_limits_are_retried_with_a_notice() -> None:
    llm = FakeStructuredLlm(
        result=QueryExtraction(title="Bohemian Rhapsody", artist="Queen"),
        errors=[rate_limited("2"), rate_limited("1")],
    )
    sleep = FakeSleep()
    node = _extract(FakeRuntime(llm), sleep=sleep)

    result = await node({"query": "Bohemian Rhapsody dei Queen"})

    assert result["query"] == "Bohemian Rhapsody"
    assert result["artist"] == "Queen"
    assert result["resolution"] == ""
    assert result["error"] is None
    assert result["notice"] == WAIT_NOTICE
    assert llm.calls == 3
    assert sleep.delays == [2.0, 1.0]


async def test_no_notice_without_a_wait() -> None:
    llm = FakeStructuredLlm(result=QueryExtraction(title="The Message", artist="Nas"))
    node = _extract(FakeRuntime(llm), sleep=FakeSleep())

    result = await node({"query": "The Message"})

    assert result["notice"] is None


async def test_daily_rate_limit_is_not_retried() -> None:
    llm = FakeStructuredLlm(errors=[rate_limited("7200")])
    sleep = FakeSleep()
    node = _extract(FakeRuntime(llm), sleep=sleep)

    result = await node({"query": "Bohemian Rhapsody dei Queen"})

    assert result["resolution"] == "llm_error"
    assert "2 hours" in result["error"]
    assert result["notice"] is None
    assert llm.calls == 1
    assert sleep.delays == []


@respx.mock
async def test_daily_rate_limit_skips_musicbrainz() -> None:
    musicbrainz_route = respx.route(host="musicbrainz.org")
    graph = build_resolve_graph(
        FakeRuntime(
            FakeStructuredLlm(errors=[rate_limited("7200")]),
            ExplodingMusicBrainz(),
        ),
        InMemorySaver(),
        sleep=FakeSleep(),
    )

    result = await graph.ainvoke(
        {"query": "Bohemian Rhapsody dei Queen"},
        {"configurable": {"thread_id": "daily-limit"}},
    )

    assert result["resolution"] == "llm_error"
    assert "2 hours" in result["error"]
    assert result.get("chosen") is None
    assert "__interrupt__" not in result
    assert not musicbrainz_route.called
