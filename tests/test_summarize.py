from __future__ import annotations

from typing import Any

import groq
import httpx

from src.llm import WAIT_NOTICE
from src.resolver import _summarize
from src.schemas import LyricsSummary, TrackRef

TRACK = TrackRef(
    mbid="11111111-1111-1111-1111-111111111111",
    title="The Message",
    artist="Nas",
)
LYRICS = "alpha beta gamma delta epsilon zeta eta theta"
WIKI = "The Message is a song by Nas. Nas said the track reflects life in the city."


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


class FakeSummaryLlm:
    def __init__(
        self,
        result: LyricsSummary | None = None,
        errors: list[Exception] | None = None,
        results: list[LyricsSummary] | None = None,
    ) -> None:
        self.result = result
        self.errors = list(errors or [])
        self.results = list(results or [])
        self.calls = 0
        self.messages: list[Any] = []

    def with_structured_output(self, schema: type[LyricsSummary]) -> FakeSummaryLlm:
        assert schema is LyricsSummary
        return self

    async def ainvoke(self, messages: list[Any]) -> LyricsSummary:
        self.calls += 1
        self.messages.append(messages)
        if self.errors:
            raise self.errors.pop(0)
        if self.results:
            return self.results.pop(0)
        assert self.result is not None
        return self.result


class ExplodingMusicBrainz:
    async def search_recordings(self, *args: Any, **kwargs: Any) -> list[dict]:
        raise AssertionError("MusicBrainz must not be called by summarize")

    async def lookup_recording(self, *args: Any, **kwargs: Any) -> dict:
        raise AssertionError("MusicBrainz must not be called by summarize")


class FakeRuntime:
    def __init__(self, llm: FakeSummaryLlm) -> None:
        self.llm = llm
        self.musicbrainz = ExplodingMusicBrainz()


def state(
    *,
    lyrics: str | None = LYRICS,
    wiki_text: str | None = WIKI,
    chosen: dict | None = None,
) -> dict[str, Any]:
    return {
        "chosen": TRACK.model_dump() if chosen is None else chosen,
        "lyrics": lyrics,
        "wiki_text": wiki_text,
    }


async def test_valid_summary_is_written_to_state() -> None:
    llm = FakeSummaryLlm(
        LyricsSummary(
            topic="Life in the city",
            summary="A short description of the song.",
            artist_view="Nas said the track reflects life in the city.",
        )
    )
    node = _summarize(FakeRuntime(llm), sleep=FakeSleep())

    result = await node(state())

    assert result["lyrics_summary"] == {
        "topic": "Life in the city",
        "summary": "A short description of the song.",
        "artist_view": "Nas said the track reflects life in the city.",
    }
    assert result["summary_status"] == "ok"
    assert result["notice"] is None
    assert llm.calls == 1


async def test_leaking_fields_are_discarded() -> None:
    llm = FakeSummaryLlm(
        LyricsSummary(
            topic="alpha beta gamma delta epsilon zeta",
            summary="This copies alpha beta gamma delta epsilon zeta directly.",
            artist_view=None,
        )
    )
    node = _summarize(FakeRuntime(llm), sleep=FakeSleep())

    result = await node(state())

    assert result["lyrics_summary"]["topic"] == ""
    assert result["lyrics_summary"]["summary"] == ""
    # A leak-check discard is final: retrying would keep reproducing lyrics.
    assert result["summary_status"] == "leak_check"
    assert llm.calls == 1


async def test_empty_summary_is_retried_until_valid() -> None:
    llm = FakeSummaryLlm(
        results=[
            LyricsSummary(topic="", summary=""),
            LyricsSummary(topic="Life in the city", summary="A short description."),
        ]
    )
    node = _summarize(FakeRuntime(llm), sleep=FakeSleep())

    result = await node(state())

    assert result["lyrics_summary"]["topic"] == "Life in the city"
    assert result["summary_status"] == "ok"
    assert llm.calls == 2


async def test_empty_summary_is_reported_after_all_attempts() -> None:
    llm = FakeSummaryLlm(
        results=[LyricsSummary(topic="", summary="") for _ in range(3)]
    )
    node = _summarize(FakeRuntime(llm), sleep=FakeSleep())

    result = await node(state())

    assert result["lyrics_summary"]["topic"] == ""
    assert result["summary_status"] == "empty"
    assert llm.calls == 3


async def test_artist_view_must_appear_in_the_wiki_text() -> None:
    llm = FakeSummaryLlm(
        LyricsSummary(
            topic="Life in the city",
            summary="A short description of the song.",
            artist_view="Nas said this in a recent interview.",
        )
    )
    node = _summarize(FakeRuntime(llm), sleep=FakeSleep())

    result = await node(state())

    assert result["lyrics_summary"]["artist_view"] is None
    assert result["summary_status"] == "ok"


async def test_no_sources_skips_the_llm() -> None:
    llm = FakeSummaryLlm(LyricsSummary(topic="unused", summary="unused"))
    node = _summarize(FakeRuntime(llm), sleep=FakeSleep())

    result = await node(state(lyrics=None, wiki_text=None))

    assert result["lyrics_summary"] is None
    assert result["summary_status"] == "not_attempted"
    assert llm.calls == 0


async def test_short_rate_limits_are_retried_with_a_notice() -> None:
    llm = FakeSummaryLlm(
        LyricsSummary(
            topic="Life in the city",
            summary="A short description of the song.",
        ),
        errors=[rate_limited("2"), rate_limited("1")],
    )
    sleep = FakeSleep()
    node = _summarize(FakeRuntime(llm), sleep=sleep)

    result = await node(state())

    assert result["lyrics_summary"]["topic"] == "Life in the city"
    assert result["notice"] == WAIT_NOTICE
    assert llm.calls == 3
    assert sleep.delays == [2.0, 1.0]


async def test_daily_rate_limit_stops_without_musicbrainz() -> None:
    llm = FakeSummaryLlm(errors=[rate_limited("7200")])
    sleep = FakeSleep()
    node = _summarize(FakeRuntime(llm), sleep=sleep)

    result = await node(state())

    assert result["resolution"] == "llm_error"
    assert "2 hours" in result["error"]
    assert result["notice"] is None
    assert llm.calls == 1
    assert sleep.delays == []


async def test_prompt_contains_capped_lyrics_and_wiki_blocks() -> None:
    long_lyrics = "x" * 5000
    long_wiki = "w" * 10_000
    llm = FakeSummaryLlm(
        LyricsSummary(topic="A theme", summary="A summary.", artist_view=None)
    )
    node = _summarize(FakeRuntime(llm), sleep=FakeSleep())

    await node(state(lyrics=long_lyrics, wiki_text=long_wiki))

    prompt = llm.messages[0][-1]["content"]
    assert "<lyrics>" in prompt and "</lyrics>" in prompt
    assert "<wiki>" in prompt and "</wiki>" in prompt
    assert "x" * 4000 in prompt and "x" * 4001 not in prompt
    assert "w" * 9000 in prompt and "w" * 9001 not in prompt
