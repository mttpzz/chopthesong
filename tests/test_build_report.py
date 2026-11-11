from __future__ import annotations

import json
import logging
from typing import Any

import pytest

from src.llm import INTERNAL_MESSAGE
from src.resolver import _build_report
from src.schemas import LyricsSummary, SampleNode, TrackRef

LYRICS = "SECRET LYRICS THAT MUST NEVER LEAK"
WIKI = "SECRET WIKI TEXT THAT MUST NEVER LEAK"


def track() -> dict[str, Any]:
    return TrackRef(mbid="root-id", title="Root", artist="Artist").model_dump(mode="json")


def summary(*, topic: str = "Life", body: str = "A short summary.") -> dict[str, Any]:
    return LyricsSummary(topic=topic, summary=body).model_dump()


def sample(
    title: str = "Sample",
    *,
    source: str = "MusicBrainz",
    mbid: str | None = "sample-id",
    children: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return SampleNode(
        title=title,
        artist="Sample Artist",
        year=1990,
        album="Sample Album",
        mbid=mbid,
        source=source,
        confidence="high" if source == "MusicBrainz" else "medium",
        samples=children or [],
    ).model_dump(mode="json")


def state(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "chosen": track(),
        "resolution": "auto",
        "genres": ["Hip-Hop"],
        "lyrics_summary": summary(),
        "summary_status": "ok",
        "samples": [sample()],
        "missing_sources": [],
        "lyrics": LYRICS,
        "wiki_text": WIKI,
    }
    data.update(overrides)
    return data


async def test_complete_state_produces_a_valid_report() -> None:
    result = await _build_report()(state())

    report = result["report"]
    assert report["track"]["mbid"] == "root-id"
    assert report["genres"] == ["Hip-Hop"]
    assert report["lyrics_summary"]["topic"] == "Life"
    assert report["summary_status"] == "ok"
    assert report["samples"][0]["title"] == "Sample"
    assert report["sources"] == [
        "MusicBrainz",
        "Deezer",
        "LRCLIB",
        "Wikipedia",
        "MusicBrainz samples",
    ]


async def test_missing_source_is_not_listed_and_gets_a_note() -> None:
    result = await _build_report()(state(missing_sources=["Deezer"]))

    report = result["report"]
    assert "Deezer" not in report["sources"]
    assert report["sources"] == [
        "MusicBrainz",
        "LRCLIB",
        "Wikipedia",
        "MusicBrainz samples",
    ]
    assert report["notes"] == ["No data from Deezer."]


async def test_musicbrainz_samples_listed_when_samples_exist() -> None:
    result = await _build_report()(
        state(missing_sources=["MusicBrainz samples"], samples=[sample()])
    )

    report = result["report"]
    assert "MusicBrainz samples" in report["sources"]
    assert "No data from MusicBrainz samples." not in report["notes"]


async def test_wikipedia_only_samples_do_not_claim_musicbrainz_samples() -> None:
    result = await _build_report()(
        state(
            missing_sources=["MusicBrainz samples"],
            samples=[sample("Wiki Song", source="Wikipedia", mbid=None)],
        )
    )

    report = result["report"]
    assert "MusicBrainz samples" not in report["sources"]
    assert report["notes"] == ["No data from MusicBrainz samples."]


async def test_empty_summary_and_nested_samples_add_their_notes_in_order() -> None:
    # Step 5b: the child needs its own MBID, otherwise the report dedupe would
    # drop it as the same track as its parent and the nested note would vanish.
    nested = sample("Child", mbid="child-id")
    result = await _build_report()(
        state(
            missing_sources=["Deezer", "LRCLIB"],
            lyrics_summary=summary(topic="", body=""),
            # Older states predate summary_status: keep the legacy note path.
            summary_status=None,
            samples=[sample(children=[nested])],
        )
    )

    report = result["report"]
    assert report["notes"] == [
        "No data from Deezer.",
        "No data from LRCLIB.",
        "Summary unavailable: the lyrics or the Wikipedia text did not pass the checks.",
        "Samples beyond the first level come from MusicBrainz only.",
    ]


@pytest.mark.parametrize(
    ("status", "note"),
    [
        (
            "leak_check",
            "Summary unavailable: the summary copied lyrics and was discarded.",
        ),
        (
            "empty",
            "Summary unavailable: the model returned an empty summary.",
        ),
        (
            "not_attempted",
            "Summary not attempted: no lyrics or Wikipedia text.",
        ),
    ],
)
async def test_summary_status_selects_a_specific_note(
    status: str, note: str
) -> None:
    result = await _build_report()(
        state(lyrics_summary=summary(topic="", body=""), summary_status=status)
    )

    report = result["report"]
    assert report["summary_status"] == status
    assert report["notes"] == [note]


async def test_report_never_contains_raw_lyrics_or_wiki_text() -> None:
    result = await _build_report()(state())

    payload = json.dumps(result["report"])
    assert LYRICS not in payload
    assert WIKI not in payload


@pytest.mark.parametrize(
    "overrides",
    [
        {"chosen": None},
        {"resolution": "cancelled"},
        {"error": INTERNAL_MESSAGE},
    ],
)
async def test_report_is_not_written_outside_a_successful_choice(
    overrides: dict[str, Any],
) -> None:
    result = await _build_report()(state(**overrides))

    assert result == {}


async def test_malformed_state_is_logged_and_reported_as_internal_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    broken = state(
        samples=[{"title": "Broken", "source": "MusicBrainz", "samples": 5}]
    )

    with caplog.at_level(logging.ERROR):
        result = await _build_report()(broken)

    assert result["resolution"] == "internal_error"
    assert result["error"] == INTERNAL_MESSAGE
    assert any(record.exc_info for record in caplog.records)
