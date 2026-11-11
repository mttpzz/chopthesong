from __future__ import annotations

from typing import Any

from src.resolver import _build_report, _dedupe_samples
from src.schemas import LyricsSummary, SampleNode, TrackRef


def node(
    title: str,
    *,
    artist: str | None = "Sample Artist",
    mbid: str | None = None,
    source: str = "MusicBrainz",
    confidence: str | None = None,
    children: list[SampleNode] | None = None,
) -> SampleNode:
    return SampleNode(
        title=title,
        artist=artist,
        mbid=mbid,
        source=source,
        confidence=confidence or ("high" if source == "MusicBrainz" else "medium"),
        samples=children or [],
    )


def titles(nodes: list[SampleNode]) -> list[str]:
    return [item.title for item in nodes]


def count(nodes: list[SampleNode]) -> int:
    return sum(1 + count(item.samples) for item in nodes)


def dump(nodes: list[SampleNode]) -> list[dict[str, Any]]:
    return [item.model_dump(mode="json") for item in nodes]


def test_no_duplicates_leave_the_tree_unchanged() -> None:
    samples = [
        node("One", mbid="one"),
        node("Two", mbid="two", children=[node("Three", mbid="three")]),
    ]

    result = _dedupe_samples(samples)

    assert dump(result) == dump(samples)


def test_same_mbid_keeps_the_shallowest_occurrence() -> None:
    samples = [node("Same", mbid="same", children=[node("Same", mbid="same")])]

    result = _dedupe_samples(samples)

    assert titles(result) == ["Same"]
    assert result[0].samples == []


def test_stronger_source_beats_a_shallower_wikipedia_node() -> None:
    samples = [
        node("Same Song", artist="X", source="Wikipedia"),
        node(
            "Parent",
            mbid="parent",
            children=[node("Same Song", artist="X", mbid="musicbrainz")],
        ),
    ]

    result = _dedupe_samples(samples)

    assert titles(result) == ["Parent"]
    assert titles(result[0].samples) == ["Same Song"]
    assert result[0].samples[0].source == "MusicBrainz"


def test_loser_takes_its_subtree_with_it() -> None:
    samples = [
        node(
            "Same Song",
            artist="X",
            source="Wikipedia",
            children=[node("Grandchild", mbid="grandchild")],
        ),
        node(
            "Parent",
            mbid="parent",
            children=[node("Same Song", artist="X", mbid="musicbrainz")],
        ),
    ]

    result = _dedupe_samples(samples)

    flattened = [item for parent in result for item in [parent, *parent.samples]]
    assert titles(flattened) == ["Parent", "Same Song"]
    assert "Grandchild" not in titles(flattened)


def test_missing_mbid_on_one_side_matches_by_title_and_artist() -> None:
    samples = [
        node("Same Song", artist="X"),
        node("Parent", mbid="parent", children=[node("Same Song", artist="X", mbid="m")]),
    ]

    result = _dedupe_samples(samples)

    assert titles(result) == ["Same Song", "Parent"]
    assert count(result) == 2
    assert result[1].samples == []


def test_same_title_with_different_artists_is_not_a_duplicate() -> None:
    samples = [
        node("Same Song", artist="First"),
        node("Same Song", artist="Second"),
    ]

    result = _dedupe_samples(samples)

    assert titles(result) == ["Same Song", "Same Song"]


def test_different_mbids_are_not_a_duplicate() -> None:
    samples = [
        node("Same Song", artist="X", mbid="first"),
        node("Same Song", artist="X", mbid="second"),
    ]

    result = _dedupe_samples(samples)

    assert [item.mbid for item in result] == ["first", "second"]


def test_missing_mbid_bridges_different_mbids_transitively() -> None:
    samples = [
        node("Same Song", artist="X", mbid="first"),
        node("Same Song", artist="X", source="Wikipedia"),
        node(
            "Parent",
            mbid="parent",
            children=[node("Same Song", artist="X", mbid="second")],
        ),
    ]

    result = _dedupe_samples(samples)

    assert titles(result) == ["Same Song", "Parent"]
    assert result[0].mbid == "first"
    assert result[1].samples == []


def test_dedupe_does_not_mutate_the_input_and_returns_copies() -> None:
    samples = [
        node("Same Song", artist="X", source="Wikipedia"),
        node(
            "Parent",
            mbid="parent",
            children=[node("Same Song", artist="X", mbid="musicbrainz")],
        ),
    ]
    before = dump(samples)

    result = _dedupe_samples(samples)

    assert dump(samples) == before
    assert result[0] is not samples[1]
    assert result[0].samples is not samples[1].samples


def state(samples: list[SampleNode]) -> dict[str, Any]:
    return {
        "chosen": TrackRef(
            mbid="root", title="Stronger", artist="Kanye West"
        ).model_dump(mode="json"),
        "resolution": "user",
        "genres": [],
        "lyrics_summary": LyricsSummary(topic="Life", summary="A summary.").model_dump(),
        "samples": dump(samples),
        "missing_sources": [],
    }


async def test_build_report_applies_the_stronger_example() -> None:
    samples = [
        node(
            "Cola Bottle Baby",
            artist="Edwin Birdsong",
            source="Wikipedia",
        ),
        node(
            "Harder, Better, Faster, Stronger",
            artist="Daft Punk",
            mbid="hbf",
            children=[
                node(
                    "Cola Bottle Baby",
                    artist="Edwin Birdsong",
                    mbid="cola-bottle-baby",
                )
            ],
        ),
    ]
    assert count(samples) == 3

    result = await _build_report()(state(samples))

    report = result["report"]
    assert count([SampleNode.model_validate(item) for item in report["samples"]]) == 2
    assert titles([SampleNode.model_validate(item) for item in report["samples"]]) == [
        "Harder, Better, Faster, Stronger"
    ]
    root = SampleNode.model_validate(report["samples"][0])
    assert titles(root.samples) == ["Cola Bottle Baby"]
    assert root.samples[0].mbid == "cola-bottle-baby"
    assert "MusicBrainz samples" in report["sources"]
    assert "Samples beyond the first level come from MusicBrainz only." in report["notes"]
