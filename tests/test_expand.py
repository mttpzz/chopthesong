from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from src import resolver
from src.resolver import _expand
from src.schemas import SampleNode, TrackRef

ROOT = "root-id"
A = "a-id"
B = "b-id"
C = "c-id"
D = "d-id"
W = "w-id"


def recording(
    mbid: str,
    title: str,
    artist: str,
    *,
    year: str = "1990-01-01",
    album: str = "Album",
) -> dict[str, Any]:
    return {
        "id": mbid,
        "score": 100,
        "title": title,
        "length": 200000,
        "artist-credit": [{"name": artist, "joinphrase": ""}],
        "releases": [{"title": album, "date": year}],
    }


class FakeMusicBrainz:
    def __init__(
        self,
        relations: dict[str, list[dict] | Exception] | None = None,
        search_results: list[dict] | None = None,
    ) -> None:
        self.relations = dict(relations or {})
        self.search_results = list(search_results or [])
        self.lookup_calls: list[str] = []
        self.search_calls: list[tuple[str, str | None, int]] = []

    async def lookup_recording(self, mbid: str, *, inc: str = "") -> dict:
        self.lookup_calls.append(mbid)
        value = self.relations.get(mbid, [])
        if isinstance(value, Exception):
            raise value
        return {
            "relations": [
                {
                    "type": "samples material",
                    "direction": "forward",
                    "recording": item,
                }
                for item in value
            ]
        }

    async def search_recordings(
        self, title: str, *, artist: str | None = None, limit: int = 15
    ) -> list[dict]:
        self.search_calls.append((title, artist, limit))
        return self.search_results


class ExplodingLlm:
    def __init__(self) -> None:
        self.calls = 0

    def with_structured_output(self, *args: Any, **kwargs: Any) -> None:
        self.calls += 1
        raise AssertionError("expand must not use the LLM")

    async def ainvoke(self, *args: Any, **kwargs: Any) -> None:
        self.calls += 1
        raise AssertionError("expand must not use the LLM")


class FakeRuntime:
    def __init__(self, musicbrainz: FakeMusicBrainz, *, report_depth: int = 2) -> None:
        self.musicbrainz = musicbrainz
        self.llm = ExplodingLlm()
        self.settings = SimpleNamespace(report_depth=report_depth)


def sample(
    mbid: str | None,
    title: str,
    *,
    source: str = "MusicBrainz",
    artist: str = "Artist",
    confidence: str = "high",
    year: int | None = 1990,
    album: str | None = "Album",
) -> dict[str, Any]:
    return SampleNode(
        title=title,
        artist=artist,
        year=year,
        album=album,
        mbid=mbid,
        source=source,
        confidence=confidence,
    ).model_dump(mode="json")


def state(
    samples: list[dict[str, Any]],
    *,
    missing_sources: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "chosen": TrackRef(mbid=ROOT, title="Root", artist="Artist").model_dump(),
        "samples": samples,
        "missing_sources": list(missing_sources or []),
    }


def flatten(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for node in nodes:
        result.append(node)
        result.extend(flatten(node.get("samples") or []))
    return result


async def test_report_depth_zero_does_not_expand() -> None:
    musicbrainz = FakeMusicBrainz({A: [recording(B, "B", "Artist")]})
    runtime = FakeRuntime(musicbrainz, report_depth=0)

    result = await _expand(runtime)(state([sample(A, "A")]))

    assert musicbrainz.lookup_calls == []
    assert result["samples"][0]["samples"] == []


async def test_report_depth_one_adds_children_only() -> None:
    musicbrainz = FakeMusicBrainz(
        {
            A: [recording(B, "B", "Artist")],
            B: [recording(C, "C", "Artist")],
        }
    )
    runtime = FakeRuntime(musicbrainz, report_depth=1)

    result = await _expand(runtime)(state([sample(A, "A")]))

    assert musicbrainz.lookup_calls == [A]
    assert result["samples"][0]["samples"][0]["mbid"] == B
    assert result["samples"][0]["samples"][0]["samples"] == []


async def test_report_depth_two_adds_grandchildren() -> None:
    musicbrainz = FakeMusicBrainz(
        {
            A: [recording(B, "B", "Artist")],
            B: [recording(C, "C", "Artist")],
            C: [recording(D, "D", "Artist")],
        }
    )
    runtime = FakeRuntime(musicbrainz, report_depth=2)

    result = await _expand(runtime)(state([sample(A, "A")]))

    assert musicbrainz.lookup_calls == [A, B]
    grandchild = result["samples"][0]["samples"][0]["samples"][0]
    assert grandchild["mbid"] == C
    assert grandchild["samples"] == []


async def test_cycle_expands_each_mbid_once() -> None:
    musicbrainz = FakeMusicBrainz(
        {
            A: [recording(B, "B", "Artist")],
            B: [recording(A, "A", "Artist")],
        }
    )
    runtime = FakeRuntime(musicbrainz, report_depth=3)

    result = await _expand(runtime)(state([sample(A, "A")]))

    assert musicbrainz.lookup_calls.count(A) == 1
    assert musicbrainz.lookup_calls.count(B) == 1
    child = result["samples"][0]["samples"][0]
    leaf = child["samples"][0]
    assert leaf["mbid"] == A
    assert leaf["samples"] == []


async def test_chosen_track_is_never_expanded() -> None:
    musicbrainz = FakeMusicBrainz({ROOT: [recording(A, "A", "Artist")]})
    runtime = FakeRuntime(musicbrainz, report_depth=2)

    result = await _expand(runtime)(state([sample(ROOT, "Root")]))

    assert musicbrainz.lookup_calls == []
    assert result["samples"][0]["samples"] == []


async def test_node_cap_stops_expansion_without_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(resolver, "max_sample_nodes", 3)
    musicbrainz = FakeMusicBrainz(
        {
            A: [
                recording(B, "B", "Artist"),
                recording(C, "C", "Artist"),
                recording(D, "D", "Artist"),
            ]
        }
    )
    runtime = FakeRuntime(musicbrainz, report_depth=2)

    result = await _expand(runtime)(state([sample(A, "A")]))

    tree = flatten(result["samples"])
    assert len(tree) == 3
    assert [node["mbid"] for node in tree] == [A, B, C]


async def test_confident_wikipedia_sample_is_resolved_and_expanded() -> None:
    musicbrainz = FakeMusicBrainz(
        relations={W: [recording(B, "B", "Artist")]},
        search_results=[recording(W, "Wiki Song", "Wiki Artist", year="1999-05-01")],
    )
    runtime = FakeRuntime(musicbrainz, report_depth=1)

    result = await _expand(
        runtime,
    )(
        state(
            [
                sample(
                    None,
                    "Wiki Song",
                    source="Wikipedia",
                    artist="Wiki Artist",
                    confidence="medium",
                    year=None,
                    album=None,
                )
            ]
        )
    )

    resolved = result["samples"][0]
    assert musicbrainz.search_calls == [("Wiki Song", "Wiki Artist", 15)]
    assert resolved["mbid"] == W
    assert resolved["year"] == 1999
    assert resolved["album"] == "Album"
    assert resolved["source"] == "Wikipedia"
    assert resolved["confidence"] == "medium"
    assert musicbrainz.lookup_calls == [W]
    assert resolved["samples"][0]["mbid"] == B


async def test_unconfident_wikipedia_sample_is_not_expanded() -> None:
    musicbrainz = FakeMusicBrainz(
        search_results=[recording(W, "Completely Different", "Other Artist")]
    )
    runtime = FakeRuntime(musicbrainz, report_depth=1)

    result = await _expand(
        runtime,
    )(
        state(
            [
                sample(
                    None,
                    "Wiki Song",
                    source="Wikipedia",
                    artist="Wiki Artist",
                    confidence="medium",
                    year=None,
                    album=None,
                )
            ]
        )
    )

    unresolved = result["samples"][0]
    assert len(musicbrainz.search_calls) == 1
    assert unresolved["mbid"] is None
    assert unresolved["year"] is None
    assert musicbrainz.lookup_calls == []
    assert unresolved["samples"] == []


async def test_source_error_keeps_the_rest_of_the_tree() -> None:
    musicbrainz = FakeMusicBrainz(
        {
            A: httpx.ConnectError("down"),
            B: [recording(C, "C", "Artist")],
        }
    )
    runtime = FakeRuntime(musicbrainz, report_depth=1)

    result = await _expand(runtime)(state([sample(A, "A"), sample(B, "B")]))

    assert result["samples"][0]["samples"] == []
    assert result["samples"][1]["samples"][0]["mbid"] == C
    assert musicbrainz.lookup_calls == [A, B]


async def test_unexpected_error_is_logged_and_skipped(
    caplog: pytest.LogCaptureFixture,
) -> None:
    musicbrainz = FakeMusicBrainz({A: RuntimeError("tool bug")})
    runtime = FakeRuntime(musicbrainz, report_depth=1)

    with caplog.at_level(logging.ERROR):
        result = await _expand(runtime)(state([sample(A, "A")]))

    assert result["samples"][0]["samples"] == []
    assert any(record.exc_info for record in caplog.records)


async def test_missing_source_added_only_when_first_level_is_empty() -> None:
    runtime = FakeRuntime(FakeMusicBrainz(), report_depth=2)

    empty = await _expand(runtime)(state([], missing_sources=["Deezer"]))
    assert empty["samples"] == []
    assert empty["missing_sources"] == ["Deezer", "MusicBrainz samples"]

    populated = await _expand(runtime)(
        state([sample(A, "A")], missing_sources=["Deezer"])
    )
    assert "missing_sources" not in populated


async def test_expand_never_calls_the_llm() -> None:
    musicbrainz = FakeMusicBrainz({A: [recording(B, "B", "Artist")]})
    runtime = FakeRuntime(musicbrainz, report_depth=2)

    await _expand(runtime)(state([sample(A, "A")]))

    assert runtime.llm.calls == 0
