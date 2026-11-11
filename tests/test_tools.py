from __future__ import annotations

from typing import Any

import httpx
import respx

from src.http import Http
from src.musicbrainz import MusicBrainzClient
from src.tools import (
    DEEZER_ALBUM,
    DEEZER_SEARCH,
    LRCLIB_SEARCH,
    WIKI_API,
    build_tools,
)

MBID = "11111111-1111-1111-1111-111111111111"
SAMPLE_MBID = "22222222-2222-2222-2222-222222222222"


class FakeRuntime:
    def __init__(self, http: Http, musicbrainz: Any = None) -> None:
        self.http = http
        self.musicbrainz = musicbrainz


def make_http(tmp_path) -> Http:
    return Http(tmp_path / "cache", ttl=3600, user_agent="test/0.1")


@respx.mock
async def test_get_genres_reads_album_genres(tmp_path) -> None:
    respx.get(DEEZER_SEARCH).mock(
        return_value=httpx.Response(
            200, json={"data": [{"artist": {"name": "Nas"}, "album": {"id": 42}}]}
        )
    )
    respx.get(f"{DEEZER_ALBUM}/42").mock(
        return_value=httpx.Response(
            200,
            json={
                "genres": {
                    "data": [{"name": "Hip-Hop"}, {"name": "hip-hop"}, {"name": "Rap"}]
                }
            },
        )
    )
    http = make_http(tmp_path)
    try:
        genres = await build_tools(FakeRuntime(http))["get_genres"].ainvoke(
            {"title": "The Message", "artist": "Nas"}
        )
    finally:
        await http.aclose()

    assert genres == ["Hip-Hop", "Rap"]


@respx.mock
async def test_get_genres_without_results_is_empty(tmp_path) -> None:
    respx.get(DEEZER_SEARCH).mock(
        return_value=httpx.Response(200, json={"data": []})
    )
    http = make_http(tmp_path)
    try:
        genres = await build_tools(FakeRuntime(http))["get_genres"].ainvoke(
            {"title": "The Message", "artist": "Nas"}
        )
    finally:
        await http.aclose()

    assert genres == []


@respx.mock
async def test_get_genres_http_error_is_empty(tmp_path) -> None:
    respx.get(DEEZER_SEARCH).mock(return_value=httpx.Response(500))
    http = make_http(tmp_path)
    try:
        genres = await build_tools(FakeRuntime(http))["get_genres"].ainvoke(
            {"title": "The Message", "artist": "Nas"}
        )
    finally:
        await http.aclose()

    assert genres == []


@respx.mock
async def test_get_genres_normalizes_typographic_dashes(tmp_path) -> None:
    route = respx.get(DEEZER_SEARCH).mock(
        return_value=httpx.Response(200, json={"data": []})
    )
    http = make_http(tmp_path)
    try:
        await build_tools(FakeRuntime(http))["get_genres"].ainvoke(
            {"title": "Walk This Way", "artist": "Run\u2010D.M.C."}
        )
    finally:
        await http.aclose()

    assert route.calls[0].request.url.params["q"] == "Run-D.M.C. Walk This Way"


@respx.mock
async def test_get_lyrics_returns_plain_lyrics(tmp_path) -> None:
    respx.get(LRCLIB_SEARCH).mock(
        return_value=httpx.Response(
            200,
            json=[
                {
                    "trackName": "The Message",
                    "artistName": "Nas",
                    "duration": 234,
                    "plainLyrics": "line one\nline two",
                    "instrumental": False,
                }
            ],
        )
    )
    http = make_http(tmp_path)
    try:
        lyrics = await build_tools(FakeRuntime(http))["get_lyrics"].ainvoke(
            {"title": "The Message", "artist": "Nas", "duration_ms": 234000}
        )
    finally:
        await http.aclose()

    assert lyrics == "line one\nline two"


@respx.mock
async def test_get_lyrics_ignores_instrumental_results(tmp_path) -> None:
    route = respx.get(LRCLIB_SEARCH).mock(
        return_value=httpx.Response(
            200,
            json=[
                {
                    "trackName": "The Message",
                    "artistName": "Nas",
                    "duration": 234,
                    "plainLyrics": None,
                    "instrumental": True,
                }
            ],
        )
    )
    http = make_http(tmp_path)
    try:
        lyrics = await build_tools(FakeRuntime(http))["get_lyrics"].ainvoke(
            {"title": "The Message", "artist": "Nas", "duration_ms": 234000}
        )
    finally:
        await http.aclose()

    assert lyrics is None
    assert route.call_count == 2  # first search, then the q=title fallback


@respx.mock
async def test_get_lyrics_rejects_a_distant_title(tmp_path) -> None:
    respx.get(LRCLIB_SEARCH).mock(
        return_value=httpx.Response(
            200,
            json=[
                {
                    "trackName": "Completely Different Song",
                    "artistName": "Nas",
                    "duration": 234,
                    "plainLyrics": "line one",
                    "instrumental": False,
                }
            ],
        )
    )
    http = make_http(tmp_path)
    try:
        lyrics = await build_tools(FakeRuntime(http))["get_lyrics"].ainvoke(
            {"title": "The Message", "artist": "Nas", "duration_ms": 234000}
        )
    finally:
        await http.aclose()

    assert lyrics is None


@respx.mock
async def test_get_lyrics_prefers_the_closest_duration(tmp_path) -> None:
    respx.get(LRCLIB_SEARCH).mock(
        return_value=httpx.Response(
            200,
            json=[
                {
                    "trackName": "The Message",
                    "artistName": "Nas",
                    "duration": 200,
                    "plainLyrics": "far version",
                    "instrumental": False,
                },
                {
                    "trackName": "The Message",
                    "artistName": "Nas",
                    "duration": 235,
                    "plainLyrics": "near version",
                    "instrumental": False,
                },
            ],
        )
    )
    http = make_http(tmp_path)
    try:
        lyrics = await build_tools(FakeRuntime(http))["get_lyrics"].ainvoke(
            {"title": "The Message", "artist": "Nas", "duration_ms": 234000}
        )
    finally:
        await http.aclose()

    assert lyrics == "near version"


@respx.mock
async def test_get_lyrics_http_error_is_none(tmp_path) -> None:
    respx.get(LRCLIB_SEARCH).mock(return_value=httpx.Response(500))
    http = make_http(tmp_path)
    try:
        lyrics = await build_tools(FakeRuntime(http))["get_lyrics"].ainvoke(
            {"title": "The Message", "artist": "Nas", "duration_ms": None}
        )
    finally:
        await http.aclose()

    assert lyrics is None


@respx.mock
async def test_get_lyrics_normalizes_typographic_dashes(tmp_path) -> None:
    route = respx.get(LRCLIB_SEARCH).mock(
        return_value=httpx.Response(200, json=[])
    )
    http = make_http(tmp_path)
    try:
        await build_tools(FakeRuntime(http))["get_lyrics"].ainvoke(
            {"title": "Walk This Way", "artist": "Run\u2010D.M.C.", "duration_ms": None}
        )
    finally:
        await http.aclose()

    params = route.calls[0].request.url.params
    assert params["track_name"] == "Walk This Way"
    assert params["artist_name"] == "Run-D.M.C."


@respx.mock
async def test_get_wiki_text_returns_the_extract(tmp_path) -> None:
    def side_effect(request: httpx.Request) -> httpx.Response:
        if request.url.params.get("list") == "search":
            return httpx.Response(
                200, json={"query": {"search": [{"title": "The Message"}]}}
            )
        return httpx.Response(
            200, json={"query": {"pages": {"1": {"extract": "Short text."}}}}
        )

    respx.get(WIKI_API).mock(side_effect=side_effect)
    http = make_http(tmp_path)
    try:
        text = await build_tools(FakeRuntime(http))["get_wiki_text"].ainvoke(
            {"title": "The Message", "artist": "Nas"}
        )
    finally:
        await http.aclose()

    assert text == "Short text."


@respx.mock
async def test_get_wiki_text_rejects_an_unrelated_page(tmp_path) -> None:
    def side_effect(request: httpx.Request) -> httpx.Response:
        assert request.url.params.get("list") == "search"
        return httpx.Response(
            200, json={"query": {"search": [{"title": "Another Song"}]}}
        )

    respx.get(WIKI_API).mock(side_effect=side_effect)
    http = make_http(tmp_path)
    try:
        text = await build_tools(FakeRuntime(http))["get_wiki_text"].ainvoke(
            {"title": "The Message", "artist": "Nas"}
        )
    finally:
        await http.aclose()

    assert text is None


@respx.mock
async def test_get_wiki_text_normalizes_typographic_dashes(tmp_path) -> None:
    route = respx.get(WIKI_API).mock(
        return_value=httpx.Response(200, json={"query": {"search": []}})
    )
    http = make_http(tmp_path)
    try:
        await build_tools(FakeRuntime(http))["get_wiki_text"].ainvoke(
            {"title": "Walk This Way", "artist": "Run\u2010D.M.C."}
        )
    finally:
        await http.aclose()

    query = route.calls[0].request.url.params["srsearch"]
    assert query == '"Walk This Way" Run-D.M.C. song'


@respx.mock
async def test_get_mb_samples_returns_forward_samples(tmp_path) -> None:
    payload = {
        "relations": [
            {
                "type": "samples material",
                "direction": "forward",
                "recording": {
                    "id": SAMPLE_MBID,
                    "title": "Sample Song",
                    "artist-credit": [{"name": "Other Artist", "joinphrase": ""}],
                    "releases": [{"title": "Sample Album", "date": "1990-05-01"}],
                },
            },
            {
                "type": "samples material",
                "direction": "backward",
                "recording": {"id": "reverse-relation"},
            },
            {"type": "remix", "direction": "forward", "recording": {"id": "remix"}},
        ]
    }
    respx.get(f"https://musicbrainz.org/ws/2/recording/{MBID}").mock(
        return_value=httpx.Response(200, json=payload)
    )
    http = make_http(tmp_path)
    try:
        musicbrainz = MusicBrainzClient(http, min_interval=0)
        samples = await build_tools(FakeRuntime(http, musicbrainz))[
            "get_mb_samples"
        ].ainvoke({"mbid": MBID})
    finally:
        await http.aclose()

    assert samples == [
        {
            "title": "Sample Song",
            "artist": "Other Artist",
            "year": 1990,
            "album": "Sample Album",
            "mbid": SAMPLE_MBID,
        }
    ]


@respx.mock
async def test_get_mb_samples_http_error_is_empty(tmp_path) -> None:
    respx.get(f"https://musicbrainz.org/ws/2/recording/{MBID}").mock(
        return_value=httpx.Response(500)
    )
    http = make_http(tmp_path)
    try:
        musicbrainz = MusicBrainzClient(http, min_interval=0)
        samples = await build_tools(FakeRuntime(http, musicbrainz))[
            "get_mb_samples"
        ].ainvoke({"mbid": MBID})
    finally:
        await http.aclose()

    assert samples == []
