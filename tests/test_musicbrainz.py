from __future__ import annotations

import httpx
import respx

from src.http import Http
from src.musicbrainz import MusicBrainzClient, lucene_escape, parse_recording

PAYLOAD = {
    "recordings": [
        {
            "id": "11111111-1111-1111-1111-111111111111",
            "score": 100,
            "title": "The Message",
            "length": 234000,
            "disambiguation": "album version",
            "artist-credit": [{"name": "Nas", "joinphrase": ""}],
            "releases": [
                {"title": "God's Son", "date": "2002-11-19"},
                {"title": "Greatest Hits", "date": "2007-01-01"},
            ],
        }
    ]
}


def test_parse_recording_takes_earliest_release() -> None:
    track = parse_recording(PAYLOAD["recordings"][0])
    assert track.mbid == "11111111-1111-1111-1111-111111111111"
    assert track.artist == "Nas"
    assert track.year == 2002
    assert track.album == "God's Son"
    assert track.duration_ms == 234000


def test_parse_recording_year_matches_the_album_release() -> None:
    track = parse_recording(
        {
            "id": "x",
            "title": "Song",
            "artist-credit": [{"name": "Artist", "joinphrase": ""}],
            "releases": [
                {
                    "title": "Greatest Hits",
                    "date": "1999-01-01",
                    "release-group": {"secondary-types": ["Compilation"]},
                },
                {"title": "Studio Album", "date": "2005-06-01"},
            ],
        }
    )

    assert track.album == "Studio Album"
    assert track.year == 2005


def test_parse_recording_joins_multiple_credits() -> None:
    track = parse_recording(
        {
            "id": "x",
            "title": "Song",
            "artist-credit": [
                {"name": "A", "joinphrase": " feat. "},
                {"name": "B", "joinphrase": ""},
            ],
        }
    )
    assert track.artist == "A feat. B"


def test_lucene_escape() -> None:
    assert lucene_escape('Say "Hello"') == 'Say \\"Hello\\"'


@respx.mock
async def test_search_builds_artist_query_and_caches(tmp_path) -> None:
    route = respx.get("https://musicbrainz.org/ws/2/recording").mock(
        return_value=httpx.Response(200, json=PAYLOAD)
    )
    http = Http(tmp_path, ttl=3600, user_agent="test/0.1")
    try:
        client = MusicBrainzClient(http, min_interval=0)
        first = await client.search_recordings("The Message", artist="Nas")
        second = await client.search_recordings("The Message", artist="Nas")
    finally:
        await http.aclose()

    assert len(first) == 1
    assert second == first
    assert route.call_count == 1  # second hit came from the disk cache
    query = route.calls[0].request.url.params["query"]
    assert 'recording:"The Message"' in query
    assert 'artist:"Nas"' in query
