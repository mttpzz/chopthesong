from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from src.service import add_spotify_links


class FakeHttp:
    """Records search calls and answers each one with a distinct fake URL."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def get_json(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        *,
        headers: dict[str, str] | None = None,
        ttl: int | None = None,
        retry_transient: bool = True,
    ) -> Any:
        assert headers and headers["Authorization"].startswith("Bearer ")
        assert ttl == 60 * 60 * 24
        assert retry_transient is False
        assert params is not None
        self.calls.append(params)
        kind = params["type"]
        label = params["q"].split(" ", 1)[0].replace(":", "-")
        return {
            f"{kind}s": {
                "items": [
                    {
                        "external_urls": {
                            "spotify": f"https://open.spotify.com/{kind}/{label}"
                        }
                    }
                ]
            }
        }


class RateLimitedHttp(FakeHttp):
    """Every search returns 429, so the client must stop after the first."""

    async def get_json(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        *,
        headers: dict[str, str] | None = None,
        ttl: int | None = None,
        retry_transient: bool = True,
    ) -> Any:
        assert params is not None
        self.calls.append(params)
        request = httpx.Request("GET", url)
        response = httpx.Response(429, request=request)
        raise httpx.HTTPStatusError("429", request=request, response=response)


def settings(**overrides: Any) -> SimpleNamespace:
    values = {"spotify_client_id": "fake-id", "spotify_client_secret": "fake-secret"}
    values.update(overrides)
    return SimpleNamespace(**values)


def report(samples: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "track": {
            "title": "Song",
            "artist": "Artist",
            "album": "Album",
            "year": 2000,
        },
        "samples": samples or [],
    }


@pytest.fixture(autouse=True)
def _fake_token(monkeypatch) -> None:
    async def token(client_id: str, client_secret: str) -> str:
        return "token"

    monkeypatch.setattr("src.spotify._access_token", token)


async def test_missing_credentials_make_no_calls() -> None:
    http = FakeHttp()
    payload = report()

    await add_spotify_links(
        payload, settings(spotify_client_id=None, spotify_client_secret=None), http
    )

    assert http.calls == []
    assert payload.get("spotify_track_url") is None
    assert payload.get("spotify_artist_url") is None
    assert payload.get("spotify_album_url") is None


async def test_blank_credentials_make_no_calls() -> None:
    http = FakeHttp()

    await add_spotify_links(
        report(), settings(spotify_client_id="  ", spotify_client_secret="secret"), http
    )

    assert http.calls == []


async def test_header_links_are_filled_from_search() -> None:
    http = FakeHttp()
    payload = report()

    await add_spotify_links(payload, settings(), http)

    assert payload["spotify_track_url"] == "https://open.spotify.com/track/track-Song"
    assert (
        payload["spotify_artist_url"]
        == "https://open.spotify.com/artist/artist-Artist"
    )
    assert (
        payload["spotify_album_url"] == "https://open.spotify.com/album/album-Album"
    )
    assert len(http.calls) == 3


async def test_nested_samples_get_their_links() -> None:
    http = FakeHttp()
    payload = report(
        [
            {
                "title": "Parent",
                "artist": "Parent Artist",
                "samples": [
                    {"title": "Child", "artist": "Child Artist", "samples": []},
                ],
            }
        ]
    )

    await add_spotify_links(payload, settings(), http)

    assert payload["samples"][0]["spotify_url"].startswith(
        "https://open.spotify.com/track/track-Parent"
    )
    assert payload["samples"][0]["spotify_artist_url"].startswith(
        "https://open.spotify.com/artist/artist-Parent"
    )
    child = payload["samples"][0]["samples"][0]
    assert child["spotify_url"].startswith(
        "https://open.spotify.com/track/track-Child"
    )
    assert child["spotify_artist_url"].startswith(
        "https://open.spotify.com/artist/artist-Child"
    )


async def test_sample_searches_stop_at_twenty() -> None:
    http = FakeHttp()
    samples = [
        {"title": f"Sample {index}", "artist": None, "samples": []}
        for index in range(25)
    ]
    payload = report(samples)

    await add_spotify_links(payload, settings(), http)

    sample_title_calls = [
        call for call in http.calls if call["q"].startswith("track:Sample")
    ]
    assert len(sample_title_calls) == 20
    assert all(payload["samples"][index]["spotify_url"] for index in range(20))
    for index in range(20, 25):
        assert payload["samples"][index].get("spotify_url") is None
    assert all(payload["samples"][index].get("spotify_artist_url") is None
               for index in range(25))


async def test_shared_budget_pays_titles_before_artists() -> None:
    http = FakeHttp()
    samples = [
        {"title": f"Sample {index}", "artist": f"Artist {index}", "samples": []}
        for index in range(12)
    ]
    payload = report(samples)

    await add_spotify_links(payload, settings(), http)

    assert all(payload["samples"][index]["spotify_url"] for index in range(12))
    assert all(
        payload["samples"][index]["spotify_artist_url"] for index in range(8)
    )
    for index in range(8, 12):
        assert payload["samples"][index].get("spotify_artist_url") is None

    sample_artist_calls = [
        call for call in http.calls if call["q"].startswith("artist:Artist ")
    ]
    assert len(sample_artist_calls) == 8
    assert len(http.calls) == 3 + 12 + 8


async def test_an_empty_artist_does_not_spend_the_budget() -> None:
    http = FakeHttp()
    samples = [
        {"title": "With Artist", "artist": "Artist A", "samples": []},
        {"title": "Empty", "artist": "", "samples": []},
        {"title": "Missing", "artist": None, "samples": []},
    ]
    payload = report(samples)

    await add_spotify_links(payload, settings(), http)

    assert payload["samples"][0]["spotify_artist_url"]
    assert payload["samples"][1].get("spotify_artist_url") is None
    assert payload["samples"][2].get("spotify_artist_url") is None
    assert len(http.calls) == 3 + 3 + 1


async def test_rate_limit_stops_every_later_search() -> None:
    http = RateLimitedHttp()
    payload = report(
        [
            {"title": "Sample", "artist": "Artist", "samples": []},
        ]
    )

    await add_spotify_links(payload, settings(), http)

    assert len(http.calls) == 1  # the header track search got the 429
    assert payload.get("spotify_track_url") is None
    assert payload["samples"][0].get("spotify_url") is None
    assert payload["samples"][0].get("spotify_artist_url") is None


async def test_report_with_links_is_json_serializable() -> None:
    payload = report(
        [{"title": "Parent", "artist": "Artist", "samples": []}]
    )

    await add_spotify_links(payload, settings(), FakeHttp())

    serialized = json.dumps(payload)
    assert "spotify_track_url" in serialized
    assert "spotify_url" in serialized
