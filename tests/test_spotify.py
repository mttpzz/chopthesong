from __future__ import annotations

import httpx
import pytest
import respx

from src.config import Settings
from src.http import Http
from src.spotify import (
    SEARCH_URL,
    TOKEN_URL,
    SpotifyClient,
    _reset_token_cache,
)

TRACK_URL = "https://open.spotify.com/track/abc"
ARTIST_URL = "https://open.spotify.com/artist/def"


@pytest.fixture(autouse=True)
def _clean_token_cache():
    """Keep the process-wide token cache from leaking between tests."""
    _reset_token_cache()
    yield
    _reset_token_cache()


def settings(tmp_path, **overrides) -> Settings:
    values = {
        "spotify_client_id": "fake-id",
        "spotify_client_secret": "fake-secret",
        "data_dir": tmp_path,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def token_response(token: str = "token", expires_in: int = 3600) -> httpx.Response:
    return httpx.Response(
        200, json={"access_token": token, "expires_in": expires_in}
    )


def search_response(kind: str, url: str | None) -> httpx.Response:
    items = [] if url is None else [{"external_urls": {"spotify": url}}]
    return httpx.Response(200, json={f"{kind}s": {"items": items}})


@respx.mock
async def test_token_is_fetched_once_and_reused(tmp_path) -> None:
    token_route = respx.post(TOKEN_URL).mock(return_value=token_response())
    respx.get(SEARCH_URL).mock(
        side_effect=lambda request: search_response(
            request.url.params["type"],
            TRACK_URL if request.url.params["type"] == "track" else ARTIST_URL,
        )
    )
    http = Http(tmp_path, ttl=60, user_agent="test")
    try:
        client = SpotifyClient("fake-id", "fake-secret", http)

        assert await client.find_track_url("Song", "Artist") == TRACK_URL
        assert await client.find_artist_url("Artist") == ARTIST_URL
    finally:
        await http.aclose()

    assert token_route.call_count == 1


@respx.mock
async def test_expired_token_is_renewed(tmp_path, monkeypatch) -> None:
    clock = [1000.0]
    monkeypatch.setattr("src.spotify.time.monotonic", lambda: clock[0])
    token_route = respx.post(TOKEN_URL).mock(return_value=token_response())
    respx.get(SEARCH_URL).mock(
        return_value=search_response("track", TRACK_URL)
    )
    http = Http(tmp_path, ttl=60, user_agent="test")
    try:
        client = SpotifyClient("fake-id", "fake-secret", http)
        assert await client.find_track_url("First", None) == TRACK_URL
        assert token_route.call_count == 1

        clock[0] += 3600
        assert await client.find_track_url("Second", None) == TRACK_URL
    finally:
        await http.aclose()

    assert token_route.call_count == 2


@respx.mock
async def test_empty_search_returns_none(tmp_path) -> None:
    respx.post(TOKEN_URL).mock(return_value=token_response())
    respx.get(SEARCH_URL).mock(return_value=search_response("track", None))
    http = Http(tmp_path, ttl=60, user_agent="test")
    try:
        client = SpotifyClient("fake-id", "fake-secret", http)
        assert await client.find_track_url("Missing", None) is None
    finally:
        await http.aclose()


@respx.mock
async def test_server_error_returns_none_without_raising(tmp_path) -> None:
    respx.post(TOKEN_URL).mock(return_value=token_response())
    respx.get(SEARCH_URL).mock(return_value=httpx.Response(500))
    http = Http(tmp_path, ttl=60, user_agent="test")
    try:
        client = SpotifyClient("fake-id", "fake-secret", http)
        assert await client.find_album_url("Album", "Artist") is None
    finally:
        await http.aclose()


@respx.mock
async def test_rate_limit_stops_the_rest_of_the_report(tmp_path, caplog) -> None:
    respx.post(TOKEN_URL).mock(return_value=token_response())
    search_route = respx.get(SEARCH_URL).mock(return_value=httpx.Response(429))
    http = Http(tmp_path, ttl=60, user_agent="test")
    try:
        client = SpotifyClient("fake-id", "fake-secret", http)

        assert await client.find_track_url("Song", "Artist") is None
        assert client.blocked
        assert await client.find_artist_url("Artist") is None
        assert await client.find_album_url("Album", "Artist") is None
    finally:
        await http.aclose()

    assert search_route.call_count == 1
    warnings = [record for record in caplog.records if record.levelname == "WARNING"]
    assert len(warnings) == 1
    assert "rate limit" in warnings[0].message


@respx.mock
async def test_token_failure_disables_links(tmp_path) -> None:
    respx.post(TOKEN_URL).mock(return_value=httpx.Response(401))
    search_route = respx.get(SEARCH_URL)
    http = Http(tmp_path, ttl=60, user_agent="test")
    try:
        client = SpotifyClient("fake-id", "fake-secret", http)
        assert await client.find_track_url("Song", "Artist") is None
    finally:
        await http.aclose()

    assert search_route.call_count == 0
