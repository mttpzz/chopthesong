from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import stat
import time

import httpx
import pytest
import respx

from src.http import CrossHostRedirect, Http, JsonCache


@pytest.mark.skipif(
    os.name == "nt",
    reason="Windows reports POSIX permission bits only as a placeholder.",
)
async def test_cache_directory_is_created_owner_only(tmp_path) -> None:
    cache_dir = tmp_path / "cache"

    http = Http(cache_dir, ttl=60, user_agent="test")
    try:
        assert stat.S_IMODE(cache_dir.stat().st_mode) == 0o700
    finally:
        await http.aclose()


@pytest.mark.skipif(
    os.name == "nt",
    reason="Windows reports POSIX permission bits only as a placeholder.",
)
async def test_existing_cache_directory_is_tightened(tmp_path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cache_dir.chmod(0o755)

    http = Http(cache_dir, ttl=60, user_agent="test")
    try:
        assert stat.S_IMODE(cache_dir.stat().st_mode) == 0o700
    finally:
        await http.aclose()


@pytest.mark.skipif(
    os.name == "nt",
    reason="Windows reports POSIX permission bits only as a placeholder.",
)
async def test_cache_database_is_created_owner_only(tmp_path) -> None:
    http = Http(tmp_path, ttl=60, user_agent="test")
    try:
        database = tmp_path / "http-cache.sqlite3"
        assert stat.S_IMODE(database.stat().st_mode) == 0o600
    finally:
        await http.aclose()


def test_json_cache_roundtrip(tmp_path) -> None:
    cache = JsonCache(tmp_path / "http-cache.sqlite3")
    entry = {"expires": 123.0, "value": {"ok": True, "text": "caffè"}}
    try:
        cache.set("key", entry)
        assert cache.get("key") == entry
        assert cache.get("missing") is None
    finally:
        cache.close()


@respx.mock
async def test_second_request_is_served_from_the_cache(tmp_path) -> None:
    route = respx.get("https://api.deezer.com/search").mock(
        return_value=httpx.Response(200, json={"data": [1]})
    )

    http = Http(tmp_path, ttl=60, user_agent="test")
    try:
        first = await http.get_json("https://api.deezer.com/search")
        second = await http.get_json("https://api.deezer.com/search")
    finally:
        await http.aclose()

    assert first == second == {"data": [1]}
    assert route.call_count == 1


@respx.mock
async def test_expired_entry_is_fetched_again(tmp_path) -> None:
    route = respx.get("https://api.deezer.com/search").mock(
        return_value=httpx.Response(200, json={"data": [1]})
    )

    http = Http(tmp_path, ttl=-1, user_agent="test")
    try:
        await http.get_json("https://api.deezer.com/search")
        await http.get_json("https://api.deezer.com/search")
    finally:
        await http.aclose()

    assert route.call_count == 2


@respx.mock
async def test_non_json_cache_row_is_ignored(tmp_path) -> None:
    url = "https://api.deezer.com/search"
    key = hashlib.sha256(json.dumps([url, []]).encode()).hexdigest()

    http = Http(tmp_path, ttl=60, user_agent="test")
    try:
        # A foreign or corrupted row must never be read as executable data.
        with sqlite3.connect(str(tmp_path / "http-cache.sqlite3")) as connection:
            connection.execute(
                "INSERT OR REPLACE INTO entries (key, expires, value) "
                "VALUES (?, ?, ?)",
                (key, time.time() + 60, "not json {{"),
            )

        route = respx.get(url).mock(
            return_value=httpx.Response(200, json={"ok": True})
        )
        value = await http.get_json(url)
    finally:
        await http.aclose()

    assert value == {"ok": True}
    assert route.call_count == 1


@respx.mock
async def test_redirect_to_the_same_host_is_followed(tmp_path) -> None:
    def redirect_or_ok(request: httpx.Request) -> httpx.Response:
        if request.url.params.get("ok") == "1":
            return httpx.Response(200, json={"ok": True})
        return httpx.Response(
            302, headers={"location": "https://en.wikipedia.org/w/api.php?ok=1"}
        )

    respx.get("https://en.wikipedia.org/w/api.php").mock(side_effect=redirect_or_ok)

    http = Http(tmp_path, ttl=60, user_agent="test")
    try:
        assert await http.get_json("https://en.wikipedia.org/w/api.php") == {"ok": True}
    finally:
        await http.aclose()


@respx.mock
async def test_redirect_to_another_host_is_refused(tmp_path) -> None:
    respx.get("https://en.wikipedia.org/w/api.php").mock(
        return_value=httpx.Response(
            302, headers={"location": "https://evil.example/collect"}
        )
    )
    collect = respx.get("https://evil.example/collect").mock(
        return_value=httpx.Response(200, json={"stolen": True})
    )

    http = Http(tmp_path, ttl=60, user_agent="test")
    try:
        with pytest.raises(CrossHostRedirect) as info:
            await http.get_json("https://en.wikipedia.org/w/api.php")
    finally:
        await http.aclose()

    assert isinstance(info.value, httpx.HTTPError)
    assert collect.call_count == 0
