from __future__ import annotations

import httpx
import respx

from src.http import Http

URL = "https://example.invalid/api"


@respx.mock
async def test_post_json_sends_form_data_and_skips_the_cache(tmp_path) -> None:
    route = respx.post(URL).mock(return_value=httpx.Response(200, json={"ok": True}))
    http = Http(tmp_path, ttl=60, user_agent="test")
    try:
        value = await http.post_json(
            URL, {"fingerprint": "THE-FINGERPRINT", "duration": 120}, cache=False
        )

        assert value == {"ok": True}
        # Nothing from the request or the response may reach the disk.
        assert len(http._cache) == 0
    finally:
        await http.aclose()

    request = route.calls[0].request
    assert request.method == "POST"
    assert request.headers["content-type"] == "application/x-www-form-urlencoded"
    assert b"fingerprint=THE-FINGERPRINT" in request.content
    assert b"duration=120" in request.content


@respx.mock
async def test_post_json_caches_by_default_and_keeps_the_body_in_the_key(
    tmp_path,
) -> None:
    route = respx.post(URL).mock(return_value=httpx.Response(200, json={"value": 1}))
    get_route = respx.get(URL).mock(
        return_value=httpx.Response(200, json={"value": 2})
    )
    http = Http(tmp_path, ttl=60, user_agent="test")
    try:
        assert await http.post_json(URL, {"fingerprint": "one"}) == {"value": 1}
        assert await http.post_json(URL, {"fingerprint": "one"}) == {"value": 1}
        assert route.call_count == 1

        # A different body is a different request, not a cache hit.
        assert await http.post_json(URL, {"fingerprint": "two"}) == {"value": 1}
        assert route.call_count == 2

        # GET keeps caching exactly as before.
        assert await http.get_json(URL) == {"value": 2}
        assert await http.get_json(URL) == {"value": 2}
        assert get_route.call_count == 1

        # GET and POST on the same URL do not share an entry.
        assert len(http._cache) == 3
    finally:
        await http.aclose()
