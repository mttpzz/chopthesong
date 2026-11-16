"""Minimal async HTTP client: JSON responses, disk cache and rate limiting."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import diskcache
import httpx


class RateLimiter:
    """Spaces requests by at least ``min_interval`` seconds (MusicBrainz: 1/s)."""

    def __init__(self, min_interval: float) -> None:
        self._min_interval = min_interval
        self._lock = asyncio.Lock()
        self._last = 0.0

    async def wait(self) -> None:
        """Block until another request may be sent."""
        async with self._lock:
            delay = self._last + self._min_interval - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            self._last = time.monotonic()


class Http:
    """Async JSON client with a disk cache. Caching keeps free APIs usable."""

    def __init__(
        self,
        cache_dir: Path,
        *,
        ttl: int,
        user_agent: str,
        timeout: float = 20.0,
    ) -> None:
        """Open the disk cache and the shared httpx client."""
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache = diskcache.Cache(str(cache_dir))
        self._ttl = ttl
        self._client = httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": user_agent},
        )

    async def aclose(self) -> None:
        """Release the HTTP client and close the cache database."""
        await self._client.aclose()
        self._cache.close()

    async def get_json(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        *,
        headers: dict[str, str] | None = None,
        limiter: RateLimiter | None = None,
        ttl: int | None = None,
        retry_transient: bool = True,
        cache: bool = True,
    ) -> Any:
        """Fetch JSON with GET, reusing a cached response while it is still fresh.

        Responses are keyed by URL and sorted params. Free APIs cannot absorb
        repeated calls, so caching matters as much as the rate limiter.
        ``headers`` are sent with the request but never enter the cache key, so
        a refreshed bearer token still reuses the cached body. Set
        ``retry_transient`` to false when the caller owns the retry policy.
        Set ``cache`` to false for responses that must never reach the disk.
        """
        return await self._json(
            "GET",
            url,
            params=params,
            headers=headers,
            limiter=limiter,
            ttl=ttl,
            retry_transient=retry_transient,
            cache=cache,
        )

    async def post_json(
        self,
        url: str,
        data: dict[str, Any] | None = None,
        *,
        headers: dict[str, str] | None = None,
        limiter: RateLimiter | None = None,
        ttl: int | None = None,
        retry_transient: bool = True,
        cache: bool = True,
    ) -> Any:
        """Send a form-encoded POST and parse the JSON response.

        Cache, retry and error rules are the same as ``get_json``. Set
        ``cache`` to false when neither the body nor the response may leave a
        trace on disk, as for audio fingerprints.
        """
        return await self._json(
            "POST",
            url,
            data=data,
            headers=headers,
            limiter=limiter,
            ttl=ttl,
            retry_transient=retry_transient,
            cache=cache,
        )

    async def _json(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        limiter: RateLimiter | None = None,
        ttl: int | None = None,
        retry_transient: bool = True,
        cache: bool = True,
    ) -> Any:
        """Perform one GET or POST request and return its JSON body."""
        key = self._cache_key(method, url, params, data) if cache else None
        if key is not None:
            entry = self._cache.get(key)
            if entry and entry["expires"] > time.time():
                return entry["value"]

        # Retry only transient throttling; other errors should fail fast.
        response = None
        for attempt in range(3):
            if limiter:
                await limiter.wait()
            if method == "POST":
                response = await self._client.post(url, data=data, headers=headers)
            else:
                response = await self._client.get(url, params=params, headers=headers)
            if (
                retry_transient
                and response.status_code in {429, 503}
                and attempt < 2
            ):
                await asyncio.sleep(2 * (attempt + 1))
                continue
            break

        assert response is not None
        response.raise_for_status()
        value = response.json()
        # Each entry keeps its own expiry, so callers can ask for a shorter TTL.
        if key is not None:
            expires = time.time() + (ttl if ttl is not None else self._ttl)
            self._cache.set(key, {"expires": expires, "value": value})
        return value

    @staticmethod
    def _cache_key(
        method: str,
        url: str,
        params: dict[str, Any] | None,
        data: dict[str, Any] | None,
    ) -> str:
        """Hash a request into a cache key; a POST body is part of it.

        GET keys keep their previous format, so existing caches stay valid.
        """
        parts: list[Any] = [url, sorted((params or {}).items())]
        if method != "GET":
            parts = [method, *parts, sorted((data or {}).items())]
        return hashlib.sha256(json.dumps(parts, default=str).encode()).hexdigest()
