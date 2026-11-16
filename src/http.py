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
    ) -> Any:
        """Fetch JSON, reusing a cached response while it is still fresh.

        Responses are keyed by URL and sorted params. Free APIs cannot absorb
        repeated calls, so caching matters as much as the rate limiter.
        ``headers`` are sent with the request but never enter the cache key, so
        a refreshed bearer token still reuses the cached body. Set
        ``retry_transient`` to false when the caller owns the retry policy.
        """
        # Sorting the params keeps equivalent requests on the same cache key.
        key = hashlib.sha256(
            json.dumps([url, sorted((params or {}).items())], default=str).encode()
        ).hexdigest()
        entry = self._cache.get(key)
        if entry and entry["expires"] > time.time():
            return entry["value"]

        # Retry only transient throttling; other errors should fail fast.
        response = None
        for attempt in range(3):
            if limiter:
                await limiter.wait()
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
        self._cache.set(
            key,
            {"expires": time.time() + (ttl if ttl is not None else self._ttl), "value": value},
        )
        return value
