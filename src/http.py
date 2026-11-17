"""Minimal async HTTP client: JSON responses, SQLite cache and rate limiting."""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

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


class CrossHostRedirect(httpx.HTTPError):
    """A redirect pointed at a host other than the request's own."""


async def _reject_cross_host_redirect(response: httpx.Response) -> None:
    """Allow redirects only within the host of the original request."""
    if not response.is_redirect:
        return
    location = response.headers.get("location")
    if not location:
        return
    target = response.url.join(location)
    if target.host.casefold() != response.url.host.casefold():
        raise CrossHostRedirect("Refusing a redirect to a different host.")


class JsonCache:
    """Small SQLite cache that stores only JSON text."""

    def __init__(self, path: Path) -> None:
        """Open the database, create the table and keep the file private."""
        self._lock = threading.Lock()
        self._connection = sqlite3.connect(str(path), check_same_thread=False)
        with self._lock, self._connection:
            self._connection.execute(
                "CREATE TABLE IF NOT EXISTS entries ("
                "key TEXT PRIMARY KEY, "
                "expires REAL NOT NULL, "
                "value TEXT NOT NULL)"
            )
        try:
            path.chmod(0o600)
        except OSError:
            # Some filesystems and Windows do not support POSIX permissions.
            pass

    def get(self, key: str) -> dict[str, Any] | None:
        """Return the stored entry, or ``None`` when absent or not valid JSON."""
        with self._lock:
            row = self._connection.execute(
                "SELECT value FROM entries WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            return None
        try:
            entry = json.loads(row[0])
        except (TypeError, ValueError):
            # Only JSON is ever read: a corrupted or foreign row is ignored.
            return None
        if not isinstance(entry, dict) or "expires" not in entry or "value" not in entry:
            return None
        return entry

    def set(self, key: str, entry: dict[str, Any]) -> None:
        """Store an entry as JSON text, replacing any previous value for the key."""
        payload = json.dumps(entry, ensure_ascii=False)
        with self._lock, self._connection:
            self._connection.execute(
                "INSERT OR REPLACE INTO entries (key, expires, value) VALUES (?, ?, ?)",
                (key, float(entry["expires"]), payload),
            )

    def close(self) -> None:
        """Close the database connection."""
        with self._lock:
            self._connection.close()


class Http:
    """Async JSON client with a SQLite cache. Caching keeps free APIs usable."""

    def __init__(
        self,
        cache_dir: Path,
        *,
        ttl: int,
        user_agent: str,
        timeout: float = 20.0,
    ) -> None:
        """Open the JSON cache and the shared httpx client."""
        # Entries are plain JSON, so nothing is ever deserialized as code. The
        # directory stays private because responses may still be sensitive.
        cache_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            cache_dir.chmod(0o700)
        except OSError:
            # Some filesystems and Windows do not support POSIX permissions.
            pass
        self._cache = JsonCache(cache_dir / "http-cache.sqlite3")
        self._ttl = ttl
        self._client = httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": user_agent},
            event_hooks={"response": [_reject_cross_host_redirect]},
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

        if response is None:
            raise RuntimeError("HTTP retry loop finished without a response")
        response.raise_for_status()
        value = response.json()
        # Each entry keeps its own expiry, so callers can ask for a shorter TTL.
        self._cache.set(
            key,
            {"expires": time.time() + (ttl if ttl is not None else self._ttl), "value": value},
        )
        return value
