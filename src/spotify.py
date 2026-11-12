"""Spotify Web API search client that turns report names into links.

Only ``GET /v1/search`` is used, with the client credentials flow. Search
responses already contain ``external_urls.spotify``, so no detail endpoint is
needed. Tokens stay in process memory; search bodies are cached on disk by
``Http`` for one day, which the Developer Terms allow as temporary metadata
caching (Section IV.3.b).
"""

from __future__ import annotations

import logging
import time
from typing import Any, cast

import httpx

from src.http import Http

logger = logging.getLogger(__name__)

TOKEN_URL = "https://accounts.spotify.com/api/token"
SEARCH_URL = "https://api.spotify.com/v1/search"
SEARCH_TTL_SECONDS = 60 * 60 * 24
TOKEN_MARGIN_SECONDS = 60
URL_PREFIX = "https://open.spotify.com/"

# Process-wide token cache: tokens last one hour and are shared by all reports.
_token: str | None = None
_token_expires_at = 0.0


def _reset_token_cache() -> None:
    """Forget the cached token; used by tests to isolate each scenario."""
    global _token, _token_expires_at
    _token = None
    _token_expires_at = 0.0


def _normalize(value: str | None) -> str:
    """Collapse whitespace so equivalent names share one cache entry."""
    return " ".join(str(value or "").split())


def _spotify_url(payload: Any, kind: str) -> str | None:
    """Read the first item's Spotify URL, ignoring unexpected shapes."""
    if not isinstance(payload, dict):
        return None
    section = payload.get(f"{kind}s")
    if not isinstance(section, dict):
        return None
    items = section.get("items")
    if not isinstance(items, list) or not items:
        return None
    first = items[0]
    if not isinstance(first, dict):
        return None
    urls = first.get("external_urls")
    if not isinstance(urls, dict):
        return None
    url = urls.get("spotify")
    if isinstance(url, str) and url.startswith(URL_PREFIX):
        return url
    return None


async def _access_token(client_id: str, client_secret: str) -> str | None:
    """Return a cached token, or request one with the client credentials flow."""
    global _token, _token_expires_at
    now = time.monotonic()
    if _token and _token_expires_at > now:
        return _token

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(
                TOKEN_URL,
                data={"grant_type": "client_credentials"},
                auth=httpx.BasicAuth(client_id, client_secret),
            )
            response.raise_for_status()
            payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("Spotify token request failed (%s)", type(exc).__name__)
        return None

    if not isinstance(payload, dict):
        return None
    token = payload.get("access_token")
    if not isinstance(token, str) or not token:
        return None
    try:
        expires_in = int(payload.get("expires_in", 3600))
    except (TypeError, ValueError):
        expires_in = 3600
    _token = token
    _token_expires_at = now + max(0, expires_in - TOKEN_MARGIN_SECONDS)
    return token


class SpotifyClient:
    """Search-only client with a per-report request budget and rate-limit fuse."""

    def __init__(self, client_id: str, client_secret: str, http: Http) -> None:
        self._client_id = client_id
        self._client_secret = client_secret
        self._http = http
        self.searches = 0
        self.blocked = False

    async def _search(self, kind: str, query: str) -> str | None:
        if self.blocked:
            return None
        token = await _access_token(self._client_id, self._client_secret)
        if token is None:
            return None

        self.searches += 1
        try:
            payload = await self._http.get_json(
                SEARCH_URL,
                {"q": query, "type": kind, "limit": 1},
                headers={"Authorization": f"Bearer {token}"},
                ttl=SEARCH_TTL_SECONDS,
                retry_transient=False,
            )
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 429:
                self.blocked = True
                logger.warning(
                    "Spotify rate limit reached; skipping the remaining links "
                    "for this report"
                )
            else:
                logger.info(
                    "Spotify search failed with %s", exc.response.status_code
                )
            return None
        except (httpx.HTTPError, ValueError, TypeError):
            return None
        return _spotify_url(payload, kind)

    async def find_track_url(self, title: str | None, artist: str | None) -> str | None:
        """Return the Spotify page for a track title, with an optional artist."""
        clean_title = _normalize(title)
        if not clean_title:
            return None
        query = f"track:{clean_title}"
        clean_artist = _normalize(artist)
        if clean_artist:
            query = f"{query} artist:{clean_artist}"
        return await self._search("track", query)

    async def find_artist_url(self, artist: str | None) -> str | None:
        """Return the Spotify page for an artist name."""
        clean_artist = _normalize(artist)
        if not clean_artist:
            return None
        return await self._search("artist", f"artist:{clean_artist}")

    async def find_album_url(self, album: str | None, artist: str | None) -> str | None:
        """Return the Spotify page for an album title, with an optional artist."""
        clean_album = _normalize(album)
        if not clean_album:
            return None
        query = f"album:{clean_album}"
        clean_artist = _normalize(artist)
        if clean_artist:
            query = f"{query} artist:{clean_artist}"
        return await self._search("album", query)


def client_from_settings(settings: Any, http: Http) -> SpotifyClient | None:
    """Build a client when both credentials are present, otherwise None."""
    client_id = cast(str, getattr(settings, "spotify_client_id", None) or "")
    client_secret = cast(str, getattr(settings, "spotify_client_secret", None) or "")
    if not client_id.strip() or not client_secret.strip():
        return None
    return SpotifyClient(client_id, client_secret, http)
