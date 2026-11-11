"""Thin async client for the MusicBrainz web service (search and lookups)."""

from __future__ import annotations

import re

from src.http import Http, RateLimiter
from src.schemas import TrackRef

BASE_URL = "https://musicbrainz.org/ws/2"


def lucene_escape(value: str) -> str:
    """Escape backslashes and quotes inside a Lucene query value."""
    return value.replace("\\", "\\\\").replace('"', '\\"')


def parse_recording(recording: dict) -> TrackRef:
    """Convert one search hit into a ``TrackRef``.

    A search response lists only a subset of a recording's releases, so album
    and year describe the earliest non-compilation release present in that
    response, not necessarily the original publication date.
    """
    credits = recording.get("artist-credit") or []
    artist = "".join(
        f"{credit.get('name', '')}{credit.get('joinphrase', '')}" for credit in credits
    ).strip()

    releases = recording.get("releases") or []
    # Compilations are a fallback only: their date is rarely the original one.
    studio_releases = [release for release in releases if not _is_compilation(release)]
    earliest = _earliest(releases)
    preferred = _earliest(studio_releases) or earliest or {}
    release_group = preferred.get("release-group") or {}

    return TrackRef(
        mbid=recording.get("id"),
        title=recording.get("title") or "",
        artist=artist or "Unknown artist",
        primary_artist=(credits[0].get("name") if credits else None),
        # Album, year and release group all come from the same preferred release.
        year=_year_from(preferred.get("date")),
        album=preferred.get("title"),
        duration_ms=recording.get("length"),
        disambiguation=recording.get("disambiguation") or None,
        release_primary_type=release_group.get("primary-type"),
        release_group_id=release_group.get("id"),
        release_count=len(
            {
                (release.get("release-group") or {}).get("id") or release.get("title")
                for release in releases
            }
        ),
        is_compilation=bool(releases) and not studio_releases,
    )


def _earliest(releases: list[dict]) -> dict | None:
    """Return the release with the oldest date, ignoring undated ones."""
    dated = [release for release in releases if release.get("date")]
    return min(dated, key=lambda release: release["date"]) if dated else None


def _is_compilation(release: dict) -> bool:
    """True when MusicBrainz marks the release group as a compilation."""
    release_group = release.get("release-group") or {}
    secondary = {str(value).lower() for value in release_group.get("secondary-types") or []}
    return "compilation" in secondary


def _year_from(date: str | None) -> int | None:
    """Extract the four-digit year from a MusicBrainz date string."""
    match = re.match(r"(\d{4})", str(date or ""))
    return int(match.group(1)) if match else None


class MusicBrainzClient:
    """Search recordings and fetch their details, one request per second."""

    def __init__(self, http: Http, *, min_interval: float = 1.05) -> None:
        self._http = http
        self._limiter = RateLimiter(min_interval)

    async def search_recordings(
        self,
        title: str,
        *,
        artist: str | None = None,
        limit: int = 15,
    ) -> list[dict]:
        """Search by title and optional artist; return the raw recordings."""
        clauses = [f'recording:"{lucene_escape(title)}"']
        if artist:
            clauses.append(f'artist:"{lucene_escape(artist)}"')
        payload = await self._http.get_json(
            f"{BASE_URL}/recording",
            params={"query": " AND ".join(clauses), "fmt": "json", "limit": limit},
            limiter=self._limiter,
        )
        return payload.get("recordings", [])

    async def lookup_recording(
        self,
        mbid: str,
        *,
        inc: str = "artist-credits+releases+genres+tags+recording-rels",
    ) -> dict:
        """Fetch one recording with credits, releases, genres and relations."""
        return await self._http.get_json(
            f"{BASE_URL}/recording/{mbid}",
            params={"inc": inc, "fmt": "json"},
            limiter=self._limiter,
        )

    async def lookup_release_group(self, mbid: str) -> dict:
        """Fetch genres and tags for a release group (an album)."""
        return await self._http.get_json(
            f"{BASE_URL}/release-group/{mbid}",
            params={"inc": "genres+tags", "fmt": "json"},
            limiter=self._limiter,
        )
