"""LangChain tools that fetch external data for a resolved track."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import httpx
from langchain_core.tools import BaseTool, tool
from rapidfuzz import fuzz

from src.matching import normalize_text
from src.musicbrainz import parse_recording

if TYPE_CHECKING:
    from src.resolver import Runtime

DEEZER_SEARCH = "https://api.deezer.com/search"
DEEZER_ALBUM = "https://api.deezer.com/album"
LRCLIB_SEARCH = "https://lrclib.net/api/search"
WIKI_API = "https://en.wikipedia.org/w/api.php"

TITLE_THRESHOLD = 80
ARTIST_THRESHOLD = 75
ARTIST_MATCH_SCORE = 85

# MusicBrainz sometimes returns typographic dashes (e.g. U+2010). Free search
# APIs treat them as different characters, so queries use the ASCII hyphen.
_DASH_TRANSLATION = str.maketrans(
    {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2015": "-",
        "\u2212": "-",
        "\u2e3a": "-",
        "\u2e3b": "-",
        "\ufe58": "-",
        "\ufe63": "-",
        "\uff0d": "-",
    }
)


def _search_text(value: str) -> str:
    """Normalize dashes and whitespace before sending text to a search API."""
    return " ".join((value or "").translate(_DASH_TRANSLATION).split())

# Network/HTTP failures and malformed payloads. Anything else is a bug and must surface.
SOURCE_ERRORS = (httpx.HTTPError, ValueError, KeyError, TypeError, AttributeError)


def build_tools(runtime: Runtime) -> dict[str, BaseTool]:
    """Build the four source tools bound to a runtime.

    The tools never raise for a missing result or a dead source: they return an
    empty value, and the gather node records the source as missing.
    """

    @tool
    async def get_genres(title: str, artist: str) -> list[str]:
        """Return Deezer genres for a track, or an empty list."""
        try:
            query = _search_text(f"{artist} {title}")
            data = await runtime.http.get_json(DEEZER_SEARCH, {"q": query, "limit": 8})
            results = data.get("data") or []
            match = next((item for item in results if _same_artist(item, artist)), None)
            if match is None:
                match = next(
                    (item for item in results if (item.get("album") or {}).get("id")),
                    None,
                )
            album_id = ((match or {}).get("album") or {}).get("id")
            if not album_id:
                return []
            album = await runtime.http.get_json(f"{DEEZER_ALBUM}/{album_id}")
            return _unique(
                genre.get("name") for genre in (album.get("genres") or {}).get("data") or []
            )
        except SOURCE_ERRORS:  # a dead or malformed source must not break the graph
            return []

    @tool
    async def get_lyrics(title: str, artist: str, duration_ms: int | None) -> str | None:
        """Return plain lyrics from LRCLIB, or None when nothing matches."""
        try:
            results = await runtime.http.get_json(
                LRCLIB_SEARCH,
                {
                    "track_name": _search_text(title),
                    "artist_name": _search_text(artist),
                },
            )
            usable = _usable_lyrics(results)
            if not usable:
                results = await runtime.http.get_json(
                    LRCLIB_SEARCH, {"q": _search_text(title)}
                )
                usable = _usable_lyrics(results)
            if not usable:
                return None

            target = normalize_text(title)
            normalized_artist = normalize_text(artist)
            usable.sort(
                key=lambda item: (
                    fuzz.WRatio(target, normalize_text(item.get("trackName") or ""))
                    < TITLE_THRESHOLD,
                    fuzz.WRatio(normalized_artist, normalize_text(item.get("artistName") or ""))
                    < ARTIST_THRESHOLD,
                    _duration_gap(item, duration_ms),
                )
            )
            best = usable[0]
            if (
                fuzz.WRatio(target, normalize_text(best.get("trackName") or ""))
                < TITLE_THRESHOLD
            ):
                return None
            return best["plainLyrics"]
        except SOURCE_ERRORS:  # a dead or malformed source must not break the graph
            return None

    @tool
    async def get_wiki_text(title: str, artist: str) -> str | None:
        """Return the English Wikipedia extract for a song, or None."""
        try:
            search_title = _search_text(title)
            search_artist = _search_text(artist)
            queries = [
                f'"{search_title}" {search_artist} song',
                f"{search_title} {search_artist}",
            ]
            target = normalize_text(title)
            for query in queries:
                payload = await runtime.http.get_json(
                    WIKI_API,
                    {
                        "action": "query",
                        "list": "search",
                        "srsearch": query,
                        "format": "json",
                        "srlimit": 5,
                    },
                )
                for hit in payload.get("query", {}).get("search", []):
                    page_title = hit.get("title") or ""
                    if fuzz.WRatio(target, normalize_text(page_title)) < TITLE_THRESHOLD:
                        continue
                    page = await runtime.http.get_json(
                        WIKI_API,
                        {
                            "action": "query",
                            "prop": "extracts",
                            "explaintext": 1,
                            "titles": page_title,
                            "format": "json",
                            "redirects": 1,
                        },
                    )
                    pages = page.get("query", {}).get("pages", {})
                    text = next(iter(pages.values()), {}).get("extract") or ""
                    if text.strip():
                        return text
            return None
        except SOURCE_ERRORS:  # a dead or malformed source must not break the graph
            return None

    @tool
    async def get_mb_samples(mbid: str) -> list[dict[str, Any]]:
        """Return the recordings a track samples, according to MusicBrainz."""
        if not mbid:
            return []
        try:
            detail = await runtime.musicbrainz.lookup_recording(
                mbid, inc="artist-credits+recording-rels"
            )
            samples: list[dict[str, Any]] = []
            for relation in detail.get("relations") or []:
                if relation.get("type") != "samples material":
                    continue
                if relation.get("direction") != "forward":
                    continue
                target = relation.get("recording") or {}
                if not target.get("id"):
                    continue
                parsed = parse_recording(target)
                samples.append(
                    {
                        "title": parsed.title or target.get("title") or "Unknown",
                        "artist": (
                            None if parsed.artist == "Unknown artist" else parsed.artist
                        ),
                        "year": parsed.year,
                        "album": parsed.album,
                        "mbid": target["id"],
                    }
                )
            return samples
        except SOURCE_ERRORS:  # a dead or malformed source must not break the graph
            return []

    return {
        "get_genres": get_genres,
        "get_lyrics": get_lyrics,
        "get_wiki_text": get_wiki_text,
        "get_mb_samples": get_mb_samples,
    }


def _usable_lyrics(results: Any) -> list[dict[str, Any]]:
    """Keep only results with plain lyrics that are not instrumental."""
    return [
        item
        for item in results or []
        if item.get("plainLyrics") and not item.get("instrumental")
    ]


def _duration_gap(item: dict[str, Any], duration_ms: int | None) -> float:
    """Distance in seconds between a result and the chosen track's duration."""
    if not duration_ms or not item.get("duration"):
        return 9999.0
    return abs(float(item["duration"]) * 1000 - duration_ms) / 1000.0


def _same_artist(item: dict[str, Any], artist: str) -> bool:
    """True when a Deezer result credits the requested artist closely enough."""
    name = (item.get("artist") or {}).get("name") or ""
    return fuzz.WRatio(normalize_text(name), normalize_text(artist)) >= ARTIST_MATCH_SCORE


def _unique(values: Iterable[Any]) -> list[str]:
    """Drop empty and case-insensitive duplicate names, keeping the first form."""
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        name = " ".join(str(value or "").split())
        key = name.lower()
        if name and key not in seen:
            seen.add(key)
            result.append(name)
    return result
