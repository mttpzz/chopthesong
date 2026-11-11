"""Title/artist scoring and de-duplication of MusicBrainz search results."""

from __future__ import annotations

import re
import unicodedata

from rapidfuzz import fuzz

from src.schemas import Candidate, TrackRef

# Auto-pick needs a strong winner that is clearly ahead of the runner-up.
AUTO_PICK_SCORE = 0.90
AUTO_PICK_MARGIN = 0.04
AUTO_PICK_TITLE_FLOOR = 0.85

_VERSION_RE = re.compile(
    r"[\(\[]\s*[^\)\]]*\b("
    r"remix|rmx|mix|version|edit|instrumental|live|acoustic|demo|"
    r"radio|extended|sped\s*up|slowed|reprise|session"
    r")\b[^\)\]]*[\)\]]",
    re.IGNORECASE,
)
_VERSION_WORDS = (
    "remix",
    "live",
    "instrumental",
    "acoustic",
    "demo",
    "radio edit",
    "extended",
    "sped up",
    "slowed",
    "reprise",
)
_PRIMARY_TYPES = {"album", "single", "ep"}


def normalize_text(value: str) -> str:
    """Lowercase and strip accents/punctuation so titles compare reliably."""
    decomposed = unicodedata.normalize("NFKD", value)
    ascii_only = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return " ".join(re.sub(r"[^a-z0-9]+", " ", ascii_only.lower()).split())


def version_tag(title: str, disambiguation: str | None = None) -> str | None:
    """Return a normalized tag (``remix``, ``live``, ...) when the title names a version."""
    match = _VERSION_RE.search(title)
    if match:
        return match.group(1).lower().replace(" ", "")
    if disambiguation:
        probe = disambiguation.lower()
        for word in _VERSION_WORDS:
            if re.search(rf"\b{re.escape(word)}\b", probe):
                return word.replace(" ", "")
    return None


def _edition_rank(candidate: Candidate) -> tuple[int, int, int, int, int]:
    """Lower is better: studio, widely released, with a duration, earlier."""
    primary = (candidate.track.release_primary_type or "").lower()
    return (
        1 if candidate.track.is_compilation else 0,
        0 if primary in _PRIMARY_TYPES else 1,
        -candidate.track.release_count,
        0 if candidate.track.duration_ms else 1,
        candidate.track.year or 9999,
    )


def score_candidate(
    track: TrackRef,
    *,
    query_title: str,
    query_artist: str | None = None,
) -> Candidate:
    """Score one search result against the user's title and optional artist.

    Without an artist the title alone decides. Disagreeing about being a
    remix/live version costs points, and so does an artist that barely
    resembles the requested one. ``logical_key`` groups editions of a song.
    """
    title_score = fuzz.WRatio(normalize_text(query_title), normalize_text(track.title)) / 100.0
    artist_score = (
        fuzz.WRatio(normalize_text(query_artist), normalize_text(track.artist)) / 100.0
        if query_artist
        else 0.0
    )

    # Keep the title dominant; let the artist break ties between similar titles.
    score = (
        0.55 * title_score + 0.45 * artist_score if query_artist else title_score
    )
    tag = version_tag(track.title, track.disambiguation)
    if (version_tag(query_title) is not None) != (tag is not None):
        score -= 0.10
    if query_artist and artist_score < 0.55:
        score -= 0.10

    logical_key = "|".join(
        [
            normalize_text(track.title),
            normalize_text(track.primary_artist or track.artist),
            "na" if not track.duration_ms else str(round(track.duration_ms / 3000)),
            tag or "",
        ]
    )
    return Candidate(
        track=track,
        score=round(max(0.0, min(score, 1.0)), 4),
        title_score=round(title_score, 4),
        artist_score=round(artist_score, 4),
        logical_key=logical_key,
    )


def canonicalize(candidates: list[Candidate]) -> list[Candidate]:
    """Collapse MusicBrainz duplicates, reissues and compilations.

    MusicBrainz stores one recording per release, so one song can appear dozens
    of times. Without this every search looks ambiguous. Candidates are grouped
    by title and lead artist: the best-documented edition represents the song,
    while remixes and live versions keep their own identity.
    """
    identities: dict[tuple[str, str, str], list[Candidate]] = {}
    for candidate in candidates:
        track = candidate.track
        identity = (
            normalize_text(track.title),
            normalize_text(track.primary_artist or track.artist),
            version_tag(track.title, track.disambiguation) or "",
        )
        identities.setdefault(identity, []).append(candidate)

    kept: list[Candidate] = []
    for group in identities.values():
        studio = [c for c in group if not c.track.is_compilation]
        winner = min(studio or group, key=_edition_rank)
        # Confidence belongs to the song identity, not to the best release.
        winner.score = max(candidate.score for candidate in group)
        kept.append(winner)

    return sorted(kept, key=lambda c: (-c.score, _edition_rank(c)))


def is_confident(candidates: list[Candidate]) -> bool:
    """True when the top candidate is strong and far enough ahead to skip asking."""
    if not candidates:
        return False
    top = candidates[0]
    if top.score < AUTO_PICK_SCORE or top.title_score < AUTO_PICK_TITLE_FLOOR:
        return False
    return len(candidates) == 1 or top.score - candidates[1].score >= AUTO_PICK_MARGIN
