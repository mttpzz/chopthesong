"""Pure checks that keep raw lyrics out of the report and ground Wikipedia samples."""

from __future__ import annotations

import re

from rapidfuzz import fuzz

from src.matching import normalize_text

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")
_SAMPLE_WORDS = ("sampl", "interpolat")
_ATTRIBUTED_SCORE = 90


def leaks(text: str | None, lyrics: str | None, *, size: int = 6) -> bool:
    """True when ``text`` copies ``size`` consecutive words from the lyrics."""
    source = _words(lyrics or "")
    candidate = _words(text or "")
    if not source or not candidate or len(candidate) < size:
        return False
    shingles = {
        tuple(source[index : index + size]) for index in range(len(source) - size + 1)
    }
    return any(
        tuple(candidate[index : index + size]) in shingles
        for index in range(len(candidate) - size + 1)
    )


def find_evidence(text: str | None, title: str | None) -> str | None:
    """Return the first sentence that mentions ``title`` as a sample/interpolation."""
    flat = " ".join((text or "").split())
    probe = normalize_text(title or "")
    if not probe:
        return None
    probe = f" {probe} "
    for sentence in _SENTENCE_RE.split(flat):
        if probe not in f" {normalize_text(sentence)} ":
            continue
        lowered = sentence.lower()
        if any(word in lowered for word in _SAMPLE_WORDS):
            return _trim(sentence)
    return None


def is_attributed(text: str | None, source: str | None) -> bool:
    """True when ``text`` is present in ``source``, allowing small edits."""
    probe = normalize_text(text or "")
    haystack = normalize_text(source or "")
    if not probe or not haystack:
        return False
    return probe in haystack or fuzz.partial_ratio(probe, haystack) >= _ATTRIBUTED_SCORE


def _words(text: str) -> list[str]:
    """Split text into lowercase words, ignoring punctuation."""
    return re.findall(r"[a-z0-9']+", text.lower())


def _trim(text: str, limit: int = 320) -> str:
    """Collapse whitespace and cut on a word boundary at ``limit`` characters."""
    clean = " ".join(text.split())
    return clean if len(clean) <= limit else clean[:limit].rsplit(" ", 1)[0] + "..."
