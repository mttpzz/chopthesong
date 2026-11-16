"""Audio recognition: Chromaprint fingerprinting and the AcoustID lookup.

The clip never leaves the machine it runs on: ``fpcalc`` reads it locally and
only the resulting fingerprint string is sent to AcoustID. The temporary file
is removed in a ``finally`` block, and the lookup bypasses the on-disk HTTP
cache, so neither the clip, the fingerprint nor the AcoustID response are
written to disk or to the logs.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import httpx

from src.config import Settings
from src.http import Http
from src.musicbrainz import MusicBrainzClient, parse_recording

logger = logging.getLogger(__name__)

ACOUSTID_URL = "https://api.acoustid.org/v2/lookup"

# AcoustID scores run from 0 to 1. Below 0.5 the server is closer to guessing:
# the MusicBrainz confirmation and the normal resolver do the rest of the work.
MIN_SCORE = 0.5

FPCALC_TIMEOUT_SECONDS = 20.0


class AudioRecognitionError(Exception):
    """Base class for failures the interface reports with a fixed message."""


class FpcalcMissingError(AudioRecognitionError):
    """``fpcalc`` is not installed, so the audio widget stays hidden."""


class FingerprintError(AudioRecognitionError):
    """The clip could not be fingerprinted."""


class FingerprintTimeoutError(FingerprintError):
    """``fpcalc`` did not finish in time."""


class AcoustIdError(AudioRecognitionError):
    """The AcoustID lookup failed."""


class AcoustIdTimeoutError(AcoustIdError):
    """The AcoustID request timed out."""


class AcoustIdQuotaError(AcoustIdError):
    """AcoustID refused the request because of its rate limit."""


@dataclass(frozen=True)
class RecognitionResult:
    """One recognized track, ready to feed the normal search."""

    title: str
    artist: str
    score: float
    recording_id: str | None = None


def fpcalc_available() -> bool:
    """True when the ``fpcalc`` binary is on PATH."""
    return shutil.which("fpcalc") is not None


async def fingerprint(audio_bytes: bytes) -> tuple[int, str]:
    """Return the clip duration in seconds and its Chromaprint fingerprint.

    The clip is written to a private temporary file for ``fpcalc`` and deleted
    as soon as the subprocess ends, whatever the outcome.
    """
    path = _write_temp(audio_bytes)
    try:
        return await _run_fpcalc(path)
    finally:
        _remove_temp(path)


def recognize(audio_bytes: bytes, settings: Settings) -> RecognitionResult | None:
    """Fingerprint a clip and look it up, in one short-lived event loop."""
    return asyncio.run(_recognize(audio_bytes, settings))


async def _recognize(
    audio_bytes: bytes, settings: Settings
) -> RecognitionResult | None:
    duration, value = await fingerprint(audio_bytes)
    http = Http(
        settings.cache_dir,
        ttl=settings.cache_ttl_seconds,
        user_agent=settings.user_agent,
    )
    try:
        return await lookup(value, duration, settings, http)
    finally:
        await http.aclose()


async def lookup(
    fingerprint: str,
    duration: int,
    settings: Settings,
    http: Http,
) -> RecognitionResult | None:
    """Return the best AcoustID match, or None when nothing passes the threshold.

    Only the fingerprint and the duration are sent; the audio never is. The
    response is not cached. With a MusicBrainz recording id, the canonical
    title and artist are fetched from MusicBrainz; if that call fails, the
    AcoustID metadata is used as it is.
    """
    data = await _request(fingerprint, duration, settings, http)
    candidate = _best_match(data)
    _log_lookup(data, duration, len(fingerprint), candidate is not None)
    if candidate is None:
        return None
    title, artist, recording_id, score = candidate
    title, artist = await _confirm(recording_id, title, artist, http)
    return RecognitionResult(
        title=title,
        artist=artist,
        score=score,
        recording_id=recording_id,
    )


async def _request(
    fingerprint: str,
    duration: int,
    settings: Settings,
    http: Http,
) -> dict:
    """Call the AcoustID lookup endpoint without touching the disk cache."""
    params = {
        "format": "json",
        "client": settings.acoustid_api_key or "",
        "duration": duration,
        "fingerprint": fingerprint,
        "meta": "recordings",
    }
    try:
        # AcoustID prefers POST for long fingerprints: they stay out of the URL.
        data = await http.post_json(
            ACOUSTID_URL,
            data=params,
            cache=False,
            retry_transient=False,
        )
    except httpx.TimeoutException as exc:
        raise AcoustIdTimeoutError("The AcoustID request timed out.") from exc
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 429:
            raise AcoustIdQuotaError("AcoustID rate limit reached.") from exc
        raise AcoustIdError("AcoustID rejected the request.") from exc
    except httpx.HTTPError as exc:
        raise AcoustIdError("AcoustID is unreachable.") from exc

    if not isinstance(data, dict) or data.get("status") != "ok":
        raise AcoustIdError("AcoustID returned an error.")
    return data


def _best_match(data: dict) -> tuple[str, str, str | None, float] | None:
    """Pick the highest-scoring recording above the threshold."""
    best: tuple[str, str, str | None, float] | None = None
    for result in data.get("results") or []:
        try:
            score = float(result.get("score") or 0.0)
        except (TypeError, ValueError):
            continue
        if score < MIN_SCORE:
            continue
        for recording in result.get("recordings") or []:
            title = str(recording.get("title") or "").strip()
            if not title:
                continue
            artist = _artist_name(recording.get("artists"))
            if best is None or score > best[3]:
                best = (title, artist, recording.get("id"), score)
            break
    return best


def _artist_name(artists: list[dict] | None) -> str:
    """Join the credited artists with their join phrases, as MusicBrainz does."""
    return "".join(
        f"{artist.get('name', '')}{artist.get('joinphrase', '')}"
        for artist in artists or []
    ).strip()


def _summarize(data: dict) -> tuple[int, float | None, int, int]:
    """Return result count, best score, results above the threshold, titled recordings."""
    results = data.get("results") or []
    scores: list[float] = []
    above = 0
    titled = 0
    for result in results:
        if not isinstance(result, dict):
            continue
        try:
            score = float(result.get("score") or 0.0)
        except (TypeError, ValueError):
            continue
        scores.append(score)
        if score < MIN_SCORE:
            continue
        above += 1
        for recording in result.get("recordings") or []:
            if isinstance(recording, dict) and str(recording.get("title") or "").strip():
                titled += 1
    best = max(scores) if scores else None
    return len(results), best, above, titled


def _log_lookup(
    data: dict,
    duration: int,
    fingerprint_chars: int,
    matched: bool,
) -> None:
    """Log one summary line per lookup: counts and scores, never the payload.

    The line deliberately leaves out the fingerprint, the API key, the raw
    response, the title and the artist, so it is safe to paste into a report.
    """
    results, best, above, titled = _summarize(data)
    logger.log(
        logging.INFO if matched else logging.WARNING,
        "AcoustID lookup: status=%s results=%s best_score=%s above_threshold=%s "
        "titled=%s match=%s duration=%ss fingerprint_chars=%s",
        data.get("status"),
        results,
        "none" if best is None else f"{best:.3f}",
        above,
        titled,
        matched,
        duration,
        fingerprint_chars,
    )


async def _confirm(
    recording_id: str | None,
    title: str,
    artist: str,
    http: Http,
) -> tuple[str, str]:
    """Replace the AcoustID labels with canonical MusicBrainz ones when possible."""
    if not recording_id:
        return title, artist
    try:
        raw = await MusicBrainzClient(http).lookup_recording(recording_id)
        track = parse_recording(raw)
    except Exception as exc:  # confirmation is optional, never a reason to fail
        logger.warning("MusicBrainz confirmation failed (%s)", type(exc).__name__)
        return title, artist
    if not track.title:
        return title, artist
    canonical_artist = track.artist if track.artist != "Unknown artist" else ""
    return track.title, canonical_artist or artist


def _write_temp(audio_bytes: bytes) -> Path:
    """Write the clip to a private temporary file and return its path."""
    handle, name = tempfile.mkstemp(prefix="chopthesong-", suffix=".audio")
    try:
        with os.fdopen(handle, "wb") as file:
            file.write(audio_bytes)
    except BaseException:
        _remove_temp(Path(name))
        raise
    return Path(name)


def _remove_temp(path: Path) -> None:
    """Delete the temporary clip, ignoring a concurrent removal."""
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:  # cleanup must not hide the real result
        logger.warning("Failed to remove the temporary clip (%s)", type(exc).__name__)


async def _run_fpcalc(path: Path) -> tuple[int, str]:
    """Run ``fpcalc -json`` on one file and parse its output."""
    try:
        process = await asyncio.create_subprocess_exec(
            "fpcalc",
            "-json",
            str(path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
    except FileNotFoundError as exc:
        raise FpcalcMissingError("fpcalc is not installed.") from exc

    try:
        stdout, _ = await asyncio.wait_for(
            process.communicate(), timeout=FPCALC_TIMEOUT_SECONDS
        )
    except TimeoutError as exc:
        process.kill()
        await process.wait()
        raise FingerprintTimeoutError("Fingerprinting timed out.") from exc

    if process.returncode:
        raise FingerprintError("fpcalc could not read the audio.")

    try:
        payload = json.loads(stdout)
        duration = int(round(float(payload["duration"])))
        value = str(payload["fingerprint"])
    except (KeyError, TypeError, ValueError) as exc:
        raise FingerprintError("fpcalc returned an unexpected result.") from exc

    if duration <= 0 or not value:
        raise FingerprintError("The clip is too short to be recognized.")

    return duration, value
