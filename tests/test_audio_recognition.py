from __future__ import annotations

import asyncio
from pathlib import Path
from urllib.parse import parse_qs

import httpx
import pytest
import respx

from src.audio_recognition import (
    ACOUSTID_URL,
    MIN_SCORE,
    AcoustIdError,
    AcoustIdQuotaError,
    AcoustIdTimeoutError,
    FingerprintError,
    FingerprintTimeoutError,
    FpcalcMissingError,
    fingerprint,
    fpcalc_available,
    lookup,
)
from src.config import Settings
from src.http import Http

FAKE_KEY = "fake-key"
THE_FINGERPRINT = "THE-FINGERPRINT"


@pytest.fixture(autouse=True)
def _no_real_key(monkeypatch):
    """Never pick up a real key from the shell environment."""
    monkeypatch.delenv("ACOUSTID_API_KEY", raising=False)


@pytest.fixture
async def http(tmp_path):
    client = Http(tmp_path, ttl=60, user_agent="test")
    yield client
    await client.aclose()


def settings(tmp_path: Path, **overrides) -> Settings:
    values = {"acoustid_api_key": FAKE_KEY, "data_dir": tmp_path}
    values.update(overrides)
    return Settings(_env_file=None, **values)


def acoustid_response(results: list[dict]) -> httpx.Response:
    return httpx.Response(200, json={"status": "ok", "results": results})


class FakeProcess:
    """Stand-in for the fpcalc subprocess."""

    def __init__(self, stdout: bytes, returncode: int) -> None:
        self._stdout = stdout
        self.returncode = returncode
        self.killed = False

    async def communicate(self) -> tuple[bytes, bytes]:
        return self._stdout, b""

    def kill(self) -> None:
        self.killed = True


class SlowProcess:
    """Stand-in for an fpcalc run that never finishes."""

    def __init__(self) -> None:
        self.returncode = None
        self.killed = False

    async def communicate(self) -> tuple[bytes, bytes]:
        await asyncio.sleep(5)
        return b"", b""

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    async def wait(self) -> int:
        return -9


def fake_fpcalc(monkeypatch, process: FakeProcess | SlowProcess, calls: list) -> None:
    async def executor(*args, **kwargs):
        calls.append(args)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", executor)


async def test_fingerprint_reads_fpcalc_json_and_deletes_the_clip(monkeypatch) -> None:
    calls: list = []
    fake_fpcalc(
        monkeypatch,
        FakeProcess(b'{"duration": 12.6, "fingerprint": "AQADtest"}', 0),
        calls,
    )

    duration, value = await fingerprint(b"audio bytes")

    assert (duration, value) == (13, "AQADtest")
    assert calls[0][:2] == ("fpcalc", "-json")
    # fpcalc receives a path, never the audio itself.
    assert Path(calls[0][2]).suffix == ".audio"
    assert not Path(calls[0][2]).exists()


async def test_fingerprint_deletes_the_clip_when_fpcalc_fails(monkeypatch) -> None:
    calls: list = []
    fake_fpcalc(monkeypatch, FakeProcess(b"", 2), calls)

    with pytest.raises(FingerprintError):
        await fingerprint(b"not audio")

    assert not Path(calls[0][2]).exists()


async def test_fingerprint_times_out_and_cleans_up(monkeypatch, tmp_path) -> None:
    calls: list = []
    process = SlowProcess()
    fake_fpcalc(monkeypatch, process, calls)
    monkeypatch.setattr("src.audio_recognition.FPCALC_TIMEOUT_SECONDS", 0.01)

    with pytest.raises(FingerprintTimeoutError):
        await fingerprint(b"audio bytes")

    assert process.killed
    assert not Path(calls[0][2]).exists()


async def test_fingerprint_reports_a_missing_fpcalc(monkeypatch) -> None:
    async def missing(*args, **kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(asyncio, "create_subprocess_exec", missing)

    with pytest.raises(FpcalcMissingError):
        await fingerprint(b"audio bytes")


async def test_fingerprint_rejects_unreadable_output(monkeypatch) -> None:
    calls: list = []
    fake_fpcalc(monkeypatch, FakeProcess(b"not json", 0), calls)

    with pytest.raises(FingerprintError):
        await fingerprint(b"audio bytes")


def test_fpcalc_available_follows_the_path(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.audio_recognition.shutil.which", lambda name: "/usr/bin/fpcalc"
    )
    assert fpcalc_available() is True

    monkeypatch.setattr("src.audio_recognition.shutil.which", lambda name: None)
    assert fpcalc_available() is False


def test_acoustid_enabled_needs_a_key() -> None:
    assert Settings(_env_file=None, acoustid_api_key=None).acoustid_enabled is False
    assert Settings(_env_file=None, acoustid_api_key="   ").acoustid_enabled is False
    assert Settings(_env_file=None, acoustid_api_key=FAKE_KEY).acoustid_enabled is True


@respx.mock
async def test_lookup_returns_the_best_result_and_confirms_it(
    tmp_path, http, monkeypatch
) -> None:
    confirmed: list[str] = []

    async def fake_lookup(self, mbid, *, inc=""):
        confirmed.append(mbid)
        return {
            "id": mbid,
            "title": "Canonical Title",
            "artist-credit": [{"name": "Canonical Artist"}],
            "releases": [],
        }

    monkeypatch.setattr(
        "src.audio_recognition.MusicBrainzClient.lookup_recording", fake_lookup
    )
    route = respx.post(ACOUSTID_URL).mock(
        return_value=acoustid_response(
            [
                {
                    "score": 0.4,
                    "recordings": [
                        {
                            "id": "low",
                            "title": "Low Title",
                            "artists": [{"name": "Low Artist"}],
                        }
                    ],
                },
                {
                    "score": 0.93,
                    "recordings": [
                        {
                            "id": "high",
                            "title": "Raw Title",
                            "artists": [{"name": "Raw Artist"}],
                        }
                    ],
                },
            ]
        )
    )

    result = await lookup(THE_FINGERPRINT, 120, settings(tmp_path), http)

    assert result is not None
    assert result.title == "Canonical Title"
    assert result.artist == "Canonical Artist"
    assert result.score == 0.93
    assert result.recording_id == "high"
    assert confirmed == ["high"]

    request = route.calls[0].request
    assert request.method == "POST"
    # The fingerprint travels in the body, never in the URL.
    assert "fingerprint" not in request.url.params
    body = parse_qs(request.content.decode())
    assert body["fingerprint"] == [THE_FINGERPRINT]
    assert body["duration"] == ["120"]
    assert body["meta"] == ["recordings"]
    assert body["client"] == [FAKE_KEY]
    # An AcoustID response must never reach the disk cache.
    assert len(http._cache) == 0


@respx.mock
@pytest.mark.parametrize("score", [MIN_SCORE - 0.01, MIN_SCORE])
async def test_lookup_applies_the_score_threshold(tmp_path, http, score) -> None:
    respx.post(ACOUSTID_URL).mock(
        return_value=acoustid_response(
            [
                {
                    "score": score,
                    "recordings": [
                        {"title": "Title", "artists": [{"name": "Artist"}]}
                    ],
                }
            ]
        )
    )

    result = await lookup(THE_FINGERPRINT, 60, settings(tmp_path), http)

    if score >= MIN_SCORE:
        assert result is not None
        assert (result.title, result.artist) == ("Title", "Artist")
    else:
        assert result is None


@respx.mock
async def test_lookup_without_results_returns_none(tmp_path, http) -> None:
    respx.post(ACOUSTID_URL).mock(return_value=acoustid_response([]))

    assert await lookup(THE_FINGERPRINT, 60, settings(tmp_path), http) is None


@respx.mock
async def test_lookup_falls_back_to_acoustid_metadata(tmp_path, http, monkeypatch) -> None:
    async def broken(self, mbid, *, inc=""):
        raise RuntimeError("MusicBrainz is down")

    monkeypatch.setattr(
        "src.audio_recognition.MusicBrainzClient.lookup_recording", broken
    )
    respx.post(ACOUSTID_URL).mock(
        return_value=acoustid_response(
            [
                {
                    "score": 0.8,
                    "recordings": [
                        {
                            "id": "mbid",
                            "title": "Raw Title",
                            "artists": [{"name": "Raw Artist"}],
                        }
                    ],
                }
            ]
        )
    )

    result = await lookup(THE_FINGERPRINT, 60, settings(tmp_path), http)

    assert result is not None
    assert (result.title, result.artist) == ("Raw Title", "Raw Artist")


@respx.mock
async def test_lookup_maps_the_rate_limit_to_a_typed_error(tmp_path, http) -> None:
    respx.post(ACOUSTID_URL).mock(return_value=httpx.Response(429))

    with pytest.raises(AcoustIdQuotaError):
        await lookup(THE_FINGERPRINT, 60, settings(tmp_path), http)


@respx.mock
async def test_lookup_maps_a_timeout_to_a_typed_error(tmp_path, http) -> None:
    respx.post(ACOUSTID_URL).mock(side_effect=httpx.ConnectTimeout("too slow"))

    with pytest.raises(AcoustIdTimeoutError):
        await lookup(THE_FINGERPRINT, 60, settings(tmp_path), http)


@respx.mock
async def test_lookup_maps_an_api_error_to_a_typed_error(tmp_path, http) -> None:
    respx.post(ACOUSTID_URL).mock(
        return_value=httpx.Response(
            200,
            json={"status": "error", "error": {"code": 4, "message": "invalid API key"}},
        )
    )

    with pytest.raises(AcoustIdError) as error:
        await lookup(THE_FINGERPRINT, 60, settings(tmp_path), http)

    assert FAKE_KEY not in str(error.value)
