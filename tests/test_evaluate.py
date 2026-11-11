from __future__ import annotations

import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from src.config import Settings
from src.llm import UNAVAILABLE_MESSAGE, LlmUnavailable
from src.schemas import LyricsSummary, QueryExtraction, WikiSampleList

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate.py"
SPEC = importlib.util.spec_from_file_location("evaluate_script", SCRIPT)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - broken checkout
    raise RuntimeError(f"cannot load {SCRIPT}")
evaluate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = evaluate
SPEC.loader.exec_module(evaluate)

NAS = "11111111-1111-1111-1111-111111111111"
FLASH = "22222222-2222-2222-2222-222222222222"
SAMPLE = "33333333-3333-3333-3333-333333333333"
OTHER = "44444444-4444-4444-4444-444444444444"

SECRET_LYRICS = "SECRET LYRICS MUST NOT LEAK"
SECRET_WIKI = "SECRET WIKI MUST NOT LEAK"


def recording(mbid: str, title: str, artist: str, year: str = "2002-11-19") -> dict:
    return {
        "id": mbid,
        "score": 100,
        "title": title,
        "length": 234000,
        "artist-credit": [{"name": artist, "joinphrase": ""}],
        "releases": [{"title": "Album", "date": year}],
    }


class FakeMusicBrainz:
    def __init__(self, recordings: list[dict]) -> None:
        self.recordings = recordings

    async def search_recordings(
        self, title: str, *, artist: str | None = None, limit: int = 15
    ) -> list[dict]:
        if artist:
            return [
                item
                for item in self.recordings
                if artist.lower() in item["artist-credit"][0]["name"].lower()
            ]
        return self.recordings

    async def lookup_recording(self, mbid: str, *, inc: str = "") -> dict:
        return {
            "relations": [
                {
                    "type": "samples material",
                    "direction": "forward",
                    "recording": {
                        "id": SAMPLE,
                        "title": "Sample Song",
                        "artist-credit": [{"name": "Sample Artist", "joinphrase": ""}],
                        "releases": [{"title": "Sample Album", "date": "1990-01-01"}],
                    },
                }
            ]
        }


class FakeHttp:
    """Deezer, LRCLIB and Wikipedia responses with recognizable secret text."""

    async def get_json(self, url: str, params: dict | None = None) -> Any:
        if "api.deezer.com" in url:
            if "/album/" in url:
                return {"genres": {"data": [{"name": "Hip-Hop"}]}}
            return {"data": [{"artist": {"name": "Nas"}, "album": {"id": 7}}]}
        if "lrclib.net" in url:
            return [
                {
                    "trackName": "The Message",
                    "artistName": "Nas",
                    "duration": 234,
                    "plainLyrics": SECRET_LYRICS,
                    "instrumental": False,
                }
            ]
        if "wikipedia.org" in url:
            if params and params.get("list") == "search":
                return {"query": {"search": [{"title": "The Message"}]}}
            return {"query": {"pages": {"1": {"extract": SECRET_WIKI}}}}
        raise AssertionError(f"unexpected URL: {url}")


class FakeLlm:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.schema: type = QueryExtraction

    def with_structured_output(
        self, schema: type, *, method: str | None = None
    ) -> FakeLlm:
        self.schema = schema
        return self

    async def ainvoke(self, messages: list) -> Any:
        if self.fail:
            raise LlmUnavailable(UNAVAILABLE_MESSAGE)
        last = messages[-1]
        content = last["content"] if isinstance(last, dict) else last.content
        if self.schema is QueryExtraction:
            return QueryExtraction(title=content, artist=None)
        if self.schema is LyricsSummary:
            return LyricsSummary(
                topic="A song about life",
                summary="A short summary.",
                artist_view=None,
            )
        assert self.schema is WikiSampleList
        return WikiSampleList(samples=[])


class FakeRuntime:
    def __init__(self, recordings: list[dict], *, fail_llm: bool = False) -> None:
        self.musicbrainz = FakeMusicBrainz(recordings)
        self.llm = FakeLlm(fail=fail_llm)
        self.http = FakeHttp()
        self.settings = SimpleNamespace(report_depth=1, groq_model="test-model")
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


def factory(recordings: list[dict], *, fail_llm: bool = False):
    def make() -> FakeRuntime:
        return FakeRuntime(recordings, fail_llm=fail_llm)

    return make


def ambiguous() -> list[dict]:
    return [
        recording(NAS, "The Message", "Nas", "1996-07-02"),
        recording(FLASH, "The Message", "Grandmaster Flash", "1982-07-01"),
    ]


def make_case(
    expected_mbid: str | None = NAS,
    *,
    artist: str | None = "Nas",
    samples: tuple[str, ...] = (),
) -> Any:
    return evaluate.TrackCase(
        id="the-message",
        query="The Message",
        artist=artist,
        expected_mbid=expected_mbid,
        expected_samples_mbids=samples,
    )


def dataset_entry(
    case_id: str,
    *,
    query: str = "The Message",
    artist: str | None = "Nas",
    expected_mbid: str | None = NAS,
    samples: tuple[str, ...] = (),
) -> dict[str, Any]:
    return {
        "id": case_id,
        "query": query,
        "artist": artist,
        "expected_mbid": expected_mbid,
        "expected_samples_mbids": list(samples),
        "notes": "test entry",
    }


def write_dataset(tmp_path: Path, entries: list[dict[str, Any]]) -> Path:
    path = tmp_path / "tracks.json"
    path.write_text(json.dumps(entries), encoding="utf-8")
    return path


def test_done_with_expected_mbid_is_resolved() -> None:
    result = evaluate.evaluate_case(
        make_case(samples=(SAMPLE,)),
        runtime_factory=factory([recording(NAS, "The Message", "Nas")]),
    )

    assert result.outcome == "resolved"
    assert result.status == "done"
    assert result.has_report is True
    assert result.resolution_ok is True
    assert result.resolved_mbid == NAS
    assert result.resolved_title == "The Message"
    assert result.resolved_artist == "Nas"
    assert result.has_genres is True
    assert result.has_summary is True
    assert result.summary_status == "ok"
    assert result.mb_samples_expected == 1
    assert result.mb_samples_found == 1
    assert result.mb_samples_recall == 1.0
    assert result.wikipedia_samples == 0
    assert result.total_nodes >= 1
    assert result.error is None


@pytest.mark.parametrize(
    ("report", "expected"),
    [
        ({"summary_status": "empty"}, "empty"),
        ({"summary_status": "leak_check"}, "leak_check"),
        ({"summary_status": "not_attempted"}, "not_attempted"),
        (
            {"notes": ["Summary not attempted: no lyrics or Wikipedia text."]},
            "not_attempted",
        ),
        (
            {"notes": ["Summary unavailable: the model returned an empty summary."]},
            "empty",
        ),
        (
            {
                "notes": [
                    "Summary unavailable: the summary copied lyrics and was discarded."
                ]
            },
            "leak_check",
        ),
        ({"notes": ["No data from Deezer."]}, "unknown"),
    ],
)
def test_summary_status_uses_report_field_then_notes(
    report: dict[str, Any], expected: str
) -> None:
    assert evaluate._summary_status(report, False) == expected


def test_summary_status_is_ok_when_a_summary_exists() -> None:
    assert evaluate._summary_status({"notes": []}, True) == "ok"


def test_recall_is_partial_and_null_when_nothing_is_expected() -> None:
    partial = evaluate.evaluate_case(
        make_case(samples=(SAMPLE, OTHER)),
        runtime_factory=factory([recording(NAS, "The Message", "Nas")]),
    )
    assert partial.mb_samples_expected == 2
    assert partial.mb_samples_found == 1
    assert partial.mb_samples_recall == 0.5
    assert partial.mb_samples_missing == [OTHER]

    empty = evaluate.evaluate_case(
        make_case(samples=()),
        runtime_factory=factory([recording(NAS, "The Message", "Nas")]),
    )
    assert empty.mb_samples_expected == 0
    assert empty.mb_samples_recall is None


def test_wrong_track_is_not_ok() -> None:
    result = evaluate.evaluate_case(
        make_case(FLASH),
        runtime_factory=factory([recording(NAS, "The Message", "Nas")]),
    )

    assert result.outcome == "wrong_track"
    assert result.has_report is True
    assert result.resolution_ok is False


def test_not_found_is_not_ok() -> None:
    result = evaluate.evaluate_case(
        make_case(),
        runtime_factory=factory([]),
    )

    assert result.outcome == "not_found"
    assert result.has_report is False
    assert result.resolution_ok is False


def test_needs_input_resumes_with_the_expected_candidate(monkeypatch: Any) -> None:
    calls: list[tuple[int | None, bool]] = []
    real_resume = evaluate.resume_search

    def recording_resume(
        thread_id: str,
        index: int | None = None,
        cancel: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        calls.append((index, cancel))
        return real_resume(thread_id, index=index, cancel=cancel, **kwargs)

    monkeypatch.setattr(evaluate, "resume_search", recording_resume)

    result = evaluate.evaluate_case(
        make_case(artist=None),
        runtime_factory=factory(ambiguous()),
    )

    assert result.outcome == "resolved"
    assert result.resolution_ok is True
    assert result.resume_index is not None
    assert calls == [(result.resume_index, False)]


def test_needs_input_without_expected_mbid_is_ambiguous(monkeypatch: Any) -> None:
    calls: list[tuple[int | None, bool]] = []
    real_resume = evaluate.resume_search

    def recording_resume(
        thread_id: str,
        index: int | None = None,
        cancel: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        calls.append((index, cancel))
        return real_resume(thread_id, index=index, cancel=cancel, **kwargs)

    monkeypatch.setattr(evaluate, "resume_search", recording_resume)

    result = evaluate.evaluate_case(
        make_case(None, artist=None),
        runtime_factory=factory(ambiguous()),
    )

    assert result.outcome == "ambiguous"
    assert result.status == "needs_input"
    assert result.resume_index is None
    assert result.candidate_count == 2
    assert calls == [(None, True)]


def test_error_is_reported_without_a_retry(monkeypatch: Any) -> None:
    calls: list[int] = []
    monkeypatch.setattr(
        evaluate,
        "resume_search",
        lambda *args, **kwargs: calls.append(1),
    )

    result = evaluate.evaluate_case(
        make_case(),
        runtime_factory=factory(
            [recording(NAS, "The Message", "Nas")], fail_llm=True
        ),
    )

    assert result.outcome == "error"
    assert result.status == "error"
    assert result.error == UNAVAILABLE_MESSAGE
    assert calls == []


@pytest.mark.parametrize(
    "entries",
    [
        [
            dataset_entry("same"),
            dataset_entry("same", query="Another Message"),
        ],
        [
            {
                **dataset_entry("bad-uuid"),
                "expected_mbid": "not-a-uuid",
            }
        ],
    ],
)
def test_invalid_dataset_exits_before_any_call(
    tmp_path: Path, monkeypatch: Any, entries: list[dict[str, Any]]
) -> None:
    path = write_dataset(tmp_path, entries)
    monkeypatch.setattr(
        evaluate,
        "start_search",
        lambda *args, **kwargs: pytest.fail("start_search must not be called"),
    )

    code = evaluate.main(["--dry-run", "--dataset", str(path)])

    assert code == 2


def test_dry_run_prints_valid_count(
    tmp_path: Path, capsys: Any
) -> None:
    path = write_dataset(
        tmp_path,
        [dataset_entry("one"), dataset_entry("two", query="Another Message")],
    )

    code = evaluate.main(["--dry-run", "--dataset", str(path)])

    captured = capsys.readouterr()
    assert code == 0
    assert "Valid dataset: 2 tracks" in captured.out


def test_main_writes_outputs_without_raw_text(tmp_path: Path) -> None:
    path = write_dataset(tmp_path, [dataset_entry("the-message", samples=(SAMPLE,))])
    out_dir = tmp_path / "out"
    generated_at = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)

    code = evaluate.main(
        [
            "--dataset",
            str(path),
            "--out",
            str(out_dir),
            "--depth",
            "1",
            "--limit",
            "1",
        ],
        runtime_factory=factory([recording(NAS, "The Message", "Nas")]),
        settings=Settings(_env_file=None),
        now=generated_at,
    )

    assert code == 0
    json_path = out_dir / "evaluation-20260102-030405.json"
    md_path = out_dir / "evaluation-20260102-030405.md"
    assert json_path.exists()
    assert md_path.exists()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["evaluated_tracks"] == 1
    assert payload["dataset_tracks"] == 1
    assert payload["report_depth"] == 1
    assert payload["limit"] == 1
    assert payload["results"][0]["outcome"] == "resolved"
    assert payload["results"][0]["summary_status"] == "ok"
    assert payload["totals"]["resolution_ok"] == 1
    assert payload["totals"]["summary_status_counts"]["ok"] == 1

    text = json_path.read_text(encoding="utf-8") + md_path.read_text(encoding="utf-8")
    assert "Summary status: ok 1" in text
    assert SECRET_LYRICS not in text
    assert SECRET_WIKI not in text
    assert '"lyrics"' not in text
    assert '"wiki_text"' not in text
    assert '"evidence"' not in text
    assert '"prompt"' not in text
    assert '"query"' not in text


def test_limit_evaluates_only_the_requested_prefix(tmp_path: Path) -> None:
    path = write_dataset(
        tmp_path,
        [
            dataset_entry("one"),
            dataset_entry("two", query="Another Message"),
        ],
    )
    out_dir = tmp_path / "out"

    code = evaluate.main(
        ["--dataset", str(path), "--out", str(out_dir), "--limit", "1"],
        runtime_factory=factory([recording(NAS, "The Message", "Nas")]),
        settings=Settings(_env_file=None),
        now=datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC),
    )

    assert code == 0
    payload = json.loads(
        (out_dir / "evaluation-20260102-030405.json").read_text(encoding="utf-8")
    )
    assert payload["dataset_tracks"] == 2
    assert payload["evaluated_tracks"] == 1
    assert len(payload["results"]) == 1
