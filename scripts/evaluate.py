"""Evaluate the resolver on a dataset of known tracks.

The script is meant to be run by hand: it uses the real service, so it consumes
Groq quota and network. Tests inject a fake runtime and never touch the network.

Output rules: only counts, outcomes and public identifiers. The raw user query,
song text, encyclopedia text and prompts never reach the output files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from langgraph.checkpoint.memory import InMemorySaver

from src.config import Settings, get_settings
from src.service import RuntimeFactory, resume_search, start_search

DEFAULT_DATASET = Path("eval/tracks.json")
DEFAULT_OUT = Path("data/eval")
MIN_DEPTH = 0
MAX_DEPTH = 3

OUTCOMES = (
    "resolved",
    "wrong_track",
    "unverified",
    "ambiguous",
    "not_found",
    "cancelled",
    "error",
)

_ALLOWED_KEYS = frozenset(
    {"id", "query", "artist", "expected_mbid", "expected_samples_mbids", "notes"}
)
_MISSING_PREFIX = "No data from "
_SUMMARY_FIELD = "lyrics_summary"


class DatasetError(ValueError):
    """The dataset is malformed; evaluation must stop before any search."""


@dataclass(frozen=True)
class TrackCase:
    """One dataset entry, validated and normalized."""

    id: str
    query: str
    artist: str | None
    expected_mbid: str | None
    expected_samples_mbids: tuple[str, ...]
    notes: str = ""


@dataclass(frozen=True)
class Dataset:
    """A validated dataset plus a content hash used as its version."""

    path: Path
    version: str
    cases: tuple[TrackCase, ...]


@dataclass
class TrackResult:
    """Metrics for one track; every field is safe to write to disk."""

    id: str
    expected_mbid: str | None
    outcome: str
    status: str
    has_report: bool = False
    resolution_ok: bool | None = None
    resolved_mbid: str | None = None
    resolved_title: str | None = None
    resolved_artist: str | None = None
    has_genres: bool | None = None
    has_summary: bool | None = None
    summary_status: str | None = None
    mb_samples_expected: int = 0
    mb_samples_found: int = 0
    mb_samples_recall: float | None = None
    mb_samples_missing: list[str] = field(default_factory=list)
    wikipedia_samples: int | None = None
    missing_sources: list[str] = field(default_factory=list)
    total_nodes: int = 0
    candidate_count: int | None = None
    resume_index: int | None = None
    duration_ms: float = 0.0
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON representation, with no raw text fields."""
        return {
            "id": self.id,
            "expected_mbid": self.expected_mbid,
            "outcome": self.outcome,
            "status": self.status,
            "has_report": self.has_report,
            "resolution_ok": self.resolution_ok,
            "resolved_mbid": self.resolved_mbid,
            "resolved_title": self.resolved_title,
            "resolved_artist": self.resolved_artist,
            "has_genres": self.has_genres,
            "has_summary": self.has_summary,
            "summary_status": self.summary_status,
            "mb_samples_expected": self.mb_samples_expected,
            "mb_samples_found": self.mb_samples_found,
            "mb_samples_recall": self.mb_samples_recall,
            "mb_samples_missing": self.mb_samples_missing,
            "wikipedia_samples": self.wikipedia_samples,
            "missing_sources": self.missing_sources,
            "total_nodes": self.total_nodes,
            "candidate_count": self.candidate_count,
            "resume_index": self.resume_index,
            "duration_ms": round(self.duration_ms, 1),
            "error": self.error,
        }


def load_dataset(path: Path) -> Dataset:
    """Read and validate a dataset, raising ``DatasetError`` with a clear message."""
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise DatasetError(f"cannot read {path}: {exc}") from exc

    try:
        text = raw.decode("utf-8-sig")
        payload = json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DatasetError(f"{path} is not valid UTF-8 JSON: {exc}") from exc

    if not isinstance(payload, list):
        raise DatasetError("the dataset must be a JSON array")
    if not payload:
        raise DatasetError("the dataset has no entries")

    seen: set[str] = set()
    cases = tuple(
        _parse_case(item, index=index, seen=seen)
        for index, item in enumerate(payload, 1)
    )
    version = f"sha256:{hashlib.sha256(raw).hexdigest()}"
    return Dataset(path=path, version=version, cases=cases)


def _parse_case(item: Any, *, index: int, seen: set[str]) -> TrackCase:
    if not isinstance(item, dict):
        raise DatasetError(f"entry {index} must be a JSON object")

    unknown = set(item) - _ALLOWED_KEYS
    if unknown:
        names = ", ".join(sorted(str(name) for name in unknown))
        raise DatasetError(f"entry {index} has unknown fields: {names}")

    case_id = _required_text(item.get("id"), f"entry {index} id")
    if case_id in seen:
        raise DatasetError(f"duplicate id: {case_id}")
    seen.add(case_id)

    query = _required_text(item.get("query"), f"entry {index} query")
    artist = _optional_text(item.get("artist"), f"entry {index} artist")
    expected_mbid = _optional_uuid(
        item.get("expected_mbid"), f"entry {index} expected_mbid"
    )

    samples_value = item.get("expected_samples_mbids", [])
    if not isinstance(samples_value, list):
        raise DatasetError(f"entry {index} expected_samples_mbids must be a list")
    expected_samples = tuple(
        _required_uuid(value, f"entry {index} expected_samples_mbids[{position}]")
        for position, value in enumerate(samples_value)
    )

    notes = item.get("notes", "")
    if not isinstance(notes, str):
        raise DatasetError(f"entry {index} notes must be a string")

    return TrackCase(
        id=case_id,
        query=query,
        artist=artist,
        expected_mbid=expected_mbid,
        expected_samples_mbids=expected_samples,
        notes=notes.strip(),
    )


def _required_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DatasetError(f"{label} must be a non-empty string")
    return value.strip()


def _optional_text(value: Any, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise DatasetError(f"{label} must be a string or null")
    text = value.strip()
    return text or None


def _required_uuid(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise DatasetError(f"{label} must be a UUID string")
    try:
        return str(UUID(value))
    except (ValueError, AttributeError) as exc:
        raise DatasetError(f"{label} is not a valid UUID: {value}") from exc


def _optional_uuid(value: Any, label: str) -> str | None:
    if value is None:
        return None
    return _required_uuid(value, label)


def _candidate_index(candidates: Sequence[Any], expected_mbid: str | None) -> int | None:
    """Return the candidate index for the expected MBID, or ``None``."""
    if not expected_mbid:
        return None
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            continue
        track = candidate.get("track")
        if isinstance(track, dict) and track.get("mbid") == expected_mbid:
            return index
    return None


def _iter_samples(nodes: Any) -> Iterator[dict[str, Any]]:
    if not isinstance(nodes, list):
        return
    for node in nodes:
        if not isinstance(node, dict):
            continue
        yield node
        yield from _iter_samples(node.get("samples"))


def _text_or_none(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _missing_sources(report: dict[str, Any]) -> list[str]:
    """Prefer a structured field, then parse the fixed notes produced by the report."""
    value = report.get("missing_sources")
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str) and item.strip()]

    missing: list[str] = []
    notes = report.get("notes")
    if not isinstance(notes, list):
        return missing
    for note in notes:
        if not isinstance(note, str) or not note.startswith(_MISSING_PREFIX):
            continue
        name = note[len(_MISSING_PREFIX) :].strip()
        if name.endswith("."):
            name = name[:-1].strip()
        if name:
            missing.append(name)
    return missing


def _summary_status(report: dict[str, Any], has_summary: bool) -> str:
    """Return the summary diagnostic, falling back to notes for older reports."""
    if has_summary:
        return "ok"
    value = report.get("summary_status")
    if value in {"empty", "leak_check", "not_attempted"}:
        return str(value)
    notes = report.get("notes")
    text = (
        " ".join(note for note in notes if isinstance(note, str))
        if isinstance(notes, list)
        else ""
    )
    if "Summary not attempted" in text:
        return "not_attempted"
    if "empty summary" in text:
        return "empty"
    if "passed the checks" in text or "copied lyrics" in text:
        return "leak_check"
    return "unknown"


def _apply_report(result: TrackResult, case: TrackCase, report: dict[str, Any]) -> None:
    """Fill the metrics that only exist when a report was produced."""
    result.has_report = True
    track = report.get("track")
    track = track if isinstance(track, dict) else {}
    result.resolved_mbid = _text_or_none(track.get("mbid"))
    result.resolved_title = _text_or_none(track.get("title"))
    result.resolved_artist = _text_or_none(track.get("artist"))

    if case.expected_mbid is None:
        result.outcome = "unverified"
        result.resolution_ok = None
    else:
        result.resolution_ok = result.resolved_mbid == case.expected_mbid
        result.outcome = "resolved" if result.resolution_ok else "wrong_track"

    genres = report.get("genres")
    result.has_genres = isinstance(genres, list) and bool(genres)

    summary = report.get(_SUMMARY_FIELD)
    result.has_summary = (
        isinstance(summary, dict)
        and bool(_text_or_none(summary.get("topic")))
        and bool(_text_or_none(summary.get("summary")))
    )
    result.summary_status = _summary_status(report, result.has_summary is True)

    samples = report.get("samples")
    flat = list(_iter_samples(samples))
    result.total_nodes = len(flat)
    result.wikipedia_samples = sum(
        1
        for node in (samples if isinstance(samples, list) else [])
        if isinstance(node, dict) and node.get("source") == "Wikipedia"
    )

    found = {
        node.get("mbid")
        for node in flat
        if node.get("source") == "MusicBrainz" and isinstance(node.get("mbid"), str)
    }
    expected = set(case.expected_samples_mbids)
    result.mb_samples_expected = len(case.expected_samples_mbids)
    result.mb_samples_found = len(expected & found)
    result.mb_samples_missing = sorted(expected - found)
    result.mb_samples_recall = (
        result.mb_samples_found / result.mb_samples_expected
        if result.mb_samples_expected
        else None
    )
    result.missing_sources = _missing_sources(report)


def _apply_final(
    result: TrackResult,
    case: TrackCase,
    final: dict[str, Any],
    started_at: float,
) -> None:
    """Translate one final service result into an outcome and its metrics."""
    status = final.get("status")
    result.status = status if isinstance(status, str) else "unknown"

    if status == "error":
        result.outcome = "error"
        result.error = _text_or_none(final.get("error")) or "service error"
    elif status == "needs_input":
        result.outcome = "ambiguous"
    elif status == "done":
        report = final.get("report")
        if isinstance(report, dict):
            _apply_report(result, case, report)
        else:
            resolution = final.get("resolution")
            result.outcome = "cancelled" if resolution == "cancelled" else "not_found"
            if case.expected_mbid is not None:
                result.resolution_ok = False
    else:
        result.outcome = "error"
        result.error = "unexpected service status"

    result.duration_ms = _elapsed_ms(started_at)


def evaluate_case(
    case: TrackCase, *, runtime_factory: RuntimeFactory | None = None
) -> TrackResult:
    """Run one dataset entry through the same service calls the UI uses."""
    started_at = time.perf_counter()
    result = TrackResult(id=case.id, expected_mbid=case.expected_mbid, outcome="", status="")
    saver = InMemorySaver()

    try:
        started = start_search(
            case.query,
            case.artist,
            checkpointer=saver,
            runtime_factory=runtime_factory,
        )
    except Exception as exc:  # a broken service call is reported, not raised
        _finish_error(result, exc, started_at)
        return result

    status = started.get("status")
    if status == "needs_input":
        candidates = started.get("candidates")
        candidates = candidates if isinstance(candidates, list) else []
        result.candidate_count = len(candidates)
        index = _candidate_index(candidates, case.expected_mbid)
        thread_id = started.get("thread_id")
        if not isinstance(thread_id, str):
            result.outcome = "error"
            result.status = "error"
            result.error = "missing thread id"
            result.duration_ms = _elapsed_ms(started_at)
            return result

        try:
            if index is None:
                resume_search(
                    thread_id,
                    cancel=True,
                    checkpointer=saver,
                    runtime_factory=runtime_factory,
                )
                result.outcome = "ambiguous"
                result.status = "needs_input"
                result.duration_ms = _elapsed_ms(started_at)
                return result

            result.resume_index = index
            resumed = resume_search(
                thread_id,
                index=index,
                checkpointer=saver,
                runtime_factory=runtime_factory,
            )
        except Exception as exc:  # a broken service call is reported, not raised
            _finish_error(result, exc, started_at)
            return result

        _apply_final(result, case, resumed, started_at)
        return result

    if status == "error":
        result.outcome = "error"
        result.status = "error"
        result.error = _text_or_none(started.get("error")) or "service error"
        result.duration_ms = _elapsed_ms(started_at)
        return result

    if status == "done":
        _apply_final(result, case, started, started_at)
        return result

    result.outcome = "error"
    result.status = str(status)
    result.error = "unexpected service status"
    result.duration_ms = _elapsed_ms(started_at)
    return result


def _finish_error(result: TrackResult, exc: Exception, started_at: float) -> None:
    result.outcome = "error"
    result.status = "error"
    result.error = _short(str(exc) or type(exc).__name__, 500)
    result.duration_ms = _elapsed_ms(started_at)


def _elapsed_ms(started_at: float) -> float:
    return (time.perf_counter() - started_at) * 1000.0


def run_evaluation(
    cases: Sequence[TrackCase],
    *,
    runtime_factory: RuntimeFactory | None = None,
    progress: bool = True,
) -> list[TrackResult]:
    """Evaluate entries in order, never in parallel, to stay under Groq limits."""
    results: list[TrackResult] = []
    total = len(cases)
    for index, case in enumerate(cases, 1):
        result = evaluate_case(case, runtime_factory=runtime_factory)
        results.append(result)
        if progress:
            print(
                f"[{index}/{total}] {result.id}: {result.outcome} "
                f"({result.duration_ms:.0f} ms)"
            )
    return results


def _totals(results: Sequence[TrackResult]) -> dict[str, Any]:
    verified = [result for result in results if result.expected_mbid is not None]
    rate_pool = [
        result
        for result in verified
        if result.outcome not in {"error", "ambiguous"}
    ]
    report_results = [result for result in results if result.has_report]
    recall_values = [
        result.mb_samples_recall
        for result in report_results
        if result.mb_samples_recall is not None
    ]
    outcomes = {
        outcome: sum(1 for result in results if result.outcome == outcome)
        for outcome in OUTCOMES
    }
    resolution_ok = sum(1 for result in rate_pool if result.resolution_ok is True)
    has_genres = sum(1 for result in report_results if result.has_genres is True)
    has_summary = sum(1 for result in report_results if result.has_summary is True)
    summary_status_counts = {
        status: sum(1 for result in report_results if result.summary_status == status)
        for status in ("ok", "empty", "leak_check", "not_attempted", "unknown")
    }

    return {
        "evaluated_tracks": len(results),
        "verified_tracks": len(verified),
        "unverified_tracks": len(results) - len(verified),
        "resolution_ok": resolution_ok,
        "resolution_rate_pool": len(rate_pool),
        "resolution_ok_rate": (
            resolution_ok / len(rate_pool) if rate_pool else None
        ),
        "report_tracks": len(report_results),
        "has_genres": has_genres,
        "has_genres_rate": (
            has_genres / len(report_results) if report_results else None
        ),
        "has_summary": has_summary,
        "has_summary_rate": (
            has_summary / len(report_results) if report_results else None
        ),
        "summary_status_counts": summary_status_counts,
        "mb_samples_recall_mean": (
            sum(recall_values) / len(recall_values) if recall_values else None
        ),
        "mb_samples_recall_tracks": len(recall_values),
        "outcomes": outcomes,
    }


def _payload(
    dataset: Dataset,
    results: Sequence[TrackResult],
    *,
    depth: int,
    limit: int | None,
    generated_at: datetime,
) -> dict[str, Any]:
    return {
        "generated_at": generated_at.isoformat(timespec="seconds"),
        "dataset": str(dataset.path),
        "dataset_version": dataset.version,
        "report_depth": depth,
        "dataset_tracks": len(dataset.cases),
        "evaluated_tracks": len(results),
        "limit": limit,
        "results": [result.as_dict() for result in results],
        "totals": _totals(results),
    }


def _markdown(
    dataset: Dataset,
    results: Sequence[TrackResult],
    *,
    depth: int,
    limit: int | None,
    generated_at: datetime,
) -> str:
    totals = _totals(results)
    evaluated = (
        f"{len(results)} of {len(dataset.cases)} tracks"
        + (f" (limit {limit})" if limit is not None else "")
    )
    lines = [
        "# Evaluation",
        "",
        f"- Generated: {generated_at.isoformat(timespec='seconds')}",
        f"- Dataset: `{dataset.path}` (version `{dataset.version}`)",
        f"- REPORT_DEPTH: {depth}",
        f"- Evaluated: {evaluated}",
        "",
        "## Per track",
        "",
        (
            "| id | outcome | MBID ok | MB samples | Wikipedia samples | Summary | "
            "Missing sources | Error |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for result in results:
        mbid_ok = (
            "-"
            if result.resolution_ok is None
            else ("yes" if result.resolution_ok else "no")
        )
        samples = (
            f"{result.mb_samples_found}/{result.mb_samples_expected}"
            if result.mb_samples_expected
            else "-"
        )
        wikipedia = (
            "-" if result.wikipedia_samples is None else str(result.wikipedia_samples)
        )
        missing = (
            _md_cell(", ".join(result.missing_sources))
            if result.missing_sources
            else "-"
        )
        error = _md_cell(_short(result.error, 160)) if result.error else "-"
        summary = result.summary_status or "-"
        lines.append(
            f"| {_md_cell(result.id)} | {result.outcome} | {mbid_ok} | {samples} | "
            f"{wikipedia} | {summary} | {missing} | {error} |"
        )

    lines.extend(
        [
            "",
            "## Totals",
            "",
            f"- Evaluated tracks: {len(results)}",
            f"- Tracks with expected MBID: {totals['verified_tracks']}",
            (
                "- Resolution OK: "
                f"{_ratio(totals['resolution_ok'], totals['resolution_rate_pool'])}"
            ),
            f"- Has genres: {_ratio(totals['has_genres'], totals['report_tracks'])}",
            f"- Has summary: {_ratio(totals['has_summary'], totals['report_tracks'])}",
            (
                "- Summary status: "
                f"{_counts(totals['summary_status_counts'])}"
            ),
            f"- MB samples recall (mean): {_mean(totals['mb_samples_recall_mean'])}",
            f"- Errors: {totals['outcomes']['error']}",
            f"- Ambiguous: {totals['outcomes']['ambiguous']}",
            f"- Not found: {totals['outcomes']['not_found']}",
            f"- Cancelled: {totals['outcomes']['cancelled']}",
            f"- Wrong track: {totals['outcomes']['wrong_track']}",
            f"- Unverified: {totals['unverified_tracks']}",
        ]
    )
    return "\n".join(lines) + "\n"


def _ratio(value: int, total: int) -> str:
    if total <= 0:
        return "-"
    return f"{value}/{total} ({value / total * 100:.1f}%)"


def _mean(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def _counts(values: dict[str, int]) -> str:
    """Render labelled counts compactly, e.g. ``ok 11, empty 1``."""
    parts = [f"{name} {count}" for name, count in values.items() if count]
    return ", ".join(parts) if parts else "-"


def _md_cell(value: Any) -> str:
    return _short(str(value), 500).replace("|", "\\|")


def _short(value: str | None, limit: int) -> str:
    text = " ".join((value or "").split())
    return text[:limit] + ("..." if len(text) > limit else "")


def _utc_now(now: datetime | None) -> datetime:
    if now is None:
        return datetime.now(UTC)
    if now.tzinfo is None:
        return now.replace(tzinfo=UTC)
    return now.astimezone(UTC)


def _unique_paths(out_dir: Path, stem: str) -> tuple[Path, Path]:
    json_path = out_dir / f"{stem}.json"
    md_path = out_dir / f"{stem}.md"
    counter = 1
    while json_path.exists() or md_path.exists():
        json_path = out_dir / f"{stem}-{counter}.json"
        md_path = out_dir / f"{stem}-{counter}.md"
        counter += 1
    return json_path, md_path


def write_outputs(
    dataset: Dataset,
    results: Sequence[TrackResult],
    *,
    out_dir: Path,
    depth: int,
    limit: int | None,
    generated_at: datetime,
) -> tuple[Path, Path]:
    """Write the JSON and Markdown reports and return their paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"evaluation-{generated_at.strftime('%Y%m%d-%H%M%S')}"
    json_path, md_path = _unique_paths(out_dir, stem)
    payload = _payload(dataset, results, depth=depth, limit=limit, generated_at=generated_at)
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    md_path.write_text(
        _markdown(dataset, results, depth=depth, limit=limit, generated_at=generated_at),
        encoding="utf-8",
    )
    return json_path, md_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the resolver on a dataset of known tracks."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--depth",
        type=int,
        default=None,
        help="Expansion depth; defaults to REPORT_DEPTH from Settings.",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the dataset and print the number of valid entries; no network.",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    runtime_factory: RuntimeFactory | None = None,
    settings: Settings | None = None,
    now: datetime | None = None,
) -> int:
    """Run the CLI; return 0 unless the dataset, settings or output are invalid."""
    args = _parser().parse_args(argv)

    try:
        dataset = load_dataset(args.dataset)
    except DatasetError as exc:
        print(f"Dataset error: {exc}", file=sys.stderr)
        return 2

    if args.limit is not None and args.limit <= 0:
        print("--limit must be greater than zero", file=sys.stderr)
        return 2

    if args.dry_run:
        suffix = (
            f"; limit will run {min(args.limit, len(dataset.cases))}"
            if args.limit is not None
            else ""
        )
        print(
            f"Valid dataset: {len(dataset.cases)} tracks "
            f"(version {dataset.version}){suffix}"
        )
        return 0

    try:
        effective_settings = settings or Settings()
    except Exception as exc:
        print(f"Settings error: {type(exc).__name__}", file=sys.stderr)
        return 2

    depth = args.depth if args.depth is not None else effective_settings.report_depth
    if not MIN_DEPTH <= depth <= MAX_DEPTH:
        print(f"--depth must be between {MIN_DEPTH} and {MAX_DEPTH}", file=sys.stderr)
        return 2

    if runtime_factory is None:
        if not (effective_settings.groq_api_key or "").strip():
            print("GROQ_API_KEY is missing; evaluation not started.", file=sys.stderr)
            return 2
        os.environ["REPORT_DEPTH"] = str(depth)
        get_settings.cache_clear()

    cases = dataset.cases[: args.limit] if args.limit is not None else dataset.cases
    generated_at = _utc_now(now)

    try:
        results = run_evaluation(cases, runtime_factory=runtime_factory)
        json_path, md_path = write_outputs(
            dataset,
            results,
            out_dir=args.out,
            depth=depth,
            limit=args.limit,
            generated_at=generated_at,
        )
    except OSError as exc:
        print(f"Output error: {exc}", file=sys.stderr)
        return 2

    totals = _totals(results)
    outcomes = totals["outcomes"]
    print(
        "Evaluation complete: "
        f"{totals['evaluated_tracks']} tracks, "
        f"resolved={outcomes['resolved']}, "
        f"wrong_track={outcomes['wrong_track']}, "
        f"unverified={outcomes['unverified']}, "
        f"ambiguous={outcomes['ambiguous']}, "
        f"not_found={outcomes['not_found']}, "
        f"cancelled={outcomes['cancelled']}, "
        f"error={outcomes['error']}"
    )
    print(f"JSON: {json_path}")
    print(f"Markdown: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
