"""Pydantic models shared by the graph, the LLM extraction and the report."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

SummaryStatus = Literal["ok", "empty", "leak_check", "not_attempted"]


class TrackRef(BaseModel):
    """A MusicBrainz recording reduced to the fields the app displays."""

    mbid: str | None = None
    title: str
    artist: str
    primary_artist: str | None = None
    year: int | None = None
    album: str | None = None
    duration_ms: int | None = None
    disambiguation: str | None = None
    release_primary_type: str | None = None
    release_group_id: str | None = None
    release_count: int = 0
    is_compilation: bool = False


class Candidate(BaseModel):
    """A track plus the scores that decide whether it can be picked automatically."""

    track: TrackRef
    score: float
    title_score: float
    artist_score: float
    logical_key: str


class QueryExtraction(BaseModel):
    """Title and artist extracted from a free-text user request."""

    title: str = Field(description="Song title, or an empty string when unclear.")
    artist: str | None = Field(
        default=None, description="Artist, or null when the request does not name one."
    )


class LyricsSummary(BaseModel):
    """Short LLM-written description of what a song is about."""

    topic: str
    summary: str
    artist_view: str | None = None


class SampleNode(BaseModel):
    """One node of the sample graph; ``samples`` nests the chain recursively."""

    title: str
    artist: str | None = None
    year: int | None = None
    album: str | None = None
    mbid: str | None = None
    relation: str = "sample"
    source: str
    # high: MusicBrainz "samples material"; medium: Wikipedia plus a quote.
    confidence: Literal["high", "medium"] = "high"
    samples: list[SampleNode] = []


class WikiSample(BaseModel):
    """A sample candidate the LLM found in the Wikipedia extract."""

    title: str
    artist: str | None = None


class WikiSampleList(BaseModel):
    """Structured output of the Wikipedia sample extraction."""

    samples: list[WikiSample] = []


class TrackReport(BaseModel):
    """Everything the app shows for a resolved track."""

    track: TrackRef
    genres: list[str] = []
    # Named after LyricsSummary: ResolveState.lyrics holds the raw lyrics.
    lyrics_summary: LyricsSummary | None = None
    summary_status: SummaryStatus | None = None
    samples: list[SampleNode] = []
    sources: list[str] = []
    notes: list[str] = []
