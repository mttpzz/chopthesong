from __future__ import annotations

from src.matching import canonicalize, is_confident, score_candidate
from src.schemas import TrackRef

THE_MESSAGE = TrackRef(
    mbid="11111111-1111-1111-1111-111111111111",
    title="The Message",
    artist="Nas",
    year=1996,
    album="It Was Written",
    duration_ms=234000,
    release_primary_type="Album",
)


def test_exact_match_is_confident() -> None:
    candidate = score_candidate(THE_MESSAGE, query_title="The Message", query_artist="Nas")
    assert candidate.score == 1.0
    assert is_confident([candidate])


def test_same_title_without_artist_is_not_confident() -> None:
    flash = TrackRef(
        mbid="22222222-2222-2222-2222-222222222222",
        title="The Message",
        artist="Grandmaster Flash",
        year=1982,
        album="The Message",
        duration_ms=234000,
        release_primary_type="Single",
    )
    kept = canonicalize(
        [
            score_candidate(THE_MESSAGE, query_title="The Message"),
            score_candidate(flash, query_title="The Message"),
        ]
    )
    assert len(kept) == 2
    assert not is_confident(kept)


def test_version_penalty_when_query_has_no_version() -> None:
    remix = TrackRef(
        mbid="33333333-3333-3333-3333-333333333333",
        title="The Message (Remix)",
        artist="Nas",
        year=1996,
        album="Single",
        duration_ms=234000,
    )
    plain = score_candidate(THE_MESSAGE, query_title="The Message", query_artist="Nas")
    remixed = score_candidate(remix, query_title="The Message", query_artist="Nas")
    assert plain.score > remixed.score


def test_canonicalize_drops_compilations_and_reissues() -> None:
    best_of = TrackRef(
        mbid="best-of",
        title="The Message",
        artist="Nas",
        year=2001,
        album="The Best of Nas",
        duration_ms=235000,
        release_primary_type="Album",
        is_compilation=True,
    )
    chilled = TrackRef(
        mbid="chilled",
        title="The Message",
        artist="Nas",
        year=2016,
        album="Chilled Hip Hop",
        duration_ms=225000,
        release_primary_type="Album",
        is_compilation=True,
    )

    kept = canonicalize(
        [
            score_candidate(track, query_title="The Message", query_artist="Nas")
            for track in (best_of, chilled, THE_MESSAGE)
        ]
    )

    assert len(kept) == 1
    assert kept[0].track.mbid == THE_MESSAGE.mbid
    assert is_confident(kept)
