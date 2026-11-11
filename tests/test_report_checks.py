from __future__ import annotations

from src.report_checks import find_evidence, is_attributed, leaks


def test_leaks_detects_six_consecutive_words() -> None:
    lyrics = "one two three four five six seven eight"
    text = "A summary that says one two three four five six out loud"

    assert leaks(text, lyrics) is True


def test_leaks_allows_five_consecutive_words() -> None:
    lyrics = "one two three four five six seven eight"
    text = "A summary that says one two three four five out loud"

    assert leaks(text, lyrics) is False


def test_leaks_is_false_without_lyrics_or_text() -> None:
    assert leaks("some words here", "") is False
    assert leaks("", "one two three four five six") is False


def test_find_evidence_returns_the_sample_sentence() -> None:
    text = (
        "The Message was released in 1982. "
        'It samples the track "Sample Song" by Sample Artist. '
        "The album was a success."
    )

    evidence = find_evidence(text, "Sample Song")

    assert evidence is not None
    assert "Sample Song" in evidence
    assert "samples" in evidence


def test_find_evidence_handles_interpolation_and_accents() -> None:
    text = "The arrangement interpolates a melody from Cafe del Mar during the chorus."

    evidence = find_evidence(text, "Café del Mar")

    assert evidence is not None
    assert "interpolates" in evidence


def test_find_evidence_rejects_a_sentence_without_a_sample_word() -> None:
    text = "The Message was released in 1982 and became a hit."

    assert find_evidence(text, "The Message") is None


def test_find_evidence_rejects_another_title() -> None:
    text = "The song samples the track Sample Song by Sample Artist."

    assert find_evidence(text, "Different Song") is None


def test_is_attributed_accepts_an_exact_substring() -> None:
    source = "The Message is a song by Nas. It samples the track Sample Song."

    assert is_attributed("a song by Nas", source) is True


def test_is_attributed_accepts_a_close_match() -> None:
    text = "the song samples a drum break"
    source = "According to the article, the song samples a drum breaks."

    assert is_attributed(text, source) is True


def test_is_attributed_rejects_unrelated_text() -> None:
    source = "The Message is a song by Nas."

    assert is_attributed("The artist wrote it in Paris", source) is False


def test_is_attributed_is_false_for_empty_values() -> None:
    assert is_attributed("", "some source") is False
    assert is_attributed("some text", "") is False
