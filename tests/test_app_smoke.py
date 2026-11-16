from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

from src import APP_NAME

APP = Path(__file__).resolve().parents[1] / "app.py"


def test_app_renders_without_exceptions() -> None:
    app = AppTest.from_file(str(APP), default_timeout=20)
    app.run()

    assert not app.exception
    assert app.title[0].value == APP_NAME
    assert app.text_input[0].label == "Song"
    assert app.text_input[1].label == "Artist (optional)"


def test_search_form_submits_the_typed_values(monkeypatch) -> None:
    calls: list[tuple[str, str | None]] = []

    def fake_start_search(
        query: str, artist: str | None, *, checkpointer: object
    ) -> dict[str, object]:
        calls.append((query, artist))
        return {"thread_id": "thread", "status": "needs_input", "candidates": []}

    monkeypatch.setattr("src.service.start_search", fake_start_search)

    app = AppTest.from_file(str(APP), default_timeout=20)
    app.run()

    assert app.form_submit_button[0].label == "Search"

    # Enter or a click on Search submits the form: with no song, nothing starts.
    app.form_submit_button[0].click().run()

    assert calls == []
    assert any(warning.value == "Type a song first." for warning in app.warning)

    app.text_input(key="song_input").set_value(" The Message ")
    app.text_input(key="artist_input").set_value(" Grandmaster Flash ")
    app.form_submit_button[0].click().run()

    assert not app.exception
    assert calls == [("The Message", "Grandmaster Flash")]


def test_report_renders_the_tree_without_showing_evidence() -> None:
    app = AppTest.from_file(str(APP), default_timeout=20)
    app.session_state["result"] = {
        "thread_id": "thread",
        "status": "done",
        "resolution": "auto",
        "report": {
            "track": {
                "title": "Track Title",
                "artist": "Artist",
                "album": "Album",
                "year": 2000,
            },
            "genres": ["Hip-Hop"],
            "lyrics_summary": {
                "topic": "Life",
                "summary": "Summary.",
                "artist_view": None,
            },
            "samples": [
                {
                    "title": "Parent Sample",
                    "artist": "Parent Artist",
                    "year": 1990,
                    "source": "MusicBrainz",
                    "confidence": "high",
                    "samples": [
                        {
                            "title": "Child Sample",
                            "artist": None,
                            "source": "Wikipedia",
                            "confidence": "medium",
                            "evidence": "SECRET WIKI EVIDENCE",
                            "samples": [],
                        }
                    ],
                }
            ],
            "sources": ["MusicBrainz"],
            "notes": ["A note."],
        },
    }

    app.run()

    assert not app.exception
    texts = [item.value for item in app.markdown]
    texts += [item.value for item in app.caption]
    assert any("Parent Sample" in text for text in texts)
    assert any("Child Sample" in text for text in texts)
    assert not any("SECRET WIKI EVIDENCE" in text for text in texts)
    assert not any("](https://open.spotify.com" in text for text in texts)
    assert any("Parent Artist" in text for text in texts)


def test_report_names_become_spotify_links() -> None:
    app = AppTest.from_file(str(APP), default_timeout=20)
    app.session_state["result"] = {
        "thread_id": "thread",
        "status": "done",
        "resolution": "auto",
        "report": {
            "track": {
                "title": "Track Title",
                "artist": "Artist",
                "album": "Album",
                "year": 2000,
            },
            "genres": [],
            "samples": [
                {
                    "title": "Parent Sample",
                    "artist": "Parent Artist",
                    "source": "MusicBrainz",
                    "confidence": "high",
                    "spotify_url": "https://open.spotify.com/track/sample-1",
                    "spotify_artist_url": (
                        "https://open.spotify.com/artist/sample-artist-1"
                    ),
                    "samples": [],
                }
            ],
            "sources": [],
            "notes": [],
            "spotify_track_url": "https://open.spotify.com/track/track-1",
            "spotify_artist_url": "https://open.spotify.com/artist/artist-1",
            "spotify_album_url": "https://open.spotify.com/album/album-1",
        },
    }

    app.run()

    assert not app.exception
    subheaders = [item.value for item in app.subheader]
    assert any(
        "[Track Title](https://open.spotify.com/track/track-1)" in value
        for value in subheaders
    )
    captions = [item.value for item in app.caption]
    assert any(
        "[Artist](https://open.spotify.com/artist/artist-1)" in value
        for value in captions
    )
    assert any(
        "[Album](https://open.spotify.com/album/album-1)" in value
        for value in captions
    )
    markdowns = [item.value for item in app.markdown]
    assert any(
        "[Parent Sample](https://open.spotify.com/track/sample-1)" in value
        for value in markdowns
    )
    assert any(
        "[Parent Artist](https://open.spotify.com/artist/sample-artist-1)" in value
        for value in markdowns
    )
