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


def test_search_button_stays_disabled_until_a_song_is_typed() -> None:
    app = AppTest.from_file(str(APP), default_timeout=20)
    app.run()

    assert app.button[0].label == "Search"
    assert app.button[0].disabled is True

    app.text_input[0].set_value("The Message").run()

    assert not app.exception
    assert app.button[0].disabled is False


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
