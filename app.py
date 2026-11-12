"""Streamlit interface for the sample-chain explorer.

The whole app runs in this file: no separate backend. The graph checkpointer
lives in ``st.session_state`` for the browser session, so closing the page
starts from scratch and nothing is written to disk.
"""

from __future__ import annotations

from typing import Any

import streamlit as st
from langgraph.checkpoint.memory import InMemorySaver

from src import APP_NAME
from src.service import resume_search, start_search


def _init_state() -> None:
    """Create the session objects once per browser session."""
    if "checkpointer" not in st.session_state:
        st.session_state["checkpointer"] = InMemorySaver()
    if "result" not in st.session_state:
        st.session_state["result"] = None
    if "query" not in st.session_state:
        st.session_state["query"] = ""
    if "artist" not in st.session_state:
        st.session_state["artist"] = None


def _run_search(query: str, artist: str | None) -> None:
    """Start a fresh search and keep it in the session."""
    st.session_state["query"] = query
    st.session_state["artist"] = artist
    with st.spinner("Searching..."):
        st.session_state["result"] = start_search(
            query,
            artist,
            checkpointer=st.session_state["checkpointer"],
        )
    st.rerun()


def _run_resume(thread_id: str, *, index: int | None = None, cancel: bool = False) -> None:
    """Resume the paused thread and keep the new result in the session."""
    with st.spinner("Loading..."):
        st.session_state["result"] = resume_search(
            thread_id,
            index=index,
            cancel=cancel,
            checkpointer=st.session_state["checkpointer"],
        )
    st.rerun()


def _unique(values: list[str]) -> list[str]:
    """Drop repeated labels while keeping the original order."""
    return list(dict.fromkeys(values))


def _candidate_label(item: dict[str, Any]) -> str:
    track = item.get("track") or {}
    head = f"{track.get('title') or 'Unknown'} - {track.get('artist') or 'Unknown artist'}"
    extra = [value for value in (track.get("album"), track.get("year")) if value]
    return head if not extra else f"{head} ({', '.join(str(value) for value in extra)})"


def _render_candidates(result: dict[str, Any]) -> None:
    candidates = result.get("candidates") or []
    st.subheader("Choose a version")
    if not candidates:
        st.warning("No versions available. Try a new search.")
        return

    selected = st.radio(
        "Candidates",
        options=list(range(len(candidates))),
        format_func=lambda index: _candidate_label(candidates[index]),
        label_visibility="collapsed",
    )
    choose, cancel = st.columns(2)
    if choose.button("Choose", type="primary"):
        _run_resume(result["thread_id"], index=selected)
    if cancel.button("Cancel"):
        _run_resume(result["thread_id"], cancel=True)


def _render_error(result: dict[str, Any]) -> None:
    st.error(result.get("error") or "Something went wrong. Try again.")
    if st.button("Try again", type="primary"):
        _run_search(st.session_state["query"], st.session_state["artist"])


def _render_summary(summary: dict[str, Any] | None) -> None:
    if not summary:
        st.write("No lyrics summary available.")
        return
    topic = summary.get("topic") or ""
    if topic:
        st.subheader(topic)
    body = summary.get("summary") or ""
    if body:
        st.write(body)
    artist_view = summary.get("artist_view")
    if artist_view:
        with st.container(key="chopthesong-artist-view"):
            st.caption(artist_view)


def _render_samples(nodes: list[dict[str, Any]]) -> None:
    for node in nodes:
        with st.container(border=True):
            title = node.get("title") or "Unknown"
            artist = node.get("artist")
            st.markdown(f"**{title}**" + (f" - {artist}" if artist else ""))
            details = []
            if node.get("year"):
                details.append(f"Year: {node['year']}")
            details.append(f"Source: {node.get('source')}")
            details.append(f"Confidence: {node.get('confidence')}")
            st.caption(" | ".join(details))
            # The Wikipedia evidence sentence is never stored, so there is nothing to show here.
            _render_samples(node.get("samples") or [])


def _render_done(result: dict[str, Any]) -> None:
    report = result.get("report")
    if not report:
        resolution = result.get("resolution")
        if resolution == "not_found":
            st.warning("No track found. Try different words.")
        elif resolution == "cancelled":
            st.info("Search cancelled.")
        else:
            st.warning("No report available. Try again.")
        return

    track = report.get("track") or {}
    st.subheader(track.get("title") or "Unknown track")
    meta = [track.get("artist"), track.get("album"), track.get("year")]
    caption = " - ".join(str(value) for value in meta if value)
    if caption:
        with st.container(key="chopthesong-track-meta"):
            st.caption(caption)

    genres = _unique(report.get("genres") or [])
    if genres:
        with st.container(key="chopthesong-genres"):
            st.pills("Genres", genres, disabled=True)

    _render_summary(report.get("lyrics_summary"))

    st.subheader("Samples")
    samples = report.get("samples") or []
    if samples:
        _render_samples(samples)
    else:
        st.write("No samples found.")

    sources = _unique(report.get("sources") or [])
    if sources:
        with st.container(key="chopthesong-sources"):
            st.pills("Sources", sources, disabled=True)

    notes = report.get("notes") or []
    if notes:
        with st.container(key="chopthesong-notes"):
            st.write("**Notes**")
            for note in notes:
                st.write(f"- {note}")


def main() -> None:
    st.set_page_config(page_title=APP_NAME, layout="centered")
    st.markdown(
        """
        <style>
        [data-testid="stTextInput"] [data-testid="InputInstructions"] {
            display: none !important;
        }
        [data-testid="stHeadingWithActionElements"] h1,
        [data-testid="stHeadingWithActionElements"] h1 span,
        [data-testid="stHeadingWithActionElements"] h2,
        [data-testid="stHeadingWithActionElements"] h2 span,
        [data-testid="stHeadingWithActionElements"] h3,
        [data-testid="stHeadingWithActionElements"] h3 span {
            color: #6E1414 !important;
        }
        .st-key-chopthesong-artist-view [data-testid="stCaptionContainer"] {
            color: #6E1414 !important;
        }
        .st-key-chopthesong-subtitle [data-testid="stCaptionContainer"] {
            color: #EDEDED !important;
            opacity: 1 !important;
            font-size: 1rem !important;
        }
        .st-key-chopthesong-notes [data-testid="stMarkdownContainer"] {
            color: #EDEDED !important;
            opacity: 1 !important;
        }
        .st-key-chopthesong-notes [data-testid="stMarkdownContainer"] strong {
            color: #6E1414 !important;
            font-size: 1.25rem !important;
        }
        .st-key-chopthesong-track-meta [data-testid="stCaptionContainer"] {
            color: #EDEDED !important;
            opacity: 1 !important;
            font-size: 1rem !important;
        }
        .st-key-chopthesong-genres [data-testid="stWidgetLabel"],
        .st-key-chopthesong-genres [data-testid="stWidgetLabel"]
        [data-testid="stMarkdownContainer"],
        .st-key-chopthesong-genres [data-testid="stWidgetLabel"] p {
            color: #EDEDED !important;
            opacity: 1 !important;
            font-size: 1rem !important;
        }
        .st-key-chopthesong-genres button {
            color: #EDEDED !important;
            background-color: transparent !important;
            border-color: rgba(237, 237, 237, 0.25) !important;
            font-size: 1rem !important;
        }
        .st-key-chopthesong-sources button {
            color: rgba(237, 237, 237, 0.6) !important;
            background-color: transparent !important;
            border-color: rgba(237, 237, 237, 0.25) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    _init_state()

    st.title(APP_NAME)
    with st.container(key="chopthesong-subtitle"):
        st.caption("From a song to its roots: samples, genres and lyrics.")

    with st.form("search_form", border=False):
        song = st.text_input("Song", key="song_input")
        artist = st.text_input("Artist (optional)", key="artist_input")
        submitted = st.form_submit_button("Search", type="primary")

    if submitted:
        if song.strip():
            _run_search(song.strip(), artist.strip() or None)
        else:
            st.warning("Type a song first.")

    result = st.session_state["result"]
    if not result:
        return

    notice = result.get("notice")
    if notice:
        st.info(notice)

    status = result.get("status")
    if status == "needs_input":
        _render_candidates(result)
    elif status == "error":
        _render_error(result)
    else:
        _render_done(result)


main()
