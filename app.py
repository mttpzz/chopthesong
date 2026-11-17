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

# A free public Space: bound the fields and how many searches one browser session may run.
MAX_FIELD_CHARS = 200
MAX_SEARCHES_PER_SESSION = 10
SEARCH_LIMIT_MESSAGE = (
    "This session has reached the limit of 10 searches. "
    "Reload the page to start a new session."
)

# Third-party text (titles, artists, albums, summaries) must stay text, not Markdown.
_MARKDOWN_CHARS = "\\[]()*_`#<>"


def _md_text(value: Any) -> str:
    """Escape Markdown so catalog or model text stays plain text."""
    return "".join(
        f"\\{char}" if char in _MARKDOWN_CHARS else char for char in str(value)
    )


def _https_url(value: Any) -> str | None:
    """Return the URL only when it is an absolute https link, else None."""
    if isinstance(value, str) and value.startswith("https://"):
        return value
    return None


def _link_or_text(label: Any, url: Any) -> str:
    """Build a Markdown link only for https URLs; otherwise return escaped text."""
    text = _md_text(label)
    safe_url = _https_url(url)
    return f"[{text}]({safe_url})" if safe_url else text


def _artist_markup(track: dict[str, Any], report: dict[str, Any]) -> str:
    """Render every credited artist as its own link, keeping the joinphrases."""
    credits = track.get("artist_credits") or []
    urls = report.get("spotify_artist_urls") or []
    if not any(credit.get("name") for credit in credits):
        return ""
    parts: list[str] = []
    for index, credit in enumerate(credits):
        url = urls[index] if index < len(urls) else None
        parts.append(_link_or_text(credit.get("name") or "", url))
        parts.append(_md_text(credit.get("joinphrase") or ""))
    return "".join(parts).strip()


# Opening animation: the Android icon note, cut into four diagonal pieces.
# Pure HTML, SVG and CSS, rendered with st.markdown(unsafe_allow_html=True)
# because st.html sanitizes the SVG away. Nothing blocks the first render.
INTRO_HTML = """
<div class="chopthesong-intro" aria-hidden="true">
  <svg class="chopthesong-intro-note" viewBox="0 0 120 120">
    <defs>
      <clipPath id="chopthesong-cut-1"><polygon points="0,0 120,0 120,18 0,84"/></clipPath>
      <clipPath id="chopthesong-cut-2"><polygon points="0,84 120,18 120,36 0,102"/></clipPath>
      <clipPath id="chopthesong-cut-3"><polygon points="0,102 120,36 120,54 0,120"/></clipPath>
      <clipPath id="chopthesong-cut-4"><polygon points="0,120 120,54 120,120"/></clipPath>
    </defs>
    <g class="chopthesong-intro-appear">
      <g class="chopthesong-intro-piece chopthesong-intro-piece-1"
         clip-path="url(#chopthesong-cut-1)">
        <ellipse cx="48" cy="92" rx="27" ry="19" transform="rotate(-20 48 92)"/>
        <rect x="68" y="26" width="10" height="70"/>
        <path d="M78 26 C96 30 110 44 112 60 L112 76 C102 64 90 56 78 52 Z"/>
      </g>
      <g class="chopthesong-intro-piece chopthesong-intro-piece-2"
         clip-path="url(#chopthesong-cut-2)">
        <ellipse cx="48" cy="92" rx="27" ry="19" transform="rotate(-20 48 92)"/>
        <rect x="68" y="26" width="10" height="70"/>
        <path d="M78 26 C96 30 110 44 112 60 L112 76 C102 64 90 56 78 52 Z"/>
      </g>
      <g class="chopthesong-intro-piece chopthesong-intro-piece-3"
         clip-path="url(#chopthesong-cut-3)">
        <ellipse cx="48" cy="92" rx="27" ry="19" transform="rotate(-20 48 92)"/>
        <rect x="68" y="26" width="10" height="70"/>
        <path d="M78 26 C96 30 110 44 112 60 L112 76 C102 64 90 56 78 52 Z"/>
      </g>
      <g class="chopthesong-intro-piece chopthesong-intro-piece-4"
         clip-path="url(#chopthesong-cut-4)">
        <ellipse cx="48" cy="92" rx="27" ry="19" transform="rotate(-20 48 92)"/>
        <rect x="68" y="26" width="10" height="70"/>
        <path d="M78 26 C96 30 110 44 112 60 L112 76 C102 64 90 56 78 52 Z"/>
      </g>
      <line class="chopthesong-intro-cut" x1="0" y1="102" x2="120" y2="36"
            pathLength="1"/>
    </g>
  </svg>
</div>
<style>
.chopthesong-intro {
  position: fixed;
  inset: 0;
  z-index: 999999;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #121212;
  pointer-events: none;
  animation: chopthesong-intro-outro 2.5s ease-in-out forwards;
}
.chopthesong-intro-note {
  width: min(46vmin, 320px);
  height: min(46vmin, 320px);
  overflow: visible;
}
.chopthesong-intro-piece ellipse,
.chopthesong-intro-piece rect,
.chopthesong-intro-piece path { fill: #189184; }
.chopthesong-intro-appear {
  transform-box: fill-box;
  transform-origin: center;
  animation: chopthesong-intro-in 0.6s ease-out both;
}
.chopthesong-intro-piece {
  transform-box: fill-box;
  transform-origin: center;
  animation-duration: 2.5s;
  animation-timing-function: cubic-bezier(.2, .7, .2, 1);
  animation-fill-mode: both;
}
/* One keyframes per piece: custom properties inside keyframes are ignored by
   some WebViews, which then animates only the cut line and leaves the note
   whole. Literal transforms work everywhere. */
.chopthesong-intro-piece-1 { animation-name: chopthesong-piece-1; }
.chopthesong-intro-piece-2 { animation-name: chopthesong-piece-2; }
.chopthesong-intro-piece-3 { animation-name: chopthesong-piece-3; }
.chopthesong-intro-piece-4 { animation-name: chopthesong-piece-4; }
.chopthesong-intro-cut {
  stroke: #121212;
  stroke-width: 4;
  stroke-linecap: round;
  stroke-dasharray: 1;
  stroke-dashoffset: 1;
  animation: chopthesong-cut 2.5s linear both;
}
@keyframes chopthesong-intro-in {
  from { opacity: 0; transform: scale(.86); }
  to { opacity: 1; transform: scale(1); }
}
@keyframes chopthesong-piece-1 {
  0%, 32% { transform: translate(0, 0) rotate(0deg); }
  64%, 100% { transform: translate(-14px, -26px) rotate(-9deg); }
}
@keyframes chopthesong-piece-2 {
  0%, 32% { transform: translate(0, 0) rotate(0deg); }
  64%, 100% { transform: translate(16px, -8px) rotate(6deg); }
}
@keyframes chopthesong-piece-3 {
  0%, 32% { transform: translate(0, 0) rotate(0deg); }
  64%, 100% { transform: translate(-16px, 8px) rotate(-6deg); }
}
@keyframes chopthesong-piece-4 {
  0%, 32% { transform: translate(0, 0) rotate(0deg); }
  64%, 100% { transform: translate(14px, 26px) rotate(9deg); }
}
@keyframes chopthesong-cut {
  0%, 22% { stroke-dashoffset: 1; opacity: 0; }
  24% { opacity: 1; }
  34% { stroke-dashoffset: 0; opacity: 1; }
  48%, 100% { stroke-dashoffset: 0; opacity: 0; }
}
@keyframes chopthesong-intro-outro {
  0%, 76% { opacity: 1; visibility: visible; }
  100% { opacity: 0; visibility: hidden; }
}
@media (prefers-reduced-motion: reduce) {
  .chopthesong-intro { animation: chopthesong-intro-still 1s linear forwards; }
  .chopthesong-intro-appear,
  .chopthesong-intro-piece { animation: none; }
  .chopthesong-intro-cut { display: none; }
}
@keyframes chopthesong-intro-still {
  0%, 70% { opacity: 1; visibility: visible; }
  100% { opacity: 0; visibility: hidden; }
}
</style>
"""


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
    if "searches" not in st.session_state:
        st.session_state["searches"] = 0


def _run_search(query: str, artist: str | None) -> None:
    """Start a fresh search and keep it in the session."""
    if st.session_state["searches"] >= MAX_SEARCHES_PER_SESSION:
        st.warning(SEARCH_LIMIT_MESSAGE)
        return
    st.session_state["searches"] += 1
    st.session_state["query"] = query
    st.session_state["artist"] = artist
    with st.spinner("Searching..."):
        st.session_state["result"] = start_search(
            query,
            artist,
            checkpointer=st.session_state["checkpointer"],
        )
    st.rerun()


def _render_intro() -> None:
    """Play the opening animation once per session, then never again.

    It is a full-page opaque overlay and the CSS owns its timing: the script
    returns at once, the page is already built underneath, and the overlay fades
    out to reveal it when the note has broken apart. ``st.html`` strips the inline
    SVG through its DOMPurify profile, so the markup goes through ``st.markdown``
    with raw HTML enabled instead.
    """
    if st.session_state.get("intro_done"):
        return
    st.session_state["intro_done"] = True
    st.markdown(INTRO_HTML, unsafe_allow_html=True)


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
    title = _md_text(track.get("title") or "Unknown")
    artist = _md_text(track.get("artist") or "Unknown artist")
    head = f"{title} - {artist}"
    extra = [
        _md_text(value)
        for value in (track.get("album"), track.get("year"))
        if value
    ]
    return head if not extra else f"{head} ({', '.join(extra)})"


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
        st.subheader(_md_text(topic))
    body = summary.get("summary") or ""
    if body:
        st.write(_md_text(body))
    artist_view = summary.get("artist_view")
    if artist_view:
        with st.container(key="chopthesong-artist-view"):
            st.caption(_md_text(artist_view))


def _render_samples(nodes: list[dict[str, Any]]) -> None:
    for node in nodes:
        with st.container(border=True):
            title = node.get("title") or "Unknown"
            label = _link_or_text(title, node.get("spotify_url"))
            artist = node.get("artist")
            artist_label = (
                _link_or_text(artist, node.get("spotify_artist_url")) if artist else ""
            )
            st.markdown(f"**{label}**" + (f" - {artist_label}" if artist else ""))
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
    title = track.get("title") or "Unknown track"
    track_url = report.get("spotify_track_url")
    st.subheader(_link_or_text(title, track_url))

    artist = track.get("artist")
    album = track.get("album")
    year = track.get("year")
    meta: list[str] = []
    if artist:
        meta.append(
            _artist_markup(track, report)
            or _link_or_text(artist, report.get("spotify_artist_url"))
        )
    if album:
        album_url = report.get("spotify_album_url")
        meta.append(_link_or_text(album, album_url))
    if year:
        meta.append(str(year))
    caption = " - ".join(meta)
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
            color: #189184 !important;
        }
        [data-testid="stMarkdownContainer"] a,
        [data-testid="stCaptionContainer"] a {
            color: #EDEDED !important;
            text-decoration: underline;
        }
        [data-testid="stHeadingWithActionElements"] a {
            color: #189184 !important;
            text-decoration: underline;
        }
        .st-key-chopthesong-artist-view [data-testid="stCaptionContainer"] {
            color: #189184 !important;
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
            color: #189184 !important;
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

    # A new browser session starts with the note breaking apart; a rerun in the
    # same session goes straight to the page.
    _render_intro()

    st.title(APP_NAME)
    with st.container(key="chopthesong-subtitle"):
        st.caption("From a song to its roots: samples, genres and lyrics.")

    with st.form("search_form", border=False):
        song = st.text_input("Song", key="song_input", max_chars=MAX_FIELD_CHARS)
        artist = st.text_input(
            "Artist (optional)", key="artist_input", max_chars=MAX_FIELD_CHARS
        )
        submitted = st.form_submit_button("Search", type="primary")

    # A plain external link, kept outside the form so Enter still submits it.
    st.caption(
        "Don't know the song? Identify it with Shazam, then type the title and artist above."
    )
    st.link_button(
        "Shazam",
        "https://www.shazam.com/",
        key="chopthesong-shazam",
    )

    if submitted:
        song = song.strip()[:MAX_FIELD_CHARS]
        artist = artist.strip()[:MAX_FIELD_CHARS]
        if song:
            _run_search(song, artist or None)
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
