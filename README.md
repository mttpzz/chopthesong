# Chop the Song

[![CI](https://github.com/mttpzz/chopthesong/actions/workflows/ci.yml/badge.svg)](https://github.com/mttpzz/chopthesong/actions/workflows/ci.yml)

Extract a title and artist from a free-text request, resolve the track on MusicBrainz,
and ask the user to choose when the search is ambiguous. The report (genres, lyrics,
samples) and the UI are the next steps.

Everything is free: MusicBrainz, Groq, LangGraph, and later Streamlit.

## What exists today

1. Extraction of title and artist from the user's request with Groq (`ChatGroq`) and
   a Pydantic schema.
2. Search on MusicBrainz (up to 25 results).
3. Title/artist scoring and canonicalization: one track per title + primary artist,
   with the best-documented edition (album/year), discarding compilations.
4. Auto-pick when the match is clear; otherwise `interrupt()` and the user chooses.
5. Clean failure when Groq returns 429, 5xx, a network error, or an invalid key: the
   graph ends with a single user-facing message, without a partial report.

## Run

Python 3.12 and [uv](https://docs.astral.sh/uv/) are required.

```powershell
uv sync --python 3.12
Copy-Item .env.example .env
```

In `.env`, set a `User-Agent` with a real contact (MusicBrainz requires it) and
`GROQ_API_KEY` (required to build the LLM).

## Tests

```powershell
uv run pytest
uv run ruff check .
```

Tests use fake clients and fake models: no real calls.

## Structure

```
src/
  llm.py          # ChatGroq client + typed errors
  resolver.py     # LangGraph graph + HITL
  matching.py     # scoring and canonicalization
  musicbrainz.py  # API client
  http.py         # disk cache + rate limiter
  schemas.py      # models
  config.py       # settings
tests/
```

## Notes

- HITL state is in memory (`InMemorySaver`): a restart cancels pending choices.
- The LLM has no fallback provider. Rate limits and provider errors are shown to the
  user with a retry message.
- Report and Streamlit UI are not connected yet: see `TODO`.

## Deploy

The app runs on [Streamlit Community Cloud](https://share.streamlit.io), which
serves it for free from a public GitHub repository. `requirements.txt` is installed
on the server; it does not install the local package, because `app.py` imports
`src` from the repository root.

Add the configuration in the app's **Settings → Secrets** page, in TOML format.
Never commit these values:

| Name | Required | Notes |
| --- | --- | --- |
| `GROQ_API_KEY` | yes | Secret. Never commit it. |
| `GROQ_MODEL` | no | Defaults to the model configured in `config.py`. |
| `USER_AGENT` | yes | A descriptive MusicBrainz User-Agent with a real contact. |
| `LANGFUSE_PUBLIC_KEY` | no | Leave empty to disable tracing. |
| `LANGFUSE_SECRET_KEY` | no | Leave empty to disable tracing. |
| `LANGFUSE_HOST` | no | Defaults to `https://cloud.langfuse.com`. |

Top-level secrets are also exposed as environment variables, which is how
`config.py` reads them. The server filesystem is ephemeral: the `data/` HTTP cache
is lost on restart, so the first search after a restart is slower.

## License

Released under the [MIT License](LICENSE).
