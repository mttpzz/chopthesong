"""Application settings, read from environment variables and the .env file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration shared by the graph and the HTTP clients."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    user_agent: str = "chopsongs/0.1 ( https://example.invalid )"
    data_dir: Path = Path("data")
    cache_ttl_seconds: int = 60 * 60 * 24 * 30  # 30 days of cached API responses.
    musicbrainz_min_interval: float = 1.05  # MusicBrainz allows one request per second.
    # Expansion rounds beyond the first-level samples: 0 disables, 3 is the maximum.
    report_depth: int = Field(default=2, ge=0, le=3)

    # Groq (free tier). No fallback: 429s and provider errors surface to the user.
    groq_api_key: str | None = None
    groq_model: str = "openai/gpt-oss-120b"

    # Langfuse Cloud tracing (optional). Empty keys disable it entirely.
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "https://cloud.langfuse.com"

    @property
    def cache_dir(self) -> Path:
        """Directory holding the on-disk HTTP cache."""
        return self.data_dir / "cache"

    @property
    def tracing_enabled(self) -> bool:
        """True only when both Langfuse keys are present and non-empty."""
        return bool(
            (self.langfuse_public_key or "").strip()
            and (self.langfuse_secret_key or "").strip()
        )


@lru_cache
def get_settings() -> Settings:
    """Return cached settings, creating the cache directory on first use."""
    settings = Settings()
    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    return settings
