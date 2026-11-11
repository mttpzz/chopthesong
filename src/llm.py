"""Groq client, typed LLM errors and the retry wrapper used by graph nodes."""

from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Awaitable, Callable

import groq
import httpx
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_groq import ChatGroq

from src.config import Settings

logger = logging.getLogger(__name__)

RATE_LIMITED_MESSAGE = "The Groq free-tier limit was reached. Try again in a few minutes."
UNAVAILABLE_MESSAGE = "Groq is not responding right now. Try again shortly."
AUTH_MESSAGE = "GROQ_API_KEY is not valid. Check the .env file."
MODEL_NOT_FOUND_MESSAGE = "Groq model not found."
UNEXPECTED_MESSAGE = "Groq returned an unexpected response. Try again."
INTERNAL_MESSAGE = "Internal error. Try again."
DAILY_LIMIT_MESSAGE_TEMPLATE = (
    "The Groq free-tier daily limit is reached. Try again in about {hours} {unit}."
)
WAIT_NOTICE = "Processing is slow, waiting a few seconds..."

# A retry-after above this threshold means the daily limit, not a short burst.
DAILY_LIMIT_SECONDS = 60.0
DEFAULT_RETRY_AFTER = 5.0


class LlmError(RuntimeError):
    """An LLM failure whose message can be shown to the user."""


class LlmRateLimited(LlmError):
    """Groq refused the call because the free-tier rate limit was reached."""

    def __init__(
        self,
        message: str,
        *,
        retry_after: float | None = None,
        daily: bool = False,
    ) -> None:
        super().__init__(message)
        # Seconds Groq asked us to wait; None when the header is missing or invalid.
        self.retry_after = retry_after
        # True when the wait is long enough to be the daily limit.
        self.daily = daily


class LlmUnavailable(LlmError):
    """Groq is unreachable or returned a server error."""


class LlmAuthError(LlmError):
    """The API key is missing, invalid or not allowed."""


class LlmModelNotFound(LlmError):
    """The configured model does not exist or is not accessible."""


def build_llm(settings: Settings) -> ChatGroq:
    """Build the shared ChatGroq client.

    The client does not retry on its own: ``invoke_with_retry`` owns retries so
    it can tell a short rate limit from the daily one. The limiter only spaces
    calls before they reach Groq.
    """
    if not (settings.groq_api_key or "").strip():
        raise LlmAuthError(AUTH_MESSAGE)

    return ChatGroq(
        model=settings.groq_model,
        api_key=settings.groq_api_key,
        temperature=0,
        max_retries=0,
        rate_limiter=InMemoryRateLimiter(requests_per_second=0.4, max_bucket_size=2),
    )


def _retry_after_seconds(value: str | None) -> float | None:
    """Parse the retry-after header, ignoring missing, invalid or negative values."""
    if value is None:
        return None
    try:
        seconds = float(value)
    except ValueError:
        return None
    if not math.isfinite(seconds) or seconds < 0:
        return None
    return seconds


def _rate_limited(exc: groq.RateLimitError) -> LlmRateLimited:
    """Turn a 429 into a typed error, flagging the daily limit and its wait time."""
    retry_after = _retry_after_seconds(exc.response.headers.get("retry-after"))
    if retry_after is not None and retry_after > DAILY_LIMIT_SECONDS:
        hours = max(1, math.ceil(retry_after / 3600))
        unit = "hour" if hours == 1 else "hours"
        return LlmRateLimited(
            DAILY_LIMIT_MESSAGE_TEMPLATE.format(hours=hours, unit=unit),
            retry_after=retry_after,
            daily=True,
        )
    return LlmRateLimited(RATE_LIMITED_MESSAGE, retry_after=retry_after)


def to_llm_error(exc: Exception) -> LlmError:
    """Map groq/httpx failures to our errors without leaking their raw message."""
    if isinstance(exc, LlmError):
        return exc
    if isinstance(exc, groq.RateLimitError):
        return _rate_limited(exc)
    if isinstance(exc, (groq.AuthenticationError, groq.PermissionDeniedError)):
        return LlmAuthError(AUTH_MESSAGE)
    if isinstance(exc, groq.NotFoundError):
        return LlmModelNotFound(MODEL_NOT_FOUND_MESSAGE)
    if isinstance(exc, (groq.APIConnectionError, httpx.TransportError)):
        return LlmUnavailable(UNAVAILABLE_MESSAGE)
    if isinstance(exc, groq.APIStatusError) and exc.status_code >= 500:
        return LlmUnavailable(UNAVAILABLE_MESSAGE)
    return LlmError(UNEXPECTED_MESSAGE)


def llm_failure_message(exc: Exception) -> str:
    """User-facing text for a failed LLM node.

    Expected failures (Groq API, network, malformed output) keep their specific message.
    Anything else is a bug on our side: it is logged with its traceback and reported as
    an internal error, so it is never blamed on Groq.
    """
    if isinstance(exc, (LlmError, groq.GroqError, httpx.HTTPError, ValueError)):
        return str(to_llm_error(exc))
    logger.error("Unexpected error in LLM node", exc_info=exc)
    return INTERNAL_MESSAGE


async def invoke_with_retry[T](
    call: Callable[[], Awaitable[T]],
    *,
    max_attempts: int = 3,
    on_wait: Callable[[str], None] | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> T:
    """Run ``call`` again on a short rate limit, without retrying other failures."""
    attempts = max(1, max_attempts)
    for attempt in range(attempts):
        try:
            return await call()
        except Exception as exc:
            error = to_llm_error(exc)
            can_retry = (
                isinstance(error, LlmRateLimited)
                and not error.daily
                and attempt < attempts - 1
            )
            if not can_retry:
                if error is exc:
                    raise
                raise error from exc

            if on_wait is not None:
                on_wait(WAIT_NOTICE)
            delay = error.retry_after if error.retry_after is not None else DEFAULT_RETRY_AFTER
            await sleep(delay)

    raise AssertionError("unreachable: the last attempt always returns or raises")
