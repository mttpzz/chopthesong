from __future__ import annotations

import groq
import httpx
import pytest

from src.config import Settings
from src.llm import (
    AUTH_MESSAGE,
    DEFAULT_RETRY_AFTER,
    MODEL_NOT_FOUND_MESSAGE,
    RATE_LIMITED_MESSAGE,
    UNAVAILABLE_MESSAGE,
    UNEXPECTED_MESSAGE,
    WAIT_NOTICE,
    LlmAuthError,
    LlmError,
    LlmModelNotFound,
    LlmRateLimited,
    LlmUnavailable,
    build_llm,
    invoke_with_retry,
    to_llm_error,
)

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
SECRET = "SECRET-TOKEN-123"


def request() -> httpx.Request:
    return httpx.Request("POST", GROQ_URL)


def response(status: int, headers: dict[str, str] | None = None) -> httpx.Response:
    return httpx.Response(status, headers=headers, request=request())


def rate_limited(retry_after: str | None = None) -> groq.RateLimitError:
    headers = {"retry-after": retry_after} if retry_after is not None else {}
    return groq.RateLimitError(
        SECRET, response=response(429, headers=headers), body=None
    )


class FakeSleep:
    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)


def test_rate_limit_maps_to_llm_rate_limited() -> None:
    mapped = to_llm_error(
        groq.RateLimitError(SECRET, response=response(429), body=None)
    )

    assert isinstance(mapped, LlmRateLimited)
    assert str(mapped) == RATE_LIMITED_MESSAGE


def test_authentication_maps_to_auth_error() -> None:
    mapped = to_llm_error(
        groq.AuthenticationError(SECRET, response=response(401), body=None)
    )

    assert isinstance(mapped, LlmAuthError)
    assert str(mapped) == AUTH_MESSAGE


def test_permission_denied_maps_to_auth_error() -> None:
    mapped = to_llm_error(
        groq.PermissionDeniedError(SECRET, response=response(403), body=None)
    )

    assert isinstance(mapped, LlmAuthError)
    assert str(mapped) == AUTH_MESSAGE


def test_model_not_found_maps_to_model_not_found() -> None:
    mapped = to_llm_error(groq.NotFoundError(SECRET, response=response(404), body=None))

    assert isinstance(mapped, LlmModelNotFound)
    assert str(mapped) == MODEL_NOT_FOUND_MESSAGE


def test_server_error_maps_to_unavailable() -> None:
    mapped = to_llm_error(
        groq.InternalServerError(SECRET, response=response(500), body=None)
    )

    assert isinstance(mapped, LlmUnavailable)
    assert str(mapped) == UNAVAILABLE_MESSAGE


def test_connection_error_maps_to_unavailable() -> None:
    mapped = to_llm_error(groq.APIConnectionError(message=SECRET, request=request()))

    assert isinstance(mapped, LlmUnavailable)
    assert str(mapped) == UNAVAILABLE_MESSAGE


def test_timeout_error_maps_to_unavailable() -> None:
    mapped = to_llm_error(groq.APITimeoutError(request=request()))

    assert isinstance(mapped, LlmUnavailable)
    assert str(mapped) == UNAVAILABLE_MESSAGE


def test_httpx_transport_error_maps_to_unavailable() -> None:
    mapped = to_llm_error(httpx.ConnectError(SECRET, request=request()))

    assert isinstance(mapped, LlmUnavailable)
    assert str(mapped) == UNAVAILABLE_MESSAGE


def test_unknown_error_uses_generic_message() -> None:
    mapped = to_llm_error(ValueError(SECRET))

    assert type(mapped) is LlmError
    assert str(mapped) == UNEXPECTED_MESSAGE


def test_mapped_messages_never_contain_raw_text() -> None:
    mapped_errors = [
        to_llm_error(groq.RateLimitError(SECRET, response=response(429), body=None)),
        to_llm_error(groq.AuthenticationError(SECRET, response=response(401), body=None)),
        to_llm_error(
            groq.InternalServerError(SECRET, response=response(500), body=None)
        ),
        to_llm_error(groq.NotFoundError(SECRET, response=response(404), body=None)),
        to_llm_error(groq.APIConnectionError(message=SECRET, request=request())),
        to_llm_error(ValueError(SECRET)),
    ]

    for mapped in mapped_errors:
        assert SECRET not in str(mapped)


def test_build_llm_requires_api_key() -> None:
    with pytest.raises(LlmAuthError, match="GROQ_API_KEY"):
        build_llm(Settings(groq_api_key=None))


def test_build_llm_uses_the_configured_model() -> None:
    llm = build_llm(Settings(groq_api_key="test-key", groq_model="test-model"))

    assert llm.model_name == "test-model"


def test_short_retry_after_is_kept_and_not_daily() -> None:
    mapped = to_llm_error(rate_limited("3"))

    assert isinstance(mapped, LlmRateLimited)
    assert mapped.retry_after == 3.0
    assert mapped.daily is False
    assert str(mapped) == RATE_LIMITED_MESSAGE


def test_long_retry_after_is_a_daily_limit() -> None:
    mapped = to_llm_error(rate_limited("7200"))

    assert isinstance(mapped, LlmRateLimited)
    assert mapped.daily is True
    assert mapped.retry_after == 7200.0
    assert "2 hours" in str(mapped)


def test_long_retry_after_rounds_hours_up_to_at_least_one() -> None:
    mapped = to_llm_error(rate_limited("90"))

    assert isinstance(mapped, LlmRateLimited)
    assert mapped.daily is True
    assert "1 hour." in str(mapped)
    assert "1 hours" not in str(mapped)


def test_missing_retry_after_leaves_the_field_empty() -> None:
    mapped = to_llm_error(rate_limited())

    assert isinstance(mapped, LlmRateLimited)
    assert mapped.retry_after is None
    assert mapped.daily is False
    assert str(mapped) == RATE_LIMITED_MESSAGE


def test_non_numeric_retry_after_is_ignored() -> None:
    mapped = to_llm_error(rate_limited("later"))

    assert isinstance(mapped, LlmRateLimited)
    assert mapped.retry_after is None
    assert mapped.daily is False


async def test_invoke_with_retry_waits_and_succeeds() -> None:
    sleep = FakeSleep()
    waits: list[str] = []
    calls = 0

    async def call() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise rate_limited("3")
        return "ok"

    result = await invoke_with_retry(call, on_wait=waits.append, sleep=sleep)

    assert result == "ok"
    assert calls == 2
    assert sleep.delays == [3.0]
    assert waits == [WAIT_NOTICE]


async def test_invoke_with_retry_uses_the_five_second_default() -> None:
    sleep = FakeSleep()
    calls = 0

    async def call() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise rate_limited()
        return "ok"

    result = await invoke_with_retry(call, sleep=sleep)

    assert result == "ok"
    assert sleep.delays == [DEFAULT_RETRY_AFTER]


async def test_invoke_with_retry_does_not_retry_a_daily_limit() -> None:
    sleep = FakeSleep()
    calls = 0

    async def call() -> str:
        nonlocal calls
        calls += 1
        raise rate_limited("7200")

    with pytest.raises(LlmRateLimited) as caught:
        await invoke_with_retry(call, sleep=sleep)

    assert calls == 1
    assert caught.value.daily is True
    assert sleep.delays == []


async def test_invoke_with_retry_stops_after_three_attempts() -> None:
    sleep = FakeSleep()
    waits: list[str] = []
    calls = 0

    async def call() -> str:
        nonlocal calls
        calls += 1
        raise rate_limited("3")

    with pytest.raises(LlmRateLimited) as caught:
        await invoke_with_retry(call, on_wait=waits.append, sleep=sleep)

    assert calls == 3
    assert caught.value.daily is False
    assert str(caught.value) == RATE_LIMITED_MESSAGE
    assert sleep.delays == [3.0, 3.0]
    assert waits == [WAIT_NOTICE, WAIT_NOTICE]


async def test_invoke_with_retry_does_not_retry_a_server_error() -> None:
    sleep = FakeSleep()
    calls = 0

    async def call() -> str:
        nonlocal calls
        calls += 1
        raise groq.InternalServerError(SECRET, response=response(500), body=None)

    with pytest.raises(LlmUnavailable) as caught:
        await invoke_with_retry(call, sleep=sleep)

    assert calls == 1
    assert str(caught.value) == UNAVAILABLE_MESSAGE
    assert sleep.delays == []
