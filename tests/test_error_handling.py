"""Expected failures keep their message; bugs are logged and never blamed on Groq."""

from __future__ import annotations

import logging

import groq
import httpx
import pytest

from src.llm import INTERNAL_MESSAGE, UNEXPECTED_MESSAGE, llm_failure_message
from src.resolver import _source_result


def _rate_limit_error() -> groq.RateLimitError:
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(429, request=request, headers={"retry-after": "3"})
    return groq.RateLimitError("rate limited", response=response, body=None)


def test_expected_groq_error_keeps_its_specific_message() -> None:
    assert "free-tier limit" in llm_failure_message(_rate_limit_error())


def test_malformed_output_is_reported_as_unexpected_response() -> None:
    # ValueError covers Pydantic ValidationError and LangChain parser errors.
    assert llm_failure_message(ValueError("bad json")) == UNEXPECTED_MESSAGE


def test_bug_is_logged_and_not_blamed_on_groq(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.ERROR):
        message = llm_failure_message(KeyError("oops"))

    assert message == INTERNAL_MESSAGE
    assert "Groq" not in message
    assert any(record.exc_info for record in caplog.records)


def test_source_error_is_silent_and_counts_as_missing(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.ERROR):
        result = _source_result("Deezer", httpx.ConnectError("down"), default=[])

    assert result == []
    assert not caplog.records


def test_source_bug_is_logged_and_counts_as_missing(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.ERROR):
        result = _source_result("LRCLIB", RuntimeError("tool bug"), default=None)

    assert result is None
    assert any("LRCLIB" in record.getMessage() for record in caplog.records)


def test_source_value_passes_through_unchanged() -> None:
    assert _source_result("Deezer", ["Rock"], default=[]) == ["Rock"]
