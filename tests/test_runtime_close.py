from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import groq
import pytest

from src.config import Settings
from src.llm import build_llm
from src.resolver import Runtime


class FakeHttp:
    def __init__(self, error: Exception | None = None) -> None:
        self.closed = False
        self.error = error

    async def aclose(self) -> None:
        self.closed = True
        if self.error is not None:
            raise self.error


class FakeModelClient:
    def __init__(self, error: Exception | None = None) -> None:
        self.closed = False
        self.error = error

    async def close(self) -> None:
        self.closed = True
        if self.error is not None:
            raise self.error


class FakeLlm:
    def __init__(
        self,
        client: FakeModelClient | None = None,
        *,
        with_async_client: bool = True,
    ) -> None:
        if with_async_client:
            self.async_client = SimpleNamespace(_client=client)


def runtime_with(http: Any, llm: Any) -> Runtime:
    """Build a Runtime without opening real clients or reading settings."""
    runtime = Runtime.__new__(Runtime)
    runtime.http = http
    runtime.llm = llm
    return runtime


async def test_aclose_closes_http_and_model_client() -> None:
    http = FakeHttp()
    client = FakeModelClient()

    await runtime_with(http, FakeLlm(client)).aclose()

    assert http.closed is True
    assert client.closed is True


async def test_model_client_is_closed_when_http_close_fails() -> None:
    http = FakeHttp(error=RuntimeError("http close failed"))
    client = FakeModelClient()

    with pytest.raises(RuntimeError, match="http close failed"):
        await runtime_with(http, FakeLlm(client)).aclose()

    assert http.closed is True
    assert client.closed is True


async def test_http_is_closed_when_model_close_fails() -> None:
    http = FakeHttp()
    client = FakeModelClient(error=RuntimeError("model close failed"))

    with pytest.raises(RuntimeError, match="model close failed"):
        await runtime_with(http, FakeLlm(client)).aclose()

    assert http.closed is True
    assert client.closed is True


async def test_missing_async_client_does_not_fail() -> None:
    http = FakeHttp()

    await runtime_with(http, FakeLlm(with_async_client=False)).aclose()

    assert http.closed is True


async def test_async_client_without_root_client_does_not_fail() -> None:
    http = FakeHttp()

    await runtime_with(http, FakeLlm(None)).aclose()

    assert http.closed is True


def test_real_chatgroq_exposes_the_async_groq_client() -> None:
    settings = Settings(groq_api_key="test-key", _env_file=None)
    llm = build_llm(settings)

    assert isinstance(llm.async_client._client, groq.AsyncGroq)


async def test_aclose_closes_the_real_chatgroq_client() -> None:
    settings = Settings(groq_api_key="test-key", _env_file=None)
    llm = build_llm(settings)
    root = llm.async_client._client

    assert root._client.is_closed is False

    await runtime_with(FakeHttp(), llm).aclose()

    assert root.is_closed() is True
    assert root._client.is_closed is True
