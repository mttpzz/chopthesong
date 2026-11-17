"""Logging helper that keeps request URLs out of network error logs."""

from __future__ import annotations

import logging

import httpx


def log_failure(
    logger: logging.Logger,
    message: str,
    exc: Exception,
    *,
    level: int = logging.ERROR,
) -> None:
    """Log a failure without ever printing the request URL.

    ``httpx`` exceptions embed the request URL in their text, and a search URL can
    contain the song title or artist, so network failures log only the exception
    type. Anything else is an internal bug: it keeps ``exc_info`` and its traceback.
    """
    if isinstance(exc, httpx.HTTPError):
        logger.log(level, "%s (%s)", message, type(exc).__name__)
    else:
        logger.log(level, message, exc_info=exc)
