"""Classify a raised exception as TRANSIENT (retry then fail over) or TERMINAL (fail over now).

This is the core of "run a mixture of models uninterrupted": a pulled model returns 404,
which classifies TERMINAL so the pool fails over to the next executor immediately rather than
hammering a model that is never coming back. Decoupled from the openai/anthropic SDKs by
duck-typing on status codes and class names, so it works across providers.
"""

from __future__ import annotations

from forge.shared.ensemble.models import FailureClass

_TERMINAL_STATUS = {400, 401, 403, 404}
_TERMINAL_NAME_HINTS = ("authentication", "notfound", "permission", "forbidden", "badrequest")
_TRANSIENT_NAME_HINTS = ("timeout", "connection", "ratelimit", "unavailable", "overloaded")


def classify(exc: BaseException) -> FailureClass:
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return FailureClass.TRANSIENT

    status = getattr(exc, "status_code", None)
    if not isinstance(status, int):
        status = getattr(exc, "status", None)
    if isinstance(status, int):
        if status in _TERMINAL_STATUS:
            return FailureClass.TERMINAL
        if status == 429 or 500 <= status < 600:
            return FailureClass.TRANSIENT

    name = type(exc).__name__.lower()
    if any(hint in name for hint in _TRANSIENT_NAME_HINTS):
        return FailureClass.TRANSIENT
    if any(hint in name for hint in _TERMINAL_NAME_HINTS):
        return FailureClass.TERMINAL

    # Unknown failure: treat as transient so a bounded retry gets one more chance,
    # then the pool fails over anyway.
    return FailureClass.TRANSIENT
