"""Core data types for the ensemble harness.

The harness fans work out across a *pool* of interchangeable executors and tolerates
partial failure. These types are backend-agnostic: an executor may be an API call, a
subprocess, or (later) a sandboxed container — the layers above only see ExecResult.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ExecStatus(StrEnum):
    OK = "ok"
    TIMEOUT = "timeout"
    ERROR = "error"
    SKIPPED = "skipped"


class FailureClass(StrEnum):
    NONE = "none"
    TRANSIENT = "transient"  # retry (bounded), then fail over
    TERMINAL = "terminal"  # do not retry; fail over and mark unhealthy


class QuorumState(StrEnum):
    FULL = "full"  # every pool produced a result
    DEGRADED = "degraded"  # some failed but quorum floor was met
    FAILED = "failed"  # fewer successes than the quorum floor


class Prompt(BaseModel):
    """A single unit of LLM work. Subprocess executors use `user` as the instruction."""

    system: str = ""
    user: str
    max_tokens: int = 4096


class ExecResult(BaseModel):
    executor: str  # label, e.g. "router:kimi-k2.6"
    status: ExecStatus
    output: str = ""
    latency_ms: int | None = None
    error: str | None = None
    failure_class: FailureClass = FailureClass.NONE
    attempts: int = 1
    meta: dict[str, Any] = Field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == ExecStatus.OK


class FanoutResult(BaseModel):
    role: str
    results: list[ExecResult]
    quorum_state: QuorumState
    quorum_floor: int

    @property
    def succeeded(self) -> list[ExecResult]:
        return [r for r in self.results if r.ok]
