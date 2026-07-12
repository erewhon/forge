"""Contracts for the sweep driver: one row per agent run, one result per sweep."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AgentRun(BaseModel):
    """One agent invocation on one repo. ``status`` is the agent's own headline (parsed
    from its rendered summary: merged/branched/advisory/no-candidates/up-to-date/...),
    falling back to ok/error by exit code when no headline is found."""

    repo: str
    agent: str  # "deps" | "upstream"
    status: str
    detail: str = ""
    exit_code: int = 0


class SweepResult(BaseModel):
    host: str = ""
    repos: list[str] = Field(default_factory=list)  # selected after include/exclude
    skipped: list[str] = Field(default_factory=list)  # filtered out by globs
    runs: list[AgentRun] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)  # per-repo infra failures (clone, ...)
