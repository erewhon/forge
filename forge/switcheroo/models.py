"""Switcheroo's failover-journal types — the durable record of one failover window that switch-back
reconciles from. Kept apart from :class:`forge.task_worker.models.RunOutcome` (the worker's
per-attempt result): a ``LeafOutcome`` is the *journalled, timestamped* projection of that outcome,
and a ``FailoverLog`` is the window that holds them plus the baton anchor it started from.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

LeafStatus = Literal["done", "failed", "skipped"]


class LeafOutcome(BaseModel):
    """One drained leaf's result, as journalled. Mirrors the worker's ``RunOutcome`` fields plus
    the wall-clock ``at`` it was recorded — enough for switch-back to say what landed, what
    reverted, and where to look (``commit_id`` in the leaf's own repo)."""

    task: str
    project: str
    status: LeafStatus
    reason: str = ""
    commit_id: str | None = None  #: The landed commit in the leaf's repo — switch-back's pointer.
    changed_files: list[str] = Field(default_factory=list)
    duration_s: float = 0.0
    at: str = ""  #: ISO-8601 UTC when the outcome was recorded.


class FailoverLog(BaseModel):
    """One failover window: when it ran, why, the baton anchor it started from, and every leaf it
    drained. Persisted to ``.forge/switcheroo/failover.json`` and consumed by switch-back."""

    started_at: str
    ended_at: str | None = None  #: None while the window is open (or if it was interrupted).
    reason: str = ""  #: Why switcheroo was invoked (operator note), e.g. "Claude 529s, all agents".
    model_tier: str = ""  #: The default tier the fleet ran under this window.

    # The baton anchor at window start — the pre-failover home-repo state switch-back diffs against.
    baton_goal: str = ""
    baton_vcs: str | None = None
    baton_change_id: str | None = None

    outcomes: list[LeafOutcome] = Field(default_factory=list)

    def _by(self, status: LeafStatus) -> list[LeafOutcome]:
        return [o for o in self.outcomes if o.status == status]

    @property
    def done(self) -> list[LeafOutcome]:
        return self._by("done")

    @property
    def failed(self) -> list[LeafOutcome]:
        return self._by("failed")

    @property
    def skipped(self) -> list[LeafOutcome]:
        return self._by("skipped")
