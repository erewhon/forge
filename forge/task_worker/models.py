from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class TaskInfo(BaseModel):
    """A task selected for autonomous execution."""

    id: str  # Nous row id
    task: str  # task title
    project: str
    status: str
    priority: int
    execution_mode: str
    model_tier: str = "auto"
    estimate: str = ""
    complexity: str = ""
    task_type: str = ""
    max_files: int | None = None
    # null-as-true for safety: if unset, assume tests are required
    requires_tests: bool = True
    deps: list[str] = Field(default_factory=list)


class ExecutionResult(BaseModel):
    """Outcome of a worker execution attempt."""

    task: str
    project: str
    success: bool
    reason: str = ""  # failure reason if !success
    files_changed: list[str] = Field(default_factory=list)
    commit_id: str = ""
    duration_seconds: float = 0
    skipped: bool = False  # true if we chose not to execute (e.g., guardrail)
    stdout_tail: str = ""  # last 500 chars of opencode stdout (for debugging)


class RunOutcome(BaseModel):
    """Structured result of ``run_one`` — the callable per-task API.

    Status semantics:
    - ``done``: change committed on the host and the task marked Done.
    - ``failed``: execution was attempted; the working copy was reverted and the task returned
      to Ready with a diagnostic note.
    - ``skipped``: nothing was attempted or landed (gate refusal, preflight miss, dirty working
      copy, or a dry-run that executed and then reverted).

    The coding pipeline's dispatcher maps this into its own journal record; this package stays a
    lower layer and never imports ``forge.coding_pipeline``.
    """

    task: str
    project: str
    status: Literal["done", "failed", "skipped"]
    reason: str = ""
    commit_id: str | None = None
    changed_files: list[str] = Field(default_factory=list)
    duration_s: float = 0.0  # wall clock of the whole attempt, not just the OpenCode call
    notes_written: bool = False  # whether a status/notes write reached Nous
