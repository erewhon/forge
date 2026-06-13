from __future__ import annotations

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
