"""Grind's typed surface: the user config (`grind.yaml`) and the loop's result records.

The config is a *runbook* — an ordered list of shell steps forming one experiment cycle, plus a
`check` command whose exit code is the machine-checkable done-signal (and whose stdout can carry a
numeric fitness score for hill-climbing). Everything else is the loop's own bookkeeping.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Step(BaseModel):
    """One command in the experiment cycle (e.g. reset the DB, load dev data, run the migration)."""

    name: str = Field(..., min_length=1)
    run: str = Field(..., min_length=1, description="Shell command, run in the repo root.")


class Check(BaseModel):
    """The machine-checkable done-signal. Exit 0 = the goal is met. When `score_regex` is set, a
    number captured from the check's stdout is a fitness score that unlocks hill-climbing (keep the
    best iteration, roll back regressions)."""

    run: str = Field(..., min_length=1, description="Shell command; exit 0 means done.")
    score_regex: str | None = Field(
        default=None,
        description="Optional regex with one capture group over the check's stdout; the captured "
        "number is the iteration's fitness. Absent => linear keep-last (no hill-climbing).",
    )
    score_goal: Literal["max", "min"] = "max"

    @field_validator("score_regex")
    @classmethod
    def _valid_regex(cls, v: str | None) -> str | None:
        if v is not None:
            try:
                if re.compile(v).groups < 1:
                    raise ValueError("score_regex must have exactly one capture group")
            except re.error as e:
                raise ValueError(f"invalid score_regex: {e}") from e
        return v


class GrindConfig(BaseModel):
    """A day's grind: a goal, the experiment cycle, and the loop bounds."""

    goal: str = Field(..., min_length=1)
    steps: list[Step] = Field(..., min_length=1)
    check: Check
    observe: list[str] = Field(
        default_factory=list,
        description="Step names whose output feeds the model. Empty => all steps + check.",
    )
    edit_paths: list[str] = Field(
        default_factory=list,
        description="Paths the model may edit (a scope hint in the prompt). Empty => no hint.",
    )
    model: str | None = Field(
        default=None,
        description="OpenCode model string, passed through verbatim to `opencode run -m`. "
        "Overridden by --model / GRIND_MODEL.",
    )
    max_iterations: int = Field(default=20, ge=1)
    step_timeout: int = Field(default=600, ge=1, description="Per-step timeout, seconds.")
    edit_timeout: int = Field(default=1800, ge=1, description="Per model-edit timeout, seconds.")
    no_progress_window: int = Field(
        default=3,
        ge=2,
        description="Stop when this many recent turns fail with an identical signature.",
    )

    @field_validator("observe")
    @classmethod
    def _observe_known(cls, v: list[str]) -> list[str]:
        return v  # cross-field validation (names ⊆ steps) happens in `resolved_observe`

    def resolved_observe(self) -> list[str]:
        """The step names whose output feeds the model — the configured subset, or all + `check`."""
        known = {s.name for s in self.steps} | {"check"}
        if not self.observe:
            return [s.name for s in self.steps] + ["check"]
        unknown = [n for n in self.observe if n not in known]
        if unknown:
            raise ValueError(f"observe names not in steps/check: {unknown}")
        return list(self.observe)


class StepResult(BaseModel):
    """The outcome of running one runbook step."""

    name: str
    exit_code: int
    output: str  # tail of stdout+stderr
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


class CycleResult(BaseModel):
    """The outcome of one full experiment cycle (all steps + the check)."""

    steps: list[StepResult]
    check: StepResult
    observation: str  # what the model sees next turn (tail of the observed steps' output)
    score: float | None = None

    @property
    def passed(self) -> bool:
        """The goal is met iff the check exited 0 (and every prior step ran clean)."""
        return self.check.ok and all(s.ok for s in self.steps)

    @property
    def failing_step(self) -> str | None:
        for s in self.steps:
            if not s.ok:
                return s.name
        return None if self.check.ok else "check"

    @property
    def reason(self) -> str:
        """A short 'why not done' string — the input to the no-progress failure signature."""
        if self.passed:
            return ""
        step = self.failing_step or "check"
        result = next((s for s in self.steps if s.name == step), self.check)
        head = result.output.strip().splitlines()
        return f"{step}: " + (head[0] if head else f"exit {result.exit_code}")


class IterationRecord(BaseModel):
    """One journalled turn of the grind loop."""

    iteration: int
    edited_files: list[str]
    blocked: bool
    passed: bool
    score: float | None
    failure_sig: str
    kept: bool  # did this iteration's state survive (vs. rolled back)?
    reason: str
