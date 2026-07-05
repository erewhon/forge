from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

StepName = Literal[
    "replan",
    "decompose",
    "boundedness",
    "review-findings",
    "review-confirm",
    "testgap-find",
    "testgap-skeptic",
]

_slug_re = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def _slug_ok(v: str) -> str:
    if not _slug_re.match(v):
        raise ValueError("case_id must be a lowercase slug (letters, digits, hyphens only)")
    return v


class GoldCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step: StepName
    case_id: Annotated[str, Field(min_length=1)]
    case_dir: Path
    schema_version: int
    holdout: bool = False
    inputs: dict[str, str] = {}
    expected: dict[str, Any] = {}
    notes: str = ""

    @field_validator("case_id", mode="after")
    @classmethod
    def _validate_slug(cls, v: str) -> str:
        return _slug_ok(v)


class GradeCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    passed: bool
    detail: str = ""


class GradeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    step: StepName
    passed: bool
    score: float = Field(ge=0.0, le=1.0)
    checks: list[GradeCheck] = []
    error: str | None = None


class CaseScore(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    case_id: str
    holdout: bool = False
    repeats: list[GradeResult] = []

    @property
    def passed_majority(self) -> bool:
        """Strict majority of non-error repeats passed."""
        valid = [r for r in self.repeats if r.error is None]
        if not valid:
            return False
        return sum(1 for r in valid if r.passed) > len(valid) / 2

    @property
    def mean_score(self) -> float:
        """Mean score across non-error repeats; 0.0 when none."""
        valid = [r for r in self.repeats if r.error is None]
        if not valid:
            return 0.0
        return sum(r.score for r in valid) / len(valid)


class StepScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step: StepName
    cases: list[CaseScore] = []

    @property
    def pass_rate(self) -> float:
        """Fraction of cases with passed_majority. 0.0 when no cases."""
        if not self.cases:
            return 0.0
        return sum(1 for c in self.cases if c.passed_majority) / len(self.cases)

    @property
    def holdout_pass_rate(self) -> float | None:
        """Pass rate among holdout cases only; None when no holdout."""
        holdout_cases = [c for c in self.cases if c.holdout]
        if not holdout_cases:
            return None
        return sum(1 for c in holdout_cases if c.passed_majority) / len(holdout_cases)

    @property
    def error_repeats(self) -> int:
        total = 0
        for c in self.cases:
            total += sum(1 for r in c.repeats if r.error is not None)
        return total


class Scorecard(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    timestamp: str
    steps: list[StepScore] = []
    notes: str = ""

    @property
    def overall_pass_rate(self) -> float:
        """Fraction of total cases passed. 0.0 when no cases."""
        total = 0
        passed = 0
        for s in self.steps:
            for c in s.cases:
                total += 1
                if c.passed_majority:
                    passed += 1
        if total == 0:
            return 0.0
        return passed / total
