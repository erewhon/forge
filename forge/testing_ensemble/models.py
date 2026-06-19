"""Data shapes for the testing-review ensemble.

Finders emit ``TestGap``s (in a ``TestGapsEnvelope``); the consolidator returns ``CanonicalGap``s
(in a ``CanonicalEnvelope``); the skeptic panel votes each canonical gap to a ``Verdict``.
``TestReport`` is the assembled, render-ready result. Severity is normalized on input so a model
writing "High" or an unknown label can't break validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field, field_validator

Severity = Literal["critical", "high", "medium", "low"]
_SEVERITIES = ("critical", "high", "medium", "low")
SEVERITY_RANK = {"critical": 4, "high": 3, "medium": 2, "low": 1}


def _norm_severity(v: object) -> str:
    s = str(v).strip().lower()
    return s if s in _SEVERITIES else "medium"


class TestGap(BaseModel):
    """One behavior of the source that the existing tests don't cover."""

    target: str  # file::function or the specific behavior under-tested
    gap_type: str = ""  # coverage | error-path | concurrency | edge-case | regression
    why_it_matters: str = ""  # the bug that could slip through untested
    suggested_test: str = ""  # a concrete test to add
    severity: Severity = "medium"

    @field_validator("severity", mode="before")
    @classmethod
    def _sev(cls, v: object) -> str:
        return _norm_severity(v)


class TestGapsEnvelope(BaseModel):
    gaps: list[TestGap] = Field(default_factory=list)


class CanonicalGap(BaseModel):
    """A deduped gap with a stable id, after the consolidator merges overlaps."""

    id: str
    target: str
    gap_type: str = ""
    why_it_matters: str = ""
    suggested_test: str = ""
    severity: Severity = "medium"
    merged_from: list[str] = Field(default_factory=list)

    @field_validator("severity", mode="before")
    @classmethod
    def _sev(cls, v: object) -> str:
        return _norm_severity(v)


class CanonicalEnvelope(BaseModel):
    gaps: list[CanonicalGap] = Field(default_factory=list)
    dropped: int = 0


@dataclass
class Verdict:
    """The skeptic panel's vote on one canonical gap."""

    status: Literal["confirmed", "tentative", "rejected"]
    votes_real: int
    votes_total: int
    severity: str
    reasonings: list[str] = field(default_factory=list)


@dataclass
class ScoredGap:
    gap: CanonicalGap
    verdict: Verdict


@dataclass
class TestReport:
    focus: str
    source_files: list[str]
    test_files: list[str]
    raw_count: int
    canonical_count: int
    dedup_ok: bool
    confirmed: list[ScoredGap]
    tentative: list[ScoredGap]
    rejected: list[ScoredGap]
