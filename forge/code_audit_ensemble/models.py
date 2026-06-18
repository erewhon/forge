"""Data shapes for the code-audit ensemble.

Finders emit ``Finding``s (in a ``FindingsEnvelope``); the consolidator returns
``CanonicalFinding``s (in a ``CanonicalEnvelope``); the skeptic panel votes each canonical
finding to a ``Verdict``. ``AuditReport`` is the assembled, render-ready result. Severity is
normalized on the way in so a model writing "High" or an unknown label can't break validation.
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


class Finding(BaseModel):
    """One issue a finder flagged in the provided code."""

    title: str
    file: str = ""
    line: str = ""
    severity: Severity = "medium"
    scenario: str = ""  # the concrete problem / how it bites
    suggestion: str = ""

    @field_validator("severity", mode="before")
    @classmethod
    def _sev(cls, v: object) -> str:
        return _norm_severity(v)


class FindingsEnvelope(BaseModel):
    findings: list[Finding] = Field(default_factory=list)


class CanonicalFinding(BaseModel):
    """A deduped finding with a stable id, after the consolidator merges overlaps."""

    id: str
    title: str
    file: str = ""
    line: str = ""
    severity: Severity = "medium"
    scenario: str = ""
    suggestion: str = ""
    merged_from: list[str] = Field(default_factory=list)

    @field_validator("severity", mode="before")
    @classmethod
    def _sev(cls, v: object) -> str:
        return _norm_severity(v)


class CanonicalEnvelope(BaseModel):
    findings: list[CanonicalFinding] = Field(default_factory=list)
    dropped: int = 0


@dataclass
class Verdict:
    """The skeptic panel's vote on one canonical finding."""

    status: Literal["confirmed", "tentative", "rejected"]
    votes_real: int
    votes_total: int
    severity: str  # severity after the panel's adjustment
    reasonings: list[str] = field(default_factory=list)


@dataclass
class ScoredFinding:
    finding: CanonicalFinding
    verdict: Verdict


@dataclass
class AuditReport:
    focus: str
    files: list[str]
    raw_count: int
    canonical_count: int
    dedup_ok: bool
    confirmed: list[ScoredFinding]
    tentative: list[ScoredFinding]
    rejected: list[ScoredFinding]
