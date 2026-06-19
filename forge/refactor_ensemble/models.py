"""Data shapes for the refactoring/best-practices ensemble.

Finders emit ``Smell``s (in a ``SmellsEnvelope``); the consolidator returns ``CanonicalSmell``s
(in a ``CanonicalEnvelope``); the skeptic panel votes each canonical smell to a ``Verdict``.
``RefactorPlan`` is the assembled, render-ready result. ``impact`` and ``effort`` are normalized on
input so a model writing "High" or an unknown label can't break validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field, field_validator

Impact = Literal["high", "medium", "low"]
Effort = Literal["small", "medium", "large"]
_IMPACTS = ("high", "medium", "low")
_EFFORTS = ("small", "medium", "large")
IMPACT_RANK = {"high": 3, "medium": 2, "low": 1}


def _norm_impact(v: object) -> str:
    s = str(v).strip().lower()
    return s if s in _IMPACTS else "medium"


def _norm_effort(v: object) -> str:
    s = str(v).strip().lower()
    return s if s in _EFFORTS else "medium"


class Smell(BaseModel):
    """One refactoring opportunity a finder flagged in the provided code."""

    location: str  # file::function or area
    smell_type: str = ""  # duplication | complexity | naming | coupling | dead-code | idiom
    proposed_refactor: str = ""
    benefit: str = ""
    risk: str = ""  # behavior-change / breaking-API risk, or "none"
    effort: Effort = "medium"
    impact: Impact = "medium"

    @field_validator("impact", mode="before")
    @classmethod
    def _imp(cls, v: object) -> str:
        return _norm_impact(v)

    @field_validator("effort", mode="before")
    @classmethod
    def _eff(cls, v: object) -> str:
        return _norm_effort(v)


class SmellsEnvelope(BaseModel):
    smells: list[Smell] = Field(default_factory=list)


class CanonicalSmell(BaseModel):
    """A deduped refactoring suggestion with a stable id, after the consolidator merges overlaps."""

    id: str
    location: str
    smell_type: str = ""
    proposed_refactor: str = ""
    benefit: str = ""
    risk: str = ""
    effort: Effort = "medium"
    impact: Impact = "medium"
    merged_from: list[str] = Field(default_factory=list)

    @field_validator("impact", mode="before")
    @classmethod
    def _imp(cls, v: object) -> str:
        return _norm_impact(v)

    @field_validator("effort", mode="before")
    @classmethod
    def _eff(cls, v: object) -> str:
        return _norm_effort(v)


class CanonicalEnvelope(BaseModel):
    smells: list[CanonicalSmell] = Field(default_factory=list)
    dropped: int = 0


@dataclass
class Verdict:
    """The skeptic panel's vote on one canonical smell."""

    status: Literal["confirmed", "tentative", "rejected"]
    votes_real: int
    votes_total: int
    impact: str  # impact after the panel's adjustment
    reasonings: list[str] = field(default_factory=list)


@dataclass
class ScoredSmell:
    smell: CanonicalSmell
    verdict: Verdict


@dataclass
class RefactorPlan:
    focus: str
    files: list[str]
    raw_count: int
    canonical_count: int
    dedup_ok: bool
    confirmed: list[ScoredSmell]
    tentative: list[ScoredSmell]
    rejected: list[ScoredSmell]
