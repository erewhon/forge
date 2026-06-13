from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class TopicConfig(BaseModel):
    question: str
    context: str = ""
    sub_questions: list[str] = []
    score_threshold: int | None = None
    slug: str | None = None


class SprintContract(BaseModel):
    sprint_id: str
    questions: list[str]
    success_criteria: list[str]
    rationale: str = ""


class ResearchFinding(BaseModel):
    question: str
    answer: str
    sources: list[str]
    confidence: Literal["high", "medium", "low"]


class SprintFindings(BaseModel):
    sprint_id: str
    findings: list[ResearchFinding]
    raw_search_notes: str = ""


class VerificationScores(BaseModel):
    source_diversity: int
    claim_verification: int
    counter_narrative: int
    depth: int
    actionability: int
    overall: int


class VerificationResult(BaseModel):
    sprint_id: str
    scores: VerificationScores
    passed: bool
    feedback: str
    follow_up_questions: list[str]


class Synthesis(BaseModel):
    question: str
    answer: str
    key_sources: list[str]
    confidence: Literal["high", "medium", "low"]
    open_questions: list[str]
    sprint_count: int
    best_score: int
    incomplete: bool
