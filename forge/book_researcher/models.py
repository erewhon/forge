from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class ChapterOutline(BaseModel):
    number: int
    title: str
    description: str
    research_questions: list[str]


class BookConfig(BaseModel):
    title: str
    description: str
    chapters: list[ChapterOutline]


class SprintContract(BaseModel):
    sprint_id: str
    chapter: int
    questions: list[str]
    success_criteria: list[str]
    priority: Literal["high", "medium", "low"]


class ResearchFinding(BaseModel):
    question: str
    answer: str
    sources: list[str]
    confidence: Literal["high", "medium", "low"]


class SprintFindings(BaseModel):
    sprint_id: str
    chapter: int
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
