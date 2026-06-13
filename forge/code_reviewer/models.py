from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RepoChanges(BaseModel):
    repo_name: str
    vcs: Literal["jj", "git"]
    commit_count: int
    commit_summaries: list[str]
    diff_stat: str
    diff_text: str
    truncated: bool


class ReviewFinding(BaseModel):
    severity: Literal["critical", "warning", "info", "positive"]
    file_path: str
    description: str


class RepoScores(BaseModel):
    security: int = Field(ge=1, le=10)  # injection, auth, secrets, unsafe deserialization
    correctness: int = Field(ge=1, le=10)  # logic errors, off-by-one, null handling, race conditions
    error_handling: int = Field(ge=1, le=10)  # boundary validation, resource cleanup, graceful degradation
    performance: int = Field(ge=1, le=10)  # N+1 queries, blocking in async, unnecessary allocations
    overall: int = Field(ge=1, le=10)  # weighted judgment


class RepoReview(BaseModel):
    repo_name: str
    findings: list[ReviewFinding]
    summary: str
    scores: RepoScores | None = None


class NightlyReport(BaseModel):
    date: str
    repos_reviewed: int
    repos_with_changes: int
    reviews: list[RepoReview]
    overall_summary: str
