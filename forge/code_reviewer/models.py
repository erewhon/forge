from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


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


class RepoReview(BaseModel):
    repo_name: str
    findings: list[ReviewFinding]
    summary: str


class NightlyReport(BaseModel):
    date: str
    repos_reviewed: int
    repos_with_changes: int
    reviews: list[RepoReview]
    overall_summary: str
