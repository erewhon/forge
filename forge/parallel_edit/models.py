from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

EditStatus = Literal["ok", "no_changes", "timeout", "error"]
CandidateKind = Literal["claude", "opencode"]
# winner is one of the candidate labels (A, B, C, ...), "tie", or "all_flawed" — validated
# against the actual run labels at parse time rather than pinned to a fixed pairwise Literal.


class CandidateSpec(BaseModel):
    """One candidate to run: which agent CLI (kind) drives which model.

    claude  -> `claude -p` with a Claude model id (e.g. claude-opus-4-8).
    opencode -> `opencode run -m <model>` where model is an opencode ref such as
                `llm/glm-5.1`, routed through the local LLM router (the open fleet).
    `display` is the human-readable label shown in reports (e.g. "opencode:llm/glm-5.1").
    """

    label: str  # A, B, ...
    kind: CandidateKind
    model: str
    display: str


class DiffStat(BaseModel):
    files_changed: int = 0
    insertions: int = 0
    deletions: int = 0


class EditRun(BaseModel):
    label: str
    model: str
    workspace_path: Path
    status: EditStatus
    diff_text: str = ""
    diff_stat: DiffStat = DiffStat()
    latency_ms: int | None = None
    returncode: int | None = None
    stdout_tail: str = ""
    stderr_tail: str = ""
    error_message: str | None = None


class DimensionScores(BaseModel):
    prompt_fidelity: int
    correctness: int
    scope_discipline: int
    code_quality: int
    completeness: int


class FileComparison(BaseModel):
    file: str
    best: str  # candidate label that handled this file best, or "equivalent"
    note: str


class JudgeVerdict(BaseModel):
    winner: str  # a candidate label (A, B, C, ...), "tie", or "all_flawed"
    scores: dict[str, DimensionScores]  # keyed by run label, e.g. {"A": ..., "B": ..., "C": ...}
    per_file_notes: list[FileComparison]
    summary: str
    recommendation: str


class ParallelEditResult(BaseModel):
    prompt: str
    repo_path: Path
    base_rev: str
    timestamp: datetime
    runs: list[EditRun]
    verdict: JudgeVerdict | None = None
    judge_model: str | None = None
    judge_error: str | None = None
