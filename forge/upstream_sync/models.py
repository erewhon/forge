"""Contracts for the upstream sync agent: layer manifest, collision verdict, sync result.

Pydantic like the sibling agents' models, so results serialize straight into the JSONL
decision log and Literal fields validate on construction.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class LayerManifest(BaseModel):
    """The additive layer: what the fork owns relative to the upstream merge-base.

    ``added`` are files the fork created (upstream has no copy at the merge-base);
    ``modified`` are upstream files the fork edited (including the old path of a rename).
    Computable, not guessed — this is the collision seat's ground truth.
    """

    added: list[str] = Field(default_factory=list)
    modified: list[str] = Field(default_factory=list)


class CollisionFinding(BaseModel):
    file: str
    reason: str


class CollisionVerdict(BaseModel):
    """The seat's judgment. ``collision=None`` means the seat could not judge (disabled,
    LLM failure, unparseable output) — unknown blocks --auto-merge, never the branch push."""

    collision: bool | None = None
    findings: list[CollisionFinding] = Field(default_factory=list)
    notes: str = ""


SyncStatus = Literal["up-to-date", "planned", "branched", "merged", "conflict", "advisory", "error"]


class SyncResult(BaseModel):
    status: SyncStatus
    reason: str = ""
    branch: str | None = None
    upstream_tip: str | None = None
    merge_base: str | None = None
    commits_behind: int = 0
    layer: LayerManifest | None = None
    overlap: list[str] = Field(default_factory=list)
    conflicted: list[str] = Field(default_factory=list)
    tests_passed: bool | None = None
    collision: CollisionVerdict | None = None
    merged_to_main: bool = False
