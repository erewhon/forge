"""Contracts for the dependabot bumper: candidates, audit findings, evidence, results.

Pydantic (like the sibling ensembles' models) so the loop can serialize evidence straight into
the JSONL decision log and Literal fields validate on construction.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

DeltaClass = Literal["patch", "minor", "major", "unknown"]


class BumpCandidate(BaseModel):
    name: str
    current: str
    latest: str
    delta: DeltaClass
    direct: bool = True


class AuditFinding(BaseModel):
    package: str
    vuln_id: str
    fix_versions: list[str] = Field(default_factory=list)
    description: str = ""
    aliases: list[str] = Field(default_factory=list)


class EvidenceBundle(BaseModel):
    """What the risk policy and the sign-off lens judge. ``complete`` is True ONLY when every
    evidence fetch succeeded — missing evidence must read as risk, never as absence of risk."""

    candidate: BumpCandidate
    findings_current: list[AuditFinding] = Field(default_factory=list)  # fixed by this bump
    findings_target: list[AuditFinding] = Field(default_factory=list)  # still present after it
    target_yanked: bool | None = None
    package_age_days: int | None = None
    changelog_url: str | None = None
    typosquat_suspect: str | None = None  # the popular name this is one edit away from
    # v2 provenance signals — best-effort: None means "could not determine" and deliberately
    # does NOT mark the evidence incomplete (they block only when provably True).
    maintainer_changed: bool | None = None  # identity differs between current and target release
    new_install_scripts: bool | None = None  # target sdist adds setup.py / changes build backend
    # v2: OpenSSF Scorecard (best-effort provenance from source repo)
    scorecard_score: float | None = None  # aggregate score from OpenSSF Scorecard
    scorecard_repo: str | None = None  # repo name echoed from the Scorecard payload
    lockfile_changes: list[str] = Field(default_factory=list)
    complete: bool = False


BumpStatus = Literal["merged", "branched", "advisory", "no-candidates", "planned", "error"]


class BumpResult(BaseModel):
    status: BumpStatus
    reason: str = ""
    candidate: BumpCandidate | None = None
    branch: str | None = None
    change_id: str | None = None
    merged_to_main: bool = False
    evidence: EvidenceBundle | None = None
    tests_passed: bool | None = None
