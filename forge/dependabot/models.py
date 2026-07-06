from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

DeltaClass = Literal["patch", "minor", "major", "unknown"]


VALID_DELTAS = frozenset(("patch", "minor", "major", "unknown"))


@dataclass
class BumpCandidate:
    name: str
    current: str
    latest: str
    delta: DeltaClass
    direct: bool = True

    def __post_init__(self) -> None:
        if self.delta not in VALID_DELTAS:  # type: ignore[comparison-overlap]
            raise ValueError(f"Invalid delta: {self.delta!r}. Must be one of {VALID_DELTAS}")


@dataclass
class AuditFinding:
    package: str
    vuln_id: str
    fix_versions: list[str] = field(default_factory=list)
    description: str = ""
    aliases: list[str] = field(default_factory=list)


@dataclass
class EvidenceBundle:
    candidate: BumpCandidate
    findings_current: list[AuditFinding] = field(default_factory=list)
    findings_target: list[AuditFinding] = field(default_factory=list)
    target_yanked: bool | None = None
    package_age_days: int | None = None
    changelog_url: str | None = None
    typosquat_suspect: str | None = None
    lockfile_changes: list[str] = field(default_factory=list)
    complete: bool = False


BumpStatus = Literal["merged", "branched", "advisory", "no-candidates", "planned", "error"]


@dataclass
class BumpResult:
    status: BumpStatus
    reason: str = ""
    candidate: BumpCandidate | None = None
    branch: str | None = None
    change_id: str | None = None
    merged_to_main: bool = False
    evidence: EvidenceBundle | None = None
    tests_passed: bool | None = None
