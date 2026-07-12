"""The uv (Python) ecosystem adapter — thin delegation to the original bumper modules.

``scan.py``, ``bump.py``, ``audit.py``, and ``supply_chain.py`` ARE the uv implementation;
this class only gives them the adapter shape. Behavior is identical to the pre-port bumper
by construction — no logic lives here.
"""

from __future__ import annotations

from pathlib import Path

from forge.dependabot.audit import run_audit
from forge.dependabot.bump import apply_bump, lockfile_delta
from forge.dependabot.models import AuditFinding, BumpCandidate, EvidenceBundle
from forge.dependabot.scan import scan_outdated
from forge.dependabot.supply_chain import collect_evidence


class UvEcosystem:
    name = "uv"

    def scan_outdated(self, repo: Path) -> list[BumpCandidate]:
        return scan_outdated(repo)

    def apply_bump(self, repo: Path, candidate: BumpCandidate) -> list[str]:
        return apply_bump(repo, candidate)

    def lockfile_delta(self, repo: Path) -> list[str]:
        return lockfile_delta(repo)

    def run_audit(self, repo: Path) -> list[AuditFinding]:
        return run_audit(repo)

    def collect_evidence(
        self,
        candidate: BumpCandidate,
        findings: list[AuditFinding],
        lock_delta: list[str],
        *,
        repo_root: Path,
    ) -> EvidenceBundle:
        return collect_evidence(candidate, findings, lock_delta, repo_root=repo_root)
