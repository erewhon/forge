"""The ecosystem port: everything the bumper does that differs per package manager.

The loop in ``autobump.py`` is ecosystem-neutral — VCS actions, the manifest-only and
green-suite gates, the sign-off, and advisory emission all work the same for any language.
What differs is how to SCAN for outdated dependencies, how to APPLY one bump, how to read
the LOCKFILE DELTA, how to AUDIT for known vulnerabilities, and what EVIDENCE exists for
the risk policy. Those five operations are this protocol; ``uv`` (Python) and ``go`` are
the current backends (same port-and-adapters shape as the TaskStore split).

Detection is by manifest presence: ``uv.lock`` selects uv, ``go.mod`` selects go. A repo
with both is an error naming them — pass ``--ecosystem`` (or ``DEPENDABOT_ECOSYSTEM``) to
choose; the bumper never guesses.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from forge.dependabot.models import AuditFinding, BumpCandidate, EvidenceBundle


class EcosystemError(RuntimeError):
    """Ecosystem detection failed (no manifest, ambiguous manifests, or a bad override)."""


class Ecosystem(Protocol):
    """One package manager's implementation of the bumper's five variable operations."""

    name: str

    def scan_outdated(self, repo: Path) -> list[BumpCandidate]:
        """Direct dependencies with newer versions available, sorted patch-first, capped."""
        ...

    def apply_bump(self, repo: Path, candidate: BumpCandidate) -> list[str]:
        """Apply one bump to the working copy; return changed files ([] = constraint-pinned)."""
        ...

    def lockfile_delta(self, repo: Path) -> list[str]:
        """Compact ``name old->new`` strings parsed from the working diff's lockfile hunks."""
        ...

    def run_audit(self, repo: Path) -> list[AuditFinding]:
        """Repo-wide known-vulnerability findings. MUST fail closed when no scanner exists."""
        ...

    def collect_evidence(
        self,
        candidate: BumpCandidate,
        findings: list[AuditFinding],
        lock_delta: list[str],
        *,
        repo_root: Path,
    ) -> EvidenceBundle:
        """The bundle the risk policy and sign-off judge. Ecosystems without provenance
        sources MUST return ``complete=False`` with an ``incomplete_reason`` — missing
        evidence demotes to the advisory track, it never reads as absence of risk."""
        ...


# Manifest markers, checked in this order. Detection requires exactly one match.
_MARKERS: dict[str, str] = {"uv": "uv.lock", "go": "go.mod", "pnpm": "pnpm-lock.yaml"}


def present_ecosystems(repo: Path) -> list[str]:
    """Every ecosystem whose marker exists in *repo*, in _MARKERS order.

    For callers that handle multi-ecosystem repos by running the bumper once per
    ecosystem (the fleet sweep) — ``detect_ecosystem`` stays strict and errors on
    ambiguity for direct single-run use."""
    return [name for name, marker in _MARKERS.items() if (repo / marker).exists()]


def detect_ecosystem(repo: Path, *, override: str | None = None) -> str:
    """The ecosystem name for *repo* — from *override* (validated) or manifest presence."""
    if override:
        marker = _MARKERS.get(override)
        if marker is None:
            supported = ", ".join(sorted(_MARKERS))
            raise EcosystemError(f"unknown ecosystem {override!r} — supported: {supported}")
        if not (repo / marker).exists():
            raise EcosystemError(
                f"ecosystem {override!r} was requested but {marker} does not exist in {repo}"
            )
        return override

    present = [name for name, marker in _MARKERS.items() if (repo / marker).exists()]
    if len(present) > 1:
        names = ", ".join(present)
        raise EcosystemError(
            f"multiple ecosystems detected in {repo} ({names}) — "
            "pass --ecosystem (or set DEPENDABOT_ECOSYSTEM) to choose; the bumper never guesses"
        )
    if not present:
        markers = ", ".join(_MARKERS[name] for name in sorted(_MARKERS))
        raise EcosystemError(f"no supported dependency manifest in {repo} (looked for: {markers})")
    return present[0]


def resolve_ecosystem(repo: Path, *, override: str | None = None) -> Ecosystem:
    """Detect and construct the adapter. Imports are local so each backend loads on demand."""
    name = detect_ecosystem(repo, override=override)
    if name == "go":
        from forge.dependabot.ecosystems.golang import GoEcosystem

        return GoEcosystem()
    if name == "pnpm":
        from forge.dependabot.ecosystems.pnpm import PnpmEcosystem

        return PnpmEcosystem()
    from forge.dependabot.ecosystems.uv import UvEcosystem

    return UvEcosystem()
