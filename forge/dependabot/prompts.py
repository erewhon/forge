"""The supply-chain sign-off lens — authored fresh (code_audit is shape precedent only).

The panel judges a manifest-only diff plus an evidence block; it cannot see the dependency's
source diff and the prompt says so plainly. The verdict contract matches the other sign-off
consumers so ``full_quorum_signoff`` parses it unchanged.
"""

from __future__ import annotations

from forge.dependabot.models import EvidenceBundle

SUPPLY_CHAIN_SIGNOFF = """You are a merge gatekeeper for an automated DEPENDENCY BUMP.

The diff must touch ONLY dependency manifests/lockfiles (pyproject.toml, uv.lock, and kin) — any
other file is an automatic block. Judge the bump against the EVIDENCE block provided with the
diff. Approve for automatic merge ONLY if ALL hold:

- The diff is manifest/lockfile-only and the lockfile churn matches the stated bump — one target
  package plus its legitimately re-pinned dependents, nothing unexplained.
- The evidence block says "evidence complete: yes". Incomplete evidence is an automatic block —
  missing evidence is risk, not absence of risk.
- No known vulnerability remains at the target version. (A vulnerability on the CURRENT version
  that the target FIXES is a POSITIVE signal — that is what a good bump looks like.)
- The version delta class matches the diff, and nothing about the release smells wrong. Weigh
  AGAINST approval: a brand-new release (less than ~2 days old), a missing changelog, a yanked
  or typosquat-suspect signal, a maintainer-identity change across the bump, new install/build
  scripts at the target, or lockfile changes beyond the target package and its pins.

You are judging METADATA, not the dependency's source code — you cannot see what the new version
actually does, so do not pretend to. Uncertainty weighs against approval. Be strict: an approval
here merges with no human review.

Respond with ONLY a JSON object: {"approve": true|false, "blockers": ["..."], "notes": "..."}"""


def render_evidence(evidence: EvidenceBundle) -> str:
    """The compact evidence block the loop passes as sign-off ``context`` — one line per
    signal, nothing omitted, the complete flag last and explicit."""
    c = evidence.candidate
    fixed = ", ".join(f.vuln_id for f in evidence.findings_current) or "none"
    residual = ", ".join(f.vuln_id for f in evidence.findings_target) or "none"
    lines = [
        "Evidence for this bump:",
        f"- package: {c.name} {c.current} -> {c.latest} (delta: {c.delta})",
        f"- vulnerabilities fixed by this bump: {fixed}",
        f"- vulnerabilities REMAINING at target: {residual}",
        f"- target yanked on PyPI: {_tri(evidence.target_yanked)}",
        "- target release age: "
        + (
            f"{evidence.package_age_days} day(s)"
            if evidence.package_age_days is not None
            else "unknown"
        ),
        f"- changelog: {evidence.changelog_url or 'none found'}",
        "- typosquat signal: "
        + (
            f"one edit from {evidence.typosquat_suspect!r}"
            if evidence.typosquat_suspect
            else "none"
        ),
        f"- maintainer identity changed across the bump: {_tri(evidence.maintainer_changed)}",
        f"- new install/build scripts at target: {_tri(evidence.new_install_scripts)}",
        f"- lockfile changes: {'; '.join(evidence.lockfile_changes) or 'none parsed'}",
        f"- evidence complete: {'yes' if evidence.complete else 'NO'}",
    ]
    return "\n".join(lines)


def _tri(value: bool | None) -> str:
    return "unknown" if value is None else ("YES" if value else "no")
