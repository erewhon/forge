"""Release-cut proposal for security-fix bumps — HUMAN-GATED by framing constraint.

Cutting a release is the highest-blast-radius action in the bumper's orbit, so this module
only ever RENDERS: the exact commands a human could run, clearly marked as not executed. No
subprocess import, no VCS calls, deliberately. The date is a parameter so the renderer stays
pure (testable without clock control).
"""

from __future__ import annotations

from datetime import date as date_type

from agents.dependabot.models import EvidenceBundle


def should_propose_release(evidence: EvidenceBundle) -> bool:
    """Propose only when the merged bump FIXED something and left nothing behind: at least one
    finding resolved by the bump, zero findings remaining at the target version."""
    return bool(evidence.findings_current) and not evidence.findings_target


def render_release_proposal(
    evidence: EvidenceBundle, *, merged_change_id: str, on: date_type
) -> str:
    """The markdown proposal appended to a merged security-fix bump's summary."""
    c = evidence.candidate
    cves = ", ".join(f.vuln_id for f in evidence.findings_current)
    tag = f"v{on.year}.{on.month:02d}.{on.day:02d}"
    return "\n".join(
        [
            "## Release proposal (human-gated — nothing has been tagged)",
            "",
            f"This merge bumped `{c.name}` {c.current} -> {c.latest}, fixing: {cves}.",
            "A security fix is worth releasing promptly. If you agree, a human runs:",
            "",
            "```",
            f"jj bookmark set {tag} -r {merged_change_id}",
            f"jj git push --bookmark {tag}",
            "```",
            "",
            "Suggested changelog entry:",
            "",
            f"- {tag}: security — bump {c.name} to {c.latest} ({cves})",
            "",
            "_These commands were NOT executed. The bumper never cuts releases._",
        ]
    )
