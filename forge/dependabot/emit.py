"""Advisory fallback: one Forge task per bump the loop refused to merge.

The task carries the policy/gate reason, the full rendered evidence, and the pushed branch so
a human (or the task worker) picks up exactly where the loop stopped. ``external_ref`` is keyed
on the durable (package, target-version) pair: re-running the bumper against the same outdated
dep skips instead of duplicating.
"""

from __future__ import annotations

from collections.abc import Callable

from agents.dependabot.models import BumpCandidate, EvidenceBundle
from agents.dependabot.prompts import render_evidence
from agents.shared.forge_emit import EmitSpec, EmitSummary, emit_tasks


def external_ref(candidate: BumpCandidate) -> str:
    return f"deps:{candidate.name.lower()}:{candidate.latest}"


def _title(candidate: BumpCandidate) -> str:
    # Comma-free (Forge Depends On cells split on commas).
    return f"Review dependency bump {candidate.name} {candidate.current} to {candidate.latest}"


def _content(
    candidate: BumpCandidate, evidence: EvidenceBundle | None, reason: str, branch: str | None
) -> str:
    lines = [
        f"The dependabot bumper declined to auto-merge `{candidate.name} "
        f"{candidate.current} -> {candidate.latest}`.",
        "",
        f"**Reason:** {reason}",
    ]
    if branch:
        lines += [
            "",
            f"**Branch:** `{branch}` (the bump is applied there — review and merge or discard)",
        ]
    if evidence is not None:
        lines += ["", render_evidence(evidence)]
    lines += ["", "---", "_Auto-proposed by `meta deps`. Review before merging._"]
    return "\n".join(lines)


def emit_advisory(
    candidate: BumpCandidate,
    evidence: EvidenceBundle | None,
    reason: str,
    *,
    project: str | None,
    branch: str | None,
    log: Callable[[str], None] = print,
) -> EmitSummary | None:
    """Emit the advisory task; silently a no-op when *project* is None (log-only runs)."""
    if not project:
        return None
    spec = EmitSpec(
        title=_title(candidate),
        content=_content(candidate, evidence, reason, branch),
        external_ref=external_ref(candidate),
    )
    return emit_tasks([spec], project=project, status="Ready", log=log)
