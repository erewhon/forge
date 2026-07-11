"""Advisory fallback: one Forge task per bump the loop refused to merge.

The task carries the policy/gate reason, the full rendered evidence, and the pushed branch so
a human (or the task worker) picks up exactly where the loop stopped. ``external_ref`` is keyed
on the durable (package, target-version) pair: re-running the bumper against the same outdated
dep skips instead of duplicating.

Reachability demotion: when ``reachable`` is provably ``False`` the advisory task gets a
numerically higher priority (lowered priority) and a spec note — this signal can ONLY demote,
never promote or affect the auto-merge gate.
"""

from __future__ import annotations

from collections.abc import Callable

from forge.dependabot.models import BumpCandidate, EvidenceBundle
from forge.dependabot.prompts import render_evidence
from forge.shared.forge_emit import EmitSpec, EmitSummary
from forge.shared.task_store import get_task_store

# Demotion constants — "lower priority by 2" means numerically higher (6 → 8).
# Clamp to 1–9 (1 = highest priority, 9 = lowest in the batch sort).
_DEMOTE_DELTA = 2
_DEMOTE_MIN = 1
_DEMOTE_MAX = 9
_IMPORT_GRAPH_NOTE = (
    "⚠ vulnerable package not imported by this repo's code (import-graph heuristic) — deprioritized"
)


def _demote(priority: int, reachable: bool | None) -> tuple[int, str | None]:
    """Return (adjusted_priority, note_or_None) for demotion wiring.

    Only when *reachable* is provably ``False``: raise priority by 2, clamped.
    ``True`` or ``None`` leaves priority unchanged and returns no note.
    """
    if reachable is False:
        new_pri = min(_DEMOTE_MAX, max(_DEMOTE_MIN, priority + _DEMOTE_DELTA))
        if new_pri != priority:
            return new_pri, _IMPORT_GRAPH_NOTE
    return priority, None


def external_ref(candidate: BumpCandidate) -> str:
    return f"deps:{candidate.name.lower()}:{candidate.latest}"


def _title(candidate: BumpCandidate) -> str:
    # Comma-free (Forge Depends On cells split on commas).
    return f"Review dependency bump {candidate.name} {candidate.current} to {candidate.latest}"


def _content(
    candidate: BumpCandidate,
    evidence: EvidenceBundle | None,
    reason: str,
    branch: str | None,
    reachability_note: str | None = None,
) -> str:
    lines = [
        f"The dependabot bumper declined to auto-merge `{candidate.name} "
        f"{candidate.current} -> {candidate.latest}`.",
        "",
        f"**Reason:** {reason}",
    ]
    if reachability_note:
        lines += ["", f"**Note:** {reachability_note}"]
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
    base_priority: int = 6,
) -> EmitSummary | None:
    """Emit the advisory task; silently a no-op when *project* is None (log-only runs).

    When *evidence* has ``reachable=False`` the task priority is lowered (numerically
    raised) by 2 and a reachability note is prepended — demote-only, never touches the
    auto-merge gate.
    """
    if not project:
        return None

    # Apply reachability demotion.
    reachability_note: str | None = None
    pri = base_priority
    if evidence is not None and evidence.reachable is not None:
        pri, reachability_note = _demote(base_priority, evidence.reachable)

    spec = EmitSpec(
        title=_title(candidate),
        content=_content(candidate, evidence, reason, branch, reachability_note),
        external_ref=external_ref(candidate),
        priority=pri,
    )
    return get_task_store().emit([spec], project=project, status="Ready", log=log)
