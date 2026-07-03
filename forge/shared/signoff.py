"""Full-quorum, cross-family sign-off gate for automated actions.

The last gate before an auto-action touches VCS: fan a structured approve/block prompt across a
diverse provider panel and approve ONLY if every seat responds AND unanimously approves. Anything
less — a dropped/degraded provider, one dissent, an unparseable verdict — fails closed. Extracted
from the testing ensemble's auto-merge loop so the coding pipeline's epic gate and the Dependabot
bumper reuse the same machinery.

Layering: this module never imports a specific ensemble. Callers supply the (already-filtered)
``SignoffSeat``s from their own roster — e.g. pr_review's ``build_reviewer_slots`` — keeping
``shared/`` free of upward deps, the same rule ``automerge`` follows.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from agents.shared.ensemble import Executor
from agents.shared.panel import run_panel


@dataclass(frozen=True)
class SignoffSeat:
    """One sign-off panel seat: a provider label and the executor that answers for it."""

    provider: str
    executor: Executor


@dataclass
class SignoffResult:
    """The full-quorum sign-off gate outcome. ``approved`` requires every seat to respond AND
    unanimously approve — anything less (a dropped/degraded provider, one dissent, an unparseable
    verdict) fails closed."""

    approved: bool
    attempted: int
    approvals: int
    providers: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    reason: str = ""


def full_quorum_signoff(
    diff_text: str,
    *,
    seats: Sequence[SignoffSeat],
    system: str,
    ref: str,
    context: str = "",
    min_seats: int = 2,
    max_tokens: int = 4096,
    timeout: float = 120.0,
) -> SignoffResult:
    """Run the sign-off panel over ``diff_text`` and return the fail-closed verdict.

    ``system`` is the caller's gatekeeper prompt and must demand the JSON verdict shape
    ``{"approve": true|false, "blockers": [...], "notes": "..."}``. ``ref`` names the change
    (branch/epic) and ``context`` is an optional caller-specific line inserted before the diff
    (e.g. "This change must contain ONLY test files."). Fewer than ``min_seats`` seats fails
    closed without any LLM call — a one-provider "quorum" is no cross-check at all.
    """
    providers = [s.provider for s in seats]
    if len(seats) < min_seats:
        return SignoffResult(
            approved=False,
            attempted=len(seats),
            approvals=0,
            providers=providers,
            reason=f"need >={min_seats} active providers for a diverse sign-off, have {len(seats)}",
        )
    parts = [f"Change: {ref}"]
    if context:
        parts.append(context)
    parts.append(f"\nDiff:\n{diff_text}")
    panel = run_panel(
        executors=[s.executor for s in seats],
        system=system,
        user="\n".join(parts),
        floor=len(seats),
        max_tokens=max_tokens,
        timeout=timeout,
    )
    approvals = sum(1 for r in panel.responses if r.get("approve") is True)
    blockers = [str(b) for r in panel.responses for b in (r.get("blockers") or [])]
    full = len(panel.responses) == panel.attempted  # every seat produced a verdict
    approved = full and approvals == panel.attempted
    got, n = len(panel.responses), panel.attempted
    reason = "" if approved else f"quorum {got}/{n}, approvals {approvals}/{n}"
    return SignoffResult(
        approved=approved,
        attempted=panel.attempted,
        approvals=approvals,
        providers=providers,
        blockers=blockers,
        reason=reason,
    )
