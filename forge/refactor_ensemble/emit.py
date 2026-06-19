"""Map the refactor ensemble's *confirmed* smells to Forge tasks (review-then-implement).

Only ``plan.confirmed`` smells become tasks — never tentative or rejected ones. Each task
carries the proposed refactor, benefit, risk, and the skeptic panel's reasoning so the
human reviewer sees the verification before approving. The generic create + dedup + cap
lives in ``agents/shared/forge_emit.py``; this module supplies the refactor-specific
title, body, and the stable ``external_ref``.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from agents.refactor_ensemble.models import IMPACT_RANK, RefactorPlan, ScoredSmell
from agents.shared.forge_emit import EmitSpec, EmitSummary, emit_tasks

# effort -> Forge estimate; refactors are sized small/medium/large.
_EFFORT_TO_ESTIMATE = {"small": "s", "medium": "m", "large": "l"}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def external_ref(smell: ScoredSmell) -> str:
    """Stable dedup key, independent of the run-assigned RF-NN ids (which change each run).

    Keyed on the durable (location, smell_type) pair: re-running ``meta refactor`` over the
    same code re-derives the same ref, so the task is skipped instead of duplicated.
    """
    loc = _norm(smell.smell.location) or "unknown"
    typ = _norm(smell.smell.smell_type) or "refactor"
    return f"refactor:{loc}:{typ}"


def _title(smell: ScoredSmell) -> str:
    loc = smell.smell.location.strip() or "code"
    if len(loc) > 80:
        loc = loc[:77] + "..."
    typ = smell.smell.smell_type.strip()
    return f"Refactor {loc} [{typ}]" if typ else f"Refactor {loc}"


def _content(s: ScoredSmell) -> str:
    sm, v = s.smell, s.verdict
    lines = [
        f"**Smell:** {sm.smell_type or 'refactor'} at `{sm.location}`",
        f"**Impact:** {v.impact} | **Effort:** {sm.effort} | "
        f"**Verified:** {v.votes_real}/{v.votes_total} skeptics",
        "",
        f"**Proposed refactor:** {sm.proposed_refactor}",
    ]
    if sm.benefit:
        lines += ["", f"**Benefit:** {sm.benefit}"]
    if sm.risk and sm.risk.strip().lower() not in ("none", ""):
        lines += ["", f"**Risk:** {sm.risk}"]
    if v.reasonings:
        lines += ["", "**Skeptic panel:**"]
        lines += [f"- {r}" for r in v.reasonings]
    lines += [
        "",
        "---",
        "_Auto-proposed by `meta refactor`. Behavior-preserving refactor — "
        "review before implementing._",
    ]
    return "\n".join(lines)


def plan_to_specs(plan: RefactorPlan, *, min_impact: str = "low") -> list[EmitSpec]:
    """Confirmed smells at or above ``min_impact``, as emit specs (order: highest impact first)."""
    floor = IMPACT_RANK.get(min_impact, 1)
    return [
        EmitSpec(
            title=_title(s),
            content=_content(s),
            external_ref=external_ref(s),
            task_type="refactor",
            estimate=_EFFORT_TO_ESTIMATE.get(s.smell.effort, "m"),
            complexity="routine",
            tags="refactor,auto-proposed",
        )
        for s in plan.confirmed
        if IMPACT_RANK.get(s.verdict.impact, 0) >= floor
    ]


def emit_plan(
    plan: RefactorPlan,
    *,
    project: str,
    min_impact: str = "low",
    dry_run: bool = False,
    max_per_run: int | None = None,
    log: Callable[[str], None] | None = None,
) -> EmitSummary:
    """Emit confirmed smells into ``project`` as review-then-implement tasks.

    Tasks are created ``Spec Needed`` / ``Manual`` so the autonomous worker won't pick
    them up until a human reviews and promotes them.
    """
    specs = plan_to_specs(plan, min_impact=min_impact)
    return emit_tasks(
        specs,
        project=project,
        status="Spec Needed",
        execution_mode="Manual",
        phase="Polish",
        priority=6,
        dry_run=dry_run,
        max_per_run=max_per_run,
        log=log,
    )
