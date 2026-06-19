"""Map the testing ensemble's *confirmed* gaps to Forge test tasks (review-then-implement).

Only ``report.confirmed`` gaps become tasks — never tentative or rejected ones. Each task
carries the suggested test, why it matters, and the skeptic panel's reasoning so the human
reviewer sees the verification before approving. The generic create + dedup + cap lives in
``agents/shared/forge_emit.py``; this module supplies the test-specific title, body, and the
stable ``external_ref``.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from agents.shared.forge_emit import EmitSpec, EmitSummary, emit_tasks
from agents.testing_ensemble.models import SEVERITY_RANK, ScoredGap, TestReport


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def external_ref(g: ScoredGap) -> str:
    """Stable dedup key, independent of the run-assigned TG-NN ids (which change each run).

    Keyed on the durable (target, gap_type) pair: re-running ``meta testing`` over the same
    code re-derives the same ref, so the task is skipped instead of duplicated.
    """
    target = _norm(g.gap.target) or "unknown"
    typ = _norm(g.gap.gap_type) or "coverage"
    return f"test:{target}:{typ}"


def _title(g: ScoredGap) -> str:
    target = g.gap.target.strip() or "code"
    if len(target) > 80:
        target = target[:77] + "..."
    typ = g.gap.gap_type.strip()
    return f"Add test: {target} [{typ}]" if typ else f"Add test: {target}"


def _content(g: ScoredGap) -> str:
    gap, v = g.gap, g.verdict
    lines = [
        f"**Gap:** {gap.gap_type or 'coverage'} at `{gap.target}`",
        f"**Severity:** {v.severity} | **Verified:** {v.votes_real}/{v.votes_total} skeptics",
    ]
    if gap.suggested_test:
        lines += ["", f"**Suggested test:** {gap.suggested_test}"]
    if gap.why_it_matters:
        lines += ["", f"**Why it matters:** {gap.why_it_matters}"]
    if v.reasonings:
        lines += ["", "**Skeptic panel:**"]
        lines += [f"- {r}" for r in v.reasonings]
    lines += [
        "",
        "---",
        "_Auto-proposed by `meta testing`. Review before implementing._",
    ]
    return "\n".join(lines)


def report_to_specs(report: TestReport, *, min_severity: str = "low") -> list[EmitSpec]:
    """Confirmed gaps at or above ``min_severity``, as emit specs (highest severity first)."""
    floor = SEVERITY_RANK.get(min_severity, 1)
    return [
        EmitSpec(
            title=_title(g),
            content=_content(g),
            external_ref=external_ref(g),
            task_type="test",
            estimate="s",  # adding a focused test is typically small
            complexity="routine",
            tags="test,auto-proposed",
        )
        for g in report.confirmed
        if SEVERITY_RANK.get(g.verdict.severity, 0) >= floor
    ]


def emit_report(
    report: TestReport,
    *,
    project: str,
    min_severity: str = "low",
    dry_run: bool = False,
    max_per_run: int | None = None,
    log: Callable[[str], None] | None = None,
) -> EmitSummary:
    """Emit confirmed gaps into ``project`` as review-then-implement test tasks.

    Tasks are created ``Spec Needed`` / ``Manual`` so the autonomous worker won't pick them
    up until a human reviews and promotes them.
    """
    specs = report_to_specs(report, min_severity=min_severity)
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
