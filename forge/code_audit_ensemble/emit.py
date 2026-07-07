"""Map the code-audit ensemble's *confirmed* findings to Forge bug-fix tasks (review-first).

Only ``report.confirmed`` findings become tasks — never tentative or rejected ones. Each task
carries the scenario, suggested fix, and the skeptic panel's reasoning so the human reviewer
sees the verification before approving. The generic create + dedup + cap lives in
``agents/shared/forge_emit.py``; this module supplies the bug-specific title, body, and the stable
``external_ref``.

Idempotency caveat: unlike the refactor/testing ensembles (whose finders emit a categorical
gap/smell type), audit findings have only a free-text title. The external_ref therefore keys on
``file`` + a slug of the title, which is stable when a re-run describes the same bug with the same
wording but can re-emit if the phrasing drifts. The per-run cap and within-run dedup still bound
the blast radius.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from agents.code_audit_ensemble.models import SEVERITY_RANK, AuditReport, ScoredFinding
from agents.shared.forge_emit import EmitSpec, EmitSummary
from agents.shared.task_store import get_task_store

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _slug(text: str, limit: int = 60) -> str:
    s = _NON_ALNUM.sub("-", text.strip().lower()).strip("-")
    return s[:limit].rstrip("-")


def external_ref(f: ScoredFinding) -> str:
    """Stable-ish dedup key from file + title slug (audit findings have no categorical type)."""
    fin = f.finding
    file = _slug(fin.file, limit=80) or "unknown"
    return f"audit:{file}:{_slug(fin.title) or 'finding'}"


def _title(f: ScoredFinding) -> str:
    fin = f.finding
    title = fin.title.strip() or "issue"
    loc = f" ({fin.file})" if fin.file else ""
    return f"Fix: {title}{loc}"[:120]


def _content(f: ScoredFinding) -> str:
    fin, v = f.finding, f.verdict
    lines = [f"**Issue:** {fin.title}"]
    if fin.file:
        where = fin.file + (f":{fin.line}" if fin.line else "")
        lines.append(f"**Location:** `{where}`")
    lines.append(
        f"**Severity:** {v.severity} | **Verified:** {v.votes_real}/{v.votes_total} skeptics"
    )
    if fin.scenario:
        lines += ["", f"**Scenario:** {fin.scenario}"]
    if fin.suggestion:
        lines += ["", f"**Suggested fix:** {fin.suggestion}"]
    if v.reasonings:
        lines += ["", "**Skeptic panel:**"]
        lines += [f"- {r}" for r in v.reasonings]
    lines += [
        "",
        "---",
        "_Auto-proposed by `meta audit`. Review before implementing._",
    ]
    return "\n".join(lines)


def report_to_specs(report: AuditReport, *, min_severity: str = "low") -> list[EmitSpec]:
    """Confirmed findings at or above ``min_severity``, as emit specs (highest severity first)."""
    floor = SEVERITY_RANK.get(min_severity, 1)
    return [
        EmitSpec(
            title=_title(f),
            content=_content(f),
            external_ref=external_ref(f),
            task_type="bug-fix",
            tags="bug,auto-proposed",
        )
        for f in report.confirmed
        if SEVERITY_RANK.get(f.verdict.severity, 0) >= floor
    ]


def emit_report(
    report: AuditReport,
    *,
    project: str,
    min_severity: str = "low",
    dry_run: bool = False,
    max_per_run: int | None = None,
    log: Callable[[str], None] | None = None,
) -> EmitSummary:
    """Emit confirmed findings into ``project`` as review-then-implement bug-fix tasks.

    Tasks are created ``Spec Needed`` / ``Manual`` so the autonomous worker won't pick them up
    until a human reviews and promotes them.
    """
    specs = report_to_specs(report, min_severity=min_severity)
    return get_task_store().emit(
        specs,
        project=project,
        status="Spec Needed",
        execution_mode="Manual",
        phase="Bugfix",
        priority=6,
        dry_run=dry_run,
        max_per_run=max_per_run,
        log=log,
    )
