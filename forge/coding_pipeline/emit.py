"""A3 - Emit the coding-pipeline task tree into the Forge project.

Maps ``LeafSpec``s to ``forge_emit.EmitSpec``s and emits the tree into the Forge
project, idempotently. Re-emission is the replan mechanic: re-emitting the whole
intended tree only creates genuinely new leaves (forge_emit dedup), logged via
``EmitSummary`` as created / skipped / capped.

Stable refs are derived from durable leaf attributes so that crash recovery and
replanning never produce duplicate Forge tasks.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from agents.coding_pipeline.models import LeafSpec, TaskTree
from agents.shared import forge_emit
from agents.shared.automerge import log_decision, slugify


def stable_ref(
    epic_slug: str, leaf: LeafSpec, *, fixup: bool = False, finding_slug: str | None = None
) -> str:
    """Return a stable, idempotency-key external_ref for *leaf*.

    Shape: ``pipeline:{epic-slug}:{leaf-slug}``, or ``pipeline:{epic-slug}:fix:{slug}`` for
    fix-up leaves generated during replanning. Fix-ups key on the FINDING's slug when given —
    the replan model invents leaf titles freely, so a title-derived ref would mint a duplicate
    every time the same finding is re-discovered (dry-run Q2); the finding slug is the stable
    identity. The epic slug stays in the fixup shape so fix-ups from different epics can never
    collide.
    """
    if fixup and finding_slug:
        return f"pipeline:{epic_slug}:fix:{slugify(finding_slug, max_len=50)}"
    slug_source = f"{leaf.feature}:{leaf.title}"
    leaf_slug = slugify(slug_source, max_len=50)
    if fixup:
        return f"pipeline:{epic_slug}:fix:{leaf_slug}"
    return f"pipeline:{epic_slug}:{leaf_slug}"


def _topo_sort(leaves: list[LeafSpec]) -> list[LeafSpec]:
    """Return *leaves* in dependency order.

    Rejects any title containing a comma (Forge Depends-On cell format constraint)
    and raises on unknown deps or cycles.
    """
    for leaf in leaves:
        if "," in leaf.title:
            raise ValueError(
                f"leaf title {leaf.title!r} contains a comma - "
                "Forge Depends On cells split on commas"
            )

    title_set = {leaf.title for leaf in leaves}

    # Validate: every dep title must exist in the tree
    for leaf in leaves:
        for dep in leaf.depends_on:
            if dep not in title_set:
                raise ValueError(
                    f"leaf {leaf.title!r} depends on {dep!r}, "
                    f"but no leaf with that title exists in the tree"
                )

    # Kahn's algorithm for topological sort
    in_degree: dict[str, int] = {leaf.title: len(leaf.depends_on) for leaf in leaves}

    queue = sorted([t for t, d in in_degree.items() if d == 0])
    result: list[str] = []

    adj: dict[str, list[str]] = {t: [] for t in title_set}
    for leaf in leaves:
        for dep in leaf.depends_on:
            adj[dep].append(leaf.title)

    while queue:
        node = queue.pop(0)
        result.append(node)
        for dependent in sorted(adj[node]):
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)
        queue.sort()

    if len(result) != len(title_set):
        remaining = title_set - set(result)
        raise ValueError(f"circular dependency detected among: {', '.join(sorted(remaining))}")

    by_title = {leaf.title: leaf for leaf in leaves}
    return [by_title[t] for t in result]


@dataclass
class EmitResult:
    """Outcome of emitting a single task tree into Forge."""

    project: str
    epic_slug: str
    created: list[forge_emit.EmitOutcome] = field(default_factory=list)
    skipped: list[forge_emit.EmitOutcome] = field(default_factory=list)
    capped: int = 0
    planned: list[forge_emit.EmitOutcome] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"emitted {len(self.created)} created / "
            f"{len(self.skipped)} skipped (dedup) / "
            f"{self.capped} capped / "
            f"{len(self.planned)} planned (dry-run)"
        )


def _build_spec(
    leaf: LeafSpec, epic_slug: str, *, fixup: bool = False, finding_slug: str | None = None
) -> forge_emit.EmitSpec:
    """Build an EmitSpec from a LeafSpec.

    The architect's tags pass through VERBATIM — every LeafSpec field is concrete
    (conservative defaults live on the model), so nothing here second-guesses A2's
    decisions. That includes ``requires_tests=False``: it stays an explicit "No"
    rather than degrading to unset.
    """
    return forge_emit.EmitSpec(
        title=leaf.title,
        content=leaf.content,
        external_ref=stable_ref(epic_slug, leaf, fixup=fixup, finding_slug=finding_slug),
        task_type=leaf.task_type,
        estimate=leaf.estimate,
        complexity=leaf.complexity,
        feature=leaf.feature,
        status=leaf.status,
        execution_mode=leaf.execution_mode,
        phase=leaf.phase,
        priority=leaf.priority,
        max_files=leaf.max_files,
        requires_tests=leaf.requires_tests,
        model_tier=leaf.model_tier,
        depends_on=(", ".join(leaf.depends_on) if leaf.depends_on else None),
    )


def emit_tree(
    tree: TaskTree,
    *,
    project: str,
    epic_slug: str,
    dry_run: bool = False,
    max_per_run: int | None = None,
    runs_dir: Path | None = None,
    log: Callable[[str], None] | None = None,
) -> EmitResult:
    """Map ``LeafSpec``s to ``EmitSpec``s and emit into *project*, idempotently.

    Steps:
    1. Sort leaves in dependency order (raises on comma titles / unknown deps / cycles).
    2. Build an ``EmitSpec`` for each leaf with stable ref + per-spec gating/guardrails.
       Every spec carries its leaf's own tags, so forge_emit's batch-level gating
       defaults never apply — the architect's tree is the single source of tagging.
    3. Delegate to ``forge_emit.emit_tasks`` for dedup + cap + creation.
    4. If *runs_dir* is given, log each emission decision as a JSONL record.
    """
    ordered = _topo_sort(tree.leaves)
    specs = [_build_spec(leaf, epic_slug) for leaf in ordered]

    summary = forge_emit.emit_tasks(
        specs,
        project=project,
        dry_run=dry_run,
        max_per_run=max_per_run,
        log=log,
    )

    if runs_dir is not None and not dry_run:
        run_dir = runs_dir / epic_slug
        decision_path = run_dir / "journal.jsonl"
        for outcome in summary.created:
            log_decision(
                {
                    "event": "emit_task",
                    "epic": epic_slug,
                    "external_ref": outcome.external_ref,
                    "title": outcome.title,
                    "action": "created",
                },
                decision_path,
            )
        for outcome in summary.skipped:
            log_decision(
                {
                    "event": "emit_task",
                    "epic": epic_slug,
                    "external_ref": outcome.external_ref,
                    "title": outcome.title,
                    "action": "skipped",
                    "reason": outcome.detail,
                },
                decision_path,
            )

    return EmitResult(
        project=project,
        epic_slug=epic_slug,
        created=summary.created,
        skipped=summary.skipped,
        capped=summary.capped,
        planned=summary.planned,
    )


def emit_fixup(
    leaf: LeafSpec,
    *,
    project: str,
    epic_slug: str,
    finding_slug: str | None = None,
    dry_run: bool = False,
    log: Callable[[str], None] | None = None,
) -> forge_emit.EmitOutcome:
    """Emit a single fix-up leaf (generated during replan from confirmed findings).

    Uses the ``pipeline:{epic-slug}:fix:{slug}`` ref shape, keyed on ``finding_slug``
    when given (stable across replans; the leaf title is model-invented). The leaf's
    own tags pass through exactly like tree emission (the replan stage decides fix-up
    autonomy).
    """
    spec = _build_spec(leaf, epic_slug, fixup=True, finding_slug=finding_slug)
    summary = forge_emit.emit_tasks([spec], project=project, dry_run=dry_run, log=log)
    for bucket in (summary.created, summary.skipped, summary.planned):
        if bucket:
            return bucket[0]
    raise RuntimeError(f"fix-up emission produced no outcome for {leaf.title!r}")


def dry_run_emit(
    tree: TaskTree,
    *,
    project: str,
    epic_slug: str,
    max_per_run: int | None = None,
    log: Callable[[str], None] | None = None,
) -> EmitResult:
    """Run emission in dry-run mode - plans without creating anything.

    Parity mirror for ensemble dry-run reports.
    """
    return emit_tree(
        tree,
        project=project,
        epic_slug=epic_slug,
        dry_run=True,
        max_per_run=max_per_run,
        log=log,
    )
