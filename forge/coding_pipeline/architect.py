"""The architect — A1 framing (this module will also grow A2 decomposition and A4 replan).

Framing is the push-back stage: the architect's first output is a scoping *position*, not a
task tree. It reads the A0 inventory and is explicitly licensed to declare the goal mis-scoped
and propose a better one (the Nous "web parity" → Tauri-platform-shim move). Nothing decomposes
until a human approves: ``FramingProposal.approved`` is forced False on every model output, and
only :func:`approve_framing` — a human-invoked action — flips it. There is no bypass flag.

Persistence rule: an existing ``framing.json`` is never silently overwritten (the human may have
edited it); re-proposing requires ``force=True``.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from agents.coding_pipeline.config import settings
from agents.coding_pipeline.inventory import render_inventory
from agents.coding_pipeline.models import (
    BoundednessCheck,
    EscalateAction,
    FixupAction,
    FramingProposal,
    GoalSpec,
    HaltAction,
    IntegrationFixAction,
    Inventory,
    LeafSpec,
    ReplanAction,
    RespecAction,
    TaskTree,
    WaveReport,
)
from agents.shared.ensemble import ApiExecutor, Pool
from agents.shared.panel import Finder, discover, structured


class ArchitectError(RuntimeError):
    """A stage of the architect could not produce a usable artifact."""


class FramingExistsError(ArchitectError):
    """A framing.json already exists; re-proposing would clobber possible human edits."""


class FramingNotApprovedError(ArchitectError):
    """Decomposition was attempted on a framing no human has approved."""


FRAMING_SYSTEM = """You are the ARCHITECT for an iterative coding pipeline. Your job right now \
is FRAMING, not planning: read the goal and the repository inventory, then take a scoping \
position a senior engineer would defend.

Mandates:
- Push back when warranted. If the inventory shows the goal as stated would duplicate existing \
work, fight the wrong battle, or solve the wrong problem, say so: restate the goal the way it \
SHOULD be scoped and set "rescoped" to true. Restating the goal verbatim is only correct when \
the goal is genuinely well-scoped.
- Study the "Goal-term overlaps" and "Forge tasks already filed" sections hard — they are the \
evidence for "this already exists" and "this is already planned".
- Give a value ordering: which slices ship user-visible value first. Polish comes last.
- Name the real risks and the blast radius of the work.
- Propose 2-4 genuinely different options with tradeoffs before recommending one.
- "epic_slug" must be a short lowercase hyphenated slug for this epic.

Respond with ONLY a JSON object matching:
{"goal_as_stated": str, "restated_goal": str, "rescoped": bool, "inventory_summary": str,
 "gap_analysis": str, "options": [{"name": str, "summary": str, "tradeoffs": str}],
 "recommendation": str, "value_ordering": [str], "risks": [str], "epic_slug": str,
 "branch": str|null}

Do not include an "approved" field — approval is a human decision, never yours."""


def _architect_pool() -> Pool:
    if settings.llm_backend == "anthropic":
        executor = ApiExecutor(
            label=f"anthropic:{settings.anthropic_model}",
            kind="anthropic",
            model=settings.anthropic_model,
        )
    else:
        executor = ApiExecutor(
            label=f"router:{settings.architect_model}",
            kind="openai",
            model=settings.architect_model,
            base_url=settings.openai_base_url,
            api_key=settings.openai_api_key,
        )
    return Pool(role="architect", executors=[executor])


def _framing_user(goal: GoalSpec, inventory: Inventory) -> str:
    parts = [
        "## Goal (as stated by the human)",
        goal.goal,
    ]
    if goal.context:
        parts += ["\n## Context / constraints", goal.context]
    if goal.value_hints:
        parts += ["\n## Value hints", *[f"- {h}" for h in goal.value_hints]]
    parts += ["\n## Repository inventory\n", render_inventory(inventory)]
    return "\n".join(parts)


def propose_framing(goal: GoalSpec, inventory: Inventory) -> FramingProposal:
    """One strong-tier structured call: goal + inventory → a FramingProposal.

    The schema is the validator — unparseable or schema-invalid output is retried and failed
    over inside the pool. ``approved`` is forced False regardless of what the model emitted,
    and a human-supplied ``goal.epic_slug`` beats the model's proposal.
    """
    result = structured(
        pool=_architect_pool(),
        schema=FramingProposal,
        system=FRAMING_SYSTEM,
        user=_framing_user(goal, inventory),
        max_tokens=settings.architect_max_tokens,
        timeout=settings.architect_timeout,
    )
    if result.value is None:
        raise ArchitectError(f"framing produced no usable proposal: {result.error}")
    proposal = result.value
    proposal.approved = False  # only approve_framing may flip this
    if goal.epic_slug:
        proposal.epic_slug = goal.epic_slug
    return proposal


# --- persistence + the approval gate -----------------------------------------


def _framing_json(run_dir: Path) -> Path:
    return run_dir / "framing.json"


def persist_framing(proposal: FramingProposal, run_dir: Path, *, force: bool = False) -> Path:
    """Write framing.json + framing.md into the run dir. Refuses to clobber an existing
    framing (a human may have edited or approved it) unless ``force``."""
    path = _framing_json(run_dir)
    if path.exists() and not force:
        raise FramingExistsError(
            f"{path} already exists — re-run with force to overwrite the existing framing"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(proposal.model_dump_json(indent=2))
    (run_dir / "framing.md").write_text(render_framing(proposal))
    return path


def load_framing(run_dir: Path) -> FramingProposal | None:
    path = _framing_json(run_dir)
    if not path.is_file():
        return None
    return FramingProposal.model_validate(json.loads(path.read_text()))


def approve_framing(run_dir: Path) -> FramingProposal:
    """The human approval action: flip ``approved`` and re-persist. Raises if no framing."""
    proposal = load_framing(run_dir)
    if proposal is None:
        raise ArchitectError(f"no framing.json in {run_dir} — run the framing stage first")
    proposal.approved = True
    _framing_json(run_dir).write_text(proposal.model_dump_json(indent=2))
    (run_dir / "framing.md").write_text(render_framing(proposal))
    return proposal


def require_approved_framing(run_dir: Path) -> FramingProposal:
    """The A1→A2 hard gate: decomposition calls this and gets an exception unless a human
    approved the framing. No bypass."""
    proposal = load_framing(run_dir)
    if proposal is None:
        raise ArchitectError(f"no framing.json in {run_dir} — run the framing stage first")
    if not proposal.approved:
        raise FramingNotApprovedError(
            "framing has not been approved by a human — read framing.md, then approve "
            "(meta build plan --approve) before decomposition"
        )
    return proposal


# --- A2: decomposition --------------------------------------------------------

DECOMPOSE_SYSTEM = """You are the ARCHITECT for an iterative coding pipeline. The framing below \
was APPROVED by a human. Your job now is DECOMPOSITION: turn it into a flat list of leaves whose \
"depends_on" fields encode the tree (epic -> features -> leaves, depth 3 max).

Every leaf must be WORKER-SHAPED — all five criteria:
1. single concern, no unresolved design choice inside it;
2. expected diff fits max_files (5 or fewer files);
3. estimate is xs, s, or m;
4. acceptance criteria checkable by the project's test suite;
5. the spec names its target files/modules.
A leaf that cannot meet all five must be tagged complexity="novel" and execution_mode="Manual" — \
'a human does this one' is a valid terminal state. Do NOT emit vague umbrella leaves.

Each leaf's "content" is a COMPLETE worker spec — the worker sees nothing else. Include: \
what/why (one short paragraph), acceptance criteria (bulleted, testable), a files hint, and \
test expectations.

Conservative autonomy tagging:
- routine mechanical leaves in well-tested areas: execution_mode="Auto-OK", requires_tests=true, \
max_files <= 5, model_tier="auto";
- anything with design latitude, on a safety path, or novel: execution_mode="Manual";
- an underspecified leaf: status="Spec Needed" and execution_mode="Manual".

Ordering: priority encodes the approved value ordering — user-visible value first (priority 2-3), \
infrastructure it depends on same, polish last (priority 5-6). Group related leaves under the \
same short "feature" name. "depends_on" entries must EXACTLY match another leaf's title. Titles \
must NOT contain commas.

Respond with ONLY a JSON object: {"leaves": [<LeafSpec>, ...]} where each LeafSpec is
{"title": str, "content": str, "feature": str, "depends_on": [str], "priority": int,
 "phase": "Feature"|"Infrastructure"|"Polish"|"Bugfix"|"Launch", "status": "Ready"|"Spec Needed",
 "execution_mode": "Manual"|"Auto-OK"|"Auto-Preferred", "complexity": "routine"|"novel"|null,
 "estimate": "xs"|"s"|"m"|"l"|"xl"|null, "task_type": "bug-fix"|"feature"|"refactor"|"docs"|
 "test"|"chore", "requires_tests": bool, "max_files": int|null,
 "model_tier": "auto"|"auto-free"|"auto-full"|null}"""

BOUNDEDNESS_SYSTEM = """You are a skeptical senior engineer reviewing ONE proposed task leaf for \
an autonomous coding worker. Judge it against five criteria, strictly — an optimistic pass here \
costs a blown worker run later. Echo the leaf title EXACTLY as given.

Respond with ONLY a JSON object:
{"leaf_title": str, "single_concern": bool, "bounded_diff": bool, "small_estimate": bool,
 "testable_acceptance": bool, "files_named": bool, "notes": str}"""


class LeafBoundedness(BoundednessCheck):
    """Wire shape for the per-leaf boundedness verdict: the check plus the echoed title that
    aligns it back to its leaf (finder results come back unordered)."""

    leaf_title: str


def _decompose_user(framing: FramingProposal, inventory: Inventory) -> str:
    return f"{render_framing(framing)}\n\n## Repository inventory\n\n{render_inventory(inventory)}"


def _leaf_summary(leaf: LeafSpec) -> str:
    return (
        f"Title: {leaf.title}\n"
        f"Estimate: {leaf.estimate or 'unset'} | max_files: {leaf.max_files or 'unset'} | "
        f"execution_mode: {leaf.execution_mode}\n\nSpec:\n{leaf.content}"
    )


def _validate_deps(leaves: list[LeafSpec]) -> None:
    """Unknown dep titles or cycles are architect errors — fix the tree, don't guess."""
    titles = {leaf.title for leaf in leaves}
    unknown = sorted(
        f"{leaf.title} -> {dep}" for leaf in leaves for dep in leaf.depends_on if dep not in titles
    )
    if unknown:
        raise ArchitectError(f"leaves depend on unknown titles: {'; '.join(unknown)}")

    indegree = {leaf.title: len(leaf.depends_on) for leaf in leaves}
    dependents: dict[str, list[str]] = {leaf.title: [] for leaf in leaves}
    for leaf in leaves:
        for dep in leaf.depends_on:
            dependents[dep].append(leaf.title)
    frontier = [t for t, n in indegree.items() if n == 0]
    resolved = 0
    while frontier:
        title = frontier.pop()
        resolved += 1
        for nxt in dependents[title]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                frontier.append(nxt)
    if resolved != len(leaves):
        stuck = sorted(t for t, n in indegree.items() if n > 0)
        raise ArchitectError(f"dependency cycle in the tree involving: {', '.join(stuck)}")


def _apply_conservative_tags(leaves: list[LeafSpec], framing: FramingProposal) -> None:
    """Deterministic post-rules — the floor under whatever the model tagged."""
    default_feature = framing.epic_slug.replace("-", " ").title()
    for leaf in leaves:
        if not leaf.feature.strip():
            leaf.feature = default_feature
        if leaf.status == "Spec Needed" or leaf.complexity == "novel":
            leaf.execution_mode = "Manual"  # underspecified/novel never runs auto
        if leaf.execution_mode != "Manual":
            leaf.requires_tests = True
            if leaf.max_files is None:
                leaf.max_files = settings.default_auto_max_files


def _apply_boundedness(leaves: list[LeafSpec]) -> None:
    """Run the per-leaf boundedness panel; a failing or missing verdict demotes a non-Manual
    leaf to Manual+novel (fail-closed). Splitting further is the model's job on a re-run."""
    if not leaves:
        return
    finders = [
        Finder(label=leaf.title, system=BOUNDEDNESS_SYSTEM, user=_leaf_summary(leaf))
        for leaf in leaves
    ]
    verdicts = discover(
        finders,
        pool=_architect_pool(),
        schema=LeafBoundedness,
        concurrency=4,
        max_tokens=1024,
        timeout=settings.architect_timeout,
    )
    by_title = {v.leaf_title: v for v in verdicts}
    for leaf in leaves:
        verdict = by_title.get(leaf.title)
        if verdict is None:
            leaf.boundedness = BoundednessCheck(
                single_concern=False,
                bounded_diff=False,
                small_estimate=False,
                testable_acceptance=False,
                files_named=False,
                notes="boundedness check unavailable — demoted fail-closed",
            )
        else:
            leaf.boundedness = BoundednessCheck(**verdict.model_dump(exclude={"leaf_title"}))
        if leaf.execution_mode != "Manual" and not leaf.boundedness.worker_shaped:
            leaf.execution_mode = "Manual"
            leaf.complexity = "novel"


def decompose(framing: FramingProposal, inventory: Inventory) -> list[LeafSpec]:
    """A2: approved framing + inventory → validated, conservatively-tagged leaves.

    Hard-refuses an unapproved framing (the object itself, independent of the run-dir gate).
    Comma titles never survive — the LeafSpec validator makes them a schema failure, which
    ``structured`` retries/fails over. Deterministic post-rules then floor the tagging, and
    the boundedness panel demotes anything not worker-shaped.
    """
    if not framing.approved:
        raise FramingNotApprovedError(
            "framing has not been approved by a human — decomposition refused"
        )
    result = structured(
        pool=_architect_pool(),
        schema=TaskTree,
        system=DECOMPOSE_SYSTEM,
        user=_decompose_user(framing, inventory),
        max_tokens=settings.decompose_max_tokens,
        timeout=settings.architect_timeout,
        predicate=lambda tree: len(tree.leaves) > 0,
    )
    if result.value is None:
        raise ArchitectError(f"decomposition produced no usable tree: {result.error}")
    leaves = result.value.leaves
    _validate_deps(leaves)
    _apply_conservative_tags(leaves, framing)
    _apply_boundedness(leaves)
    return leaves


def persist_tree(leaves: list[LeafSpec], run_dir: Path) -> Path:
    """Write tree.json (+ a human-readable tree.md for triage) into the run dir."""
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "tree.json"
    path.write_text(TaskTree(leaves=leaves).model_dump_json(indent=2))
    (run_dir / "tree.md").write_text(render_tree(leaves))
    return path


def load_tree(run_dir: Path) -> list[LeafSpec] | None:
    path = run_dir / "tree.json"
    if not path.is_file():
        return None
    return TaskTree.model_validate(json.loads(path.read_text())).leaves


def render_tree(leaves: list[LeafSpec]) -> str:
    """One line per leaf for the human triage pass — tags visible, specs in tree.json."""
    lines = [f"# Task tree — {len(leaves)} leaves", ""]
    for leaf in leaves:
        b = leaf.boundedness
        shaped = "?" if b is None else ("✓" if b.worker_shaped else "✗")
        deps = f" ← {', '.join(leaf.depends_on)}" if leaf.depends_on else ""
        lines.append(
            f"- [{leaf.execution_mode}/{leaf.status}] **{leaf.title}** "
            f"(p{leaf.priority}, {leaf.estimate or '?'}, {leaf.feature}, shaped:{shaped}){deps}"
        )
    return "\n".join(lines)


# --- A4: replan -----------------------------------------------------------------

REPLAN_SYSTEM = """You are the ARCHITECT for an iterative coding pipeline, running the REPLAN \
stage after a wave of automated work. You receive the approved framing, the current tree, the \
wave report (what landed, what failed and why, the suite result, CONFIRMED review findings), \
per-leaf attempt counts, and a list of leaves already escalated to a human — those are handled; \
do not touch them.

Propose the smallest set of actions that keeps the epic converging:
- "fixup": one new leaf per CONFIRMED finding (use its finding_slug). Never for unconfirmed ones.
- "respec": a failed leaf under the attempt cap gets a REWRITTEN, worker-shaped spec — fix what \
the failure diagnostics show went wrong (smaller scope, explicit files, sharper acceptance).
- "integration_fix": only when the suite is red although the landed leaves individually passed — \
one leaf targeting the interaction, complexity "novel", execution_mode "Manual".
- "split_subtree": repeated failures across one feature — park it for re-decomposition.
- "halt": the framing itself was invalidated by what landed. Prefer halt over guessing.

New/rewritten leaves follow the same rules as decomposition: complete worker specs (what/why, \
testable acceptance criteria, files hint, test expectations), conservative tagging (novel or \
underspecified -> Manual), comma-free titles.

Respond with ONLY a JSON object:
{"actions": [
  {"kind": "fixup", "finding_slug": str, "leaf": <LeafSpec>} |
  {"kind": "respec", "leaf_title": str, "revised": <LeafSpec>, "rationale": str} |
  {"kind": "split_subtree", "feature": str, "rationale": str} |
  {"kind": "integration_fix", "leaf": <LeafSpec>, "rationale": str} |
  {"kind": "halt", "reason": str}
]}
where <LeafSpec> is {"title": str, "content": str, "feature": str, "depends_on": [str],
 "priority": int, "phase": str, "status": "Ready"|"Spec Needed", "execution_mode":
 "Manual"|"Auto-OK"|"Auto-Preferred", "complexity": "routine"|"novel"|null, "estimate":
 "xs"|"s"|"m"|"l"|"xl"|null, "task_type": str, "requires_tests": bool, "max_files": int|null,
 "model_tier": "auto"|"auto-free"|"auto-full"|null}

An empty actions list is a valid answer when the wave landed clean."""


class ReplanEnvelope(BaseModel):
    """Wire shape for the replan call — the discriminated union does the heavy lifting."""

    actions: list[ReplanAction] = []


def deterministic_escalations(report: WaveReport, attempts: dict[str, int]) -> list[EscalateAction]:
    """The pre-rule that never goes near a model: a failed leaf at the attempt cap escalates
    to a human, full stop.

    Public because it is also the orchestrator's degrade path when the model replan
    call fails — the wave must still escalate capped leaves and persist its record."""
    return [
        EscalateAction(leaf_title=outcome.leaf, diagnostics=outcome.reason)
        for outcome in report.failed
        if attempts.get(outcome.leaf, 0) >= settings.max_leaf_attempts
    ]


def _replan_user(
    framing: FramingProposal,
    tree: list[LeafSpec],
    report: WaveReport,
    attempts: dict[str, int],
    escalated: list[EscalateAction],
) -> str:
    landed = "\n".join(f"- {o.leaf} (commit {o.commit_id})" for o in report.landed) or "none"
    failed = (
        "\n".join(
            f"- {o.leaf} (attempts so far: {attempts.get(o.leaf, 0)}): {o.reason}"
            for o in report.failed
            if o.leaf not in {e.leaf_title for e in escalated}
        )
        or "none"
    )
    findings = (
        "\n".join(f"- [{f.severity}] {f.slug}: {f.summary}" for f in report.findings if f.confirmed)
        or "none"
    )
    suite_tail = report.suite.output_tail if report.suite else ""
    suite = "GREEN" if report.suite_green else f"RED\n```\n{suite_tail}\n```"
    escalated_lines = "\n".join(f"- {e.leaf_title}" for e in escalated) or "none"
    return (
        f"## Approved framing (summary)\n\nGoal: {framing.restated_goal}\n"
        f"Recommendation: {framing.recommendation}\n\n"
        f"## Current tree\n\n{render_tree(tree)}\n\n"
        f"## Wave {report.wave} report\n\n"
        f"Suite: {suite}\n\nLanded:\n{landed}\n\nFailed (under the attempt cap):\n{failed}\n\n"
        f"CONFIRMED findings:\n{findings}\n\n"
        f"Already escalated to a human (do not touch):\n{escalated_lines}\n"
    )


def replan(
    framing: FramingProposal,
    tree: list[LeafSpec],
    report: WaveReport,
    attempts: dict[str, int],
) -> list[ReplanAction]:
    """A4: wave report → typed replan actions.

    Deterministic pre-rules run first and without any model: leaves at the attempt cap
    escalate to a human. The model is consulted only when there is judgment work left —
    confirmed findings to turn into fix-ups, under-cap failures to respec, or an
    integration-red suite. New/rewritten leaves get the same conservative tagging floor
    as decomposition, and a replan that wants more new leaves than the emission cap
    halts instead (a replan that big means the framing is wrong).
    """
    escalated = deterministic_escalations(report, attempts)
    escalated_titles = {e.leaf_title for e in escalated}

    confirmed = [f for f in report.findings if f.confirmed]
    failed_under_cap = [o for o in report.failed if o.leaf not in escalated_titles]
    integration_red = bool(report.landed) and not report.suite_green
    if not confirmed and not failed_under_cap and not integration_red:
        return list(escalated)

    result = structured(
        pool=_architect_pool(),
        schema=ReplanEnvelope,
        system=REPLAN_SYSTEM,
        user=_replan_user(framing, tree, report, attempts, escalated),
        max_tokens=settings.decompose_max_tokens,
        timeout=settings.architect_timeout,
    )
    if result.value is None:
        raise ArchitectError(f"replan produced no usable actions: {result.error}")

    actions: list[ReplanAction] = []
    new_leaves: list[LeafSpec] = []
    creations = 0
    for action in result.value.actions:
        if isinstance(action, RespecAction | EscalateAction):
            title = action.leaf_title
            if title in escalated_titles:
                continue  # already human-owned; the model was told not to touch these
        if isinstance(action, FixupAction | IntegrationFixAction):
            creations += 1
            new_leaves.append(action.leaf)
        if isinstance(action, RespecAction):
            new_leaves.append(action.revised)
        actions.append(action)

    _apply_conservative_tags(new_leaves, framing)

    from agents.shared.forge_emit import settings as emit_settings

    if creations > emit_settings.max_per_run:
        return [
            *escalated,
            HaltAction(
                reason=(
                    f"replan wants {creations} new leaves, over the emission cap "
                    f"({emit_settings.max_per_run}) — a replan that big means the framing "
                    f"needs human review"
                )
            ),
        ]
    return [*escalated, *actions]


def render_framing(p: FramingProposal) -> str:
    """Human-readable framing.md — what the human actually reads before approving."""
    approval = "yes" if p.approved else "NO — review and approve before decomposition"
    lines = [
        f"# Framing — {p.epic_slug}",
        f"\n**Approved:** {approval}",
        f"\n## Goal as stated\n\n{p.goal_as_stated}",
    ]
    if p.rescoped:
        lines.append(f"\n## ⚠ Architect push-back — goal re-scoped\n\n{p.restated_goal}")
    else:
        lines.append(f"\n## Restated goal\n\n{p.restated_goal}")
    if p.inventory_summary:
        lines.append(f"\n## Inventory summary\n\n{p.inventory_summary}")
    if p.gap_analysis:
        lines.append(f"\n## Gap analysis\n\n{p.gap_analysis}")
    if p.options:
        lines.append("\n## Options considered\n")
        for o in p.options:
            lines.append(f"### {o.name}\n\n{o.summary}")
            if o.tradeoffs:
                lines.append(f"\n*Tradeoffs:* {o.tradeoffs}")
    lines.append(f"\n## Recommendation\n\n{p.recommendation}")
    if p.value_ordering:
        lines.append("\n## Value ordering (ship first → last)\n")
        lines.extend(f"{i + 1}. {v}" for i, v in enumerate(p.value_ordering))
    if p.risks:
        lines.append("\n## Risks\n")
        lines.extend(f"- {r}" for r in p.risks)
    if p.branch:
        lines.append(f"\n## Suggested epic branch\n\n`{p.branch}`")
    return "\n".join(lines)
