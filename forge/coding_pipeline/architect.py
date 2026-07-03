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

from agents.coding_pipeline.config import settings
from agents.coding_pipeline.inventory import render_inventory
from agents.coding_pipeline.models import FramingProposal, GoalSpec, Inventory
from agents.shared.ensemble import ApiExecutor, Pool
from agents.shared.panel import structured


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
