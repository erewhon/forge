"""A structured-output verification panel built on the ensemble harness.

Fan a JSON prompt across N diverse models concurrently and return each member's parsed response
plus a quorum flag; the caller aggregates the responses however it likes (e.g. median-score an
adversarial research-verification panel). This is the shared piece that makes the researchers
harness consumers — they reuse `Pool` / `ApiExecutor` instead of re-implementing fan-out + parse +
quorum. Synchronous (wraps the async harness in ``asyncio.run``) so the existing synchronous
research loops can call it directly.

Two shapes:
- **Uniform** (`run_panel`): every member gets the *same* system prompt — N independent graders,
  median-aggregated to kill single-model bias.
- **Perspective-diverse** (`run_member_panel` + `build_lens_members`): each member gets a *distinct
  lens* system prompt, so the union of their challenges covers orthogonal failure modes instead of
  redundantly flagging the same obvious gap. Diversity catches what redundancy can't.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field

from agents.shared.ensemble import ApiExecutor, ExecResult, Executor, Pool, Prompt
from agents.shared.llm import extract_json


@dataclass
class PanelResult:
    responses: list[dict] = field(default_factory=list)  # parsed JSON, one per member that answered
    member_labels: list[str] = field(default_factory=list)  # labels aligned with `responses`
    attempted: int = 0
    quorum_met: bool = False


@dataclass
class PanelMember:
    """One panel seat: an executor paired with the (possibly lens-specialised) system prompt it
    runs. ``label`` is for reporting only — it defaults to the executor's own label."""

    executor: Executor
    system: str
    label: str | None = None


def build_router_executors(
    models: Sequence[str], *, base_url: str, api_key: str
) -> list[ApiExecutor]:
    """One OpenAI-compatible (router) executor per model name — the panel members."""
    return [
        ApiExecutor(label=f"router:{m}", kind="openai", model=m, base_url=base_url, api_key=api_key)
        for m in models
    ]


def build_lens_members(
    lenses: Sequence[tuple[str, str]],
    models: Sequence[str],
    *,
    base_url: str,
    api_key: str,
    base_system: str,
) -> list[PanelMember]:
    """One panel member per lens — each gets ``base_system`` followed by its lens directive, with
    router models assigned round-robin so the panel is diverse in *both* viewpoint and vendor.

    With 5 lenses and 3 models the members cycle models[0], models[1], models[2], models[0],
    models[1] — every lens still scores all dimensions, but each hunts its own failure mode hardest.
    """
    if not models:
        return []
    members: list[PanelMember] = []
    for i, (name, directive) in enumerate(lenses):
        model = models[i % len(models)]
        executor = ApiExecutor(
            label=f"router:{model}", kind="openai", model=model, base_url=base_url, api_key=api_key
        )
        system = f"{base_system}\n\n{directive}".strip() if directive else base_system
        members.append(PanelMember(executor=executor, system=system, label=f"{model}/{name}"))
    return members


async def _run_member_panel(
    members: Sequence[PanelMember], user: str, *, floor: int, max_tokens: int, timeout: float
) -> PanelResult:
    async def _one(member: PanelMember) -> ExecResult:
        role = f"panel:{member.label or member.executor.label}"
        pool = Pool(role=role, executors=[member.executor])
        prompt = Prompt(system=member.system, user=user, max_tokens=max_tokens)
        return await pool.run(prompt, timeout=timeout)

    results = await asyncio.gather(*(_one(m) for m in members))
    responses: list[dict] = []
    labels: list[str] = []
    for member, result in zip(members, results):
        if not result.ok:
            continue
        data = extract_json(result.output)
        if data:  # transport-ok but unparseable JSON is dropped, like a failed member
            responses.append(data)
            labels.append(member.label or result.executor)
    return PanelResult(
        responses=responses,
        member_labels=labels,
        attempted=len(members),
        quorum_met=len(responses) >= floor,
    )


def run_member_panel(
    *,
    members: Sequence[PanelMember],
    user: str,
    floor: int,
    max_tokens: int = 4096,
    timeout: float = 120.0,
) -> PanelResult:
    """Fan a per-member (lens-specialised) prompt set across the panel; return parsed responses +
    quorum. Each member runs its own system prompt against the shared ``user`` message. Members
    that error or return unparseable JSON are dropped; ``quorum_met`` is whether at least ``floor``
    members produced a usable response.
    """
    return asyncio.run(
        _run_member_panel(members, user, floor=floor, max_tokens=max_tokens, timeout=timeout)
    )


def run_panel(
    *,
    executors: Sequence[Executor],
    system: str,
    user: str,
    floor: int,
    max_tokens: int = 4096,
    timeout: float = 120.0,
) -> PanelResult:
    """Uniform panel: every member runs the same ``system`` prompt. Thin wrapper over
    ``run_member_panel`` for the N-independent-graders case."""
    members = [PanelMember(executor=e, system=system) for e in executors]
    return run_member_panel(
        members=members, user=user, floor=floor, max_tokens=max_tokens, timeout=timeout
    )
