"""A structured-output verification panel built on the ensemble harness.

Fan a JSON prompt across N diverse models concurrently and return each member's parsed response
plus a quorum flag; the caller aggregates the responses however it likes (e.g. median-score an
adversarial research-verification panel). This is the shared piece that makes the researchers
harness consumers — they reuse `fanout` / `ApiExecutor` / `Pool` instead of re-implementing
fan-out + parse + quorum. Synchronous (wraps the async harness in ``asyncio.run``) so the existing
synchronous research loops can call it directly.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field

from agents.shared.ensemble import ApiExecutor, Executor, Pool, Prompt
from agents.shared.ensemble.pool import fanout
from agents.shared.llm import extract_json


@dataclass
class PanelResult:
    responses: list[dict] = field(default_factory=list)  # parsed JSON, one per member that answered
    member_labels: list[str] = field(default_factory=list)  # labels aligned with `responses`
    attempted: int = 0
    quorum_met: bool = False


def build_router_executors(
    models: Sequence[str], *, base_url: str, api_key: str
) -> list[ApiExecutor]:
    """One OpenAI-compatible (router) executor per model name — the panel members."""
    return [
        ApiExecutor(label=f"router:{m}", kind="openai", model=m, base_url=base_url, api_key=api_key)
        for m in models
    ]


async def _run_panel(
    executors: Sequence[Executor], prompt: Prompt, *, floor: int, timeout: float
) -> PanelResult:
    pools = [Pool(role=f"panel:{e.label}", executors=[e]) for e in executors]
    fan = await fanout("panel", pools, prompt, timeout=timeout, quorum_floor=floor)
    responses: list[dict] = []
    labels: list[str] = []
    for r in fan.results:
        if not r.ok:
            continue
        data = extract_json(r.output)
        if data:  # transport-ok but unparseable JSON is dropped, like a failed member
            responses.append(data)
            labels.append(r.executor)
    return PanelResult(
        responses=responses,
        member_labels=labels,
        attempted=len(pools),
        quorum_met=len(responses) >= floor,
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
    """Fan a JSON prompt across ``executors`` concurrently; return parsed responses + quorum.

    Members that error or return unparseable JSON are dropped; ``quorum_met`` is whether at least
    ``floor`` members produced a usable response.
    """
    prompt = Prompt(system=system, user=user, max_tokens=max_tokens)
    return asyncio.run(_run_panel(executors, prompt, floor=floor, timeout=timeout))
