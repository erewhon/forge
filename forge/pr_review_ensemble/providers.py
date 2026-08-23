"""Reviewer roster: each seat is a failover ``Pool`` — a primary model plus backups pulled in only
when the primary is down. Every model resolves on the local LLM router (which holds all provider
creds server-side), so the whole roster is one endpoint + one key.

The roster is the single source of reviewers shared by the PR-review ensemble, the coding-pipeline
epic gate, wave-verify, the testing ensemble, and the Dependabot bumper — reconfigure it here and
every reviewer changes at once. Default: three diverse primary seats, two of them local — Claude
Sonnet (the frontier anchor), Ling 3 flash on hekaton (seated 2026-08-22; qualeval v2 code_review
0.91, the best local reviewer ever measured), and Gemma 4 26B on the talos B70 card 1 (seated
2026-08-22; board #1 composite 0.93, code_review 0.84, the analyst seat: adversarial 0.90 /
research 0.92, and the fleet's only Google-family model). Lightning (NVIDIA, code_review 0.84,
~100 t/s) is the fast local backup on both local seats and holds a primary seat in the all-local
roster. gpt-oss-120b (code_review 0.64, FP-heavy) lost its review seat on the 2026-08-22 board
but stays reachable as a backup; cheaper cloud backups: GLM, MiniMax M3, Kimi. Diversity
(distinct model families) over count. A disabled seat becomes a ``SkipExecutor`` slot:
attempted-but-never-ok for quorum accounting, without a doomed network call.

``build_local_reviewer_slots`` is the all-local roster (Ling 3 / Gemma 4 / Lightning — three
families, Ant/Google/NVIDIA, zero cloud seats), serving the ROUTINE lane and the ``shadow``
pass run side by side with production to decide whether sonnet can be unseated.

Serving contract for the Gemma seat: its 16k completion budget cannot be enforced server-side,
so reviewers must send generous max_tokens — ``review_max_tokens`` (16384) covers it; do not
lower it below 8192 or the v1 truncate-mid-flight trait returns. Ling 3 decodes on hekaton CPU
(~7 t/s with DSpark): a big-diff review can near the 300s per-provider timeout, at which point
the seat fails over to Lightning by design.
"""

from __future__ import annotations

from dataclasses import dataclass

from forge.pr_review_ensemble.config import settings
from forge.shared.ensemble import (
    ApiExecutor,
    ExecResult,
    ExecStatus,
    Executor,
    FailureClass,
    Pool,
    Prompt,
)


@dataclass
class ReviewerSlot:
    """One seat in the ensemble: its identity plus the failover Pool that runs it (primary first,
    then backups). ``model`` is the primary model's alias, for display/accounting."""

    provider: str
    model: str
    pool: Pool
    skipped_reason: str | None = None  # set when the slot is a no-op skip (disabled / no creds)

    @property
    def active(self) -> bool:
        return self.skipped_reason is None


class SkipExecutor:
    """An Executor that never calls out — represents a disabled/unconfigured seat.

    Returns a SKIPPED / TERMINAL ExecResult so ``fanout`` counts the slot as attempted but not
    successful (matching the MVP, where a skipped provider still occupied a quorum slot) without
    spending a request that would only 401/“disabled”.
    """

    def __init__(self, label: str, reason: str) -> None:
        self.label = label
        self._reason = reason

    async def run(self, prompt: Prompt, *, timeout: float) -> ExecResult:
        return ExecResult(
            executor=self.label,
            status=ExecStatus.SKIPPED,
            error=self._reason,
            failure_class=FailureClass.TERMINAL,
        )


def _router_executor(label: str, model: str) -> ApiExecutor:
    """An OpenAI-compat executor pointed at the local LLM router (holds every provider's creds
    server-side, so a bare alias like ``glm``/``m3``/``kimi``/``coder`` just works)."""
    return ApiExecutor(
        label=label,
        kind="openai",
        model=model,
        base_url=settings.local_base_url,
        api_key=settings.local_api_key,
    )


def _anthropic_primary() -> ApiExecutor:
    """The premium Claude seat's primary executor: routed through the LiteLLM proxy by default (no
    per-shell ANTHROPIC_API_KEY), or the native SDK when ``anthropic_base_url`` is cleared."""
    model = settings.anthropic_model
    label = f"sonnet:{model}"
    if settings.anthropic_base_url:
        return ApiExecutor(
            label=label,
            kind="openai",
            model=model,
            base_url=settings.anthropic_base_url,
            api_key=settings.anthropic_api_key,
        )
    return ApiExecutor(label=label, kind="anthropic", model=model)


def _failover_slot(
    provider: str, primary: Executor, model: str, backups: list[str]
) -> ReviewerSlot:
    """A seat whose Pool tries ``primary`` first, then each backup alias (via the router)."""
    executors: list[Executor] = [primary]
    executors += [_router_executor(f"{provider}:backup:{m}", m) for m in backups]
    return ReviewerSlot(provider, model, Pool(role=f"review:{provider}", executors=executors))


def _sonnet_slot() -> ReviewerSlot:
    """Premium Claude seat: sonnet primary, local ``coder`` break-glass backup. Honors the
    ``anthropic_enabled`` toggle — flip it off to drop the seat during an Anthropic outage."""
    if not settings.anthropic_enabled:
        pool = Pool(
            role="review:sonnet-5", executors=[SkipExecutor("sonnet-5", "disabled in config")]
        )
        return ReviewerSlot(
            "sonnet-5", settings.anthropic_model, pool, skipped_reason="disabled in config"
        )
    return _failover_slot("sonnet-5", _anthropic_primary(), settings.anthropic_model, ["coder"])


def _ling3_slot() -> ReviewerSlot:
    """Local Ant/Bailing seat: primary is Ling-3.0-flash on hekaton (router alias `ling`,
    llama-server-ling3 :5393, DSpark drafter — this seat's diffs stay in the homelab). qualeval
    v2 0.92 composite, code_review 0.91 — the best local reviewer ever measured (seated
    2026-08-22, displacing gpt-oss from the review seat). Its known flaw (tu-07: fabricates
    instead of surfacing tool errors) never fires in a review seat — reviews use no tools.
    Slow CPU decode: Lightning (local, fast, review 0.84) is the first backup, then kimi."""
    primary = _router_executor("ling3", "ling")
    return _failover_slot("ling3", primary, "ling", ["lightning", "kimi"])


def _gemma_slot() -> ReviewerSlot:
    """Local Google seat: primary is Gemma 4 26B on the talos B70 card 1 (router alias `gemma`,
    gemma-server :5392, ~62 t/s Vulkan — diffs stay in the homelab). qualeval v2 board #1
    (0.93): code_review 0.84 (ties Lightning), adversarial 0.90 / research 0.92 — the panel's
    analyst. Sampling params are held server-side; the 16k completion budget is NOT — callers
    must keep max_tokens generous (review_max_tokens 16384 does). gpt-oss (local, the demoted
    executor) is the first backup, then glm (family-diverse cloud)."""
    primary = _router_executor("gemma", "gemma")
    return _failover_slot("gemma", primary, "gemma", ["gpt-oss", "glm"])


def _lightning_slot() -> ReviewerSlot:
    """Local NVIDIA seat: primary is Nemotron 3.5 Lightning on the talos B70 (router alias
    `lightning`, ~100 t/s — this seat's diffs stay in the homelab). The serving side carries the
    load-bearing flags (reasoning-budget 3000, temp 0.6, MTP), so a bare alias gets the tuned
    0.91/0.84-review config. Zen-hosted MiniMax m3 then kimi as cloud backups (family-diverse)."""
    primary = _router_executor("lightning", "lightning")
    return _failover_slot("lightning", primary, "lightning", ["m3", "kimi"])


def build_reviewer_slots() -> list[ReviewerSlot]:
    """The frontier roster, in a stable order: sonnet (anchor) plus the two strongest local
    reviewers on the 2026-08-22 board — Ling 3 (review 0.91) and Gemma 4 (review 0.84 + the
    analyst profile). Each seat is a router-backed failover chain; Lightning covers the ling3
    seat's failover so a hekaton outage degrades to a fast local reviewer, not a cloud seat."""
    return [_sonnet_slot(), _ling3_slot(), _gemma_slot()]


def build_local_reviewer_slots() -> list[ReviewerSlot]:
    """The all-local roster: Ling 3 (Ant), Gemma 4 (Google), Lightning (NVIDIA) — three
    distinct families with zero cloud seats, every diff staying in the homelab, and three
    complementary failure modes (slow-but-thorough, few-FP analyst, fast-and-disciplined).
    Serves the ROUTINE lane (see ``roster_for_lane``) and the ``shadow`` pass; the frontier
    roster keeps the gates where sonnet's unique catches clustered in the shadow trial.
    Rewired 2026-08-22 from Lightning/gpt-oss/coder-next on the qualeval v2 board: gpt-oss
    review 0.64 and coder-next 0.67 were the weakest links; both stay reachable as backups."""
    return [_ling3_slot(), _gemma_slot(), _lightning_slot()]


def roster_for_lane(lane: str) -> list[ReviewerSlot]:
    """Lane split (2026-08-18, from the 7-run shadow trial): ``"local"`` — the all-local trio,
    for routine work (dependency bumps, config moves, mechanical passes) where it reviewed at
    parity and costs nothing; anything else — the frontier roster, for the merge-blocking gates
    and security-adjacent diffs where sonnet's unique catches clustered. Unknown values fall
    through to frontier on purpose: the expensive-but-safer roster is the fail-closed default."""
    return build_local_reviewer_slots() if lane == "local" else build_reviewer_slots()


# Capability-ordered rotation for the aggregator/digest failover pool. All seats route through the
# router, so ordering is by synthesis capability; a `preferred` seat is promoted to the front.
# After sonnet: Lightning first among locals for synthesis duty (instruction 0.99, disciplined,
# ~100 t/s), then Ling 3 (instruction 1.00, best reviewer, but slow hekaton CPU decode), then
# Gemma (instruction 1.00 but the most verbose thinker — fine work, slow synthesis). Only
# providers actually present in the given slots are used, so this order spans both rosters.
ROTATION_ORDER = ("sonnet-5", "lightning", "ling3", "gemma")


def rotation_pool(slots: list[ReviewerSlot], *, role: str, preferred: str | None = None) -> Pool:
    """A failover Pool over the *active* seats in capability-rotation order.

    Shared by the aggregator (synthesize N reviews) and the digest (one resilient pass). Inactive
    (skipped) seats are excluded; `preferred` (if active) leads, then ROTATION_ORDER fills in. Each
    seat's primary executor is reused (ApiExecutor is stateless), so the pool rotates over the same
    models the ensemble reviewed with.
    """
    active = {s.provider: s for s in slots if s.active}
    order = [preferred] if preferred in active else []
    order += [p for p in ROTATION_ORDER if p in active and p not in order]
    return Pool(role=role, executors=[active[p].pool.executors[0] for p in order])
