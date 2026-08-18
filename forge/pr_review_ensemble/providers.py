"""Reviewer roster: each seat is a failover ``Pool`` — a primary model plus backups pulled in only
when the primary is down. Every model resolves on the local LLM router (which holds all provider
creds server-side), so the whole roster is one endpoint + one key.

The roster is the single source of reviewers shared by the PR-review ensemble, the coding-pipeline
epic gate, wave-verify, the testing ensemble, and the Dependabot bumper — reconfigure it here and
every reviewer changes at once. Default: three diverse primary seats, two of them local — Claude
Sonnet (the frontier anchor), the self-hosted gpt-oss-120b on delphi (seated 2026-08-18, replacing
the metered GLM seat at ~equal review score and adding a model family the fleet lacked), and the
self-hosted Nemotron 3.5 Lightning on the talos B70 (seated 2026-08-16; qualeval v2 code_review
0.84, the fleet's strongest local reviewer by ~0.17, at ~100 t/s). Cheaper backups: GLM, a local
coder model, MiniMax M3, Kimi. Diversity (distinct model families) over count. A disabled seat
becomes a ``SkipExecutor`` slot: attempted-but-never-ok for quorum accounting, without a doomed
network call.

``build_local_reviewer_slots`` is the all-local shadow roster (Lightning/gpt-oss/Coder-Next —
three families, zero cloud seats), run by the ``shadow`` pass side by side with production to
decide whether sonnet can be unseated.
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


def _gptoss_slot() -> ReviewerSlot:
    """Local OpenAI-family seat: primary is gpt-oss-120b on delphi (router alias `gpt-oss`,
    ~49 t/s Vulkan — this seat's diffs stay in the homelab). qualeval v2 0.87 composite,
    code_review 0.64: FP-heavy on defect probes but passes the clean-code canary — it over-reports
    on buggy code, doesn't invent bugs in clean code, and the quorum labeling absorbs the noise.
    Seated 2026-08-18 replacing the metered GLM seat (GLM-family review ≤0.67) at ~equal review
    score and zero marginal cost. GLM stays reachable as the first backup, then kimi (both
    family-diverse cloud)."""
    primary = _router_executor("gpt-oss", "gpt-oss")
    return _failover_slot("gpt-oss", primary, "gpt-oss", ["glm", "kimi"])


def _lightning_slot() -> ReviewerSlot:
    """Local NVIDIA seat: primary is Nemotron 3.5 Lightning on the talos B70 (router alias
    `lightning`, ~100 t/s — this seat's diffs stay in the homelab). The serving side carries the
    load-bearing flags (reasoning-budget 3000, temp 0.6, MTP), so a bare alias gets the tuned
    0.91/0.84-review config. Zen-hosted MiniMax m3 then kimi as cloud backups (family-diverse)."""
    primary = _router_executor("lightning", "lightning")
    return _failover_slot("lightning", primary, "lightning", ["m3", "kimi"])


def _coder_next_slot() -> ReviewerSlot:
    """Local Qwen seat (shadow roster only): primary is Qwen3-Coder-Next on archimedes (router
    alias `coder-next`, vLLM FP8, ~46 t/s). qualeval v2 0.90 composite but code_review 0.67 and a
    terse non-thinker — a builder, not a reviewer; it sits in the shadow roster to give the
    all-local trio its third model family. Backup is the local `coder` role alias (qwen3.6 on
    hypatia), keeping the seat fully local on failover."""
    primary = _router_executor("coder-next", "coder-next")
    return _failover_slot("coder-next", primary, "coder-next", ["coder"])


def build_reviewer_slots() -> list[ReviewerSlot]:
    """The roster, in a stable order: three primary seats, each a router-backed failover chain."""
    return [_sonnet_slot(), _gptoss_slot(), _lightning_slot()]


def build_local_reviewer_slots() -> list[ReviewerSlot]:
    """The all-local roster: Lightning (NVIDIA), gpt-oss-120b (OpenAI family), Coder-Next
    (Qwen) — three distinct families with zero cloud seats, every diff staying in the homelab.
    Serves the ROUTINE lane (see ``roster_for_lane``) and the ``shadow`` pass; the frontier
    roster keeps the gates where sonnet's unique catches clustered in the shadow trial."""
    return [_lightning_slot(), _gptoss_slot(), _coder_next_slot()]


def roster_for_lane(lane: str) -> list[ReviewerSlot]:
    """Lane split (2026-08-18, from the 7-run shadow trial): ``"local"`` — the all-local trio,
    for routine work (dependency bumps, config moves, mechanical passes) where it reviewed at
    parity and costs nothing; anything else — the frontier roster, for the merge-blocking gates
    and security-adjacent diffs where sonnet's unique catches clustered. Unknown values fall
    through to frontier on purpose: the expensive-but-safer roster is the fail-closed default."""
    return build_local_reviewer_slots() if lane == "local" else build_reviewer_slots()


# Capability-ordered rotation for the aggregator/digest failover pool. All seats route through the
# router, so ordering is by review capability; a `preferred` seat is promoted to the front.
# Lightning first among locals (review 0.84); gpt-oss ahead of coder-next for synthesis duty
# (instruction 1.00 and a disciplined reasoner vs a terse non-thinker) despite near-equal review
# scores (0.64 vs 0.67). Local seats outrank any metered cloud fallback.
ROTATION_ORDER = ("sonnet-5", "lightning", "gpt-oss", "coder-next")


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
