"""Orchestrate the ensemble: fan a diff across reviewer pools, then synthesize.

Consumer #2 of the shared ensemble harness (parallel_edit's judge was #1). The runner reduces to:
build pools → ``fanout`` (resilient, quorum-aware) → ``AggregateCombiner`` → map to EnsembleResult.
The per-provider timeout/error capture, TRANSIENT-retry/TERMINAL-failover, quorum accounting, and
aggregator rotation + concat fallback all live in the harness now.
"""

from __future__ import annotations

from datetime import UTC, datetime

from forge.pr_review_ensemble.aggregator import build_aggregator
from forge.pr_review_ensemble.config import settings
from forge.pr_review_ensemble.models import (
    EnsembleResult,
    ProviderReview,
    ProviderStatus,
    QuorumState,
)
from forge.pr_review_ensemble.prompts import AGGREGATOR_SYSTEM_PROMPT, REVIEW_SYSTEM_PROMPT
from forge.pr_review_ensemble.providers import ReviewerSlot, build_reviewer_slots
from forge.shared.ensemble import Combiner, ExecResult, ExecStatus, Prompt
from forge.shared.ensemble import QuorumState as HarnessQuorumState
from forge.shared.ensemble.pool import fanout

_STATUS_MAP: dict[ExecStatus, ProviderStatus] = {
    ExecStatus.OK: "ok",
    ExecStatus.TIMEOUT: "timeout",
    ExecStatus.ERROR: "error",
    ExecStatus.SKIPPED: "skipped",
}
_QUORUM_MAP: dict[HarnessQuorumState, QuorumState] = {
    HarnessQuorumState.FULL: "full",
    HarnessQuorumState.DEGRADED: "degraded",
    HarnessQuorumState.FAILED: "failed",
}


def _to_provider_review(slot: ReviewerSlot, res: ExecResult) -> ProviderReview:
    return ProviderReview(
        provider=slot.provider,
        model=slot.model,
        status=_STATUS_MAP[res.status],
        response_text=res.output,
        latency_ms=res.latency_ms,
        error_message=res.error,
    )


async def run_ensemble(
    *,
    diff_text: str,
    pr_ref: str,
    slots: list[ReviewerSlot] | None = None,
    aggregator: Combiner | None = None,
    system_prompt: str = REVIEW_SYSTEM_PROMPT,
    aggregator_system: str = AGGREGATOR_SYSTEM_PROMPT,
    aggregator_noun: str = "reviews",
    user_preamble: str = "",
) -> EnsembleResult:
    """Run every reviewer concurrently, then synthesize if quorum is met.

    ``system_prompt`` / ``aggregator_system`` / ``user_preamble`` let other passes reuse the
    fan-out machinery with their own lens (e.g. the supply-chain audit). ``slots`` / ``aggregator``
    are injectable for testing; production builds them from config.
    """
    if slots is None:
        slots = build_reviewer_slots()

    diff_lines = diff_text.count("\n") + 1
    timestamp = datetime.now(UTC)
    user_message = (
        f"{user_preamble}Pull request: {pr_ref}\n"
        f"Diff size: {diff_lines} lines\n\nDiff:\n{diff_text}"
    )
    prompt = Prompt(system=system_prompt, user=user_message, max_tokens=settings.review_max_tokens)

    fan = await fanout(
        "review",
        [s.pool for s in slots],
        prompt,
        timeout=settings.per_provider_timeout_seconds,
        quorum_floor=settings.quorum_floor,
    )
    # fanout preserves pool order, so results line up with slots positionally.
    reviews = [_to_provider_review(s, r) for s, r in zip(slots, fan.results, strict=True)]
    succeeded = [r for r in reviews if r.status == "ok"]

    aggregated_review: str | None = None
    aggregator_provider: str | None = None
    aggregator_used_fallback = False
    if fan.quorum_state != HarnessQuorumState.FAILED:
        combiner = aggregator or build_aggregator(
            slots,
            pr_ref=pr_ref,
            n_reviews=len(fan.succeeded),
            system=aggregator_system,
            noun=aggregator_noun,
        )
        combined = await combiner.combine(fan.succeeded)
        aggregated_review = combined.text
        aggregator_provider = combined.combiner
        aggregator_used_fallback = combined.used_fallback

    return EnsembleResult(
        pr_ref=pr_ref,
        timestamp=timestamp,
        diff_lines=diff_lines,
        reviews=reviews,
        aggregated_review=aggregated_review,
        aggregator_provider=aggregator_provider,
        aggregator_used_fallback=aggregator_used_fallback,
        quorum_state=_QUORUM_MAP[fan.quorum_state],
        quorum_floor=fan.quorum_floor,
        providers_attempted=[s.provider for s in slots],
        providers_succeeded=[r.provider for r in succeeded],
    )
