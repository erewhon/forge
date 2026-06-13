from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from agents.pr_review_ensemble.aggregator import aggregate_reviews
from agents.pr_review_ensemble.config import settings
from agents.pr_review_ensemble.models import EnsembleResult, ProviderName, ProviderReview
from agents.pr_review_ensemble.prompts import REVIEW_SYSTEM_PROMPT
from agents.pr_review_ensemble.providers import all_providers, call_provider


async def run_ensemble(*, diff_text: str, pr_ref: str) -> EnsembleResult:
    diff_lines = diff_text.count("\n") + 1
    user_message = (
        f"Pull request: {pr_ref}\n"
        f"Diff size: {diff_lines} lines\n\n"
        f"Diff:\n{diff_text}"
    )
    timestamp = datetime.now(UTC)

    providers_attempted: list[ProviderName] = all_providers()
    coros = [
        call_provider(p, system_prompt=REVIEW_SYSTEM_PROMPT, user_message=user_message)
        for p in providers_attempted
    ]
    reviews: list[ProviderReview] = await asyncio.gather(*coros)

    successful = [r for r in reviews if r.status == "ok"]
    providers_succeeded: list[ProviderName] = [r.provider for r in successful]

    if len(successful) < settings.quorum_floor:
        return EnsembleResult(
            pr_ref=pr_ref,
            timestamp=timestamp,
            diff_lines=diff_lines,
            reviews=reviews,
            aggregated_review=None,
            aggregator_provider=None,
            quorum_state="failed",
            quorum_floor=settings.quorum_floor,
            providers_attempted=providers_attempted,
            providers_succeeded=providers_succeeded,
        )

    quorum_state = "full" if len(successful) == len(providers_attempted) else "degraded"
    aggregated_review, aggregator_provider = await aggregate_reviews(
        successful_reviews=successful, pr_ref=pr_ref
    )

    return EnsembleResult(
        pr_ref=pr_ref,
        timestamp=timestamp,
        diff_lines=diff_lines,
        reviews=reviews,
        aggregated_review=aggregated_review,
        aggregator_provider=aggregator_provider,
        quorum_state=quorum_state,
        quorum_floor=settings.quorum_floor,
        providers_attempted=providers_attempted,
        providers_succeeded=providers_succeeded,
    )
