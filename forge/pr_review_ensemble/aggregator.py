from __future__ import annotations

from typing import cast

from agents.pr_review_ensemble.config import settings
from agents.pr_review_ensemble.models import ProviderName, ProviderReview
from agents.pr_review_ensemble.prompts import AGGREGATOR_SYSTEM_PROMPT
from agents.pr_review_ensemble.providers import all_providers, call_provider


def _format_for_aggregator(r: ProviderReview) -> str:
    return f"### Reviewer: {r.provider} ({r.model})\n\n{r.response_text}"


async def aggregate_reviews(
    *, successful_reviews: list[ProviderReview], pr_ref: str
) -> tuple[str, ProviderName]:
    """Synthesize N successful reviews into one advisory. Returns (text, aggregator)."""
    aggregator: ProviderName
    if settings.aggregator_provider in all_providers():
        aggregator = cast(ProviderName, settings.aggregator_provider)
    else:
        aggregator = successful_reviews[0].provider

    user_message = (
        f"Pull request: {pr_ref}\n"
        f"Number of independent reviews: {len(successful_reviews)}\n\n"
        + "\n\n".join(_format_for_aggregator(r) for r in successful_reviews)
        + "\n\nProduce the synthesized advisory."
    )

    result = await call_provider(
        aggregator, system_prompt=AGGREGATOR_SYSTEM_PROMPT, user_message=user_message
    )
    if result.status != "ok":
        fallback = (
            f"_(Aggregator [{aggregator}] failed: {result.error_message}. "
            "Showing concatenated raw reviews below.)_\n\n"
            + "\n\n".join(_format_for_aggregator(r) for r in successful_reviews)
        )
        return fallback, aggregator

    return result.response_text, aggregator
