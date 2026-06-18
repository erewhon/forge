"""Build the aggregator: a failover Pool of synthesizers wrapped in AggregateCombiner.

The MVP picked one aggregator provider and concatenated on failure. The shared harness's
``AggregateCombiner`` runs the synthesizer through a failover ``Pool`` — so we get the rotation
the MVP deferred (preferred provider, then a capability order ending in the always-reachable
local model as break-glass) plus deterministic concatenation when every synthesizer is down.
"""

from __future__ import annotations

from agents.pr_review_ensemble.config import settings
from agents.pr_review_ensemble.prompts import AGGREGATOR_SYSTEM_PROMPT
from agents.pr_review_ensemble.providers import ReviewerSlot, rotation_pool
from agents.shared.ensemble import AggregateCombiner


def build_aggregator(
    slots: list[ReviewerSlot], *, pr_ref: str, n_reviews: int
) -> AggregateCombiner:
    """Assemble the aggregator over the *active* providers, in rotation order.

    Reuses each active slot's executor (ApiExecutor is stateless), so the aggregator rotates over
    the same models that reviewed. Inactive (skipped) providers are excluded from the rotation.
    """
    pool = rotation_pool(slots, role="aggregator", preferred=settings.aggregator_provider)

    header = (
        f"Pull request: {pr_ref}\n"
        f"Number of independent reviews: {n_reviews}\n\n"
        "Synthesize these independent reviews into one advisory:"
    )
    return AggregateCombiner(
        pool=pool,
        system=AGGREGATOR_SYSTEM_PROMPT,
        header=header,
        timeout=settings.per_provider_timeout_seconds,
        max_tokens=settings.aggregator_max_tokens,
    )
