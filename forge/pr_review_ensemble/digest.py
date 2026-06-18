"""Large-PR digest pass: one resilient navigational digest of a big feature PR.

Unlike the review pass (fan out N reviewers, then synthesize), the digest wants a single coherent
reading guide — so it runs ONE prompt through a failover ``Pool`` (rotation: preferred → anthropic
→ opencode_zen → local break-glass) and takes whichever model answers. This MVP is single-shot:
if the diff is larger than ``digest_max_diff_chars`` it fails loudly rather than silently
truncating; the chunked map-reduce path is the planned follow-on (then this becomes the "fits in
context" branch of a size-guarded hybrid).
"""

from __future__ import annotations

from datetime import UTC, datetime

from agents.pr_review_ensemble.config import settings
from agents.pr_review_ensemble.models import DigestResult
from agents.pr_review_ensemble.prompts import DIGEST_SYSTEM_PROMPT
from agents.pr_review_ensemble.providers import ReviewerSlot, build_reviewer_slots, rotation_pool
from agents.shared.ensemble import Pool, Prompt


def build_digest_pool(slots: list[ReviewerSlot]) -> Pool:
    """A failover pool over active providers, strongest-first (same rotation as the aggregator)."""
    return rotation_pool(slots, role="digest", preferred=settings.aggregator_provider)


async def run_digest(
    *,
    diff_text: str,
    pr_ref: str,
    slots: list[ReviewerSlot] | None = None,
    pool: Pool | None = None,
) -> DigestResult:
    """Produce a navigational digest of a (large) PR diff. ``slots``/``pool`` are injectable."""
    diff_lines = diff_text.count("\n") + 1
    diff_chars = len(diff_text)
    timestamp = datetime.now(UTC)

    base = DigestResult(
        pr_ref=pr_ref, timestamp=timestamp, diff_lines=diff_lines, diff_chars=diff_chars
    )

    # Fail loud rather than truncate: a diff over budget is the case the chunked path will own.
    if diff_chars > settings.digest_max_diff_chars:
        return base.model_copy(
            update={
                "oversize": True,
                "error": (
                    f"diff is {diff_chars} chars, over the single-pass budget of "
                    f"{settings.digest_max_diff_chars}. Chunked (map-reduce) digest pending; "
                    "raise PR_REVIEW_ENSEMBLE_DIGEST_MAX_DIFF_CHARS to force a single-shot attempt."
                ),
            }
        )

    if pool is None:
        pool = build_digest_pool(slots if slots is not None else build_reviewer_slots())
    if not pool.executors:
        return base.model_copy(update={"error": "no active providers for the digest pool"})

    user = f"Pull request: {pr_ref}\nDiff size: {diff_lines} lines\n\nDiff:\n{diff_text}"
    prompt = Prompt(system=DIGEST_SYSTEM_PROMPT, user=user, max_tokens=settings.digest_max_tokens)
    result = await pool.run(prompt, timeout=settings.per_provider_timeout_seconds)

    if not result.ok:
        return base.model_copy(update={"model": result.executor, "error": result.error})
    return base.model_copy(update={"digest": result.output, "model": result.executor})
