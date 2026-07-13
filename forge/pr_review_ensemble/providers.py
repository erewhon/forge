"""Reviewer roster: each seat is a failover ``Pool`` — a primary model plus backups pulled in only
when the primary is down. Every model resolves on the local LLM router (which holds all provider
creds server-side), so the whole roster is one endpoint + one key.

The roster is the single source of reviewers shared by the PR-review ensemble, the coding-pipeline
epic gate, wave-verify, the testing ensemble, and the Dependabot bumper — reconfigure it here and
every reviewer changes at once. Default: three diverse primary seats (Claude Sonnet, GLM, MiniMax
M3) with cheaper backups (a local coder model, Kimi). Diversity (distinct model families) over
count. A disabled seat becomes a ``SkipExecutor`` slot: attempted-but-never-ok for quorum
accounting, without a doomed network call.
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


def _glm_slot() -> ReviewerSlot:
    return _failover_slot("glm", _router_executor("glm", "glm"), "glm", ["kimi"])


def _m3_slot() -> ReviewerSlot:
    return _failover_slot("m3", _router_executor("m3", "m3"), "m3", ["kimi"])


def build_reviewer_slots() -> list[ReviewerSlot]:
    """The roster, in a stable order: three primary seats, each a router-backed failover chain."""
    return [_sonnet_slot(), _glm_slot(), _m3_slot()]


# Capability-ordered rotation for the aggregator/digest failover pool. All seats route through the
# router, so ordering is by review capability; a `preferred` seat is promoted to the front.
ROTATION_ORDER = ("sonnet-5", "glm", "m3")


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
