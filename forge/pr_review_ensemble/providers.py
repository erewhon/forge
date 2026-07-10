"""Reviewer slots: each provider reduced to a shared-harness Pool of one ApiExecutor.

The MVP hand-rolled three async provider calls plus timeout/error capture; that is exactly what
``forge.shared.ensemble`` now provides. Each provider becomes a single-executor ``Pool`` (so a
slot can gain backup executors later without touching the runner), and the runner fans out across
them. A disabled or unconfigured provider becomes a ``SkipExecutor`` slot: it counts as an
attempted-but-never-ok member (preserving the MVP's quorum accounting) without making a doomed
network call.
"""

from __future__ import annotations

from dataclasses import dataclass

from forge.pr_review_ensemble.config import settings
from forge.pr_review_ensemble.models import ProviderName
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
    """One provider in the ensemble: its identity plus the failover Pool that runs it."""

    provider: ProviderName
    model: str
    pool: Pool
    skipped_reason: str | None = None  # set when the slot is a no-op skip (disabled / no creds)

    @property
    def active(self) -> bool:
        return self.skipped_reason is None


class SkipExecutor:
    """An Executor that never calls out — represents a disabled/unconfigured provider.

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


def _slot(
    provider: ProviderName, model: str, executor: Executor, reason: str | None
) -> ReviewerSlot:
    return ReviewerSlot(
        provider=provider,
        model=model,
        pool=Pool(role=f"review:{provider}", executors=[executor]),
        skipped_reason=reason,
    )


def _anthropic_slot() -> ReviewerSlot:
    model = settings.anthropic_model
    label = f"anthropic:{model}"
    if not settings.anthropic_enabled:
        return _slot(
            "anthropic", model, SkipExecutor(label, "disabled in config"), "disabled in config"
        )
    if settings.anthropic_base_url:
        # Real Claude, routed through the local LiteLLM proxy (OpenAI-compat): centralized creds,
        # no per-shell ANTHROPIC_API_KEY. Empty base_url falls back to the native Anthropic SDK.
        executor = ApiExecutor(
            label=label,
            kind="openai",
            model=model,
            base_url=settings.anthropic_base_url,
            api_key=settings.anthropic_api_key,
        )
    else:
        executor = ApiExecutor(label=label, kind="anthropic", model=model)
    return _slot("anthropic", model, executor, None)


def _local_slot() -> ReviewerSlot:
    model = settings.local_model
    label = f"local:{model}"
    if not settings.local_enabled:
        return _slot(
            "local", model, SkipExecutor(label, "disabled in config"), "disabled in config"
        )
    executor = ApiExecutor(
        label=label,
        kind="openai",
        model=model,
        base_url=settings.local_base_url,
        api_key=settings.local_api_key,
    )
    return _slot("local", model, executor, None)


def _opencode_zen_slot() -> ReviewerSlot:
    model = settings.opencode_zen_model
    label = f"opencode_zen:{model}"
    if not settings.opencode_zen_enabled:
        return _slot(
            "opencode_zen", model, SkipExecutor(label, "disabled in config"), "disabled in config"
        )
    if not settings.opencode_zen_api_key:
        reason = "no api key configured"
        return _slot("opencode_zen", model, SkipExecutor(label, reason), reason)
    executor = ApiExecutor(
        label=label,
        kind="openai",
        model=model,
        base_url=settings.opencode_zen_base_url,
        api_key=settings.opencode_zen_api_key,
    )
    return _slot("opencode_zen", model, executor, None)


def build_reviewer_slots() -> list[ReviewerSlot]:
    """The ensemble roster, in a stable order. Diversity (distinct model families) > count."""
    return [_anthropic_slot(), _local_slot(), _opencode_zen_slot()]


# Capability-ordered rotation; "local" (the local LLM router) is the structural break-glass and
# always sits last. A `preferred` provider is promoted to the front when it is active.
ROTATION_ORDER = ("anthropic", "opencode_zen", "local")


def rotation_pool(slots: list[ReviewerSlot], *, role: str, preferred: str | None = None) -> Pool:
    """A failover Pool over the *active* providers in capability-rotation order.

    Shared by the aggregator (synthesize N reviews) and the digest (one resilient pass). Inactive
    (skipped) providers are excluded; `preferred` (if active) leads, then ROTATION_ORDER fills in.
    Each active slot's executor is reused (ApiExecutor is stateless), so the pool rotates over the
    same models the ensemble uses.
    """
    active = {s.provider: s for s in slots if s.active}
    order = [preferred] if preferred in active else []
    order += [p for p in ROTATION_ORDER if p in active and p not in order]
    return Pool(role=role, executors=[active[p].pool.executors[0] for p in order])
