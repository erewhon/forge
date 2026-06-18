from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel

ProviderName = Literal["anthropic", "local", "opencode_zen"]
ProviderStatus = Literal["ok", "timeout", "error", "skipped"]
QuorumState = Literal["full", "degraded", "failed"]


class ProviderReview(BaseModel):
    provider: ProviderName
    model: str
    status: ProviderStatus
    response_text: str = ""
    latency_ms: int | None = None
    error_message: str | None = None


class DigestResult(BaseModel):
    """One single-pass digest run. ``digest`` is None when the pass failed or the diff was too
    large for the single-shot MVP (``oversize=True`` — chunked path pending)."""

    pr_ref: str
    timestamp: datetime
    diff_lines: int
    diff_chars: int
    digest: str | None = None
    model: str | None = None  # executor label that produced the digest (after any failover)
    oversize: bool = False  # diff exceeded digest_max_diff_chars; not attempted single-shot
    error: str | None = None


class EnsembleResult(BaseModel):
    pr_ref: str
    timestamp: datetime
    diff_lines: int
    reviews: list[ProviderReview]
    aggregated_review: str | None
    # The executor label that actually synthesized (e.g. "anthropic:claude-sonnet-4-6") or
    # "fallback:concat" — a label, not just a provider name, since the aggregator rotates.
    aggregator_provider: str | None
    aggregator_used_fallback: bool = False
    quorum_state: QuorumState
    quorum_floor: int
    providers_attempted: list[ProviderName]
    providers_succeeded: list[ProviderName]
