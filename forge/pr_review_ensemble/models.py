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
