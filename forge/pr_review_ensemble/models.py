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


DigestStrategy = Literal["single", "map_reduce"]


class DigestResult(BaseModel):
    """One digest run. ``digest`` is None only when the pass could not produce anything usable."""

    pr_ref: str
    timestamp: datetime
    diff_lines: int
    diff_chars: int
    digest: str | None = None
    model: str | None = None  # executor label that produced the digest (or "fallback:concat")
    strategy: DigestStrategy = "single"  # "single" (fit in context) or "map_reduce" (chunked)
    chunks: int = 0  # number of map chunks (0 for single-pass)
    chunks_dropped: int = 0  # chunks beyond digest_max_chunks that were not summarized
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
