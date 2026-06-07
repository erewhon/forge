"""Ensemble harness — fan work across a resilient pool of models, then combine.

Core failover layer (Phase 1): Executor protocol + ApiExecutor, failure classification,
Pool failover, fanout-to-quorum, and Combiner with a deterministic fallback. Consumers
(pr_review_ensemble, parallel_edit) reduce to: build pools, fanout or run, then combine.
"""

from __future__ import annotations

from agents.shared.ensemble.classify import classify
from agents.shared.ensemble.combiner import AggregateCombiner, Combiner, CombineResult
from agents.shared.ensemble.executor import ApiExecutor, Executor
from agents.shared.ensemble.models import (
    ExecResult,
    ExecStatus,
    FailureClass,
    FanoutResult,
    Prompt,
    QuorumState,
)
from agents.shared.ensemble.pool import Pool, fanout

__all__ = [
    "AggregateCombiner",
    "ApiExecutor",
    "CombineResult",
    "Combiner",
    "ExecResult",
    "ExecStatus",
    "Executor",
    "FailureClass",
    "FanoutResult",
    "Pool",
    "Prompt",
    "QuorumState",
    "classify",
    "fanout",
]
