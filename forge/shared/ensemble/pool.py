"""Pools and fan-out — the resilience layer.

A Pool is an ordered set of interchangeable executors for one role. `Pool.run` is failover:
try the preferred executor, retry briefly on TRANSIENT failures, skip to the next on TERMINAL,
and keep going until one succeeds. `fanout` runs several pools concurrently and reports a quorum
state so callers can degrade gracefully instead of aborting when some members fail.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field

from forge.shared.ensemble.executor import Executor
from forge.shared.ensemble.models import (
    ExecResult,
    ExecStatus,
    FailureClass,
    FanoutResult,
    Prompt,
    QuorumState,
)


@dataclass
class Pool:
    """A role backed by an ordered list of executors tried with failover."""

    role: str
    executors: list[Executor] = field(default_factory=list)
    max_attempts_per_executor: int = 2  # bounded retries on TRANSIENT before failing over
    retry_backoff_s: float = 0.5

    async def run(
        self,
        prompt: Prompt,
        *,
        timeout: float,
        validate: Callable[[str], bool] | None = None,
    ) -> ExecResult:
        """Try executors with failover; return the first result that succeeds (and validates).

        ``validate`` guards against the failure mode endpoint failover can't see: a call that
        succeeds at the transport layer but returns an unusable payload (e.g. a judge that emits
        malformed JSON). Output that fails validation is demoted to a TRANSIENT failure, so the
        same model is retried and then failed over — exactly like a 5xx — until output is usable.
        """
        last: ExecResult | None = None
        total_attempts = 0

        for executor in self.executors:
            for attempt in range(self.max_attempts_per_executor):
                total_attempts += 1
                result = await executor.run(prompt, timeout=timeout)
                result.attempts = total_attempts
                if result.ok:
                    if validate is None or validate(result.output):
                        return result
                    # Transport OK but payload unusable — treat as transient and keep trying.
                    result.status = ExecStatus.ERROR
                    result.failure_class = FailureClass.TRANSIENT
                    result.error = "output failed validation"
                last = result
                if result.failure_class == FailureClass.TERMINAL:
                    break  # never coming back — move to the next executor
                if attempt + 1 < self.max_attempts_per_executor and self.retry_backoff_s > 0:
                    await asyncio.sleep(self.retry_backoff_s)

        if last is not None:
            return last
        return ExecResult(
            executor=f"pool:{self.role}",
            status=ExecStatus.SKIPPED,
            error="pool has no executors",
            failure_class=FailureClass.TERMINAL,
            attempts=total_attempts,
        )


async def fanout(
    role: str,
    pools: list[Pool],
    prompt: Prompt,
    *,
    timeout: float,
    quorum_floor: int,
) -> FanoutResult:
    """Run each pool's failover concurrently; report a quorum state over the successes."""
    results = await asyncio.gather(*(pool.run(prompt, timeout=timeout) for pool in pools))
    n_ok = sum(1 for r in results if r.ok)

    if n_ok < quorum_floor:
        state = QuorumState.FAILED
    elif n_ok == len(pools):
        state = QuorumState.FULL
    else:
        state = QuorumState.DEGRADED

    return FanoutResult(
        role=role,
        results=list(results),
        quorum_state=state,
        quorum_floor=quorum_floor,
    )


async def map_items[I, R](
    items: Sequence[I],
    fn: Callable[[I], Awaitable[R]],
    *,
    concurrency: int,
) -> list[R]:
    """Map an async ``fn`` over ``items`` with at most ``concurrency`` in flight; results stay
    aligned to input order.

    This is the bounded fan-out under map-reduce passes (summarize each chunk) and per-item
    verification (run a skeptic panel per finding) — the boilerplate ``Semaphore`` + ``gather`` that
    consumers kept re-rolling. ``fn`` owns its own error handling (return an error-encoded result so
    one bad item doesn't sink the batch); a raised exception propagates out of the gather.
    """
    sem = asyncio.Semaphore(max(1, concurrency))

    async def _one(item: I) -> R:
        async with sem:
            return await fn(item)

    return list(await asyncio.gather(*(_one(item) for item in items)))
