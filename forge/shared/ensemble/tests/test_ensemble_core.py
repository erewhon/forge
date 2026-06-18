"""Core failover tests — no network; executors are scripted fakes."""

from __future__ import annotations

import asyncio

from agents.shared.ensemble.classify import classify
from agents.shared.ensemble.combiner import AggregateCombiner
from agents.shared.ensemble.models import (
    ExecResult,
    ExecStatus,
    FailureClass,
    Prompt,
    QuorumState,
)
from agents.shared.ensemble.pool import Pool, fanout, map_items


class FakeExecutor:
    """Returns scripted ExecResults in sequence; the last entry repeats."""

    def __init__(self, label: str, script: list[ExecResult]) -> None:
        self.label = label
        self._script = script
        self._i = 0

    async def run(self, prompt: Prompt, *, timeout: float) -> ExecResult:
        result = self._script[min(self._i, len(self._script) - 1)]
        self._i += 1
        return result.model_copy()


def _ok(label: str = "x", output: str = "OK") -> ExecResult:
    return ExecResult(executor=label, status=ExecStatus.OK, output=output)


def _fail(label: str = "x", fc: FailureClass = FailureClass.TERMINAL) -> ExecResult:
    return ExecResult(executor=label, status=ExecStatus.ERROR, error="boom", failure_class=fc)


class _HTTPErr(Exception):
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        super().__init__(str(status_code))


def test_classify_status_codes() -> None:
    assert classify(_HTTPErr(404)) is FailureClass.TERMINAL  # pulled model
    assert classify(_HTTPErr(401)) is FailureClass.TERMINAL
    assert classify(_HTTPErr(429)) is FailureClass.TRANSIENT
    assert classify(_HTTPErr(503)) is FailureClass.TRANSIENT
    assert classify(TimeoutError()) is FailureClass.TRANSIENT
    assert classify(ConnectionError()) is FailureClass.TRANSIENT


def test_failover_skips_terminal_to_next() -> None:
    pool = Pool(
        role="judge",
        executors=[
            FakeExecutor("a", [_fail("a", FailureClass.TERMINAL)]),
            FakeExecutor("b", [_ok("b")]),
        ],
        retry_backoff_s=0,
    )
    result = asyncio.run(pool.run(Prompt(user="hi"), timeout=1))
    assert result.ok
    assert result.executor == "b"


def test_failover_retries_transient_then_succeeds() -> None:
    pool = Pool(
        role="judge",
        executors=[FakeExecutor("a", [_fail("a", FailureClass.TRANSIENT), _ok("a")])],
        max_attempts_per_executor=2,
        retry_backoff_s=0,
    )
    result = asyncio.run(pool.run(Prompt(user="hi"), timeout=1))
    assert result.ok
    assert result.executor == "a"
    assert result.attempts == 2


def test_failover_all_dead_returns_last_failure() -> None:
    pool = Pool(
        role="judge",
        executors=[
            FakeExecutor("a", [_fail("a", FailureClass.TERMINAL)]),
            FakeExecutor("b", [_fail("b", FailureClass.TERMINAL)]),
        ],
        retry_backoff_s=0,
    )
    result = asyncio.run(pool.run(Prompt(user="hi"), timeout=1))
    assert not result.ok
    assert result.executor == "b"


def test_validate_demotes_bad_output_and_retries() -> None:
    # First reply is transport-OK but invalid (e.g. malformed JSON); the second validates.
    pool = Pool(
        role="judge",
        executors=[FakeExecutor("a", [_ok("a", "not json"), _ok("a", "{good}")])],
        max_attempts_per_executor=2,
        retry_backoff_s=0,
    )
    result = asyncio.run(pool.run(Prompt(user="hi"), timeout=1, validate=lambda t: t == "{good}"))
    assert result.ok
    assert result.output == "{good}"
    assert result.attempts == 2


def test_validate_fails_over_to_next_executor() -> None:
    # Primary always emits invalid output; pool must fail over to a model that validates.
    pool = Pool(
        role="judge",
        executors=[
            FakeExecutor("a", [_ok("a", "garbage")]),
            FakeExecutor("b", [_ok("b", "{good}")]),
        ],
        max_attempts_per_executor=2,
        retry_backoff_s=0,
    )
    result = asyncio.run(pool.run(Prompt(user="hi"), timeout=1, validate=lambda t: t == "{good}"))
    assert result.ok
    assert result.executor == "b"


def _single(label: str, ok: bool) -> Pool:
    script = [_ok(label) if ok else _fail(label)]
    return Pool(role="reviewer", executors=[FakeExecutor(label, script)], retry_backoff_s=0)


def test_fanout_quorum_states() -> None:
    degraded = asyncio.run(
        fanout(
            "reviewer",
            [_single("a", True), _single("b", True), _single("c", False)],
            Prompt(user="x"),
            timeout=1,
            quorum_floor=2,
        )
    )
    assert degraded.quorum_state is QuorumState.DEGRADED
    assert len(degraded.succeeded) == 2

    failed = asyncio.run(
        fanout(
            "reviewer",
            [_single("a", True), _single("b", False)],
            Prompt(user="x"),
            timeout=1,
            quorum_floor=2,
        )
    )
    assert failed.quorum_state is QuorumState.FAILED

    full = asyncio.run(
        fanout(
            "reviewer",
            [_single("a", True), _single("b", True)],
            Prompt(user="x"),
            timeout=1,
            quorum_floor=1,
        )
    )
    assert full.quorum_state is QuorumState.FULL


def test_aggregate_combiner_falls_back_to_concat() -> None:
    dead = Pool(
        role="aggregator",
        executors=[FakeExecutor("agg", [_fail("agg", FailureClass.TERMINAL)])],
        retry_backoff_s=0,
    )
    combiner = AggregateCombiner(pool=dead, system="synthesize", timeout=1)
    inputs = [_ok("r1", "first review"), _ok("r2", "second review")]
    out = asyncio.run(combiner.combine(inputs))
    assert out.used_fallback
    assert "first review" in out.text
    assert "second review" in out.text


# --- map_items (bounded-concurrency fan-out over a runtime work-list) ---


def test_map_items_preserves_order() -> None:
    async def double(x: int) -> int:
        return x * 2

    out = asyncio.run(map_items([1, 2, 3, 4], double, concurrency=2))
    assert out == [2, 4, 6, 8]


def test_map_items_empty_is_empty() -> None:
    async def boom(_x: int) -> int:  # must never be called
        raise AssertionError("fn should not run for empty input")

    assert asyncio.run(map_items([], boom, concurrency=4)) == []


def test_map_items_bounds_concurrency() -> None:
    inflight = 0
    max_inflight = 0

    async def tracked(x: int) -> int:
        nonlocal inflight, max_inflight
        inflight += 1
        max_inflight = max(max_inflight, inflight)
        await asyncio.sleep(0.01)  # hold the slot so overlap is observable
        inflight -= 1
        return x

    out = asyncio.run(map_items(list(range(6)), tracked, concurrency=2))
    assert out == list(range(6))
    assert max_inflight <= 2  # the semaphore never let more than `concurrency` run at once
