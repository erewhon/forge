"""Run-scoped token accounting for autonomous loops.

A loop that only counts *attempts* cannot bound *spend* — one attempt's thinking-token budget can
be arbitrarily expensive (the concurrent-workers gate lesson). This ledger accumulates the token
usage of every in-process LLM call of a run, persists it so an epic's spend is cumulative and
resumable across waves and restarts, and lets the loop fail closed when a budget is exhausted.

The executor records into the *ambient* ledger via :func:`record_usage`, so no call site has to
thread a ledger through its signature: a run binds one with :func:`active_ledger`, and every API
call underneath it counts. Outside such a block, recording is a no-op.

Scope note: the dominant spend — the worker subprocess (opencode/claude) — is out of process and
not visible here. What this bounds is the pipeline's OWN reasoning spend (architect replan, wave
review, sign-off), which is exactly where an unbounded loop's cost runs away.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from pydantic import BaseModel


class Usage(BaseModel):
    """Cumulative token usage for a run/epic."""

    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class UsageLedger:
    """A thread-safe, optionally-persisted token counter for one run.

    Given a ``path``, it loads any existing total on construction (so a resumed epic keeps
    accumulating) and rewrites it on every record — persistence is best-effort and never raises
    into a wave. Without a path it is an in-memory counter (tests, one-shot calls).
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._usage = Usage()
        if path is not None and path.is_file():
            try:
                self._usage = Usage.model_validate_json(path.read_text())
            except (ValueError, OSError):
                self._usage = Usage()  # a corrupt ledger restarts at zero, never crashes the run

    @property
    def usage(self) -> Usage:
        with self._lock:
            return self._usage.model_copy()

    def record(self, input_tokens: int, output_tokens: int) -> None:
        """Add one call's usage. Negative/None counts coerce to 0 (a provider that omits usage
        must not corrupt the total)."""
        with self._lock:
            self._usage.input_tokens += max(0, int(input_tokens or 0))
            self._usage.output_tokens += max(0, int(output_tokens or 0))
            self._usage.calls += 1
            self._save_locked()

    def _save_locked(self) -> None:
        if self._path is None:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(self._usage.model_dump_json(indent=2))
        except OSError:
            pass  # persistence is best-effort; the in-memory total is still authoritative


_active: ContextVar[UsageLedger | None] = ContextVar("forge_active_usage_ledger", default=None)


def record_usage(input_tokens: int, output_tokens: int) -> None:
    """Record token usage against the active ledger, if any. A no-op when no run is accounting, so
    executors can call it unconditionally."""
    ledger = _active.get()
    if ledger is not None:
        ledger.record(input_tokens, output_tokens)


@contextmanager
def active_ledger(ledger: UsageLedger):
    """Bind *ledger* as the ambient accounting sink for the duration of the block."""
    token = _active.set(ledger)
    try:
        yield ledger
    finally:
        _active.reset(token)
