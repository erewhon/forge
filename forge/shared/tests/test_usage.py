"""Token accounting ledger — accumulation, persistence/resume, and the ambient record_usage sink."""

from __future__ import annotations

from pathlib import Path

from forge.shared.usage import UsageLedger, active_ledger, record_usage


def test_records_accumulate():
    led = UsageLedger()
    led.record(100, 20)
    led.record(50, 10)
    assert led.usage.input_tokens == 150
    assert led.usage.output_tokens == 30
    assert led.usage.total_tokens == 180
    assert led.usage.calls == 2


def test_negative_or_none_counts_coerce_to_zero():
    led = UsageLedger()
    led.record(-5, None)  # type: ignore[arg-type]
    led.record(10, 3)
    assert led.usage.total_tokens == 13
    assert led.usage.calls == 2


def test_persists_and_resumes_cumulative_total(tmp_path: Path):
    path = tmp_path / "usage.json"
    led = UsageLedger(path)
    led.record(200, 40)
    assert path.is_file()
    # a fresh ledger over the same file continues the running total (resume across waves/runs)
    resumed = UsageLedger(path)
    assert resumed.usage.total_tokens == 240
    resumed.record(10, 0)
    assert UsageLedger(path).usage.total_tokens == 250


def test_corrupt_ledger_file_restarts_at_zero(tmp_path: Path):
    path = tmp_path / "usage.json"
    path.write_text("{not valid json")
    led = UsageLedger(path)
    assert led.usage.total_tokens == 0  # never crashes the run over a bad ledger


def test_in_memory_ledger_needs_no_path():
    led = UsageLedger()
    led.record(1, 1)
    assert led.usage.total_tokens == 2  # no file written, no error


def test_record_usage_is_a_noop_without_an_active_ledger():
    record_usage(999, 999)  # must not raise when nothing is accounting


def test_active_ledger_captures_ambient_records():
    led = UsageLedger()
    with active_ledger(led):
        record_usage(300, 100)
        record_usage(10, 5)
    assert led.usage.total_tokens == 415
    # outside the block, recording no longer hits this ledger
    record_usage(1000, 1000)
    assert led.usage.total_tokens == 415


def test_active_ledger_restores_the_previous_sink():
    outer, inner = UsageLedger(), UsageLedger()
    with active_ledger(outer):
        record_usage(1, 0)
        with active_ledger(inner):
            record_usage(5, 0)
        record_usage(2, 0)
    assert inner.usage.total_tokens == 5
    assert outer.usage.total_tokens == 3
