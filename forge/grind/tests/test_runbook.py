"""Runbook cycle execution + scoring, against real shell commands in a tmp dir."""

from __future__ import annotations

from forge.grind.models import GrindConfig
from forge.grind.runbook import run_cycle, score_improves


def _cfg(check_run: str, *, score_regex=None, observe=None) -> GrindConfig:
    return GrindConfig(
        goal="g",
        steps=[{"name": "prep", "run": "echo prepped"}, {"name": "act", "run": "echo acted"}],
        check={"run": check_run, "score_regex": score_regex},
        observe=observe or [],
    )


def test_cycle_passes_when_check_exits_zero(tmp_path):
    result = run_cycle(_cfg("true"), tmp_path)
    assert result.passed
    assert result.failing_step is None
    assert result.reason == ""


def test_cycle_fails_and_names_failing_check(tmp_path):
    result = run_cycle(_cfg("echo boom >&2; exit 1"), tmp_path)
    assert not result.passed
    assert result.failing_step == "check"
    assert "check:" in result.reason


def test_cycle_fails_on_a_step_before_check(tmp_path):
    cfg = GrindConfig(
        goal="g",
        steps=[{"name": "prep", "run": "exit 2"}, {"name": "act", "run": "echo acted"}],
        check={"run": "true"},
    )
    result = run_cycle(cfg, tmp_path)
    assert not result.passed
    assert result.failing_step == "prep"  # first failing step wins


def test_score_parsed_from_check_stdout(tmp_path):
    result = run_cycle(
        _cfg("echo RECONCILED=42; exit 1", score_regex="RECONCILED=([0-9]+)"), tmp_path
    )
    assert result.score == 42.0


def test_observation_respects_observe_filter(tmp_path):
    result = run_cycle(_cfg("echo checked", observe=["act", "check"]), tmp_path)
    assert "acted" in result.observation
    assert "checked" in result.observation
    assert "prepped" not in result.observation  # 'prep' not observed


def test_score_improves_direction():
    assert score_improves(5.0, 3.0, "max")
    assert not score_improves(3.0, 5.0, "max")
    assert score_improves(3.0, 5.0, "min")
    assert score_improves(1.0, None, "max")  # any score beats no baseline
    assert not score_improves(None, 1.0, "max")
