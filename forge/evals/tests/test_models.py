"""Tests for eval models and settings."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from agents.evals.config import EvalsSettings
from agents.evals.models import (
    CaseScore,
    GoldCase,
    GradeCheck,
    GradeResult,
    Scorecard,
    StepScore,
)

# -- config --


def test_settings_defaults():
    s = EvalsSettings()
    assert s.model == "coder"
    assert s.repeats == 3
    assert s.temperature == 0.0
    assert s.timeout == 240.0
    assert s.max_tokens == 16_000
    assert s.openai_base_url == "http://localhost:4010/v1"
    assert s.goldsets_dir == Path("/agents/evals/goldsets")
    assert s.runs_dir == Path("/eval-runs")


def test_settings_env_override(monkeypatch):
    monkeypatch.setenv("EVALS_MODEL", "gpt-4o")
    monkeypatch.setenv("EVALS_REPEATS", "7")
    # Force reload by creating fresh instance
    s = EvalsSettings()
    assert s.model == "gpt-4o"
    assert s.repeats == 7


# -- GoldCase --


def _make_gold_case(**overrides: object) -> GoldCase:
    return GoldCase(
        step="replan",
        case_id="test-case",
        case_dir=Path("/goldsets/test-case"),
        schema_version=1,
        inputs={"diff": "a.patch"},
        expected={"action": "replan"},
        **overrides,
    )


def test_gold_case_valid():
    c = _make_gold_case()
    assert c.step == "replan"
    assert c.holdout is False


@pytest.mark.parametrize(
    "cid",
    ["Upper", "has space", "123_456", "aB"],
)
def test_gold_case_slug_rejects_non_slug(cid: str):
    with pytest.raises(ValidationError, match="slug"):
        GoldCase(
            step="replan",
            case_id=cid,
            case_dir=Path("/x"),
            schema_version=1,
        )


def test_gold_case_slug_rejects_empty():
    with pytest.raises(ValidationError):
        GoldCase(
            step="replan",
            case_id="",
            case_dir=Path("/x"),
            schema_version=1,
        )


def test_gold_case_rejects_uppercase_slug():
    with pytest.raises(ValidationError, match="slug"):
        GoldCase(
            step="replan",
            case_id="Upper",
            case_dir=Path("/x"),
            schema_version=1,
        )


def test_gold_case_json_roundtrip():
    c = _make_gold_case(notes="some notes")
    raw = c.model_dump_json()
    restored = GoldCase.model_validate_json(raw)
    assert restored.case_id == c.case_id
    assert restored.notes == "some notes"


# -- GradeResult / GradeCheck --


def test_grade_result_defaults():
    r = GradeResult(case_id="x", step="replan", passed=True, score=0.8)
    assert r.checks == []
    assert r.error is None


def test_grade_result_error():
    r = GradeResult(case_id="x", step="replan", passed=False, score=0.0, error="timeout")
    assert r.error == "timeout"


def test_grade_check_default_detail():
    check = GradeCheck(name="foo", passed=True)
    assert check.detail == ""


# -- CaseScore --


def test_case_score_passed_majority():
    repeats = [
        GradeResult(case_id="x", step="replan", passed=True, score=1.0),
        GradeResult(case_id="x", step="replan", passed=True, score=1.0),
        GradeResult(case_id="x", step="replan", passed=False, score=0.0),
    ]
    cs = CaseScore(case_id="x", repeats=repeats)
    assert cs.passed_majority is True


def test_case_score_failed_majority():
    repeats = [
        GradeResult(case_id="x", step="replan", passed=False, score=0.0),
        GradeResult(case_id="x", step="replan", passed=False, score=0.0),
        GradeResult(case_id="x", step="replan", passed=True, score=1.0),
    ]
    cs = CaseScore(case_id="x", repeats=repeats)
    assert cs.passed_majority is False


def test_case_score_zero_repeats():
    cs = CaseScore(case_id="x")
    assert cs.passed_majority is False
    assert cs.mean_score == 0.0


def test_case_score_all_errors():
    repeats = [
        GradeResult(case_id="x", step="replan", passed=False, score=0.0, error="e"),
        GradeResult(case_id="x", step="replan", passed=False, score=0.0, error="e"),
    ]
    cs = CaseScore(case_id="x", repeats=repeats)
    assert cs.passed_majority is False
    assert cs.mean_score == 0.0


def test_case_score_mean_score():
    repeats = [
        GradeResult(case_id="x", step="replan", passed=True, score=0.5),
        GradeResult(case_id="x", step="replan", passed=True, score=1.0),
        GradeResult(case_id="x", step="replan", passed=False, score=0.0, error="e"),
    ]
    cs = CaseScore(case_id="x", repeats=repeats)
    assert cs.mean_score == pytest.approx(0.75)


# -- StepScore --


def test_step_score_pass_rate():
    cases = [
        CaseScore(case_id="a", repeats=[_valid_result()]),
        CaseScore(case_id="b", repeats=[_invalid_result()]),
    ]
    ss = StepScore(step="replan", cases=cases)
    assert ss.pass_rate == pytest.approx(0.5)


def test_step_score_empty():
    ss = StepScore(step="replan")
    assert ss.pass_rate == pytest.approx(0.0)
    assert ss.holdout_pass_rate is None
    assert ss.error_repeats == 0


def test_step_score_holdout_pass_rate():
    cases = [
        CaseScore(case_id="a", holdout=True, repeats=[_valid_result()]),
        CaseScore(case_id="b", holdout=False, repeats=[_invalid_result()]),
        CaseScore(case_id="c", holdout=True, repeats=[_invalid_result()]),
    ]
    ss = StepScore(step="replan", cases=cases)
    assert ss.holdout_pass_rate == pytest.approx(0.5)
    assert ss.error_repeats == 0


def test_step_score_holdout_none_when_no_holdout():
    ss = StepScore(step="replan", cases=[CaseScore(case_id="a")])
    assert ss.holdout_pass_rate is None


def test_step_score_error_repeats():
    cases = [
        CaseScore(
            case_id="a",
            repeats=[
                GradeResult(case_id="a", step="replan", passed=False, score=0.0, error="e"),
                _valid_result(),
            ],
        ),
    ]
    ss = StepScore(step="replan", cases=cases)
    assert ss.error_repeats == 1


def test_step_score_zero_division_safety():
    """No ZeroDivisionError when cases or repeats are empty."""
    ss = StepScore(step="replan")
    _ = ss.pass_rate
    _ = ss.holdout_pass_rate


# -- Scorecard --


def _valid_result():
    return GradeResult(case_id="x", step="replan", passed=True, score=1.0)


def _invalid_result():
    return GradeResult(case_id="x", step="replan", passed=False, score=0.0)


def test_scorecard_overall_pass_rate():
    cases = [
        CaseScore(case_id="a", repeats=[_valid_result()]),
        CaseScore(case_id="b", repeats=[_valid_result()]),
        CaseScore(case_id="c", repeats=[_invalid_result()]),
    ]
    ss = StepScore(step="replan", cases=cases)
    sc = Scorecard(model="coder", timestamp="2025-01-01T00:00:00Z", steps=[ss])
    assert sc.overall_pass_rate == pytest.approx(2 / 3)


def test_scorecard_empty():
    sc = Scorecard(model="coder", timestamp="2025-01-01T00:00:00Z")
    assert sc.overall_pass_rate == pytest.approx(0.0)


def test_scorecard_json_roundtrip():
    ss = StepScore(
        step="replan",
        cases=[CaseScore(case_id="a", repeats=[_valid_result()])],
    )
    sc = Scorecard(model="coder", timestamp="2025-01-01T00:00:00Z", steps=[ss])
    raw = sc.model_dump_json()
    restored = Scorecard.model_validate_json(raw)
    assert restored.model == sc.model
    assert len(restored.steps) == len(sc.steps)
    assert restored.steps[0].cases[0].case_id == "a"
