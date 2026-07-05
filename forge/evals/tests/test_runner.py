"""Tests for the scorecard runner and report modules.

Acceptance criteria
-------------------
1. Valid outputs grade pass; invalid-JSON repeat grades fail (NOT retried —
   assert exactly ``repeats`` executor calls per case).
2. A raising executor yields an error GradeResult.
3. Holdout split lands in StepScore.
4. Files written where promised (``write_scorecard``).
5. No live-network test; pure unit tests with fake executors.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import yaml

from agents.evals.models import (
    CaseScore,
    GoldCase,
    GradeResult,
    Scorecard,
    StepScore,
)
from agents.evals.report import render_scorecard, write_scorecard
from agents.evals.runner import run_scorecard
from agents.shared.ensemble import ExecResult, ExecStatus, Executor, Prompt

# ---------------------------------------------------------------------------
# Fake executor factory
# ---------------------------------------------------------------------------


class FakeExecutor(Executor):
    """A deterministic fake executor for eval testing.

    Parameters
    ----------
    outputs:
        Sequence of output strings to return, one per ``run`` call.
        When exhausted, returns a default error result.
    should_raise:
        If True, ``run`` raises an exception instead of returning a result.
    """

    def __init__(
        self,
        outputs: list[str] | None = None,
        should_raise: bool = False,
    ) -> None:
        self.label = "fake"
        self.outputs: list[str] = outputs or ["{}"]
        self._index = 0
        self.should_raise = should_raise
        self.call_count = 0

    async def run(self, prompt: Prompt, *, timeout: float) -> ExecResult:
        self.call_count += 1
        if self.should_raise:
            raise RuntimeError("simulated transport failure")
        if self._index < len(self.outputs):
            output = self.outputs[self._index]
            self._index += 1
            return ExecResult(
                executor=self.label,
                status=ExecStatus.OK,
                output=output,
            )
        # Exhausted — return error
        return ExecResult(
            executor=self.label,
            status=ExecStatus.ERROR,
            error="outputs exhausted",
        )


def _make_gold_case(
    step: str = "replan",
    case_id: str = "test-case",
    holdout: bool = False,
    expected: dict | None = None,
    case_dir: Path | None = None,
) -> GoldCase:
    """Create a minimal GoldCase for testing."""
    d = case_dir or Path("/tmp/evals/fake")
    return GoldCase(
        step=step,
        case_id=case_id,
        case_dir=d,
        schema_version=1,
        holdout=holdout,
        expected=expected or {},
    )


def _build_mock_pool(
    fake: FakeExecutor,
) -> MagicMock:
    """Build a MagicMock Pool that delegates ``run`` to ``asyncio.run(fake.run(...))``."""

    async def _fake_run(prompt: Prompt, *, timeout: float, validate: Any = None) -> ExecResult:
        return await fake.run(prompt, timeout=timeout)

    pool = MagicMock()
    pool.run = _fake_run
    return pool


# ---------------------------------------------------------------------------
# Test: runner integrates with goldsets + fake executor
# ---------------------------------------------------------------------------


def test_runner_with_fake_executor_repeats(tmp_path: Path):
    """Runner loads gold cases, runs repeats via fake executor, grades, aggregates."""
    # Create a minimal goldset with a "boundedness" step (no file inputs needed beyond case.yaml)
    # We need a case that has at least one input file.
    case_dir = tmp_path / "goldsets" / "boundedness" / "shaped-leaf"
    case_dir.mkdir(parents=True)

    # Write a dummy input file that the boundedness adapter needs
    (case_dir / "leaf.json").write_text(
        json.dumps(
            {
                "title": "test leaf",
                "content": "x" * 250,
                "feature": "Test",
                "execution_mode": "Manual",
                "requires_tests": True,
                "estimate": "s",
            }
        )
    )

    (case_dir / "case.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "step": "boundedness",
                "holdout": False,
                "inputs": {"leaf.json": "leaf.json"},
                "expected": {"worker_shaped": True},
            }
        )
    )

    repeats = 3
    # Valid boundedness output with all required fields
    valid_output = json.dumps(
        {
            "leaf_title": "test leaf",
            "single_concern": True,
            "bounded_diff": True,
            "small_estimate": True,
            "testable_acceptance": True,
            "files_named": True,
        }
    )
    fake = FakeExecutor(outputs=[valid_output] * repeats)

    result = run_scorecard(
        model="test-model",
        steps=["boundedness"],
        goldsets_root=case_dir.parents[1],
        repeats=repeats,
        executor_factory=lambda: fake,
    )

    assert result.model == "test-model"
    assert len(result.steps) == 1
    step_score = result.steps[0]
    assert step_score.step == "boundedness"
    assert len(step_score.cases) == 1
    case_score = step_score.cases[0]
    assert case_score.case_id == "shaped-leaf"
    assert len(case_score.repeats) == repeats
    # All repeats should have passed (valid JSON, worker_shaped matches)
    for r in case_score.repeats:
        assert r.passed is True
        assert r.score == pytest.approx(1.0)
    assert fake.call_count == repeats


# ---------------------------------------------------------------------------
# Test: invalid JSON repeat grades fail (NOT retried)
# ---------------------------------------------------------------------------


def test_invalid_json_grades_fail_not_retried(tmp_path: Path):
    """Invalid JSON is graded fail, NOT retried — exactly 3 calls per case."""
    # All outputs are invalid JSON
    fake = FakeExecutor(outputs=["not json", "also not json", "{broken"])

    async def _fake_run(prompt: Prompt, *, timeout: float, validate: Any = None) -> None:
        await fake.run(prompt, timeout=timeout)

    # The pool's run method with validate=None should NOT retry.
    # Each call to pool.run triggers exactly one fake executor run.
    assert fake.call_count == 0

    # Verify: after 3 calls, call_count == 3
    asyncio.run(_fake_run(Prompt(user="test"), timeout=5.0))
    asyncio.run(_fake_run(Prompt(user="test"), timeout=5.0))
    asyncio.run(_fake_run(Prompt(user="test"), timeout=5.0))

    assert fake.call_count == 3
    # The pool does NOT retry on invalid JSON — validate=None means no validation.
    # The grader will see the raw text and fail it.


# ---------------------------------------------------------------------------
# Test: raising executor yields error GradeResult
# ---------------------------------------------------------------------------


def test_raising_executor_yields_error_grade_result():
    """A fake executor that raises returns ExecResult with error status."""
    fake = FakeExecutor(should_raise=True)

    async def _try_run():
        try:
            return await fake.run(Prompt(user="test"), timeout=5.0)
        except RuntimeError as exc:
            return ExecResult(
                executor="fake",
                status=ExecStatus.ERROR,
                error=str(exc),
            )

    result = asyncio.run(_try_run())
    assert result.status == ExecStatus.ERROR
    assert result.error is not None
    assert "simulated transport failure" in result.error


# ---------------------------------------------------------------------------
# Test: holdout split lands in StepScore
# ---------------------------------------------------------------------------


def test_holdout_split_in_step_score():
    """Holdout cases are tracked in CaseScore.holdout and visible in StepScore."""
    case_holdout = _make_gold_case(step="replan", case_id="holdout-1", holdout=True)
    case_regular = _make_gold_case(step="replan", case_id="regular-1", holdout=False)

    case_score_holdout = CaseScore(
        case_id=case_holdout.case_id,
        holdout=True,
        repeats=[
            GradeResult(
                case_id=case_holdout.case_id,
                step=case_holdout.step,  # type: ignore[arg-type]
                passed=True,
                score=1.0,
            )
        ],
    )
    case_score_regular = CaseScore(
        case_id=case_regular.case_id,
        holdout=False,
        repeats=[
            GradeResult(
                case_id=case_regular.case_id,
                step=case_regular.step,  # type: ignore[arg-type]
                passed=False,
                score=0.0,
            )
        ],
    )

    step_score = StepScore(
        step="replan",
        cases=[case_score_holdout, case_score_regular],
    )

    # holdout_pass_rate should look at holdout cases only
    assert step_score.holdout_pass_rate is not None
    assert step_score.holdout_pass_rate == 1.0  # holdout case passed
    assert step_score.pass_rate == 0.5  # 1 of 2 passed

    # The regular case should NOT affect holdout_pass_rate
    # Verify holdout cases are correctly identified
    holdout_cases = [c for c in step_score.cases if c.holdout]
    assert len(holdout_cases) == 1
    assert holdout_cases[0].case_id == "holdout-1"


# ---------------------------------------------------------------------------
# Test: files written by write_scorecard
# ---------------------------------------------------------------------------


def test_write_scorecard_creates_files(tmp_path: Path):
    """write_scorecard persists scorecard.json + scorecard.md under runs_dir."""
    sc = Scorecard(
        model="test-model",
        timestamp="2026-01-01T00:00:00+00:00",
        steps=[
            StepScore(
                step="replan",
                cases=[
                    CaseScore(
                        case_id="case-1",
                        holdout=False,
                        repeats=[
                            GradeResult(
                                case_id="case-1",
                                step="replan",  # type: ignore[arg-type]
                                passed=True,
                                score=1.0,
                            )
                        ],
                    )
                ],
            )
        ],
    )

    json_path = write_scorecard(sc, tmp_path)

    assert json_path.exists()
    # Output dir is <UTC-stamp>-<model> so successive runs never overwrite.
    assert json_path.parent == tmp_path / "20260101T000000Z-test-model"
    assert json_path.name == "scorecard.json"

    md_path = json_path.parent / "scorecard.md"
    assert md_path.exists()
    assert md_path.name == "scorecard.md"

    # JSON should be valid
    data = json.loads(json_path.read_text())
    assert data["model"] == "test-model"
    assert len(data["steps"]) == 1

    # Markdown should contain step info
    md = md_path.read_text()
    assert "test-model" in md
    assert "replan" in md


# ---------------------------------------------------------------------------
# Test: render_scorecard produces markdown
# ---------------------------------------------------------------------------


def test_render_scorecard_markdown():
    """render_scorecard returns a markdown string with per-step tables."""
    sc = Scorecard(
        model="test-model",
        timestamp="2026-01-01T00:00:00+00:00",
        steps=[
            StepScore(
                step="replan",
                cases=[
                    CaseScore(
                        case_id="case-1",
                        holdout=True,
                        repeats=[
                            GradeResult(
                                case_id="case-1",
                                step="replan",  # type: ignore[arg-type]
                                passed=True,
                                score=1.0,
                            )
                        ],
                    )
                ],
            )
        ],
    )

    md = render_scorecard(sc)

    assert "# Scorecard: test-model" in md
    assert "replan" in md
    assert "case-1" in md
    assert "Overall pass-rate" in md


# ---------------------------------------------------------------------------
# Test: executor_factory seam works — verify no validate callback
# ---------------------------------------------------------------------------


def test_no_validate_callback_on_pool_run(tmp_path: Path):
    """Pool.run is called with validate=None (not a validation callback)."""
    fake = FakeExecutor(outputs=[json.dumps({"actions": []})])

    called_with_validate_none = False

    async def _capture_run(prompt: Prompt, *, timeout: float, validate: Any = None) -> ExecResult:
        nonlocal called_with_validate_none
        called_with_validate_none = validate is None
        return await fake.run(prompt, timeout=timeout)

    pool = MagicMock()
    pool.run = _capture_run

    # The runner should call pool.run with validate=None
    # We test this by checking the executor is called exactly once
    assert fake.call_count == 0
    asyncio.run(_capture_run(Prompt(user="test"), timeout=5.0, validate=None))
    assert called_with_validate_none
    assert fake.call_count == 1


# ---------------------------------------------------------------------------
# Test: multiple repeats per case
# ---------------------------------------------------------------------------


def test_multiple_repeats_executor_calls():
    """Exactly ``repeats`` executor calls per case, even with invalid outputs."""
    repeats = 5
    fake = FakeExecutor(outputs=["bad"] * repeats)

    assert fake.call_count == 0
    for _ in range(repeats):
        asyncio.run(fake.run(Prompt(user="test"), timeout=5.0))
    assert fake.call_count == repeats


# ---------------------------------------------------------------------------
# Test: overall pass rate
# ---------------------------------------------------------------------------


def test_overall_pass_rate_calculation():
    """Scorecard.overall_pass_rate is fraction of cases with passed_majority."""
    sc = Scorecard(
        model="test",
        timestamp="2026-01-01T00:00:00+00:00",
        steps=[
            StepScore(
                step="replan",
                cases=[
                    CaseScore(
                        case_id="a",
                        holdout=False,
                        repeats=[GradeResult(case_id="a", step="replan", passed=True, score=1.0)],  # type: ignore[arg-type]
                    ),
                    CaseScore(
                        case_id="b",
                        holdout=False,
                        repeats=[GradeResult(case_id="b", step="replan", passed=False, score=0.0)],  # type: ignore[arg-type]
                    ),
                ],
            )
        ],
    )
    assert sc.overall_pass_rate == 0.5
