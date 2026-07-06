"""Tests for agents.evals.main — monkeypatched run_scorecard, tmp dirs."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from agents.evals.config import settings
from agents.evals.main import (
    _cmd_baseline,
    _cmd_compare,
    _cmd_run,
)
from agents.evals.models import Scorecard

# The module under test — we patch constants here at import time.
_evals_main = None


@pytest.fixture(autouse=True)
def _patch_paths(tmp_path: Path):
    """Point settings and baseline paths at tmp dirs."""
    global _evals_main
    import agents.evals.main as _mod

    _evals_main = _mod
    goldsets = tmp_path / "goldsets"
    goldsets.mkdir(parents=True)

    with patch.object(settings, "goldsets_dir", goldsets):
        with patch.object(settings, "runs_dir", tmp_path / "runs"):
            with patch.object(_mod, "BASELINES_DIR", tmp_path / "baselines"):
                yield


@pytest.fixture()
def mock_scorecard() -> Scorecard:
    """Build a minimal Scorecard for patching."""
    return Scorecard(
        model="test-model",
        timestamp="2026-01-01T00:00:00+00:00",
        steps=[],
    )


# ---------------------------------------------------------------------------
# Tests: run writes + prints
# ---------------------------------------------------------------------------


def test_run_writes_and_prints(mock_scorecard: Scorecard, tmp_path: Path):
    """run calls run_scorecard, writes scorecard files, prints markdown."""
    goldsets = settings.goldsets_dir

    with patch("agents.evals.main.run_scorecard", return_value=mock_scorecard) as mock_run:
        with patch("agents.evals.main.write_scorecard") as mock_write:
            mock_write.return_value = goldsets / "scorecard.json"

            captured = []

            def capture(*args, **kwargs):
                captured.append(" ".join(str(a) for a in args))

            with patch("builtins.print", capture):
                rc = _cmd_run(["--model", "test-model", "--goldsets", str(goldsets)])

    assert rc == 0
    mock_run.assert_called_once_with(
        model="test-model",
        steps=None,
        goldsets_root=goldsets,
        repeats=None,
    )
    # Scorecards land in runs_dir — never inside the package next to the goldsets.
    mock_write.assert_called_once_with(mock_scorecard, settings.runs_dir)
    assert any("Scorecard written to:" in s for s in captured)


# ---------------------------------------------------------------------------
# Tests: baseline refuses overwrite without --force
# ---------------------------------------------------------------------------


def test_baseline_refuses_overwrite_without_force(mock_scorecard: Scorecard, tmp_path: Path):
    """Baseline refuses to overwrite existing file without --force."""
    goldsets = settings.goldsets_dir
    baseline_file = _evals_main._baseline_file("test-model")
    baseline_file.parent.mkdir(parents=True, exist_ok=True)
    baseline_file.write_text("{}", encoding="utf-8")

    captured = []

    def capture(*args, **kwargs):
        captured.append(" ".join(str(a) for a in args))

    with patch("agents.evals.main.run_scorecard", return_value=mock_scorecard):
        with patch("builtins.print", capture):
            rc = _cmd_baseline(["--model", "test-model", "--goldsets", str(goldsets)])

    assert rc == 1
    assert any("already exists" in s for s in captured)


# ---------------------------------------------------------------------------
# Tests: baseline overwrites with --force
# ---------------------------------------------------------------------------


def test_baseline_overwrites_with_force(mock_scorecard: Scorecard, tmp_path: Path):
    """Baseline overwrites existing file when --force is passed."""
    goldsets = settings.goldsets_dir
    baseline_file = _evals_main._baseline_file("test-model")
    baseline_file.parent.mkdir(parents=True, exist_ok=True)
    baseline_file.write_text("{}", encoding="utf-8")

    with patch("agents.evals.main.run_scorecard", return_value=mock_scorecard) as mock_run:
        captured = []

        def capture(*args, **kwargs):
            captured.append(" ".join(str(a) for a in args))

        with patch("builtins.print", capture):
            rc = _cmd_baseline(["--model", "test-model", "--goldsets", str(goldsets), "--force"])

    assert rc == 0
    mock_run.assert_called_once()
    assert baseline_file.exists()
    data = json.loads(baseline_file.read_text())
    assert data["model"] == "test-model"


# ---------------------------------------------------------------------------
# Tests: baseline creates baselines dir
# ---------------------------------------------------------------------------


def test_baseline_creates_baselines_dir(mock_scorecard: Scorecard, tmp_path: Path):
    """Baseline creates the baselines directory if it doesn't exist."""
    goldsets = settings.goldsets_dir
    baseline_file = _evals_main._baseline_file("test-model")
    assert not baseline_file.parent.exists()

    with patch("agents.evals.main.run_scorecard", return_value=mock_scorecard):
        with patch("builtins.print"):
            rc = _cmd_baseline(["--model", "test-model", "--goldsets", str(goldsets)])

    assert rc == 0
    assert baseline_file.exists()


def test_baselines_are_per_model(mock_scorecard: Scorecard, tmp_path: Path):
    """Each model gets its own baseline file — one model's baseline never
    blocks or overwrites another's."""
    goldsets = settings.goldsets_dir
    other = _evals_main._baseline_file("other-model")
    other.parent.mkdir(parents=True, exist_ok=True)
    other.write_text("{}", encoding="utf-8")

    with patch("agents.evals.main.run_scorecard", return_value=mock_scorecard):
        with patch("builtins.print"):
            rc = _cmd_baseline(["--model", "test-model", "--goldsets", str(goldsets)])

    assert rc == 0  # other-model's existing baseline is irrelevant
    assert _evals_main._baseline_file("test-model").exists()
    assert other.read_text() == "{}"  # untouched


# ---------------------------------------------------------------------------
# Tests: compare exits 2 without baseline
# ---------------------------------------------------------------------------


def test_compare_exits_2_without_baseline(tmp_path: Path):
    """Compare exits 2 when no baseline file exists."""
    goldsets = settings.goldsets_dir
    assert not _evals_main._baseline_file("test-model").exists()

    captured = []

    def capture(*args, **kwargs):
        captured.append(" ".join(str(a) for a in args))

    with patch("builtins.print", capture):
        rc = _cmd_compare(["--model", "test-model", "--goldsets", str(goldsets)])

    assert rc == 2
    assert any("No baseline found" in s for s in captured)


# ---------------------------------------------------------------------------
# Tests: compare renders deltas with one baseline step
# ---------------------------------------------------------------------------


def test_compare_renders_deltas_with_one_step(tmp_path: Path):
    """Compare renders per-step deltas with REAL StepScore objects on both
    sides — the both-present branch once crashed on attribute-vs-dict access,
    and the baseline JSON does not carry the computed rate fields at all."""
    from agents.evals.models import CaseScore, GradeResult, StepScore

    def step_score(passed: bool) -> StepScore:
        return StepScore(
            step="replan",
            cases=[
                CaseScore(
                    case_id="case-1",
                    holdout=True,
                    repeats=[
                        GradeResult(
                            case_id="case-1",
                            step="replan",
                            passed=passed,
                            score=1.0 if passed else 0.0,
                        )
                    ],
                )
            ],
        )

    goldsets = settings.goldsets_dir
    baseline_file = _evals_main._baseline_file("test-model")
    baseline_file.parent.mkdir(parents=True, exist_ok=True)
    baseline = Scorecard(
        model="test-model",
        timestamp="2026-01-01T00:00:00+00:00",
        steps=[step_score(passed=False)],  # baseline: 0% pass
    )
    baseline_file.write_text(baseline.model_dump_json(), encoding="utf-8")

    fresh = Scorecard(
        model="test-model",
        timestamp="2026-01-02T00:00:00+00:00",
        steps=[step_score(passed=True)],  # fresh: 100% pass
    )

    captured = []

    def capture(*args, **kwargs):
        captured.append(" ".join(str(a) for a in args))

    with patch("agents.evals.main.run_scorecard", return_value=fresh):
        with patch("builtins.print", capture):
            rc = _cmd_compare(["--model", "test-model", "--goldsets", str(goldsets)])

    assert rc == 0
    full_output = "\n".join(captured)
    assert "replan" in full_output
    assert "0%" in full_output and "100%" in full_output  # both sides rendered
    assert "+100%" in full_output  # the delta itself
    assert "REGRESSION" not in full_output


def test_compare_flags_holdout_regression(tmp_path: Path):
    """A holdout drop is flagged explicitly — the load-bearing gate."""
    from agents.evals.models import CaseScore, GradeResult, StepScore

    def step_score(passed: bool) -> StepScore:
        return StepScore(
            step="replan",
            cases=[
                CaseScore(
                    case_id="case-1",
                    holdout=True,
                    repeats=[
                        GradeResult(
                            case_id="case-1",
                            step="replan",
                            passed=passed,
                            score=1.0 if passed else 0.0,
                        )
                    ],
                )
            ],
        )

    goldsets = settings.goldsets_dir
    baseline_file = _evals_main._baseline_file("test-model")
    baseline_file.parent.mkdir(parents=True, exist_ok=True)
    baseline_file.write_text(
        Scorecard(
            model="test-model",
            timestamp="2026-01-01T00:00:00+00:00",
            steps=[step_score(passed=True)],
        ).model_dump_json(),
        encoding="utf-8",
    )

    fresh = Scorecard(
        model="test-model",
        timestamp="2026-01-02T00:00:00+00:00",
        steps=[step_score(passed=False)],
    )

    captured = []

    def capture(*args, **kwargs):
        captured.append(" ".join(str(a) for a in args))

    with patch("agents.evals.main.run_scorecard", return_value=fresh):
        with patch("builtins.print", capture):
            rc = _cmd_compare(["--model", "test-model", "--goldsets", str(goldsets)])

    assert rc == 0
    full_output = "\n".join(captured)
    assert "HOLDOUT REGRESSION" in full_output


# ---------------------------------------------------------------------------
# Tests: unknown --step errors naming the bad key
# ---------------------------------------------------------------------------


def test_unknown_step_errors_naming_bad_key(tmp_path: Path):
    """Unknown --step values are rejected with the bad key named."""
    goldsets = settings.goldsets_dir

    captured = []

    def capture(*args, **kwargs):
        captured.append(" ".join(str(a) for a in args))

    with patch("builtins.print", capture):
        rc = _cmd_run(
            [
                "--model",
                "test-model",
                "--step",
                "nonexistent-step",
                "--goldsets",
                str(goldsets),
            ]
        )

    assert rc == 1
    assert any("nonexistent-step" in s for s in captured)


def test_unknown_step_baseline(mock_scorecard: Scorecard, tmp_path: Path):
    """Unknown --step in baseline is rejected with the bad key named."""
    goldsets = settings.goldsets_dir

    captured = []

    def capture(*args, **kwargs):
        captured.append(" ".join(str(a) for a in args))

    with patch("builtins.print", capture):
        rc = _cmd_baseline(
            ["--model", "test-model", "--step", "bogus", "--goldsets", str(goldsets)]
        )

    assert rc == 1
    assert any("bogus" in s for s in captured)


# ---------------------------------------------------------------------------
# Tests: registry resolves evals and load_main() imports
# ---------------------------------------------------------------------------


def test_registry_resolves_evals():
    """REGISTRY contains an 'evals' entry."""
    from agents.registry import REGISTRY

    names = [cmd.name for cmd in REGISTRY]
    assert "evals" in names

    evals_cmd = next(cmd for cmd in REGISTRY if cmd.name == "evals")
    assert evals_cmd.module == "agents.evals.main"
    assert evals_cmd.exposes_mcp is False


def test_registry_load_main_imports(tmp_path: Path):
    """AgentCommand.load_main() for evals returns a callable."""
    from agents.registry import REGISTRY

    evals_cmd = next(cmd for cmd in REGISTRY if cmd.name == "evals")
    main_func = evals_cmd.load_main()
    assert callable(main_func)
