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
                with patch.object(
                    _mod,
                    "BASELINE_FILE",
                    tmp_path / "baselines" / ".json",
                ):
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
    assert any("Scorecard written to:" in s for s in captured)


# ---------------------------------------------------------------------------
# Tests: baseline refuses overwrite without --force
# ---------------------------------------------------------------------------


def test_baseline_refuses_overwrite_without_force(mock_scorecard: Scorecard, tmp_path: Path):
    """Baseline refuses to overwrite existing file without --force."""
    goldsets = settings.goldsets_dir
    baseline_dir = _evals_main.BASELINE_FILE.parent
    baseline_dir.mkdir(parents=True, exist_ok=True)
    _evals_main.BASELINE_FILE.write_text("{}", encoding="utf-8")

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
    baseline_dir = _evals_main.BASELINE_FILE.parent
    baseline_dir.mkdir(parents=True, exist_ok=True)
    _evals_main.BASELINE_FILE.write_text("{}", encoding="utf-8")

    with patch("agents.evals.main.run_scorecard", return_value=mock_scorecard) as mock_run:
        captured = []

        def capture(*args, **kwargs):
            captured.append(" ".join(str(a) for a in args))

        with patch("builtins.print", capture):
            rc = _cmd_baseline(["--model", "test-model", "--goldsets", str(goldsets), "--force"])

    assert rc == 0
    mock_run.assert_called_once()
    assert _evals_main.BASELINE_FILE.exists()
    data = json.loads(_evals_main.BASELINE_FILE.read_text())
    assert data["model"] == "test-model"


# ---------------------------------------------------------------------------
# Tests: baseline creates baselines dir
# ---------------------------------------------------------------------------


def test_baseline_creates_baselines_dir(mock_scorecard: Scorecard, tmp_path: Path):
    """Baseline creates the baselines directory if it doesn't exist."""
    goldsets = settings.goldsets_dir
    baseline_dir = _evals_main.BASELINE_FILE.parent
    # Ensure it doesn't exist
    if baseline_dir.exists():
        import shutil

        shutil.rmtree(baseline_dir)

    with patch("agents.evals.main.run_scorecard", return_value=mock_scorecard):
        with patch("builtins.print"):
            rc = _cmd_baseline(["--model", "test-model", "--goldsets", str(goldsets)])

    assert rc == 0
    assert baseline_dir.exists()
    assert _evals_main.BASELINE_FILE.exists()


# ---------------------------------------------------------------------------
# Tests: compare exits 2 without baseline
# ---------------------------------------------------------------------------


def test_compare_exits_2_without_baseline(tmp_path: Path):
    """Compare exits 2 when no baseline file exists."""
    goldsets = settings.goldsets_dir
    baseline_file = _evals_main.BASELINE_FILE
    # Ensure baseline does not exist
    if baseline_file.parent.exists():
        import shutil

        shutil.rmtree(baseline_file.parent)

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


def test_compare_renders_deltas_with_one_step(mock_scorecard: Scorecard, tmp_path: Path):
    """Compare renders a per-step delta table with one baseline step."""
    goldsets = settings.goldsets_dir
    baseline_dir = _evals_main.BASELINE_FILE.parent
    baseline_dir.mkdir(parents=True, exist_ok=True)

    baseline_data = {
        "model": "test-model",
        "timestamp": "2026-01-01T00:00:00+00:00",
        "steps": [
            {
                "step": "replan",
                "pass_rate": 0.5,
                "holdout_pass_rate": None,
                "error_repeats": 0,
                "cases": [],
            }
        ],
    }
    _evals_main.BASELINE_FILE.write_text(json.dumps(baseline_data), encoding="utf-8")

    fresh = Scorecard(
        model="test-model",
        timestamp="2026-01-02T00:00:00+00:00",
        steps=[],
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
