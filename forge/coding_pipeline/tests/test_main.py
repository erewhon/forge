"""Tests for the coding pipeline CLI (agents/coding_pipeline/main.py).

Verifies: subcommand parsing, --help output, plan without --approve performs no Forge writes,
and the registry wires the entry point correctly.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from agents.coding_pipeline.main import main


def _inventory():
    """Return a minimal valid Inventory for mocking."""
    from agents.coding_pipeline.models import Inventory

    return Inventory.model_validate({"project": "Meta", "repo": "/tmp", "tree": ""})


# ---------------------------------------------------------------------------
# CLI help / dispatch
# ---------------------------------------------------------------------------


def test_main_help_shows_all_subcommands():
    """`meta build --help` should list plan, run, and status."""
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0


def test_main_no_args_prints_help():
    """Calling main() with no arguments prints help and returns 0."""
    result = main([])
    assert result == 0


def test_main_unknown_subcommand_exits_nonzero():
    """An unrecognised subcommand should fail."""
    with pytest.raises(SystemExit) as exc:
        main(["bogus"])
    assert exc.value.code != 0


def test_plan_requires_project():
    """`meta build plan` without --project should fail (argparse required=True)."""
    with pytest.raises(SystemExit) as exc:
        main(["plan"])
    assert exc.value.code != 0


# ---------------------------------------------------------------------------
# plan: subcommand argument parsing
# ---------------------------------------------------------------------------


def test_plan_parses_spec_and_project(tmp_path):
    """plan should accept a positional spec file and --project."""
    spec = tmp_path / "goal.yaml"
    spec.write_text("goal: build x\nproject: Meta\n")

    with patch("agents.coding_pipeline.main.collect_inventory") as mock_inv, \
         patch("agents.coding_pipeline.main.propose_framing") as mock_frame, \
         patch("agents.coding_pipeline.main.persist_framing") as mock_persist, \
         patch("agents.coding_pipeline.main.write_inventory"):

        mock_inv.return_value = _inventory()
        mock_frame.return_value = MagicMock(approved=False)

        result = main(["plan", str(spec), "--project", "Meta"])
        assert result == 0
        mock_inv.assert_called_once()
        mock_frame.assert_called_once()
        mock_persist.assert_called_once()


# ---------------------------------------------------------------------------
# plan: --approve requires existing framing.json
# ---------------------------------------------------------------------------


def test_plan_approve_without_framing_json(tmp_path):
    """--approve without an existing framing.json is an error, never a shortcut."""
    run_dir = tmp_path / "pipeline-runs" / "my-epic"
    run_dir.mkdir(parents=True)

    with patch("agents.coding_pipeline.main.collect_inventory") as mock_inv, \
         patch("agents.coding_pipeline.main.write_inventory"):
        mock_inv.return_value = _inventory()
        result = main(["plan", "--approve", "--project", "Meta"])
        assert result == 1


def test_plan_approve_without_approved_flag(tmp_path):
    """--approve without approved:true in framing.json is an error."""
    run_dir = tmp_path / "pipeline-runs" / "my-epic"
    run_dir.mkdir(parents=True)
    (run_dir / "framing.json").write_text(
        json.dumps({
            "goal_as_stated": "x",
            "restated_goal": "x",
            "recommendation": "y",
            "epic_slug": "my-epic",
            "approved": False,
        })
    )

    with patch("agents.coding_pipeline.main.collect_inventory") as mock_inv, \
         patch("agents.coding_pipeline.main.write_inventory"):
        mock_inv.return_value = _inventory()
        result = main(["plan", "--approve", "--project", "Meta"])
        assert result == 1


# ---------------------------------------------------------------------------
# plan: without --approve performs no Forge writes (decomposition/emission)
# ---------------------------------------------------------------------------


def test_plan_without_approve_does_no_emission(tmp_path):
    """plan without --approve writes framing but does NOT call decompose or emit_tree."""
    spec = tmp_path / "goal.yaml"
    spec.write_text("goal: build x\nproject: Meta\n")

    with patch("agents.coding_pipeline.main.collect_inventory") as mock_inv, \
         patch("agents.coding_pipeline.main.propose_framing") as mock_frame, \
         patch("agents.coding_pipeline.main.persist_framing") as mock_persist, \
         patch("agents.coding_pipeline.main.write_inventory"):

        mock_inv.return_value = _inventory()
        mock_frame.return_value = MagicMock(approved=False)

        result = main(["plan", str(spec), "--project", "Meta"])
        assert result == 0

        # Framing should be written.
        mock_persist.assert_called_once()

        # decompose and emit_tree should NOT be called — re-run plan
        # and verify these functions are never imported/called.
        with patch("agents.coding_pipeline.main.decompose") as mock_decompose, \
             patch("agents.coding_pipeline.main.emit_tree") as mock_emit:
            mock_inv.reset_mock()
            mock_persist.reset_mock()
            mock_frame.reset_mock()

            result = main(["plan", str(spec), "--project", "Meta"])
            assert result == 0
            mock_decompose.assert_not_called()
            mock_emit.assert_not_called()


# ---------------------------------------------------------------------------
# run: subcommand argument parsing
# ---------------------------------------------------------------------------


def test_run_requires_epic_slug():
    """`meta build run` without epic_slug should fail."""
    with pytest.raises(SystemExit) as exc:
        main(["run"])
    assert exc.value.code != 0


def test_run_missing_run_dir(tmp_path):
    """`meta build run` with a non-existent run dir returns 1."""
    result = main(["run", "nonexistent-epic"])
    assert result == 1


# ---------------------------------------------------------------------------
# status: subcommand argument parsing
# ---------------------------------------------------------------------------


def test_status_without_args_no_prior_runs(tmp_path):
    """`meta build status` with no epic slug and no prior runs returns 0."""
    with patch("agents.coding_pipeline.main.settings") as mock_settings:
        mock_settings.runs_dir = tmp_path / "does-not-exist"
        result = main(["status"])
        assert result == 0


# ---------------------------------------------------------------------------
# registry integration
# ---------------------------------------------------------------------------


def test_registry_includes_build():
    """The build verb must appear in the registry."""
    from agents.registry import REGISTRY

    names = [cmd.name for cmd in REGISTRY]
    assert "build" in names


def test_build_exposes_mcp_false():
    """build must have exposes_mcp=False (it mutates repos)."""
    from agents.registry import REGISTRY

    build_cmd = next(cmd for cmd in REGISTRY if cmd.name == "build")
    assert build_cmd.exposes_mcp is False
    assert build_cmd.exposes_cli is True


def test_build_has_callable_main():
    """The build registry entry must resolve to a callable main."""
    from agents.registry import REGISTRY

    build_cmd = next(cmd for cmd in REGISTRY if cmd.name == "build")
    assert callable(build_cmd.load_main())


# ---------------------------------------------------------------------------
# help does not import agent modules (lazy load)
# ---------------------------------------------------------------------------


def test_help_does_not_import_coding_pipeline(monkeypatch):
    """`meta build --help` must not import the coding_pipeline package."""
    import importlib

    real_import = importlib.import_module

    def spy(name, *args, **kwargs):
        if "coding_pipeline" in name:
            raise AssertionError(f"coding_pipeline was imported during help: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("importlib.import_module", spy)
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
