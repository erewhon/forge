"""CLI entry point tests — main(…) argument parsing, flag routing, exit codes."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agents.dependabot.main import main
from agents.dependabot.models import BumpCandidate, BumpResult


def _result(status: str = "planned", **over) -> BumpResult:
    c = BumpCandidate(name="idna", current="3.11", latest="3.15", delta="minor")
    base = dict(status=status, candidate=c, branch="deps/idna-3-15")
    base.update(over)
    return BumpResult(**base)


@pytest.fixture
def auto_bump_mock(monkeypatch):
    """Return a configurable MagicMock patched into auto_bump."""
    mock = MagicMock(return_value=_result())
    monkeypatch.setattr("agents.dependabot.main.auto_bump", mock)
    return mock


def test_dry_run_passes_flag(auto_bump_mock, capsys):
    result = main(["--dry-run"])
    assert result == 0
    out = capsys.readouterr().out
    assert "planned" in out


def test_auto_merge_flag_reaches_auto_bump(auto_bump_mock):
    result = main(["--auto-merge"])
    assert result == 0
    assert auto_bump_mock.call_args[1]["auto_merge"] is True


def test_advisory_status_exits_1(auto_bump_mock, capsys):
    auto_bump_mock.return_value = _result(status="advisory", reason="major delta")
    result = main([])
    assert result == 1
    out = capsys.readouterr().out
    assert "advisory" in out


def test_error_status_exits_1(auto_bump_mock, capsys):
    auto_bump_mock.return_value = _result(status="error", reason="no repo found")
    result = main([])
    assert result == 1


def test_merged_exits_0(auto_bump_mock, capsys):
    auto_bump_mock.return_value = _result(status="merged", merged_to_main=True)
    result = main(["--auto-merge"])
    assert result == 0


def test_branched_exits_0(auto_bump_mock, capsys):
    auto_bump_mock.return_value = _result(status="branched")
    result = main([])
    assert result == 0


def test_no_candidates_exits_0(auto_bump_mock, capsys):
    auto_bump_mock.return_value = _result(status="no-candidates")
    result = main([])
    assert result == 0


def test_explicit_repo_path(auto_bump_mock, tmp_path):
    result = main(["--repo", str(tmp_path)])
    assert result == 0
    assert auto_bump_mock.call_args[0][0] == tmp_path.resolve()


def test_custom_project(auto_bump_mock):
    result = main(["--project", "TestProject"])
    assert result == 0
    assert auto_bump_mock.call_args[1]["project"] == "TestProject"


def test_no_repo_prints_error(capsys, monkeypatch):
    monkeypatch.setattr("agents.dependabot.main.find_repo_root", lambda _: None)
    result = main([])
    assert result == 1
    err = capsys.readouterr().err
    assert "no jj/git repo found" in err
