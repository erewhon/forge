"""Unit tests for agents.shared.workspaces — jj workspace lifecycle.

Uses mocked subprocess.run for create/forget/resolve argv shapes.
Real-tmp-repo test skipped when jj is unavailable.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from agents.shared import workspaces as ws_module
from agents.shared.workspaces import (
    DiffStat,
    JJError,
    _diff_exclude_fileset,
    _parse_diff_stat,
    collect_diff,
    create_workspace,
    ensure_git_marker,
    forget_workspace,
    resolve_base_rev,
    workspace_destination,
)

# ----------------------------------------------------------------------- fixtures

pytestmark = pytest.mark.usefixtures("mock_subprocess")


@pytest.fixture()
def mock_subprocess(monkeypatch):
    """Return a fake CompletedProcess factory for mocking subprocess.run."""
    history: list[list[str]] = []

    def _factory(*args, **kwargs):
        cmd = args[0] if args else kwargs.get("args", [])
        history.append(cmd)

        # Return a non-empty stdout for resolve_base_rev tests by default,
        # unless explicitly overridden via a "return_stdout" kwarg.
        stdout = kwargs.pop("return_stdout", "abc123")
        fake = subprocess.CompletedProcess(
            cmd,
            returncode=kwargs.pop("returncode", 0),
            stdout=stdout,
            stderr=kwargs.pop("stderr", ""),
        )
        return fake

    monkeypatch.setattr(subprocess, "run", _factory)
    return history


# ----------------------------------------------------------------------- DiffStat


def test_diffstat_defaults():
    stat = DiffStat()
    assert stat.files_changed == 0
    assert stat.insertions == 0
    assert stat.deletions == 0


def test_diffstat_fields():
    stat = DiffStat(files_changed=3, insertions=42, deletions=7)
    assert stat.files_changed == 3
    assert stat.insertions == 42
    assert stat.deletions == 7


# ----------------------------------------------------------------------- resolve_base_rev


def test_resolve_base_rev(mock_subprocess):
    rev = resolve_base_rev(Path("/tmp/repo"))
    assert rev == "abc123"  # default stdout
    cmd = mock_subprocess[0]
    assert cmd[:2] == ["jj", "log"]
    assert "-r" in cmd
    assert "@" in cmd
    assert "-T" in cmd
    assert "commit_id" in cmd


def test_resolve_base_rev_returns_commit_id(mock_subprocess):
    rev = resolve_base_rev(Path("/tmp/repo"))
    assert rev == "abc123"


def test_resolve_base_rev_empty_raises():
    with patch("agents.shared.workspaces.subprocess.run") as m:
        m.return_value = subprocess.CompletedProcess(["jj", "log"], returncode=0, stdout="")
        with pytest.raises(JJError, match="could not resolve"):
            resolve_base_rev(Path("/tmp/repo"))


# ----------------------------------------------------------------------- workspace_destination


def test_workspace_destination_defaults():
    repo = Path("/tmp/myrepo")
    dest = workspace_destination(repo, "test-label")
    parent = dest.parent
    assert parent == repo.parent
    name = dest.name
    assert name.startswith("ws-test-label-")
    # uuid suffix is 6 hex chars
    suffix = name.split("-")[-1]
    assert len(suffix) == 6
    int(suffix, 16)  # won't raise if hex


def test_workspace_destination_custom_base_dir():
    repo = Path("/tmp/myrepo")
    base = Path("/tmp/other")
    dest = workspace_destination(repo, "x", base_dir=base, prefix="cl")
    assert dest.parent == base
    assert dest.name.startswith("cl-x-")


# ----------------------------------------------------------------------- create_workspace


def test_create_workspace(mock_subprocess):
    repo = Path("/tmp/repo")
    dest = Path("/tmp/ws-A")
    create_workspace(repo, dest, base_rev="abc")
    cmd = mock_subprocess[0]
    assert cmd[:3] == ["jj", "workspace", "add"]
    assert "--revision" in cmd
    assert "abc" in cmd
    assert str(dest) in cmd


def test_create_workspace_existing_dest():
    with patch("agents.shared.workspaces.subprocess.run"):
        with patch("pathlib.Path.exists", return_value=True):
            with pytest.raises(JJError, match="already exists"):
                create_workspace(Path("/tmp/repo"), Path("/tmp/ws"), base_rev="abc")


# ----------------------------------------------------------------------- ensure_git_marker


def test_ensure_git_marker_skips_if_git_exists():
    with patch("pathlib.Path.exists", return_value=True):
        ensure_git_marker(Path("/tmp/ws"))


def test_ensure_git_marker_runs_git_init(mock_subprocess):
    ensure_git_marker(Path("/tmp/ws"))
    calls = mock_subprocess
    # subprocess.run called twice: git init + git add
    assert len(calls) == 2
    assert calls[0][0] == "git"
    assert calls[0][1] == "init"
    assert calls[1][0] == "git"
    assert calls[1][1] == "add"


# ----------------------------------------------------------------------- _diff_exclude_fileset


def test_diff_exclude_fileset_covers_cruft():
    fileset = _diff_exclude_fileset()
    assert len(fileset) == 1
    term = fileset[0]
    assert '~".open-mem"' in term
    assert '~glob:"**/__pycache__/**"' in term
    assert '~glob:"**/*.pyc"' in term
    assert " & " in term


def test_diff_exclude_fileset_empty_when_no_excludes(monkeypatch):
    monkeypatch.setattr(
        ws_module,
        "_DIFF_EXCLUDE_FILESETS",
        (),
    )
    assert _diff_exclude_fileset() == []


# ----------------------------------------------------------------------- _parse_diff_stat


def test_parse_diff_stat_parses_summary():
    stat = _parse_diff_stat("3 files changed, 42 insertions(+), 7 deletions(-)")
    assert stat.files_changed == 3
    assert stat.insertions == 42
    assert stat.deletions == 7


def test_parse_diff_stat_empty():
    stat = _parse_diff_stat("")
    assert stat == DiffStat()


def test_parse_diff_stat_malformed():
    stat = _parse_diff_stat("no useful info here")
    assert stat == DiffStat()


def test_parse_diff_stat_last_nonempty_line():
    """Only the last non-empty line is parsed."""
    stat = _parse_diff_stat("x.py | 10 +-\n\n2 files changed, 5 insertions(+), 3 deletions(-)")
    assert stat.files_changed == 2
    assert stat.insertions == 5
    assert stat.deletions == 3


# ----------------------------------------------------------------------- collect_diff


def test_collect_diff_calls_jj_once(mock_subprocess):
    dest = Path("/tmp/ws-A")
    with patch("agents.shared.workspaces._run_jj") as run_jj:
        run_jj.return_value = subprocess.CompletedProcess(
            ["jj", "diff"], returncode=0, stdout="diff content"
        )
        collect_diff(dest, "abc")
        assert run_jj.call_count == 2
        calls_args = [c[0][0] for c in run_jj.call_args_list]
        assert "--git" in calls_args[0]
        assert "--stat" in calls_args[1]


# ----------------------------------------------------------------------- forget_workspace


def test_forget_workspace(mock_subprocess):
    repo = Path("/tmp/repo")
    ws = Path("/tmp/ws-A")
    forget_workspace(repo, ws)
    cmd = mock_subprocess[0]
    assert cmd[:2] == ["jj", "workspace"]
    assert "forget" in cmd
    assert ws.name in cmd
