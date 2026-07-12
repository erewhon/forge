"""CLI wiring: flags reach sync_upstream, statuses map to exit codes."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from forge.upstream_sync import main as m
from forge.upstream_sync.models import SyncResult


@pytest.fixture
def sync_mock(monkeypatch):
    mock = MagicMock(return_value=SyncResult(status="branched", branch="upstream-sync/x"))
    monkeypatch.setattr(m, "sync_upstream", mock)
    return mock


def test_flags_reach_sync(sync_mock, tmp_path, capsys):
    assert m.main(["--repo", str(tmp_path), "--dry-run", "--auto-merge", "--project", "P"]) == 0
    kw = sync_mock.call_args[1]
    assert kw["dry_run"] is True and kw["auto_merge"] is True and kw["project"] == "P"


@pytest.mark.parametrize(
    ("status", "code"),
    [
        ("up-to-date", 0),
        ("planned", 0),
        ("branched", 0),
        ("merged", 0),
        ("conflict", 1),
        ("advisory", 1),
        ("error", 1),
    ],
)
def test_exit_codes(sync_mock, tmp_path, capsys, status, code):
    sync_mock.return_value = SyncResult(status=status)
    assert m.main(["--repo", str(tmp_path)]) == code


def test_render_tells_the_story(capsys):
    result = SyncResult(
        status="advisory",
        reason="green-suite gate failed",
        branch="upstream-sync/2026-07-12-abcd1234",
        commits_behind=3,
        tests_passed=False,
    )
    out = m.render_sync(result)
    assert "forge upstream — advisory" in out
    assert "3 commit(s)" in out
    assert "Suite: RED" in out
