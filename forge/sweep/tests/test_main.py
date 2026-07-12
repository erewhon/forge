"""CLI wiring: flags reach sweep, overrides land in settings, summary renders."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from forge.sweep import main as m
from forge.sweep.config import settings
from forge.sweep.models import AgentRun, SweepResult


@pytest.fixture
def sweep_mock(monkeypatch):
    mock = MagicMock(return_value=(SweepResult(host="h"), 0))
    monkeypatch.setattr(m, "sweep", mock)
    return mock


def test_flags_reach_sweep(sweep_mock, capsys):
    assert m.main(["--dry-run", "--auto-merge"]) == 0
    kw = sweep_mock.call_args[1]
    assert kw["dry_run"] is True and kw["auto_merge"] is True


def test_host_and_only_override_settings(sweep_mock, monkeypatch, capsys):
    monkeypatch.setattr(settings, "host", "old")
    monkeypatch.setattr(settings, "include", ["*"])
    m.main(["--host", "new-host", "--only", "me/*"])
    assert settings.host == "new-host"
    assert settings.include == ["me/*"]


def test_driver_failure_exit_code_passes_through(sweep_mock, capsys):
    sweep_mock.return_value = (SweepResult(errors=["SWEEP_HOST is not set"]), 2)
    assert m.main([]) == 2


def test_render_shows_rows_and_errors():
    result = SweepResult(
        host="code-public",
        repos=["me/one", "me/two"],
        skipped=["other/x"],
        runs=[
            AgentRun(
                repo="me/one",
                agent="deps",
                status="advisory",
                detail="no Go-native provenance evidence source yet",
            ),
            AgentRun(repo="me/one", agent="upstream", status="up-to-date"),
        ],
        errors=["me/two: clone exploded"],
    )
    out = m.render_sweep(result)
    assert "code-public" in out
    assert "2 repo(s) swept, 1 filtered out, 2 agent run(s), 1 repo error(s)" in out
    assert "me/one" in out and "advisory" in out and "up-to-date" in out
    assert "no Go-native provenance" in out
    assert "clone exploded" in out
