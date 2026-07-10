"""The Nous daily-note sink is opt-in: default delivery is markdown-only, no Nous imports."""

from __future__ import annotations

import sys

import pytest

from forge.code_reviewer.config import settings
from forge.code_reviewer.main import _deliver
from forge.code_reviewer.models import NightlyReport


@pytest.fixture
def report() -> NightlyReport:
    return NightlyReport(
        date="2026-07-10",
        repos_reviewed=1,
        repos_with_changes=0,
        reviews=[],
        overall_summary="quiet night",
    )


def test_nous_sink_is_off_by_default():
    assert settings.nous_sink is False


def test_default_delivery_prints_markdown_and_never_imports_nous(report, monkeypatch, capsys):
    # Force a clean slate so the lazy-import assertion is meaningful even if another test
    # (or the nous-enabled path) already pulled the writer in.
    monkeypatch.delitem(sys.modules, "forge.code_reviewer.writer", raising=False)
    monkeypatch.delitem(sys.modules, "forge.shared.nous_http", raising=False)
    monkeypatch.setattr(settings, "nous_sink", False)

    _deliver(report)

    out = capsys.readouterr().out
    assert "quiet night" in out  # the rendered markdown reached stdout
    assert "forge.code_reviewer.writer" not in sys.modules
    assert "forge.shared.nous_http" not in sys.modules


def test_dry_run_prints_markdown_without_nous(report, monkeypatch, capsys):
    monkeypatch.delitem(sys.modules, "forge.code_reviewer.writer", raising=False)
    monkeypatch.setattr(settings, "nous_sink", True)  # dry-run wins even with the sink on

    _deliver(report, dry_run=True)

    out = capsys.readouterr().out
    assert "DRY RUN" in out
    assert "quiet night" in out
    assert "forge.code_reviewer.writer" not in sys.modules


def test_enabled_sink_invokes_the_nous_writer(report, monkeypatch, capsys):
    monkeypatch.setattr(settings, "nous_sink", True)
    calls: list[NightlyReport] = []
    monkeypatch.setattr(
        "forge.code_reviewer.writer.append_to_daily_note",
        lambda r: calls.append(r) or {"blocksAdded": 3},
    )

    _deliver(report)

    assert calls == [report]
    assert "3 blocks appended" in capsys.readouterr().out


def test_enabled_sink_reports_skip(report, monkeypatch, capsys):
    monkeypatch.setattr(settings, "nous_sink", True)
    monkeypatch.setattr("forge.code_reviewer.writer.append_to_daily_note", lambda r: None)

    _deliver(report)

    assert "Not appended" in capsys.readouterr().out
