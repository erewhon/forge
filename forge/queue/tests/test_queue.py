"""`forge queue` tests — row semantics, report rendering, and CLI wiring.

Rendering is pure (rows in, text out); the CLI test fakes the task store so no backend
is touched. Store-side ``queue()`` behavior lives with each adapter's own test file.
"""

from __future__ import annotations

import pytest

from forge.queue.main import main, render_queue
from forge.queue.models import QueueRow


def _row(**overrides) -> QueueRow:
    base = dict(project="Meta", task="A task", status="Ready", execution_mode="Manual")
    base.update(overrides)
    return QueueRow(**base)


class TestQueueRow:
    def test_dispatchable_is_ready_auto_unblocked(self):
        assert _row(execution_mode="Auto-OK").is_dispatchable
        assert _row(execution_mode="Auto-Preferred").is_dispatchable

    def test_manual_ready_is_not_dispatchable(self):
        assert not _row(execution_mode="Manual").is_dispatchable

    def test_blocked_auto_is_not_dispatchable(self):
        assert not _row(execution_mode="Auto-OK", blocked=True, blocked_by=["dep"]).is_dispatchable

    def test_non_ready_auto_is_not_dispatchable(self):
        assert not _row(execution_mode="Auto-OK", status="Spec Needed").is_dispatchable
        assert not _row(execution_mode="Auto-OK", status="In Progress").is_dispatchable


class TestRenderQueue:
    def test_groups_by_project_alphabetically(self):
        rows = [
            _row(project="Nous", task="n-task"),
            _row(project="Meta", task="m-task"),
        ]
        out = render_queue(rows)
        assert out.index("Meta — 1 open") < out.index("Nous — 1 open")

    def test_rows_sorted_by_priority_then_title(self):
        rows = [
            _row(task="zebra", priority=3),
            _row(task="apple", priority=3),
            _row(task="urgent", priority=1),
        ]
        out = render_queue(rows)
        assert out.index("urgent") < out.index("apple") < out.index("zebra")

    def test_header_counts_open_and_auto_ready(self):
        rows = [
            _row(task="a", execution_mode="Auto-OK"),
            _row(task="b", execution_mode="Auto-OK", blocked=True, blocked_by=["a"]),
            _row(task="c", execution_mode="Manual"),
        ]
        assert "Meta — 3 open, 1 auto-ready" in render_queue(rows)

    def test_auto_row_shows_mode_with_tier(self):
        out = render_queue([_row(execution_mode="Auto-OK", model_tier="sonnet")])
        assert "Auto-OK:sonnet" in out

    def test_manual_row_never_shows_tier(self):
        out = render_queue([_row(execution_mode="Manual", model_tier="sonnet")])
        assert "sonnet" not in out

    def test_feature_shown_in_parens(self):
        out = render_queue([_row(feature="Soft Serve Automation")])
        assert "(Soft Serve Automation)" in out

    def test_blocked_rows_list_blockers(self):
        out = render_queue([_row(blocked=True, blocked_by=["dep-1", "dep-2"])])
        assert "[blocked by: dep-1, dep-2]" in out

    def test_long_blocker_lists_collapse_to_a_count(self):
        row = _row(blocked=True, blocked_by=["d1", "d2", "d3", "d4"])
        out = render_queue([row])
        assert "[blocked by: d1, d2 +2 more]" in out
        assert "d3" not in out

    def test_auto_only_drops_manual_rows_and_empty_projects(self):
        rows = [
            _row(project="Meta", task="auto-task", execution_mode="Auto-OK"),
            _row(project="Meta", task="manual-task"),
            _row(project="Nous", task="only-manual"),
        ]
        out = render_queue(rows, auto_only=True)
        assert "auto-task" in out
        assert "manual-task" not in out
        assert "Nous" not in out
        # the header counts auto-mode rows, not all open rows — say so
        assert "Meta — 1 auto-mode, 1 auto-ready" in out

    def test_empty_messages(self):
        assert render_queue([]) == "No open tasks."
        assert render_queue([_row()], auto_only=True) == "No auto-mode tasks open."

    def test_missing_project_gets_a_placeholder_group(self):
        assert "(no project) — 1 open" in render_queue([_row(project="")])


class TestMain:
    @pytest.fixture
    def fake_store(self, monkeypatch):
        class _Store:
            def __init__(self):
                self.seen: dict = {}
                self.rows: list[QueueRow] = []

            def queue(self, *, project=None):
                self.seen["project"] = project
                return self.rows

        store = _Store()
        monkeypatch.setattr("forge.queue.main.get_task_store", lambda: store)
        return store

    def test_prints_report_and_exits_zero(self, fake_store, capsys):
        fake_store.rows = [_row(task="the-task")]
        assert main([]) == 0
        assert "the-task" in capsys.readouterr().out

    def test_project_flag_narrows_the_store_query(self, fake_store, capsys):
        assert main(["--project", "Meta"]) == 0
        assert fake_store.seen["project"] == "Meta"

    def test_auto_flag_filters(self, fake_store, capsys):
        fake_store.rows = [
            _row(task="auto-task", execution_mode="Auto-OK"),
            _row(task="manual-task"),
        ]
        assert main(["--auto"]) == 0
        out = capsys.readouterr().out
        assert "auto-task" in out
        assert "manual-task" not in out
