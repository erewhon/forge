"""GitBugTaskStore tests — an in-memory git-bug fake, no binary, no repo.

Mirrors the GitHubTaskStore suite: same conventions (task_conventions), same TaskStore
surface, plus the reader/writer least-agency split the backend was born with.
"""

from __future__ import annotations

import pytest

from forge.shared.forge_emit import EmitSpec
from forge.shared.git_bug_task_store import (
    GitBugTaskStore,
    _Bug,
)
from forge.shared.task_conventions import format_meta_block


class _FakeGitBug:
    """In-memory stand-in for the git-bug CLI, bound to one repo."""

    def __init__(self) -> None:
        self.bugs: dict[str, _Bug] = {}
        self._next = 1
        self.comments: list[tuple[str, str]] = []

    # --- reader ---
    def list_bugs(self) -> list[_Bug]:
        # summaries: no description/first_comment_id, like `git-bug bug --format json`
        return [
            _Bug(id=b.id, title=b.title, state=b.state, labels=list(b.labels))
            for b in self.bugs.values()
        ]

    def show_bug(self, bug_id: str) -> _Bug:
        return self.bugs[bug_id]

    # --- writer ---
    def create_bug(self, *, title: str, body: str) -> str:
        bug_id = f"bug-{self._next}"
        self._next += 1
        self.bugs[bug_id] = _Bug(
            id=bug_id,
            title=title,
            state="open",
            labels=[],
            description=body,
            first_comment_id=f"{bug_id}-c0",
            comments=[body],
        )
        return bug_id

    def add_label(self, bug_id: str, label: str) -> None:
        if label not in self.bugs[bug_id].labels:
            self.bugs[bug_id].labels.append(label)

    def remove_label(self, bug_id: str, label: str) -> None:
        self.bugs[bug_id].labels = [lbl for lbl in self.bugs[bug_id].labels if lbl != label]

    def close_bug(self, bug_id: str) -> None:
        self.bugs[bug_id].state = "closed"

    def open_bug(self, bug_id: str) -> None:
        self.bugs[bug_id].state = "open"

    def add_comment(self, bug_id: str, body: str) -> None:
        self.comments.append((bug_id, body))
        self.bugs[bug_id].comments.append(body)

    def edit_comment(self, comment_id: str, body: str) -> None:
        for bug in self.bugs.values():
            if bug.first_comment_id == comment_id:
                bug.description = body
                bug.comments[0] = body
                return
        raise KeyError(comment_id)


def _store(gb: _FakeGitBug | None = None) -> tuple[GitBugTaskStore, _FakeGitBug]:
    gb = gb or _FakeGitBug()
    return GitBugTaskStore(gb=gb, project="Meta"), gb


def _make(
    gb: _FakeGitBug,
    title: str,
    *,
    status: str = "Ready",
    mode: str = "Auto-OK",
    priority: int = 3,
    deps: str = "",
    ref: str = "",
    feature: str = "",
    body: str = "Spec body",
) -> str:
    fields = {
        "external_ref": ref or f"ref:{title}",
        "execution_mode": mode,
        "priority": str(priority),
    }
    if deps:
        fields["depends_on"] = deps
    if feature:
        fields["feature"] = feature
    bug_id = gb.create_bug(title=title, body=f"{format_meta_block(fields)}\n\n{body}")
    label = {
        "spec needed": "status:spec-needed",
        "ready": "status:ready",
        "in progress": "status:in-progress",
    }.get(status.lower())
    if label:
        gb.add_label(bug_id, label)
    if status.lower() == "done":
        gb.close_bug(bug_id)
    return bug_id


# --- emit ---------------------------------------------------------------------------


def test_emit_creates_bugs_with_meta_and_status_label():
    store, gb = _store()
    spec = EmitSpec(
        external_ref="pipeline:toy:leaf-a",
        title="leaf-a",
        content="Do the thing",
        task_type="feature",
        status="Ready",
        execution_mode="Auto-OK",
        priority=2,
    )
    summary = store.emit([spec], project="Meta")
    assert len(summary.created) == 1
    bug = next(iter(gb.bugs.values()))
    assert bug.title == "leaf-a"
    assert "status:ready" in bug.labels
    assert "external_ref: pipeline:toy:leaf-a" in bug.description
    assert "Do the thing" in bug.description


def test_emit_dedupes_by_external_ref():
    store, gb = _store()
    _make(gb, "existing", ref="pipeline:toy:leaf-a")
    spec = EmitSpec(
        external_ref="pipeline:toy:leaf-a", title="renamed", content="c", task_type="chore"
    )
    summary = store.emit([spec], project="Meta")
    assert len(summary.skipped) == 1 and not summary.created


def test_emit_respects_the_cap():
    store, _ = _store()
    specs = [
        EmitSpec(external_ref=f"r:{i}", title=f"t{i}", content="c", task_type="chore")
        for i in range(4)
    ]
    summary = store.emit(specs, project="Meta", max_per_run=2)
    assert len(summary.created) == 2 and summary.capped == 2


def test_emit_dry_run_creates_nothing():
    store, gb = _store()
    spec = EmitSpec(external_ref="r:1", title="t", content="c", task_type="chore")
    summary = store.emit([spec], project="Meta", dry_run=True)
    assert len(summary.planned) == 1 and not gb.bugs


# --- status round-trip ----------------------------------------------------------------


def test_done_closes_and_strips_status_labels():
    store, gb = _store()
    bug_id = _make(gb, "leaf-a", status="In Progress")
    store.update_status("leaf-a", "Done", notes="landed")
    bug = gb.bugs[bug_id]
    assert bug.state == "closed"
    assert not [lbl for lbl in bug.labels if lbl.startswith("status:")]
    assert gb.comments and "landed" in gb.comments[0][1]


def test_reopen_from_done_swaps_label_and_reopens():
    store, gb = _store()
    bug_id = _make(gb, "leaf-a", status="Done")
    store.update_status("leaf-a", "Ready")
    bug = gb.bugs[bug_id]
    assert bug.state == "open"
    assert "status:ready" in bug.labels


def test_escalation_updates_execution_mode_in_meta():
    store, gb = _store()
    bug_id = _make(gb, "leaf-a", status="Ready", mode="Auto-OK")
    store.update_status("leaf-a", "Spec Needed", execution_mode="Manual")
    bug = gb.bugs[bug_id]
    assert "execution_mode: Manual" in bug.description
    assert "status:spec-needed" in bug.labels
    assert store.find_task("leaf-a").execution_mode == "Manual"


def test_update_status_unknown_task_raises():
    store, _ = _store()
    with pytest.raises(ValueError, match="not found"):
        store.update_status("ghost", "Done")


# --- next_ready / worker_gate ----------------------------------------------------------


def test_next_ready_dep_gating_and_priority_order():
    store, gb = _store()
    _make(gb, "blocked-leaf", priority=1, deps="not-done-dep")
    _make(gb, "not-done-dep", status="In Progress")
    _make(gb, "low-priority", priority=5)
    _make(gb, "high-priority", priority=2)
    picked = store.next_ready(["Meta"])
    assert picked is not None and picked.task == "high-priority"


def test_next_ready_unblocks_when_dep_is_done():
    store, gb = _store()
    _make(gb, "leaf-a", priority=1, deps="the-dep")
    _make(gb, "the-dep", status="Done")
    picked = store.next_ready(["Meta"])
    assert picked is not None and picked.task == "leaf-a"


def test_next_ready_skips_manual_and_wrong_project():
    store, gb = _store()
    _make(gb, "manual-leaf", mode="Manual")
    assert store.next_ready(["Meta"]) is None
    assert store.next_ready(["OtherProject"]) is None


def test_worker_gate_fails_closed_on_blockers():
    store, gb = _store()
    _make(gb, "leaf-a", deps="missing-dep")
    reason = store.worker_gate("leaf-a")
    assert "missing-dep" in reason
    assert store.worker_gate("ghost") != ""


def test_worker_gate_clean_for_ready_auto_unblocked():
    store, gb = _store()
    _make(gb, "leaf-a")
    assert store.worker_gate("leaf-a") == ""


# --- list_rows / get_spec / in_progress -------------------------------------------------


def test_list_rows_parses_status_blockers_and_feature_filter():
    store, gb = _store()
    _make(gb, "leaf-a", feature="F1", deps="leaf-b")
    _make(gb, "leaf-b", status="Done", feature="F2")
    rows = {r.task: r for r in store.list_rows("Meta")}
    assert rows["leaf-a"].blocked is False  # dep is Done
    assert rows["leaf-b"].status == "Done"
    only_f1 = store.list_rows("Meta", feature="F1")
    assert [r.task for r in only_f1] == ["leaf-a"]
    no_done = store.list_rows("Meta", include_done=False)
    assert [r.task for r in no_done] == ["leaf-a"]


def test_get_spec_carries_metadata_and_body():
    store, gb = _store()
    _make(gb, "leaf-a", deps="leaf-b", body="The actual spec")
    _make(gb, "leaf-b", status="Done")
    spec = store.get_spec("leaf-a")
    assert "The actual spec" in spec
    assert "leaf-b: done" in spec
    assert "pipeline-meta" not in spec  # meta block stripped from the body section


def test_in_progress_titles_filters_by_ref_prefix():
    store, gb = _store()
    _make(gb, "leaf-a", status="In Progress", ref="pipeline:toy:leaf-a")
    _make(gb, "leaf-b", status="In Progress", ref="pipeline:other:leaf-b")
    _make(gb, "leaf-c", status="Ready", ref="pipeline:toy:leaf-c")
    assert store.in_progress_titles("pipeline:toy:") == ["leaf-a"]


# --- least agency: reader/writer split ---------------------------------------------------


class _ReaderOnlyGitBug:
    def __init__(self, gb: _FakeGitBug) -> None:
        self._gb = gb

    def list_bugs(self):
        return self._gb.list_bugs()

    def show_bug(self, bug_id: str):
        return self._gb.show_bug(bug_id)


def test_every_read_path_works_with_a_reader_only_client():
    gb = _FakeGitBug()
    store = GitBugTaskStore(reader=_ReaderOnlyGitBug(gb), project="Meta")
    _make(gb, "leaf-a")
    _make(gb, "leaf-b", status="In Progress", ref="pipeline:toy:leaf-b")
    assert store.find_task("leaf-a").task == "leaf-a"
    assert store.next_ready(["Meta"]).task == "leaf-a"
    assert {r.task for r in store.list_rows("Meta")} == {"leaf-a", "leaf-b"}
    assert store.worker_gate("leaf-a") == ""
    assert "Spec body" in store.get_spec("leaf-a")
    assert store.in_progress_titles("pipeline:toy:") == ["leaf-b"]


def test_read_only_store_write_fails_fast():
    gb = _FakeGitBug()
    store = GitBugTaskStore(reader=_ReaderOnlyGitBug(gb), project="Meta")
    _make(gb, "leaf-a")
    with pytest.raises(PermissionError, match="read-only"):
        store.update_status("leaf-a", "Done")
    spec = EmitSpec(external_ref="r:1", title="t", content="c", task_type="chore")
    assert len(store.emit([spec], project="Meta", dry_run=True).planned) == 1
    with pytest.raises(PermissionError, match="read-only"):
        store.emit([spec], project="Meta")


def test_gb_and_reader_are_mutually_exclusive():
    gb = _FakeGitBug()
    with pytest.raises(ValueError, match="not both"):
        GitBugTaskStore(gb=gb, reader=_ReaderOnlyGitBug(gb), project="Meta")


# --- factory ------------------------------------------------------------------------------


def test_factory_returns_git_bug_store(monkeypatch):
    from forge.shared import task_store as ts

    monkeypatch.setattr(ts.settings, "backend", "git-bug")
    store = ts.get_task_store()
    assert isinstance(store, GitBugTaskStore)


def test_factory_unknown_backend_names_git_bug(monkeypatch):
    from forge.shared import task_store as ts

    monkeypatch.setattr(ts.settings, "backend", "jira")
    with pytest.raises(ValueError, match="git-bug"):
        ts.get_task_store()


def test_queue_filters_done_and_carries_project_feature_blockers():
    store, gb = _store()
    _make(gb, "shipped", status="Done", feature="A")
    _make(gb, "free", status="Ready", mode="Auto-OK", feature="A")
    _make(gb, "stuck", status="Ready", mode="Manual", deps="free", feature="B")
    rows = {r.task: r for r in store.queue()}
    assert set(rows) == {"free", "stuck"}  # Done rows never appear
    assert rows["free"].project == "Meta"
    assert rows["free"].feature == "A"
    assert rows["free"].is_dispatchable
    assert rows["stuck"].blocked and rows["stuck"].blocked_by == ["free"]
    assert not rows["stuck"].is_dispatchable


def test_queue_is_single_project():
    store, gb = _store()
    _make(gb, "leaf", status="Ready")
    assert store.queue(project="meta")  # case-insensitive match on the store's project
    assert store.queue(project="Other") == []
