"""GitHubTaskStore tests — the issue-backed adapter, driven by an in-memory fake gh client.

No subprocess, no network: ``_FakeGh`` stands in for the ``gh`` CLI, so every op is exercised
against real issue state. Covers the pure meta block helpers, emission (create / dedup / dry-run
/ cap / per-spec overrides), status transitions (label swap, Done-closes, reopen, meta rewrite),
and the read path (find / next_ready ordering + blocked exclusion / worker_gate / list_rows /
in_progress_titles).
"""

from __future__ import annotations

import pytest

from forge.shared import github_task_store as gh_store
from forge.shared import task_store
from forge.shared.forge_emit import EmitSpec
from forge.shared.github_task_store import (
    _STATUS_TO_LABEL,
    GitHubTaskStore,
    _Issue,
    format_meta_block,
    parse_meta_block,
    set_meta_field,
    strip_meta_block,
)


class _FakeGh:
    """In-memory stand-in for the gh CLI, bound to one repo."""

    def __init__(self) -> None:
        self.issues: dict[int, _Issue] = {}
        self._next = 1
        self.comments: list[tuple[int, str]] = []
        self.labels_ensured: set[str] = set()

    def list_issues(self, *, state: str, label: str | None = None) -> list[_Issue]:
        out = []
        for issue in self.issues.values():
            if state == "open" and issue.state != "open":
                continue
            if state == "closed" and issue.state != "closed":
                continue
            if label and label not in issue.labels:
                continue
            out.append(issue)
        return out

    def create_issue(self, *, title: str, body: str, labels: list[str]) -> int:
        n = self._next
        self._next += 1
        self.issues[n] = _Issue(number=n, title=title, body=body, state="open", labels=list(labels))
        return n

    def edit_labels(self, number: int, *, add: list[str], remove: list[str]) -> None:
        issue = self.issues[number]
        labels = [lbl for lbl in issue.labels if lbl not in remove]
        for lbl in add:
            if lbl not in labels:
                labels.append(lbl)
        issue.labels = labels

    def edit_body(self, number: int, body: str) -> None:
        self.issues[number].body = body

    def close_issue(self, number: int) -> None:
        self.issues[number].state = "closed"

    def reopen_issue(self, number: int) -> None:
        self.issues[number].state = "open"

    def comment(self, number: int, body: str) -> None:
        self.comments.append((number, body))

    def ensure_label(self, name: str, *, color: str, description: str) -> None:
        self.labels_ensured.add(name)


def _store(gh: _FakeGh | None = None) -> tuple[GitHubTaskStore, _FakeGh]:
    gh = gh or _FakeGh()
    return GitHubTaskStore(gh=gh, project="Meta"), gh


def _make(
    gh: _FakeGh,
    title: str,
    *,
    status="Ready",
    mode="Auto-OK",
    priority=3,
    deps="",
    ref=None,
    feature="",
) -> int:
    fields = {
        "external_ref": ref or f"pipeline:toy:{title}",
        "execution_mode": mode,
        "priority": str(priority),
    }
    if deps:
        fields["depends_on"] = deps
    if feature:
        fields["feature"] = feature
    n = gh.create_issue(
        title=title, body=f"{format_meta_block(fields)}\n\nspec for {title}", labels=[]
    )
    st = status.lower()
    if st == "done":
        gh.close_issue(n)
    elif label := _STATUS_TO_LABEL.get(st):
        gh.edit_labels(n, add=[label], remove=[])
    return n


# --- pure meta block helpers -----------------------------------------------


def test_meta_block_round_trips():
    fields = {
        "external_ref": "pipeline:e:a",
        "execution_mode": "Auto-OK",
        "priority": "4",
        "depends_on": "Add units, Wire flag",
    }
    body = f"{format_meta_block(fields)}\n\nthe spec body here"
    assert parse_meta_block(body) == fields
    assert strip_meta_block(body) == "the spec body here"


def test_set_meta_field_preserves_spec_and_other_fields():
    body = f"{format_meta_block({'external_ref': 'r', 'execution_mode': 'Auto-OK'})}\n\nspec"
    updated = set_meta_field(body, "execution_mode", "Manual")
    assert parse_meta_block(updated)["execution_mode"] == "Manual"
    assert parse_meta_block(updated)["external_ref"] == "r"
    assert strip_meta_block(updated) == "spec"


def test_parse_meta_block_absent_is_empty():
    assert parse_meta_block("just a plain body, no meta") == {}
    assert strip_meta_block("just a plain body") == "just a plain body"


# --- emission --------------------------------------------------------------


def test_emit_creates_issue_with_meta_and_status_label():
    store, gh = _store()
    spec = EmitSpec(
        title="Add parser",
        content="write the parser",
        external_ref="pipeline:e:add-parser",
        task_type="feature",
        status="Ready",
        execution_mode="Auto-OK",
        priority=4,
        max_files=3,
        requires_tests=True,
        depends_on="Add units",
    )
    summary = store.emit([spec], project="Meta")
    assert len(summary.created) == 1
    issue = gh.issues[1]
    assert issue.title == "Add parser"
    assert "status:ready" in issue.labels
    meta = parse_meta_block(issue.body)
    assert meta["external_ref"] == "pipeline:e:add-parser"
    assert meta["execution_mode"] == "Auto-OK"
    assert meta["requires_tests"] == "true"
    assert meta["depends_on"] == "Add units"
    assert strip_meta_block(issue.body) == "write the parser"


def test_emit_dedups_by_external_ref():
    store, gh = _store()
    _make(gh, "Existing", ref="pipeline:e:dup")
    spec = EmitSpec(title="New title", content="c", external_ref="pipeline:e:dup", status="Ready")
    summary = store.emit([spec], project="Meta")
    assert summary.created == []
    assert len(summary.skipped) == 1
    assert len(gh.issues) == 1  # nothing created


def test_emit_bootstraps_status_labels():
    store, gh = _store()
    spec = EmitSpec(title="Bootstrap", content="c", external_ref="pipeline:e:b", status="Ready")
    store.emit([spec], project="Meta")
    assert gh.labels_ensured == {
        "status:spec-needed",
        "status:ready",
        "status:in-progress",
        "status:done",
    }


def test_emit_dry_run_creates_nothing_and_touches_no_labels():
    store, gh = _store()
    spec = EmitSpec(title="Planned", content="c", external_ref="pipeline:e:p", status="Ready")
    summary = store.emit([spec], project="Meta", dry_run=True)
    assert len(summary.planned) == 1
    assert gh.issues == {}
    assert gh.labels_ensured == set()  # dry-run must not mutate the repo


def test_emit_respects_cap():
    store, gh = _store()
    specs = [
        EmitSpec(title=f"L{i}", content="c", external_ref=f"pipeline:e:{i}", status="Ready")
        for i in range(4)
    ]
    summary = store.emit(specs, project="Meta", max_per_run=2)
    assert len(summary.created) == 2
    assert summary.capped == 2


# --- status transitions ----------------------------------------------------


def test_update_status_to_done_closes_and_comments():
    store, gh = _store()
    n = _make(gh, "Ship it", status="In Progress")
    store.update_status("Ship it", "Done", notes="all green")
    assert gh.issues[n].state == "closed"
    assert "status:in-progress" not in gh.issues[n].labels
    assert gh.comments and "all green" in gh.comments[-1][1]


def test_update_status_ready_reopens_and_swaps_label():
    store, gh = _store()
    n = _make(gh, "Redo", status="Done")  # closed
    store.update_status("Redo", "Ready")
    assert gh.issues[n].state == "open"
    assert "status:ready" in gh.issues[n].labels


def test_update_status_rewrites_execution_mode_in_meta():
    store, gh = _store()
    n = _make(gh, "Escalate me", status="Ready", mode="Auto-OK")
    store.update_status("Escalate me", "Spec Needed", execution_mode="Manual")
    assert parse_meta_block(gh.issues[n].body)["execution_mode"] == "Manual"
    assert "status:spec-needed" in gh.issues[n].labels


def test_update_status_missing_issue_raises():
    store, _ = _store()
    with pytest.raises(ValueError, match="issue not found"):
        store.update_status("ghost", "Ready")


# --- read path -------------------------------------------------------------


def test_find_task_builds_taskinfo():
    store, gh = _store()
    _make(gh, "Parser", status="Ready", mode="Auto-OK", priority=2, deps="Units")
    info = store.find_task("Parser")
    assert info is not None
    assert info.task == "Parser"
    assert info.project == "Meta"
    assert info.execution_mode == "Auto-OK"
    assert info.deps == ["Units"]
    assert store.find_task("nope") is None


def test_next_ready_prefers_then_priority_and_excludes_blocked_and_manual():
    store, gh = _store()
    _make(gh, "pref", status="Ready", mode="Auto-Preferred", priority=5)
    _make(gh, "ok1", status="Ready", mode="Auto-OK", priority=1)
    _make(gh, "manual", status="Ready", mode="Manual", priority=0)
    _make(gh, "blocked", status="Ready", mode="Auto-OK", priority=0, deps="ok1")
    pick = store.next_ready(["Meta"])
    assert pick is not None and pick.task == "pref"  # Auto-Preferred leads


def test_next_ready_unblocks_when_dependency_done():
    store, gh = _store()
    _make(gh, "dep", status="Done")
    _make(gh, "leaf", status="Ready", mode="Auto-OK", priority=1, deps="dep")
    pick = store.next_ready(["Meta"])
    assert pick is not None and pick.task == "leaf"


def test_next_ready_project_filter_excludes_other_projects():
    store, gh = _store()
    _make(gh, "leaf", status="Ready")
    assert store.next_ready(["Nous"]) is None  # this repo is "Meta"
    assert store.next_ready([]) is not None  # empty = no narrowing


def test_worker_gate_allows_ready_auto_unblocked_else_reason():
    store, gh = _store()
    _make(gh, "go", status="Ready", mode="Auto-OK")
    _make(gh, "manual", status="Ready", mode="Manual")
    _make(gh, "stuck", status="Ready", mode="Auto-OK", deps="go")
    assert store.worker_gate("go") == ""
    assert "not Auto" in store.worker_gate("manual")
    assert "blocked" in store.worker_gate("stuck").lower()
    assert "not found" in store.worker_gate("ghost")


def test_get_spec_includes_metadata_and_blocked_guardrail():
    store, gh = _store()
    _make(gh, "dep", status="Ready")  # not Done → blocks
    _make(gh, "leaf", status="Ready", mode="Auto-OK", priority=4, deps="dep")
    spec = store.get_spec("leaf")
    assert "## Task Metadata" in spec
    assert "**Status:** Ready" in spec
    assert "> **Blocked:**" in spec  # dep not Done
    assert "spec for leaf" in spec  # the body, meta block stripped


def test_list_rows_resolves_blocked_and_filters():
    store, gh = _store()
    _make(gh, "done-dep", status="Done", feature="A")
    _make(gh, "free", status="Ready", feature="A")
    _make(gh, "stuck", status="Ready", deps="free", feature="B")
    rows = {r.task: r for r in store.list_rows("Meta")}
    assert rows["stuck"].blocked and rows["stuck"].blocked_by == ["free"]
    assert not rows["free"].blocked
    # feature filter
    feat_a = store.list_rows("Meta", feature="A")
    assert {r.task for r in feat_a} == {"done-dep", "free"}
    # include_done=False drops the closed one
    assert "done-dep" not in {r.task for r in store.list_rows("Meta", include_done=False)}


def test_in_progress_titles_filters_by_ref_prefix():
    store, gh = _store()
    _make(gh, "mine", status="In Progress", ref="pipeline:toy-epic:mine")
    _make(gh, "other", status="In Progress", ref="pipeline:other:x")
    _make(gh, "ready", status="Ready", ref="pipeline:toy-epic:ready")  # not in progress
    assert store.in_progress_titles("pipeline:toy-epic:") == ["mine"]


# --- factory wiring --------------------------------------------------------


def test_factory_selects_github_backend(monkeypatch):
    monkeypatch.setattr(task_store.settings, "backend", "github")
    monkeypatch.setattr(gh_store.settings, "repo", "owner/repo")
    store = task_store.get_task_store()
    assert isinstance(store, GitHubTaskStore)


def test_construction_requires_repo(monkeypatch):
    monkeypatch.setattr(gh_store.settings, "repo", "")
    with pytest.raises(ValueError, match="GITHUB_TASK_STORE_REPO"):
        GitHubTaskStore()


# --- least agency: GhReader / GhWriter split ----------------------------------------


class _ReaderOnlyGh:
    """A GhReader and nothing more — proves the read paths never touch write surface."""

    def __init__(self, gh: _FakeGh) -> None:
        self._gh = gh

    def list_issues(self, *, state: str, label: str | None = None):
        return self._gh.list_issues(state=state, label=label)


def _read_only_store() -> tuple[GitHubTaskStore, _FakeGh]:
    gh = _FakeGh()
    return GitHubTaskStore(reader=_ReaderOnlyGh(gh), project="Meta"), gh


def test_every_read_path_works_with_a_reader_only_client():
    store, gh = _read_only_store()
    _make(gh, "leaf-a", status="Ready", mode="Auto-OK")
    _make(gh, "leaf-b", status="In Progress", ref="pipeline:toy:leaf-b")

    assert store.find_task("leaf-a").task == "leaf-a"
    assert store.next_ready(["Meta"]).task == "leaf-a"
    assert {r.task for r in store.list_rows("Meta")} == {"leaf-a", "leaf-b"}
    assert store.worker_gate("leaf-a") == ""
    assert "leaf-a" in store.get_spec("leaf-a")
    assert store.in_progress_titles("pipeline:toy:") == ["leaf-b"]


def test_read_only_store_write_fails_fast_with_the_reason():
    store, gh = _read_only_store()
    _make(gh, "leaf-a", status="Ready")
    with pytest.raises(PermissionError, match="read-only.*edit_labels"):
        store.update_status("leaf-a", "Done")
    with pytest.raises(PermissionError, match="read-only"):
        store.ensure_labels()


def test_gh_and_reader_are_mutually_exclusive():
    gh = _FakeGh()
    with pytest.raises(ValueError, match="not both"):
        GitHubTaskStore(gh=gh, reader=_ReaderOnlyGh(gh), project="Meta")


def test_read_only_emit_dry_run_works_but_create_fails_fast():
    # emit's dedup listing is a read; dry-run never writes, so it must succeed on a
    # reader-only store — the write capability is only demanded when actually creating.
    from forge.shared.forge_emit import EmitSpec

    store, gh = _read_only_store()
    spec = EmitSpec(external_ref="x:1", title="t", content="c", task_type="chore")
    summary = store.emit([spec], project="Meta", dry_run=True)
    assert len(summary.planned) == 1
    with pytest.raises(PermissionError, match="read-only"):
        store.emit([spec], project="Meta")
