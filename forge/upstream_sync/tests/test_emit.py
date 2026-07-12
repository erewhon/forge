"""Advisory emission: stable dedupe ref, comma-free title, content carries the story."""

from __future__ import annotations

from forge.upstream_sync import emit as em
from forge.upstream_sync.models import (
    CollisionFinding,
    CollisionVerdict,
    LayerManifest,
    SyncResult,
)


class _FakeStore:
    def __init__(self):
        self.calls = []

    def emit(self, specs, **kw):
        self.calls.append((specs, kw))

        class _S:
            def line(self):
                return "emitted 1"

        return _S()


def _result(**over) -> SyncResult:
    base = dict(
        status="advisory",
        reason="green-suite gate failed",
        branch="upstream-sync/2026-07-12-abcd1234",
        upstream_tip="abcd1234deadbeef",
        merge_base="0000aaaa11112222",
        commits_behind=7,
        layer=LayerManifest(added=["SPRINKLES.md"], modified=["README.md"]),
        overlap=["README.md"],
    )
    base.update(over)
    return SyncResult(**base)


def test_emit_files_spec_with_stable_ref_and_clean_title(monkeypatch):
    store = _FakeStore()
    monkeypatch.setattr(em, "get_task_store", lambda: store)
    summary = em.emit_advisory("sprinkles", _result(), "abc fix things", project="P")
    assert summary is not None
    (specs, kw) = store.calls[0]
    spec = specs[0]
    assert kw["project"] == "P"
    assert spec.external_ref == "upstream:sprinkles:abcd1234dead"
    assert "," not in spec.title
    assert "7 commits behind" in spec.title
    assert "upstream-sync/2026-07-12-abcd1234" in spec.content
    assert "abc fix things" in spec.content


def test_conflict_content_lists_files_not_branch(monkeypatch):
    store = _FakeStore()
    monkeypatch.setattr(em, "get_task_store", lambda: store)
    result = _result(
        status="conflict",
        branch=None,
        conflicted=["README.md", "core.txt"],
        reason="textual merge conflict in 2 file(s)",
    )
    em.emit_advisory("sprinkles", result, "", project="P")
    content = store.calls[0][0][0].content
    assert "`README.md`" in content and "`core.txt`" in content
    assert "**Branch:**" not in content
    assert "merge conflict" in store.calls[0][0][0].title


def test_collision_findings_render(monkeypatch):
    store = _FakeStore()
    monkeypatch.setattr(em, "get_task_store", lambda: store)
    result = _result(
        reason="collision seat blocked",
        collision=CollisionVerdict(
            collision=True,
            findings=[CollisionFinding(file="pkg/web/routes.go", reason="layer hooks this")],
            notes="check route registration",
        ),
    )
    em.emit_advisory("sprinkles", result, "", project="P")
    content = store.calls[0][0][0].content
    assert "pkg/web/routes.go" in content and "layer hooks this" in content
    assert "check route registration" in content


def test_no_project_files_nothing(monkeypatch):
    store = _FakeStore()
    monkeypatch.setattr(em, "get_task_store", lambda: store)
    assert em.emit_advisory("sprinkles", _result(), "", project=None) is None
    assert store.calls == []
