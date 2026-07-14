"""Leaf provenance notes — payload shape and note writing on a real git repo, plus a real-jj
integration proving the note lands on the POST-rebase commit (change ids are stable, shas are not).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from forge.coding_pipeline import provenance as prov
from forge.coding_pipeline.models import LeafOutcome
from forge.shared.git_notes import read_note
from forge.task_worker.models import TaskInfo


def _git(repo: Path, *args: str) -> str:
    res = subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", *args],
        capture_output=True,
        text=True,
        cwd=repo,
    )
    assert res.returncode == 0, res.stderr
    return res.stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "-b", "main")
    (tmp_path / "a.txt").write_text("one\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-m", "auto: leaf one")
    return tmp_path


def _task(**over) -> TaskInfo:
    base = dict(
        id="1",
        task="Leaf one",
        project="Demo",
        status="Ready",
        priority=3,
        execution_mode="Auto-OK",
        model_tier="auto-full",
        task_type="feature",
        complexity="routine",
        external_ref="pipeline:demo:leaf-one",
    )
    base.update(over)
    return TaskInfo(**base)


def _done(repo: Path) -> LeafOutcome:
    return LeafOutcome(
        leaf="Leaf one",
        status="done",
        commit_id=_git(repo, "rev-parse", "--short", "HEAD"),
        changed_files=["a.txt"],
        duration_s=12.5,
    )


# --- pure helpers ---------------------------------------------------------------------


def test_spec_sha256_is_deterministic_and_prefixed():
    assert prov.spec_sha256("hello") == prov.spec_sha256("hello")
    assert prov.spec_sha256("hello") != prov.spec_sha256("world")
    assert prov.spec_sha256("x").startswith("sha256:")


def test_payload_carries_outcome_and_task_fields():
    payload = prov.leaf_note_payload(
        _done_stub(), _task(), wave=3, spec_sha="sha256:abc", timestamp="2026-07-13T00:00:00+00:00"
    )
    assert payload["schema"] == 1
    assert payload["kind"] == "leaf"
    assert payload["leaf"] == "Leaf one"
    assert payload["wave"] == 3
    assert payload["spec_sha256"] == "sha256:abc"
    assert payload["changed_files"] == ["a.txt"]
    assert payload["duration_s"] == 12.5
    assert payload["external_ref"] == "pipeline:demo:leaf-one"
    assert payload["model_tier"] == "auto-full"
    assert payload["timestamp"] == "2026-07-13T00:00:00+00:00"


def test_payload_omits_routing_fields_when_task_is_unknown():
    payload = prov.leaf_note_payload(
        _done_stub(), None, wave=1, spec_sha=None, timestamp="2026-07-13T00:00:00+00:00"
    )
    assert "external_ref" not in payload
    assert "model_tier" not in payload
    assert payload["spec_sha256"] is None  # still recorded, explicitly absent


def _done_stub() -> LeafOutcome:
    return LeafOutcome(
        leaf="Leaf one",
        status="done",
        commit_id="deadbeef",
        changed_files=["a.txt"],
        duration_s=12.5,
    )


# --- resolve + write on a real git repo -----------------------------------------------


def test_resolve_git_commit_expands_short_sha(repo: Path):
    short = _git(repo, "rev-parse", "--short", "HEAD")
    assert prov.resolve_git_commit(repo, short) == _git(repo, "rev-parse", "HEAD")


def test_resolve_git_commit_returns_none_for_unknown(repo: Path):
    assert prov.resolve_git_commit(repo, "0000000000000000000000000000000000000000") is None
    assert prov.resolve_git_commit(repo, "") is None


def test_write_leaf_note_round_trips_on_git(repo: Path):
    sha = prov.write_leaf_note(
        repo,
        _done(repo),
        _task(),
        wave=2,
        spec_sha="sha256:abc",
        timestamp="2026-07-13T00:00:00+00:00",
        log=lambda m: None,
    )
    assert sha == _git(repo, "rev-parse", "HEAD")
    payload = read_note(repo, prov.LEAF_NOTE_REF, sha)
    assert payload["leaf"] == "Leaf one"
    assert payload["external_ref"] == "pipeline:demo:leaf-one"
    assert payload["wave"] == 2


def test_write_leaf_note_unresolvable_commit_warns_and_skips(repo: Path):
    warnings: list[str] = []
    bad = LeafOutcome(leaf="ghost", status="done", commit_id="0" * 40)
    sha = prov.write_leaf_note(
        repo,
        bad,
        _task(),
        wave=1,
        spec_sha=None,
        timestamp="2026-07-13T00:00:00+00:00",
        log=warnings.append,
    )
    assert sha is None
    assert any("cannot resolve commit" in w for w in warnings)


# --- record over a wave's outcomes ----------------------------------------------------


def test_record_writes_notes_only_for_landed_leaves(repo: Path):
    landed = _done(repo)
    failed = LeafOutcome(leaf="Leaf two", status="failed", reason="demoted", commit_id=None)
    skipped = LeafOutcome(leaf="Leaf three", status="skipped", commit_id=None)

    prov.record_leaf_provenance(
        repo,
        [landed, failed, skipped],
        find=lambda title: _task() if title == "Leaf one" else None,
        fetch_spec=lambda title: f"spec for {title}",
        wave=4,
        timestamp="2026-07-13T00:00:00+00:00",
        log=lambda m: None,
    )
    head = _git(repo, "rev-parse", "HEAD")
    payload = read_note(repo, prov.LEAF_NOTE_REF, head)
    assert payload is not None
    assert payload["leaf"] == "Leaf one"
    assert payload["wave"] == 4
    assert payload["spec_sha256"] == prov.spec_sha256("spec for Leaf one")


def test_record_degrades_when_lookups_fail(repo: Path):
    """A note still lands (no routing fields / spec hash) when find and fetch_spec both raise."""

    def boom(title: str):
        raise RuntimeError("store down")

    prov.record_leaf_provenance(
        repo,
        [_done(repo)],
        find=boom,
        fetch_spec=boom,
        wave=1,
        timestamp="2026-07-13T00:00:00+00:00",
        log=lambda m: None,
    )
    payload = read_note(repo, prov.LEAF_NOTE_REF, _git(repo, "rev-parse", "HEAD"))
    assert payload["leaf"] == "Leaf one"
    assert payload["spec_sha256"] is None
    assert "external_ref" not in payload  # task unknown → routing fields omitted


# --- real jj: the note lands on the POST-rebase commit --------------------------------


needs_jj = pytest.mark.skipif(shutil.which("jj") is None, reason="jj not on PATH")


def _jj(args: list[str], cwd: Path) -> str:
    env = os.environ | {"JJ_USER": "test", "JJ_EMAIL": "test@test.invalid"}
    proc = subprocess.run(["jj", *args], cwd=cwd, capture_output=True, text=True, env=env)
    assert proc.returncode == 0, f"jj {args}: {proc.stderr}"
    return proc.stdout.strip()


@needs_jj
def test_note_lands_on_post_rebase_commit(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _jj(["git", "init"], cwd=repo)
    (repo / "base.txt").write_text("base\n")
    _jj(["describe", "-m", "base"], cwd=repo)
    base = _jj(["log", "-r", "@", "--no-graph", "-T", "commit_id"], cwd=repo)

    # A landed leaf: a described sibling off base, addressed by its stable change id.
    _jj(["new", base], cwd=repo)
    (repo / "leaf.txt").write_text("leaf\n")
    _jj(["describe", "-m", "auto: the leaf"], cwd=repo)
    change = _jj(["log", "-r", "@", "--no-graph", "-T", "change_id.short()"], cwd=repo)
    sha_before = _jj(["log", "-r", change, "--no-graph", "-T", "commit_id"], cwd=repo)

    # Advance base and rebase the leaf onto it: the change id is unchanged, the git sha is not.
    _jj(["new", base], cwd=repo)
    (repo / "base.txt").write_text("base moved\n")
    _jj(["describe", "-m", "base2"], cwd=repo)
    base2 = _jj(["log", "-r", "@", "--no-graph", "-T", "change_id.short()"], cwd=repo)
    _jj(["rebase", "-r", change, "-d", base2], cwd=repo)
    sha_after = _jj(["log", "-r", change, "--no-graph", "-T", "commit_id"], cwd=repo)
    assert sha_after != sha_before  # the rebase rewrote the commit

    outcome = LeafOutcome(
        leaf="the leaf", status="done", commit_id=change, changed_files=["leaf.txt"]
    )
    landed = prov.write_leaf_note(
        repo,
        outcome,
        _task(task="the leaf"),
        wave=1,
        spec_sha=None,
        timestamp="2026-07-13T00:00:00+00:00",
        log=lambda m: None,
    )
    assert landed == sha_after  # resolved to the CURRENT (post-rebase) sha, not the stale one

    # The note is a real git object on the post-rebase commit.
    body = _git(repo, "notes", f"--ref={prov.LEAF_NOTE_REF}", "show", sha_after)
    assert '"kind": "leaf"' in body
    assert read_note(repo, prov.LEAF_NOTE_REF, sha_after)["leaf"] == "the leaf"
