"""git-notes provenance plumbing — write/read round-trips on a real temp git repo."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from forge.shared.git_notes import read_note, write_note


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
    _git(tmp_path, "commit", "-m", "c1")
    return tmp_path


def _head(repo: Path) -> str:
    return _git(repo, "rev-parse", "HEAD")


def test_round_trip(repo: Path):
    payload = {"schema": 1, "kind": "gate", "approved": True, "seats": [{"provider": "x"}]}
    write_note(repo, "pipeline/gate", _head(repo), payload)
    assert read_note(repo, "pipeline/gate", _head(repo)) == payload


def test_missing_note_returns_none(repo: Path):
    assert read_note(repo, "pipeline/gate", _head(repo)) is None


def test_ref_is_stored_under_refs_notes(repo: Path):
    write_note(repo, "pipeline/gate", _head(repo), {"schema": 1})
    refs = _git(repo, "for-each-ref", "--format=%(refname)", "refs/notes/pipeline")
    assert "refs/notes/pipeline/gate" in refs


def test_write_force_replaces_existing(repo: Path):
    head = _head(repo)
    write_note(repo, "pipeline/gate", head, {"schema": 1, "approved": False})
    write_note(repo, "pipeline/gate", head, {"schema": 1, "approved": True})
    assert read_note(repo, "pipeline/gate", head) == {"schema": 1, "approved": True}


def test_distinct_refs_do_not_collide(repo: Path):
    head = _head(repo)
    write_note(repo, "pipeline/gate", head, {"kind": "gate"})
    write_note(repo, "pipeline/provenance", head, {"kind": "leaf"})
    assert read_note(repo, "pipeline/gate", head) == {"kind": "gate"}
    assert read_note(repo, "pipeline/provenance", head) == {"kind": "leaf"}


def test_malformed_note_raises(repo: Path):
    head = _head(repo)
    # Write a non-JSON note body directly via git, then read it back.
    _git(repo, "notes", "--ref=pipeline/gate", "add", "-f", "-m", "not json", head)
    with pytest.raises(ValueError, match="not valid JSON"):
        read_note(repo, "pipeline/gate", head)
