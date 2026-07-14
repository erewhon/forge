"""Journal mirror — append-only refs/pipeline/<epic> commit chain and resume-from-clone hydrate,
on a real git repo (plain plumbing, no jj needed)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from forge.coding_pipeline import journal_mirror as jm


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
    r = tmp_path / "repo"
    r.mkdir()
    _git(r, "init", "-b", "main")
    return r


def _run_dir(tmp_path: Path, **files: str) -> Path:
    d = tmp_path / "runs" / "toy"
    d.mkdir(parents=True)
    default = {"framing.json": '{"epic_slug": "toy"}', "journal.jsonl": '{"event": "x"}\n'}
    for name, content in {**default, **files}.items():
        (d / name).write_text(content)
    return d


# --- mirror write ---------------------------------------------------------------------


def test_mirror_snapshots_run_dir_into_the_ref(repo: Path, tmp_path: Path):
    run_dir = _run_dir(tmp_path, **{"wave-0001.json": '{"wave": 1}'})
    commit = jm.mirror_run_dir(repo, run_dir, "toy", message="wave 1: toy", log=lambda m: None)
    assert commit == _git(repo, "rev-parse", "refs/pipeline/toy")
    # every top-level file is retrievable from the ref via git show
    assert _git(repo, "show", "refs/pipeline/toy:journal.jsonl") == '{"event": "x"}'
    assert _git(repo, "show", "refs/pipeline/toy:wave-0001.json") == '{"wave": 1}'
    assert "wave 1: toy" in _git(repo, "log", "-1", "--format=%s", "refs/pipeline/toy")


def test_chain_is_append_only(repo: Path, tmp_path: Path):
    run_dir = _run_dir(tmp_path)
    first = jm.mirror_run_dir(repo, run_dir, "toy", message="wave 1", log=lambda m: None)
    (run_dir / "wave-0002.json").write_text('{"wave": 2}')
    second = jm.mirror_run_dir(repo, run_dir, "toy", message="wave 2", log=lambda m: None)

    assert first != second
    assert _git(repo, "rev-list", "--count", "refs/pipeline/toy") == "2"
    # the second commit parents the first — history IS the audit chain
    assert _git(repo, "rev-parse", "refs/pipeline/toy~1") == first
    assert _git(repo, "log", "-1", "--format=%P", "refs/pipeline/toy") == first


def test_mirror_of_absent_or_empty_run_dir_is_a_noop(repo: Path, tmp_path: Path):
    assert (
        jm.mirror_run_dir(repo, tmp_path / "nope", "toy", message="m", log=lambda m: None) is None
    )
    empty = tmp_path / "runs" / "empty"
    empty.mkdir(parents=True)
    assert jm.mirror_run_dir(repo, empty, "empty", message="m", log=lambda m: None) is None
    # neither wrote a ref
    assert not _git(repo, "for-each-ref", "refs/pipeline")


def test_mirror_failure_degrades_to_a_warning(tmp_path: Path):
    not_a_repo = tmp_path / "plain"
    not_a_repo.mkdir()
    run_dir = _run_dir(tmp_path)
    warnings: list[str] = []
    commit = jm.mirror_run_dir(not_a_repo, run_dir, "toy", message="m", log=warnings.append)
    assert commit is None
    assert any("mirroring run dir" in w for w in warnings)


def test_mirror_skips_subdirectories(repo: Path, tmp_path: Path):
    run_dir = _run_dir(tmp_path)
    (run_dir / "logs").mkdir()
    (run_dir / "logs" / "a.log").write_text("noise")
    jm.mirror_run_dir(repo, run_dir, "toy", message="m", log=lambda m: None)
    names = _git(repo, "ls-tree", "--name-only", "refs/pipeline/toy").splitlines()
    assert "logs" not in names
    assert "journal.jsonl" in names


# --- hydrate (resume from clone) ------------------------------------------------------


def test_hydrate_materializes_run_dir_from_ref(repo: Path, tmp_path: Path):
    source = _run_dir(tmp_path, **{"tree.json": '{"leaves": []}'})
    jm.mirror_run_dir(repo, source, "toy", message="wave 1", log=lambda m: None)

    # a fresh machine: the ref exists, but no local run dir
    fresh = tmp_path / "fresh" / "toy"
    logs: list[str] = []
    assert jm.hydrate_run_dir(repo, fresh, "toy", log=logs.append) is True
    assert (fresh / "journal.jsonl").read_text() == '{"event": "x"}\n'
    assert (fresh / "tree.json").read_text() == '{"leaves": []}'
    assert (fresh / "framing.json").read_text() == '{"epic_slug": "toy"}'
    assert any("hydrated" in m for m in logs)


def test_hydrate_is_a_noop_when_local_framing_exists(repo: Path, tmp_path: Path):
    source = _run_dir(tmp_path)
    jm.mirror_run_dir(repo, source, "toy", message="wave 1", log=lambda m: None)

    local = tmp_path / "local" / "toy"
    local.mkdir(parents=True)
    (local / "framing.json").write_text('{"epic_slug": "local-wins"}')
    assert jm.hydrate_run_dir(repo, local, "toy", log=lambda m: None) is False
    # local state is write-primary — never overwritten by the ref
    assert (local / "framing.json").read_text() == '{"epic_slug": "local-wins"}'


def test_hydrate_is_a_noop_when_ref_is_absent(repo: Path, tmp_path: Path):
    fresh = tmp_path / "fresh" / "toy"
    assert jm.hydrate_run_dir(repo, fresh, "toy", log=lambda m: None) is False
    assert not fresh.exists()


def test_mirror_then_hydrate_round_trips(repo: Path, tmp_path: Path):
    source = _run_dir(tmp_path, **{"wave-0001.json": '{"wave": 1}', "inventory.md": "# inv"})
    jm.mirror_run_dir(repo, source, "toy", message="wave 1", log=lambda m: None)
    fresh = tmp_path / "clone-runs" / "toy"
    jm.hydrate_run_dir(repo, fresh, "toy", log=lambda m: None)
    assert {p.name for p in fresh.iterdir()} == {p.name for p in source.iterdir()}
    for p in source.iterdir():
        assert (fresh / p.name).read_bytes() == p.read_bytes()
