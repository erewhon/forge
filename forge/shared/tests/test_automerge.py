"""Tests for the shared auto-merge gates and VCS actions.

The pure gates (classifier, slug) are exercised directly; the branch/advance-main actions run
against a real temporary **git** repo (universally available) — jj is the structurally-analogous
production path, validated live rather than here.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from agents.shared.automerge import (
    advance_main,
    classify_manifest_only,
    classify_tests_only,
    find_repo_root,
    is_manifest_path,
    is_test_path,
    push_branch,
    slugify,
    working_diff,
)
from agents.task_worker.vcs import VCSError

# --- pure classifiers -------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("test_foo.py", True),
        ("foo_test.py", True),
        ("conftest.py", True),
        ("pkg/tests/thing.py", True),
        ("src/__tests__/x.js", True),
        ("web/button.test.ts", True),
        ("web/button.spec.tsx", True),
        ("crate/tests/it.rs", True),
        ("handler_test.go", True),
        ("src/foo.py", False),
        ("pyproject.toml", False),
        ("README.md", False),
    ],
)
def test_is_test_path(path, expected):
    assert is_test_path(path) is expected


def test_is_manifest_path():
    assert is_manifest_path("pyproject.toml")
    assert is_manifest_path("pkg/package.json")
    assert is_manifest_path("Cargo.lock")
    assert not is_manifest_path("src/app.py")
    assert not is_manifest_path("tests/test_app.py")


def test_classify_all_test_files_ok():
    v = classify_tests_only(Path("/nope"), changed=["tests/test_a.py", "pkg/b_test.py"])
    assert v.ok
    assert v.non_test == []


def test_classify_blocks_non_test_file():
    v = classify_tests_only(Path("/nope"), changed=["tests/test_a.py", "src/app.py"])
    assert not v.ok
    assert v.non_test == ["src/app.py"]
    assert "non-test" in v.reason


def test_classify_blocks_manifest_even_under_tests_dir():
    # A manifest is unsafe regardless of location — it can add a dependency.
    v = classify_tests_only(Path("/nope"), changed=["tests/test_a.py", "pyproject.toml"])
    assert not v.ok
    assert v.manifests == ["pyproject.toml"]
    assert "manifest" in v.reason


def test_classify_empty_is_blocked():
    v = classify_tests_only(Path("/nope"), changed=[])
    assert not v.ok
    assert "no changes" in v.reason


def test_jj_push_branch_argv_has_no_allow_new(monkeypatch):
    """jj 0.42 dropped --allow-new (new bookmarks push by default). Caught live by the
    dependabot smoke: the first real advisory push failed on this exact argv."""
    import agents.shared.automerge as am

    captured: list[list[str]] = []

    def fake_run(cmd, cwd, timeout=30):
        captured.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(am, "_run", fake_run)
    monkeypatch.setattr(am, "detect_vcs", lambda p: "jj")
    monkeypatch.setattr(am, "commit", lambda p, m: "changeid1")

    result = am.push_branch(Path("/tmp"), "deps/foo", "msg")
    assert result.pushed
    push_cmd = captured[-1]
    assert "--bookmark" in push_cmd
    assert "--allow-new" not in push_cmd


def test_working_copy_base_and_repark_jj(monkeypatch):
    import agents.shared.automerge as am

    captured: list[list[str]] = []

    def fake_run(cmd, cwd, timeout=30):
        captured.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "abc123def", "")

    monkeypatch.setattr(am, "_run", fake_run)
    monkeypatch.setattr(am, "detect_vcs", lambda p: "jj")

    assert am.working_copy_base(Path("/tmp")) == "abc123def"
    am.repark_working_copy(Path("/tmp"), "abc123def")
    assert captured[-1] == ["jj", "new", "abc123def"]


def test_repark_git_checks_out_base_branch(monkeypatch):
    import agents.shared.automerge as am

    captured: list[list[str]] = []

    def fake_run(cmd, cwd, timeout=30):
        captured.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "main", "")

    monkeypatch.setattr(am, "_run", fake_run)
    monkeypatch.setattr(am, "detect_vcs", lambda p: "git")

    assert am.working_copy_base(Path("/tmp")) == "main"
    am.repark_working_copy(Path("/tmp"), "main")
    assert captured[-1] == ["git", "checkout", "main"]


def test_manifest_only_pure_bump_ok():
    v = classify_manifest_only(Path("/nope"), changed=["pyproject.toml", "uv.lock"])
    assert v.ok
    assert v.non_manifest == []


def test_manifest_only_blocks_source_file_alongside_bump():
    # The classic classifier-inversion guard: a bump that also touches source must NOT pass.
    v = classify_manifest_only(Path("/nope"), changed=["uv.lock", "agents/shared/llm.py"])
    assert not v.ok
    assert v.non_manifest == ["agents/shared/llm.py"]
    assert "agents/shared/llm.py" in v.reason


def test_manifest_only_blocks_source_only_change():
    v = classify_manifest_only(Path("/nope"), changed=["src/app.py"])
    assert not v.ok
    assert v.non_manifest == ["src/app.py"]


def test_manifest_only_blocks_test_files_too():
    # Manifest-only means ONLY manifests — even harmless-looking test files block.
    v = classify_manifest_only(Path("/nope"), changed=["uv.lock", "tests/test_a.py"])
    assert not v.ok


def test_manifest_only_empty_is_blocked():
    v = classify_manifest_only(Path("/nope"), changed=[])
    assert not v.ok
    assert "no changes" in v.reason


def test_slugify():
    assert slugify("Add test: foo::bar [edge-case]") == "add-test-foo-bar-edge-case"
    assert slugify("   ") == "change"
    assert len(slugify("x" * 100)) <= 40


# --- VCS actions against a real temp git repo -------------------------------


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr}")
    return result.stdout.strip()


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=repo, capture_output=True, text=True)
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "Test")
    (repo / "app.py").write_text("def f():\n    return 1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "init")
    return repo


def test_push_branch_and_advance_main_git(git_repo: Path):
    # A tests-only working-copy change.
    (git_repo / "tests").mkdir()
    (git_repo / "tests" / "test_f.py").write_text(
        "from app import f\n\ndef test_f():\n    assert f() == 1\n"
    )

    # Classify the working copy before committing.
    verdict = classify_tests_only(git_repo)
    assert verdict.ok, verdict.reason

    pushed = push_branch(git_repo, "auto-tests/add-f", "test: cover f()", push=False)
    assert pushed.vcs == "git"
    assert pushed.change_id
    assert not pushed.pushed  # push=False
    assert _git(git_repo, "rev-parse", "--abbrev-ref", "HEAD") == "auto-tests/add-f"
    # The test file is committed on the branch, and main hasn't moved yet.
    assert "test_f.py" in _git(git_repo, "show", "--name-only", "--format=", "HEAD")
    branch_sha = _git(git_repo, "rev-parse", "--short", "HEAD")
    assert _git(git_repo, "rev-parse", "--short", "main") != branch_sha

    merged = advance_main(git_repo, pushed.change_id, push=False)
    assert merged.merged_to_main
    assert _git(git_repo, "rev-parse", "--short", "main") == branch_sha


def test_advance_main_refuses_when_main_checked_out(git_repo: Path):
    sha = _git(git_repo, "rev-parse", "--short", "HEAD")
    # Still on main from the fixture.
    with pytest.raises(VCSError, match="checked out"):
        advance_main(git_repo, sha, push=False)


def test_working_diff_includes_new_test_file(git_repo: Path):
    (git_repo / "tests").mkdir()
    (git_repo / "tests" / "test_new.py").write_text("def test_x():\n    assert True\n")
    diff = working_diff(git_repo)
    assert "tests/test_new.py" in diff
    assert "def test_x" in diff


def test_find_repo_root(git_repo: Path):
    nested = git_repo / "a" / "b"
    nested.mkdir(parents=True)
    assert find_repo_root(nested) == git_repo.resolve()
    assert find_repo_root(Path("/")) is None
