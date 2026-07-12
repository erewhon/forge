"""Shared fixture: a real upstream/fork/origin git repo trio in tmp_path.

Real git repos (not fakes) — the sync loop is mostly git plumbing, and scripted repos test
the actual merge/worktree/push behavior the agent lives on. LLM seat and test gate are
mocked per-test; git is not.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


def g(repo: Path, *args: str) -> str:
    """Run git in *repo*, asserting success."""
    result = subprocess.run(["git", *args], cwd=str(repo), capture_output=True, text=True)
    assert result.returncode == 0, f"git {' '.join(args)}: {result.stderr}"
    return result.stdout.strip()


def _configure_user(repo: Path) -> None:
    g(repo, "config", "user.email", "test@example.com")
    g(repo, "config", "user.name", "Test")


def upstream_commit(upstream: Path, path: str, content: str, message: str) -> None:
    (upstream / path).parent.mkdir(parents=True, exist_ok=True)
    (upstream / path).write_text(content)
    g(upstream, "add", "-A")
    g(upstream, "commit", "-qm", message)


@pytest.fixture
def repos(tmp_path):
    """(upstream, fork, bare_origin): fork = upstream clone + additive layer, pushed to a
    local bare origin. The fork's remotes: ``upstream`` -> the upstream repo, ``origin``
    -> the bare. The layer: SPRINKLES.md added, README.md modified."""
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    g(upstream, "init", "-q", "-b", "main")
    _configure_user(upstream)
    (upstream / "core.txt").write_text("line1\nline2\nline3\n")
    (upstream / "README.md").write_text("# soft serve\n")
    g(upstream, "add", "-A")
    g(upstream, "commit", "-qm", "upstream base")

    bare = tmp_path / "origin.git"
    subprocess.run(["git", "init", "-q", "--bare", str(bare)], check=True)

    fork = tmp_path / "fork"
    subprocess.run(
        ["git", "clone", "-q", str(upstream), str(fork)], check=True, capture_output=True
    )
    _configure_user(fork)
    g(fork, "remote", "rename", "origin", "upstream")
    g(fork, "remote", "add", "origin", str(bare))
    (fork / "SPRINKLES.md").write_text("the fork's additive layer\n")
    (fork / "README.md").write_text("# soft serve\nwith sprinkles\n")
    g(fork, "add", "-A")
    g(fork, "commit", "-qm", "fork: additive layer")
    g(fork, "push", "-q", "-u", "origin", "main")

    return upstream, fork, bare
