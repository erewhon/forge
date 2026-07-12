"""Layer manifest classification against a real git history."""

from __future__ import annotations

from forge.upstream_sync.gitops import git
from forge.upstream_sync.layer import compute_layer
from forge.upstream_sync.tests.conftest import g


def test_added_and_modified_classified(repos):
    _, fork, _ = repos
    merge_base = git(fork, "merge-base", "main", "refs/remotes/upstream/main")
    layer = compute_layer(fork, merge_base, git(fork, "rev-parse", "main"))
    assert layer.added == ["SPRINKLES.md"]
    assert layer.modified == ["README.md"]


def test_rename_counts_old_path_as_modified_and_new_as_added(repos):
    _, fork, _ = repos
    g(fork, "mv", "core.txt", "renamed.txt")
    g(fork, "commit", "-qm", "fork: rename core")
    merge_base = git(fork, "merge-base", "main", "refs/remotes/upstream/main")
    layer = compute_layer(fork, merge_base, git(fork, "rev-parse", "main"))
    # Upstream changes to core.txt will collide; renamed.txt is fork-owned.
    assert "core.txt" in layer.modified
    assert "renamed.txt" in layer.added


def test_delete_counts_as_modified(repos):
    _, fork, _ = repos
    g(fork, "rm", "-q", "core.txt")
    g(fork, "commit", "-qm", "fork: drop core")
    merge_base = git(fork, "merge-base", "main", "refs/remotes/upstream/main")
    layer = compute_layer(fork, merge_base, git(fork, "rev-parse", "main"))
    assert "core.txt" in layer.modified
