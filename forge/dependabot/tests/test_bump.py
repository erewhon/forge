from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from forge.dependabot.bump import (
    BumpError,
    apply_bump,
    lockfile_delta,
)
from forge.dependabot.models import BumpCandidate


@pytest.fixture
def candidate():
    return BumpCandidate(name="httpx", current="0.27.0", latest="0.28.0", delta="minor")


# ---- apply_bump ----


class TestApplyBump:
    """Success path, non-zero exit -> revert, and no-change lock."""

    def test_success_returns_changed_files(self, candidate):
        """apply_bump success path returns changed files (subprocess + vcs mocked)."""
        repo = Path("/fake/repo")
        with (
            patch("subprocess.run") as mock_run,
            patch("forge.dependabot.bump.get_changed_files") as mock_changed,
        ):
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            mock_changed.return_value = ["uv.lock"]

            result = apply_bump(repo, candidate)

            mock_run.assert_called_once_with(
                ["uv", "lock", "--upgrade-package", "httpx"],
                capture_output=True,
                text=True,
                timeout=None,
                cwd=repo,
            )
            assert result == ["uv.lock"]

    def test_non_zero_exit_calls_revert_and_raises(self, candidate):
        """Non-zero uv exit -> revert_changes called exactly once and BumpError raised."""
        repo = Path("/fake/repo")
        with (
            patch("subprocess.run") as mock_run,
            patch("forge.dependabot.bump.revert_changes") as mock_revert,
        ):
            mock_run.return_value = MagicMock(
                returncode=1,
                stdout="",
                stderr="error: resolution failed",
            )

            with pytest.raises(BumpError, match="resolution failed"):
                apply_bump(repo, candidate, timeout=60)

            mock_revert.assert_called_once_with(repo)

    def test_no_change_lock_returns_empty(self, candidate):
        """No-change lock -> [] and NO revert."""
        repo = Path("/fake/repo")
        with (
            patch("subprocess.run") as mock_run,
            patch("forge.dependabot.bump.get_changed_files") as mock_changed,
            patch("forge.dependabot.bump.revert_changes") as mock_revert,
        ):
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            mock_changed.return_value = []

            result = apply_bump(repo, candidate)

            assert result == []
            mock_revert.assert_not_called()


# ---- lockfile_delta ----


class TestLockfileDelta:
    """Extract version deltas from a crafted uv.lock diff fixture."""

    def _make_diff(self, body: str) -> str:
        """Wrap *body* as a git-style uv.lock diff."""
        return "--- a/uv.lock\n+++ b/uv.lock\n@@ -1,0 +0,0 @@\n" + body

    def test_extract_single_delta(self):
        """lockfile_delta extracts ``foo 1.0.0->1.0.1`` from a crafted uv.lock diff."""
        body = '-name = "foo"\n+name = "foo"\n-version = "1.0.0"\n+version = "1.0.1"\n'
        diff_text = self._make_diff(body)
        with patch("forge.dependabot.bump.working_diff", return_value=diff_text):
            result = lockfile_delta(Path("/fake/repo"))
        assert result == ["foo 1.0.0->1.0.1"]

    def test_extract_multiple_deltas(self):
        """Multiple packages bumped in the same diff."""
        body = (
            '-name = "foo"\n'
            '+name = "foo"\n'
            '-version = "1.0.0"\n'
            '+version = "1.0.1"\n'
            '-name = "bar"\n'
            '+name = "bar"\n'
            '-version = "2.0.0"\n'
            '+version = "2.1.0"\n'
        )
        diff_text = self._make_diff(body)
        with patch("forge.dependabot.bump.working_diff", return_value=diff_text):
            result = lockfile_delta(Path("/fake/repo"))
        assert result == ["foo 1.0.0->1.0.1", "bar 2.0.0->2.1.0"]

    def test_no_version_change_returns_empty(self):
        """Only name lines, no version diff -> empty."""
        body = 'name = "foo"\n'
        diff_text = self._make_diff(body)
        with patch("forge.dependabot.bump.working_diff", return_value=diff_text):
            result = lockfile_delta(Path("/fake/repo"))
        assert result == []

    def test_later_files_in_diff_are_not_misattributed(self):
        """A file appearing AFTER uv.lock in the diff must end the lock section — its
        ``version = "..."`` lines (e.g. pyproject's project version) are not package bumps."""
        diff_text = (
            "--- a/uv.lock\n+++ b/uv.lock\n@@ -1,0 +0,0 @@\n"
            ' name = "foo"\n-version = "1.0.0"\n+version = "1.0.1"\n'
            "--- a/pyproject.toml\n+++ b/pyproject.toml\n@@ -1,0 +0,0 @@\n"
            ' name = "my-project"\n-version = "0.1.0"\n+version = "0.2.0"\n'
        )
        with patch("forge.dependabot.bump.working_diff", return_value=diff_text):
            result = lockfile_delta(Path("/fake/repo"))
        assert result == ["foo 1.0.0->1.0.1"]  # pyproject's version change ignored

    def test_context_name_line_groups_versions(self):
        """Real bump diffs keep ``name = ...`` as an unchanged context line."""
        body = ' name = "foo"\n-version = "1.0.0"\n+version = "1.0.1"\n'
        with patch("forge.dependabot.bump.working_diff", return_value=self._make_diff(body)):
            result = lockfile_delta(Path("/fake/repo"))
        assert result == ["foo 1.0.0->1.0.1"]

    def test_vcsonerror_returns_empty(self):
        """VCS error -> best-effort empty list."""
        from forge.task_worker.vcs import VCSError

        with patch("forge.dependabot.bump.working_diff") as mock_diff:
            mock_diff.side_effect = VCSError("vcs error")
            result = lockfile_delta(Path("/fake/repo"))
        assert result == []
