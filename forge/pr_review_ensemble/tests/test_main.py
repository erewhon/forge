"""CLI input/output wiring for the review ensemble — the work-deployable GitHub PR paths.

The ensemble itself is exercised elsewhere; these tests pin the plumbing added to run at work:
fetching a PR diff via ``gh pr diff``, posting the advisory back via ``gh pr comment``, input
precedence, and the ``--post-comment`` guard. The ``gh`` boundary is mocked — no subprocess.
"""

from __future__ import annotations

import argparse
import asyncio

import pytest

from agents.pr_review_ensemble import main


def _args(**overrides) -> argparse.Namespace:
    base = dict(pr=None, repo=None, diff_file=None, post_comment=False, pr_ref=None)
    base.update(overrides)
    return argparse.Namespace(**base)


def test_read_diff_from_pr_calls_gh_pr_diff(monkeypatch):
    calls = []
    monkeypatch.setattr(main, "_gh", lambda cmd: calls.append(cmd) or "THE DIFF")
    diff = main._read_diff(_args(pr=42, repo="owner/repo"))
    assert diff == "THE DIFF"
    assert calls == [["pr", "diff", "42", "--repo", "owner/repo"]]


def test_pr_overrides_diff_file(monkeypatch, tmp_path):
    f = tmp_path / "d.diff"
    f.write_text("file diff")
    monkeypatch.setattr(main, "_gh", lambda cmd: "pr diff")
    assert main._read_diff(_args(pr=7, diff_file=str(f))) == "pr diff"  # --pr wins


def test_read_diff_without_pr_reads_file(monkeypatch, tmp_path):
    f = tmp_path / "d.diff"
    f.write_text("file diff")
    assert main._read_diff(_args(diff_file=str(f))) == "file diff"


def test_post_pr_comment_builds_gh_command(monkeypatch):
    calls = []
    monkeypatch.setattr(main, "_gh", lambda cmd: calls.append(cmd) or "")
    main._post_pr_comment(42, "owner/repo", "the advisory")
    assert calls == [["pr", "comment", "42", "--body", "the advisory", "--repo", "owner/repo"]]


def test_maybe_post_only_when_flag_and_pr(monkeypatch):
    posted = []
    monkeypatch.setattr(main, "_post_pr_comment", lambda pr, repo, body: posted.append((pr, body)))
    main._maybe_post("adv", _args(post_comment=False, pr=1), label="Advisory")
    main._maybe_post("adv", _args(post_comment=True, pr=None), label="Advisory")
    assert posted == []  # neither: flag off, then no pr
    main._maybe_post("adv", _args(post_comment=True, pr=9, repo="o/r"), label="Advisory")
    assert posted == [(9, "adv")]


def test_default_pr_ref():
    assert main._default_pr_ref(_args(pr=5, repo="o/r")) == "o/r#5"
    assert main._default_pr_ref(_args(pr=5)) == "#5"
    assert main._default_pr_ref(_args()) == "(unspecified)"


def test_post_comment_requires_pr():
    with pytest.raises(SystemExit, match="--post-comment requires --pr"):
        asyncio.run(main._run(_args(post_comment=True, pr=None)))
