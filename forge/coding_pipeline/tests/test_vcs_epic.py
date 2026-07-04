"""Epic branch + gate tests — bookmark lifecycle on a real temp git repo (the automerge test
precedent); the sign-off wiring with fake seats and a mocked quorum."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from types import SimpleNamespace

from agents.coding_pipeline import vcs_epic as ve
from agents.coding_pipeline.models import FramingProposal
from agents.shared.signoff import SignoffResult


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


def _framing() -> FramingProposal:
    return FramingProposal(
        goal_as_stated="g",
        restated_goal="restated goal",
        recommendation="r",
        epic_slug="toy",
        approved=True,
    )


# --- bookmark lifecycle -------------------------------------------------------------


def test_ensure_creates_branch_at_tip_once(repo):
    branch = ve.ensure_epic_bookmark(repo, "toy", push=False, log=lambda m: None)
    assert branch == "pipeline/toy"
    first = _git(repo, "rev-parse", branch)
    assert first == _git(repo, "rev-parse", "HEAD")

    # a new commit lands; ensure must NOT move the existing bookmark
    (repo / "a.txt").write_text("two\n")
    _git(repo, "commit", "-am", "c2")
    ve.ensure_epic_bookmark(repo, "toy", push=False, log=lambda m: None)
    assert _git(repo, "rev-parse", branch) == first


def test_update_advances_branch_to_tip(repo):
    ve.ensure_epic_bookmark(repo, "toy", push=False, log=lambda m: None)
    (repo / "a.txt").write_text("two\n")
    _git(repo, "commit", "-am", "c2")
    ve.update_epic_bookmark(repo, "toy", push=False, log=lambda m: None)
    assert _git(repo, "rev-parse", "pipeline/toy") == _git(repo, "rev-parse", "HEAD")


def test_push_failure_is_a_warning_not_an_error(repo):
    # no remote configured: the push fails, the call still succeeds
    warnings: list[str] = []
    ve.ensure_epic_bookmark(repo, "toy", push=True, log=warnings.append)
    assert any("push" in w for w in warnings)


def test_epic_diff_is_merge_base_to_tip(repo):
    # epic branch gets a commit main doesn't have
    _git(repo, "checkout", "-b", "pipeline/toy")
    (repo / "b.txt").write_text("epic work\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "epic c2")
    _git(repo, "checkout", "main")

    diff = ve.epic_diff(repo, "toy")
    assert "epic work" in diff
    assert "b.txt" in diff


def test_no_vcs_raises(tmp_path):
    with pytest.raises(Exception, match="No VCS"):
        ve.ensure_epic_bookmark(tmp_path, "toy", push=False, log=lambda m: None)


# --- the gate -----------------------------------------------------------------------


def test_gate_passes_framing_context_and_fails_closed(monkeypatch, tmp_path):
    calls: dict = {}

    def fake_signoff(diff, **kwargs):
        calls["diff"] = diff
        calls.update(kwargs)
        return SignoffResult(approved=True, attempted=2, approvals=2, providers=["a", "b"])

    monkeypatch.setattr(ve, "epic_diff", lambda repo, slug, main="main": "THE-EPIC-DIFF")
    monkeypatch.setattr(ve, "full_quorum_signoff", fake_signoff)

    result = ve.run_epic_gate(tmp_path, "toy", _framing(), seats=[object(), object()])
    assert result.approved
    assert calls["diff"] == "THE-EPIC-DIFF"
    assert calls["ref"] == "pipeline/toy"
    assert "restated goal" in calls["context"]  # the gate judges against the approved framing
    assert calls["system"] == ve.EPIC_SIGNOFF_SYSTEM


def test_gate_blocks_empty_diff_without_any_llm_call(monkeypatch, tmp_path):
    def boom(*a, **k):
        raise AssertionError("no sign-off call for an empty diff")

    monkeypatch.setattr(ve, "epic_diff", lambda repo, slug, main="main": "")
    monkeypatch.setattr(ve, "full_quorum_signoff", boom)
    result = ve.run_epic_gate(tmp_path, "toy", _framing(), seats=[])
    assert not result.approved
    assert "empty epic diff" in result.reason


# --- rendering ------------------------------------------------------------------------


def test_render_approved_instructs_human_merge_only():
    result = SignoffResult(approved=True, attempted=3, approvals=3, providers=["a", "b", "c"])
    out = ve.render_epic_gate(result, "toy", tip="abc123")
    assert "APPROVED" in out
    assert "HUMAN merge" in out
    assert "pipeline/toy" in out and "abc123" in out
    assert "advance" not in out.lower() or "never advances" in out  # no auto-merge language
    # the EXACT merge command a human runs — the render is the pipeline's terminal action
    assert "jj bookmark set main -r pipeline/toy && jj git push --bookmark main" in out
    assert "3/3" in out and "a, b, c" in out  # quorum provenance visible


def test_approved_gate_never_touches_vcs(tmp_path, monkeypatch):
    """Approval RENDERS, never merges: an approved run_epic_gate must issue only
    read commands (diff) — no bookmark moves, no pushes, no commits (dry-run Q5)."""
    commands: list[list[str]] = []

    def spy_run(cmd, cwd):
        commands.append(cmd)
        return SimpleNamespace(returncode=0, stdout="diff --git a/x b/x\n+y\n", stderr="")

    monkeypatch.setattr(ve, "_run", spy_run)
    monkeypatch.setattr(ve, "detect_vcs", lambda repo: "jj")
    monkeypatch.setattr(
        ve,
        "full_quorum_signoff",
        lambda *a, **k: SignoffResult(approved=True, attempted=2, approvals=2, providers=["x", "y"]),
    )
    result = ve.run_epic_gate(tmp_path, "toy", _framing())
    assert result.approved
    mutating = ("bookmark", "push", "commit", "describe", "new", "set")
    for cmd in commands:
        assert not any(m in cmd for m in mutating), f"gate issued a mutating command: {cmd}"


def test_render_blocked_lists_blockers():
    result = SignoffResult(
        approved=False,
        attempted=3,
        approvals=2,
        reason="quorum 3/3, approvals 2/3",
        blockers=["test weakened in leaf 2"],
    )
    out = ve.render_epic_gate(result, "toy")
    assert "BLOCKED" in out
    assert "quorum 3/3" in out
    assert "test weakened" in out


def test_jj_push_argv_has_no_allow_new(monkeypatch):
    """jj 0.42 dropped --allow-new (new bookmarks push by default). Verify the argv."""
    captured: list[list[str]] = []

    def fake_run(cmd, cwd=None):
        captured.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(ve, "_run", fake_run)

    ve._push_branch(Path("/tmp"), "jj", "pipeline/toy", lambda m: None)

    # The command should contain --bookmark but NOT --allow-new
    cmd = captured[0]
    assert "--bookmark" in cmd
    assert "--allow-new" not in cmd
