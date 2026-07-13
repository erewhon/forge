"""Orchestration tests for the auto-merge loop (no LLM, no VCS, no test runner).

Every IO boundary is mocked so the decision flow itself is under test: push-vs-merge, and that each
gate fails *closed* — reverting and falling back to the Forge emit rather than merging.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from forge.shared.automerge import PushResult
from forge.shared.automerge import TestsOnlyVerdict as _TestsOnlyVerdict
from forge.testing_ensemble import autotest as at
from forge.testing_ensemble.autotest import SignoffResult
from forge.testing_ensemble.generate import GeneratedTest, GeneratedTestsEnvelope
from forge.testing_ensemble.models import CanonicalGap, ScoredGap, Verdict
from forge.testing_ensemble.models import TestReport as _TestReport


def _report(*, sev: str = "high", n: int = 1) -> _TestReport:
    gaps = []
    for i in range(n):
        gap = CanonicalGap(
            id=f"TG-{i}",
            target=f"mod::func{i}",
            gap_type="edge-case",
            why_it_matters="w",
            suggested_test="s",
            severity=sev,
        )
        verdict = Verdict(status="confirmed", votes_real=2, votes_total=2, severity=sev)
        gaps.append(ScoredGap(gap=gap, verdict=verdict))
    return _TestReport(
        focus="f",
        source_files=["m.py"],
        test_files=[],
        raw_count=n,
        canonical_count=n,
        dedup_ok=True,
        confirmed=gaps,
        tentative=[],
        rejected=[],
    )


class _Summary:
    def line(self) -> str:
        return "created 1, skipped 0"


@pytest.fixture
def wired(monkeypatch):
    """Install the happy path across every boundary; return spies on the action functions."""
    monkeypatch.setattr(at, "detect_vcs", lambda p: "git")
    monkeypatch.setattr(at, "run_review", lambda paths, focus: _report())
    monkeypatch.setattr(at, "collect_context", lambda paths: ("CTX", ["m.py"], []))
    monkeypatch.setattr(
        at,
        "generate_tests",
        lambda ctx, gaps, log=None: GeneratedTestsEnvelope(
            tests=[
                GeneratedTest(
                    gap_target="mod::func0",
                    test_file="tests/test_mod.py",
                    mode="append",
                    code="def test_x():\n    assert True\n",
                )
            ]
        ),
    )
    monkeypatch.setattr(at, "apply_generated", lambda repo, env, log=None: ["tests/test_mod.py"])
    monkeypatch.setattr(
        at,
        "classify_tests_only",
        lambda repo, **k: _TestsOnlyVerdict(ok=True, changed=["tests/test_mod.py"]),
    )
    monkeypatch.setattr(at, "run_tests", lambda repo: (True, "ok"))
    monkeypatch.setattr(at, "working_diff", lambda repo: "DIFF")
    monkeypatch.setattr(
        at,
        "_signoff",
        lambda diff, pr_ref: SignoffResult(
            approved=True, attempted=2, approvals=2, providers=["anthropic", "local"]
        ),
    )
    push = MagicMock(
        return_value=PushResult(
            vcs="git", branch="auto-tests/mod-func0", change_id="abc123", pushed=True
        )
    )
    monkeypatch.setattr(at, "push_branch", push)
    advance = MagicMock(
        return_value=PushResult(
            vcs="git", branch="main", change_id="abc123", pushed=True, merged_to_main=True
        )
    )
    monkeypatch.setattr(at, "advance_main", advance)
    revert = MagicMock()
    monkeypatch.setattr(at, "revert_changes", revert)
    emit = MagicMock(return_value=_Summary())
    monkeypatch.setattr(at, "emit_report", emit)
    monkeypatch.setattr(at, "log_decision", lambda rec, path: path)
    return {"push": push, "advance": advance, "revert": revert, "emit": emit}


def test_default_pushes_branch_without_merge(wired):
    r = at.auto_test(["m.py"], repo_path=Path("/repo"), log=lambda m: None)
    assert r.status == "branched"
    assert r.change_id == "abc123"
    wired["push"].assert_called_once()
    wired["advance"].assert_not_called()
    wired["revert"].assert_not_called()


def test_auto_merge_advances_main(wired):
    r = at.auto_test(["m.py"], repo_path=Path("/repo"), auto_merge=True, log=lambda m: None)
    assert r.status == "merged"
    assert r.merged_to_main
    wired["advance"].assert_called_once()


def test_tests_only_gate_blocks_reverts_and_emits(wired, monkeypatch):
    monkeypatch.setattr(
        at,
        "classify_tests_only",
        lambda repo, **k: _TestsOnlyVerdict(
            ok=False,
            changed=["src/x.py"],
            non_test=["src/x.py"],
            reason="non-test file(s): src/x.py",
        ),
    )
    r = at.auto_test(["m.py"], repo_path=Path("/repo"), project="Meta", log=lambda m: None)
    assert r.status == "blocked"
    assert "tests-only" in r.reason
    wired["revert"].assert_called_once()
    wired["emit"].assert_called_once()
    wired["push"].assert_not_called()
    assert r.emitted is not None


def test_green_gate_blocks(wired, monkeypatch):
    monkeypatch.setattr(at, "run_tests", lambda repo: (False, "boom"))
    r = at.auto_test(["m.py"], repo_path=Path("/repo"), project="Meta", log=lambda m: None)
    assert r.status == "blocked"
    assert r.tests_passed is False
    wired["revert"].assert_called_once()
    wired["push"].assert_not_called()


def test_signoff_gate_blocks(wired, monkeypatch):
    monkeypatch.setattr(
        at,
        "_signoff",
        lambda diff, pr_ref: SignoffResult(
            approved=False,
            attempted=2,
            approvals=1,
            providers=["anthropic", "local"],
            reason="quorum 2/2, approvals 1/2",
        ),
    )
    r = at.auto_test(["m.py"], repo_path=Path("/repo"), project="Meta", log=lambda m: None)
    assert r.status == "blocked"
    assert "sign-off" in r.reason
    wired["push"].assert_not_called()
    wired["emit"].assert_called_once()


def test_blocked_without_project_skips_emit(wired, monkeypatch):
    monkeypatch.setattr(at, "run_tests", lambda repo: (False, "boom"))
    r = at.auto_test(["m.py"], repo_path=Path("/repo"), log=lambda m: None)  # no project
    assert r.status == "blocked"
    wired["emit"].assert_not_called()
    assert r.emitted is None


def test_no_confirmed_gaps_is_noop(wired, monkeypatch):
    monkeypatch.setattr(at, "run_review", lambda paths, focus: _report(n=0))
    r = at.auto_test(["m.py"], repo_path=Path("/repo"), log=lambda m: None)
    assert r.status == "no-gaps"
    wired["push"].assert_not_called()


def test_min_severity_filters_out_low_gaps(wired, monkeypatch):
    monkeypatch.setattr(at, "run_review", lambda paths, focus: _report(sev="low"))
    r = at.auto_test(["m.py"], repo_path=Path("/repo"), min_severity="high", log=lambda m: None)
    assert r.status == "no-gaps"


def test_dry_run_plans_without_generating(wired, monkeypatch):
    gen = MagicMock()
    monkeypatch.setattr(at, "generate_tests", gen)
    r = at.auto_test(["m.py"], repo_path=Path("/repo"), dry_run=True, log=lambda m: None)
    assert r.status == "planned"
    assert r.gaps_targeted == 1
    gen.assert_not_called()


def test_no_vcs_is_error(wired, monkeypatch):
    monkeypatch.setattr(at, "detect_vcs", lambda p: "")
    r = at.auto_test(["m.py"], repo_path=Path("/repo"), log=lambda m: None)
    assert r.status == "error"


def test_render_auto_smoke(wired):
    r = at.auto_test(["m.py"], repo_path=Path("/repo"), log=lambda m: None)
    out = at.render_auto(r)
    assert "branched" in out
    assert "auto-tests/mod-func0" in out


def test_signoff_seats_shared_gate_from_active_slots(monkeypatch):
    """_signoff wiring: only active roster slots become seats, with the testing prompt."""
    from types import SimpleNamespace

    slots = [
        SimpleNamespace(provider="anthropic", active=True, pool=SimpleNamespace(executors=["e1"])),
        SimpleNamespace(provider="dead", active=False, pool=SimpleNamespace(executors=["e2"])),
        SimpleNamespace(provider="local", active=True, pool=SimpleNamespace(executors=["e3"])),
    ]
    monkeypatch.setattr(at, "build_reviewer_slots", lambda: slots)
    captured: dict = {}

    def fake_gate(diff_text, **kwargs):
        captured["diff"] = diff_text
        captured.update(kwargs)
        return SignoffResult(approved=True, attempted=2, approvals=2)

    monkeypatch.setattr(at, "full_quorum_signoff", fake_gate)
    at._signoff("DIFF", pr_ref="auto-tests/foo")
    assert [s.provider for s in captured["seats"]] == ["anthropic", "local"]
    # The seat's executor is now the whole failover Pool (not just the primary), so a down primary
    # fails over to its backup instead of dropping the seat.
    assert [s.executor for s in captured["seats"]] == [slots[0].pool, slots[2].pool]
    assert captured["system"] == at._SIGNOFF_SYSTEM
    assert captured["ref"] == "auto-tests/foo"
    assert "ONLY test files" in captured["context"]
