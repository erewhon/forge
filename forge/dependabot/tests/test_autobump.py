"""Loop tests — every collaborator mocked; assert gate ordering and fail-closed endings."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from agents.dependabot import autobump as ab
from agents.dependabot.models import AuditFinding, BumpCandidate, EvidenceBundle
from agents.shared.automerge import ManifestOnlyVerdict, PushResult
from agents.shared.signoff import SignoffResult


def _candidate(delta: str = "minor") -> BumpCandidate:
    return BumpCandidate(name="idna", current="3.11", latest="3.15", delta=delta)


def _evidence(candidate: BumpCandidate, **over) -> EvidenceBundle:
    base = dict(
        candidate=candidate,
        target_yanked=False,
        package_age_days=30,
        changelog_url="https://example.com/CHANGES",
        lockfile_changes=["idna 3.11->3.15"],
        complete=True,
    )
    base.update(over)
    return EvidenceBundle(**base)


def _push(branch: str = "deps/idna-3-15") -> PushResult:
    return PushResult(vcs="jj", branch=branch, change_id="abc123", pushed=True)


def _approved(n: int = 2) -> SignoffResult:
    return SignoffResult(approved=True, attempted=n, approvals=n, providers=["a", "l"])


def _blocked() -> SignoffResult:
    return SignoffResult(
        approved=False,
        attempted=2,
        approvals=1,
        providers=["a", "l"],
        reason="quorum 2/2, approvals 1/2",
        blockers=["suspicious lockfile churn"],
    )


@pytest.fixture
def loop(monkeypatch, tmp_path):
    """Patch every collaborator to a happy-path default; tests override per-case."""
    mocks = {
        "detect_vcs": patch.object(ab, "detect_vcs", return_value="jj"),
        "scan_outdated": patch.object(ab, "scan_outdated", return_value=[_candidate()]),
        "run_audit": patch.object(ab, "run_audit", return_value=[]),
        "apply_bump": patch.object(ab, "apply_bump", return_value=["uv.lock"]),
        "lockfile_delta": patch.object(ab, "lockfile_delta", return_value=["idna 3.11->3.15"]),
        "collect_evidence": patch.object(
            ab, "collect_evidence", side_effect=lambda c, f, d: _evidence(c)
        ),
        "classify_manifest_only": patch.object(
            ab,
            "classify_manifest_only",
            return_value=ManifestOnlyVerdict(ok=True, changed=["uv.lock"]),
        ),
        "run_tests": patch.object(ab, "run_tests", return_value=(True, "ok")),
        "working_diff": patch.object(ab, "working_diff", return_value="DIFF"),
        "_signoff": patch.object(ab, "_signoff", return_value=_approved()),
        "push_branch": patch.object(ab, "push_branch", return_value=_push()),
        "advance_main": patch.object(ab, "advance_main", return_value=_push()),
        "emit_advisory": patch.object(ab, "emit_advisory", return_value=None),
        "revert_changes": patch.object(ab, "revert_changes"),
        "get_changed_files": patch.object(ab, "get_changed_files", return_value=[]),
        "working_copy_base": patch.object(ab, "working_copy_base", return_value="base123"),
        "repark_working_copy": patch.object(ab, "repark_working_copy"),
    }
    monkeypatch.setattr(ab.settings, "auto_log_path", tmp_path / "auto.jsonl")
    started = {name: p.start() for name, p in mocks.items()}
    yield started
    for p in mocks.values():
        p.stop()


def test_clean_bump_with_auto_merge_advances_main(loop, tmp_path):
    result = ab.auto_bump(tmp_path, auto_merge=True, log=lambda m: None)
    assert result.status == "merged"
    assert result.merged_to_main
    loop["advance_main"].assert_called_once()
    assert (tmp_path / "auto.jsonl").exists()  # decision logged


def test_default_pushes_branch_only(loop, tmp_path):
    result = ab.auto_bump(tmp_path, log=lambda m: None)
    assert result.status == "branched"
    assert not result.merged_to_main
    loop["push_branch"].assert_called_once()
    loop["advance_main"].assert_not_called()
    # The bump branch stays a side head — the working copy returns to where it started.
    loop["repark_working_copy"].assert_called_once_with(tmp_path, "base123")


def test_merged_run_does_not_repark(loop, tmp_path):
    # After advance_main the bump commit IS main; continuing from it is correct.
    ab.auto_bump(tmp_path, auto_merge=True, log=lambda m: None)
    loop["repark_working_copy"].assert_not_called()


def test_advisory_reparks_too(loop, tmp_path):
    loop["collect_evidence"].side_effect = lambda c, f, d: _evidence(c, complete=False)
    result = ab.auto_bump(tmp_path, log=lambda m: None)
    assert result.status == "advisory"
    loop["repark_working_copy"].assert_called_once_with(tmp_path, "base123")


def test_policy_ineligible_goes_advisory_without_expensive_gates(loop, tmp_path):
    loop["scan_outdated"].return_value = [_candidate(delta="major")]
    result = ab.auto_bump(tmp_path, log=lambda m: None)
    assert result.status == "advisory"
    assert "major" in result.reason
    loop["run_tests"].assert_not_called()
    loop["_signoff"].assert_not_called()
    loop["advance_main"].assert_not_called()
    loop["push_branch"].assert_called_once()  # the bump still ships as a reviewable branch
    loop["emit_advisory"].assert_called_once()


def test_incomplete_evidence_goes_advisory(loop, tmp_path):
    loop["collect_evidence"].side_effect = lambda c, f, d: _evidence(c, complete=False)
    result = ab.auto_bump(tmp_path, log=lambda m: None)
    assert result.status == "advisory"
    assert "incomplete" in result.reason
    loop["advance_main"].assert_not_called()


def test_manifest_gate_miss_goes_advisory(loop, tmp_path):
    loop["classify_manifest_only"].return_value = ManifestOnlyVerdict(
        ok=False, changed=["uv.lock", "app.py"], non_manifest=["app.py"], reason="app.py changed"
    )
    result = ab.auto_bump(tmp_path, auto_merge=True, log=lambda m: None)
    assert result.status == "advisory"
    assert "manifest-only gate" in result.reason
    loop["advance_main"].assert_not_called()


def test_red_suite_goes_advisory(loop, tmp_path):
    loop["run_tests"].return_value = (False, "1 failed")
    result = ab.auto_bump(tmp_path, auto_merge=True, log=lambda m: None)
    assert result.status == "advisory"
    assert result.tests_passed is False
    loop["_signoff"].assert_not_called()  # no LLM spend on a red suite
    loop["advance_main"].assert_not_called()


def test_signoff_block_goes_advisory_with_detail(loop, tmp_path):
    loop["_signoff"].return_value = _blocked()
    result = ab.auto_bump(tmp_path, auto_merge=True, log=lambda m: None)
    assert result.status == "advisory"
    assert "quorum 2/2, approvals 1/2" in result.reason
    assert "suspicious lockfile churn" in result.reason
    loop["advance_main"].assert_not_called()


def test_fixed_cve_bump_rides_auto_track(loop, tmp_path):
    fixed = AuditFinding(package="idna", vuln_id="PYSEC-2026-215", fix_versions=["3.15"])
    loop["collect_evidence"].side_effect = lambda c, f, d: _evidence(c, findings_current=[fixed])
    result = ab.auto_bump(tmp_path, log=lambda m: None)
    assert result.status == "branched"
    assert result.evidence is not None
    assert result.evidence.findings_current[0].vuln_id == "PYSEC-2026-215"


def test_no_candidates(loop, tmp_path):
    loop["scan_outdated"].return_value = []
    result = ab.auto_bump(tmp_path, log=lambda m: None)
    assert result.status == "no-candidates"
    loop["run_audit"].assert_not_called()


def test_constraint_pinned_bump_is_a_skip(loop, tmp_path):
    loop["apply_bump"].return_value = []
    result = ab.auto_bump(tmp_path, log=lambda m: None)
    assert result.status == "no-candidates"
    assert "constraint-pinned" in result.reason
    loop["push_branch"].assert_not_called()


def test_dry_run_stops_after_selection(loop, tmp_path):
    result = ab.auto_bump(tmp_path, dry_run=True, log=lambda m: None)
    assert result.status == "planned"
    assert result.branch == "deps/idna-3-15"
    loop["apply_bump"].assert_not_called()
    loop["_signoff"].assert_not_called()


def test_dirty_working_copy_refuses_to_run(loop, tmp_path):
    """push_branch commits the WHOLE working copy — running over uncommitted work would scoop
    it into the bump branch (the incident that added this guard)."""
    loop["get_changed_files"].return_value = ["agents/dependabot/main.py"]
    result = ab.auto_bump(tmp_path, log=lambda m: None)
    assert result.status == "error"
    assert "not clean" in result.reason
    assert "agents/dependabot/main.py" in result.reason
    loop["apply_bump"].assert_not_called()
    loop["push_branch"].assert_not_called()


def test_dirty_working_copy_still_allows_dry_run(loop, tmp_path):
    loop["get_changed_files"].return_value = ["some/file.py"]
    result = ab.auto_bump(tmp_path, dry_run=True, log=lambda m: None)
    assert result.status == "planned"  # read-only path is exempt


def test_no_vcs_is_an_error(loop, tmp_path):
    loop["detect_vcs"].return_value = ""
    result = ab.auto_bump(tmp_path, log=lambda m: None)
    assert result.status == "error"
    loop["scan_outdated"].assert_not_called()


def test_advisory_push_failure_also_reparks(loop, tmp_path):
    """The advisory path's push can fail after its commit landed (live finding: sandboxed ssh)
    — the cleanup must repark so the bump commit doesn't become the mainline's parent."""
    from agents.task_worker.vcs import VCSError

    loop["collect_evidence"].side_effect = lambda c, f, d: _evidence(c, complete=False)
    loop["push_branch"].side_effect = VCSError("ssh exploded")
    result = ab.auto_bump(tmp_path, log=lambda m: None)
    assert result.status == "error"
    loop["revert_changes"].assert_called_once()
    loop["repark_working_copy"].assert_called_once_with(tmp_path, "base123")


def test_vcs_failure_after_gates_is_fail_closed_error(loop, tmp_path):
    """The post-gate VCS action can still fail (live finding: stale sideways bookmark).
    That must be a logged error result with cleanup — never a traceback, never a half-merge."""
    from agents.task_worker.vcs import VCSError

    loop["push_branch"].side_effect = VCSError("refusing to move bookmark sideways")
    result = ab.auto_bump(tmp_path, auto_merge=True, log=lambda m: None)
    assert result.status == "error"
    assert "after all gates passed" in result.reason
    assert result.tests_passed is True  # the gates DID pass; only the action failed
    loop["advance_main"].assert_not_called()
    loop["revert_changes"].assert_called_once()
    loop["repark_working_copy"].assert_called_once_with(tmp_path, "base123")
    assert (tmp_path / "auto.jsonl").exists()  # the failure is in the decision log


def test_render_bump_shows_the_story(loop, tmp_path):
    result = ab.auto_bump(tmp_path, auto_merge=True, log=lambda m: None)
    out = ab.render_bump(result)
    assert "meta deps — merged" in out
    assert "idna 3.11 -> 3.15 (minor)" in out
    assert "merged to main" in out
