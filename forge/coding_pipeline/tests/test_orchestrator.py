"""Wave-loop tests — every IO boundary (Forge, dispatch, verify, replan, VCS) mocked.

The composition logic is under test: exit conditions (dry / waiting / gate / max / halt /
aborted), the suite-red -> replan path, action application, and resume numbering.
"""

from __future__ import annotations

import pytest

from agents.coding_pipeline import orchestrator as orc
from agents.coding_pipeline.models import (
    EscalateAction,
    FixupAction,
    FramingProposal,
    HaltAction,
    LeafOutcome,
    LeafSpec,
    RespecAction,
    SuiteResult,
    WavePlan,
    WaveReport,
)


def _framing() -> FramingProposal:
    return FramingProposal(
        goal_as_stated="g",
        restated_goal="g",
        recommendation="r",
        epic_slug="toy-epic",
        approved=True,
    )


def _leaf(title: str = "leaf-a") -> LeafSpec:
    return LeafSpec(title=title, content="spec", feature="Toy")


def _plan(*titles: str, **counts) -> WavePlan:
    return WavePlan(feature="Toy", project="Meta", dispatch=list(titles), **counts)


def _report(wave: int = 1, passed: bool = True, findings=()) -> WaveReport:
    return WaveReport(wave=wave, suite=SuiteResult(passed=passed), findings=list(findings))


def _done(title: str) -> LeafOutcome:
    return LeafOutcome(leaf=title, status="done", commit_id="abc")


@pytest.fixture
def wired(monkeypatch, tmp_path):
    """Happy single-wave world; tests override individual boundaries."""
    monkeypatch.setattr(orc.settings, "runs_dir", tmp_path)
    monkeypatch.setattr(orc.settings, "wave_size", 4)
    monkeypatch.setattr(orc, "require_approved_framing", lambda run_dir: _framing())
    monkeypatch.setattr(orc, "_load_tree", lambda run_dir: [_leaf()])
    monkeypatch.setattr(orc, "reconcile", lambda feature: [])
    monkeypatch.setattr(orc, "get_changed_files", lambda repo: [])
    monkeypatch.setattr(orc, "wave_start_rev", lambda repo: "c0")

    plans = iter([_plan("leaf-a"), _plan(done=1)])  # one wave, then dry
    monkeypatch.setattr(orc, "plan_wave", lambda *a, **k: next(plans))
    monkeypatch.setattr(orc, "run_wave", lambda plan, repo, **k: [_done(t) for t in plan.dispatch])
    monkeypatch.setattr(orc, "verify_wave", lambda repo, *, wave, from_change: _report(wave))
    monkeypatch.setattr(orc, "replan", lambda f, t, r, a: [])
    monkeypatch.setattr(orc, "update_task_status", lambda *a, **k: None)
    monkeypatch.setattr(orc, "emit_fixup", lambda *a, **k: None)
    monkeypatch.setattr(orc, "ensure_epic_bookmark", lambda repo, slug, log: "pipeline/toy-epic")
    monkeypatch.setattr(orc, "update_epic_bookmark", lambda repo, slug, log: "pipeline/toy-epic")
    return tmp_path


def _run(**overrides):
    kwargs = dict(
        project="Meta",
        feature="Toy",
        epic_slug="toy-epic",
        repo=overrides.pop("repo", None) or __import__("pathlib").Path("/repo"),
        log=lambda m: None,
    )
    kwargs.update(overrides)
    return orc.run_epic(**kwargs)


# --- exit conditions --------------------------------------------------------------


def test_one_wave_then_dry(wired):
    result = _run()
    assert result.status == "dry"
    assert result.waves_run == 1
    assert result.dispatched == ["leaf-a"]
    # the wave record was persisted with resume-correct numbering
    assert (wired / "toy-epic" / "wave-0001.json").exists()


def test_waiting_on_human_exits_cleanly(wired, monkeypatch):
    monkeypatch.setattr(orc, "plan_wave", lambda *a, **k: _plan(ready_manual=2))
    result = _run()
    assert result.status == "waiting-on-human"
    assert result.waves_run == 0
    assert "2 manual" in result.notes[-1]


def test_dry_run_plans_without_dispatching(wired, monkeypatch):
    dispatched = []
    monkeypatch.setattr(orc, "run_wave", lambda *a, **k: dispatched.append(1))
    result = _run(dry_run=True)
    assert result.status == "planned"
    assert result.dispatched == ["leaf-a"]
    assert dispatched == []


def test_wave_gate_stops_after_one_wave(wired, monkeypatch):
    plans = iter([_plan("leaf-a"), _plan("leaf-b")])
    monkeypatch.setattr(orc, "plan_wave", lambda *a, **k: next(plans))
    result = _run(wave_gate=True)
    assert result.status == "wave-gate"
    assert result.waves_run == 1


def test_max_waves_bounds_the_run(wired, monkeypatch):
    monkeypatch.setattr(orc, "plan_wave", lambda *a, **k: _plan("leaf-a"))
    result = _run(max_waves=2)
    assert result.status == "max-waves"
    assert result.waves_run == 2


def test_halt_action_stops_the_run(wired, monkeypatch):
    monkeypatch.setattr(
        orc, "replan", lambda f, t, r, a: [HaltAction(reason="framing invalidated")]
    )
    result = _run()
    assert result.status == "halted"
    assert result.waves_run == 1


def test_dispatch_error_aborts_with_forge_untouched(wired, monkeypatch):
    def boom(plan, repo, **k):
        raise orc.DispatchError("another dispatch holds the lock")

    monkeypatch.setattr(orc, "run_wave", boom)
    result = _run()
    assert result.status == "aborted"
    assert "lock" in result.notes[-1]


def test_dirty_working_copy_aborts_before_dispatch(wired, monkeypatch):
    monkeypatch.setattr(orc, "get_changed_files", lambda repo: ["stray.py"])
    dispatched = []
    monkeypatch.setattr(orc, "run_wave", lambda *a, **k: dispatched.append(1))
    result = _run()
    assert result.status == "aborted"
    assert dispatched == []


def test_unapproved_framing_refuses_the_run(wired, monkeypatch):
    from agents.coding_pipeline.architect import FramingNotApprovedError

    def refuse(run_dir):
        raise FramingNotApprovedError("not approved")

    monkeypatch.setattr(orc, "require_approved_framing", refuse)
    with pytest.raises(FramingNotApprovedError):
        _run()


# --- the replan wiring ---------------------------------------------------------------


def test_suite_red_report_reaches_replan_and_loop_continues(wired, monkeypatch):
    seen_reports = []
    monkeypatch.setattr(
        orc, "verify_wave", lambda repo, *, wave, from_change: _report(wave, passed=False)
    )
    monkeypatch.setattr(orc, "replan", lambda f, t, r, a: seen_reports.append(r) or [])
    result = _run()
    assert result.status == "dry"  # second plan is dry: the loop continued past the red wave
    assert len(seen_reports) == 1
    assert not seen_reports[0].suite_green


def test_replan_receives_attempt_counts_from_journal(wired, monkeypatch):
    monkeypatch.setattr(
        orc,
        "run_wave",
        lambda plan, repo, **k: [LeafOutcome(leaf="leaf-a", status="failed", reason="tests red")],
    )
    seen = {}
    monkeypatch.setattr(orc, "replan", lambda f, t, r, a: seen.update(a) or [])
    monkeypatch.setattr(orc, "count_attempts_for_all", lambda d, titles: {t: 7 for t in titles})
    _run()
    assert seen == {"leaf-a": 7}


def test_replan_failure_degrades_to_deterministic_escalations(wired, monkeypatch):
    # e2e dry-run regression: a failed model replan crashed the run, losing the wave
    # record while the journal kept counting attempts. It must degrade instead: capped
    # leaves still escalate, the record persists, the loop continues.
    monkeypatch.setattr(
        orc,
        "run_wave",
        lambda plan, repo, **k: [LeafOutcome(leaf="leaf-a", status="failed", reason="tests red")],
    )
    monkeypatch.setattr(orc, "count_attempts_for_all", lambda d, titles: {t: 99 for t in titles})

    def boom(f, t, r, a):
        raise orc.ArchitectError("replan produced no usable actions: output failed validation")

    monkeypatch.setattr(orc, "replan", boom)
    status_writes = []
    monkeypatch.setattr(
        orc, "update_task_status", lambda t, s, notes="": status_writes.append((t, s))
    )

    result = _run()

    assert ("leaf-a", "Spec Needed") in status_writes  # escalation survived the degrade
    assert (wired / "toy-epic" / "wave-0001.json").exists()  # record persisted
    journal = (wired / "toy-epic" / "journal.jsonl").read_text()
    assert "replan-degraded" in journal
    assert any("degraded" in n for n in result.notes)
    assert result.status == "dry"  # the loop continued past the failure


def test_replan_failure_under_cap_degrades_to_no_actions(wired, monkeypatch):
    monkeypatch.setattr(
        orc,
        "run_wave",
        lambda plan, repo, **k: [
            LeafOutcome(leaf="leaf-a", status="failed", reason="no file changes")
        ],
    )

    def boom(f, t, r, a):
        raise orc.ArchitectError("output failed validation")

    monkeypatch.setattr(orc, "replan", boom)
    result = _run()
    assert result.status == "dry"
    assert (wired / "toy-epic" / "wave-0001.json").exists()


def test_escalation_action_flips_task_to_spec_needed(wired, monkeypatch):
    status_writes = []
    monkeypatch.setattr(
        orc, "update_task_status", lambda t, s, notes="": status_writes.append((t, s))
    )
    monkeypatch.setattr(
        orc,
        "replan",
        lambda f, t, r, a: [EscalateAction(leaf_title="leaf-a", diagnostics="boom")],
    )
    _run()
    assert ("leaf-a", "Spec Needed") in status_writes


def test_fixup_action_emits_idempotently(wired, monkeypatch):
    emitted = []

    class _Outcome:
        external_ref = "pipeline:toy-epic:fix:x"
        action = "created"

    monkeypatch.setattr(
        orc, "emit_fixup", lambda leaf, **k: emitted.append(leaf.title) or _Outcome()
    )
    monkeypatch.setattr(
        orc,
        "replan",
        lambda f, t, r, a: [FixupAction(finding_slug="x", leaf=_leaf("fix the thing"))],
    )
    _run()
    assert emitted == ["fix the thing"]


def test_respec_action_reopens_with_revised_spec(wired, monkeypatch):
    writes = []
    monkeypatch.setattr(
        orc, "update_task_status", lambda t, s, notes="": writes.append((t, s, notes))
    )
    monkeypatch.setattr(
        orc,
        "replan",
        lambda f, t, r, a: [
            RespecAction(leaf_title="leaf-a", revised=_leaf("leaf-a"), rationale="shrink")
        ],
    )
    _run()
    task, status, notes = writes[0]
    assert (task, status) == ("leaf-a", "Ready")
    assert "shrink" in notes and "spec" in notes.lower()


def test_epic_bookmark_updated_at_each_wave_checkpoint(wired, monkeypatch):
    updates = []
    monkeypatch.setattr(orc, "update_epic_bookmark", lambda repo, slug, log: updates.append(slug))
    result = _run()
    assert result.waves_run == 1
    assert updates == ["toy-epic"]  # re-pushed once per wave


# --- resume numbering ------------------------------------------------------------------


def test_wave_numbering_resumes_across_runs(wired, monkeypatch):
    # a previous run left wave-0002; this run's wave must be 0003
    from agents.coding_pipeline.journal import persist_wave

    persist_wave(
        wired,
        "toy-epic",
        __import__("agents.coding_pipeline.models", fromlist=["WaveRecord"]).WaveRecord(
            wave=2, report=WaveReport(wave=2)
        ),
    )

    result = _run()
    assert result.waves_run == 1
    assert (wired / "toy-epic" / "wave-0003.json").exists()
