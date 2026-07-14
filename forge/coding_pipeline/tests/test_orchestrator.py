"""Wave-loop tests — every IO boundary (Forge, dispatch, verify, replan, VCS) mocked.

The composition logic is under test: exit conditions (dry / waiting / gate / max / halt /
aborted), the suite-red -> replan path, action application, and resume numbering.
"""

from __future__ import annotations

import pytest

from forge.coding_pipeline import orchestrator as orc
from forge.coding_pipeline.models import (
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


class _FakeStore:
    """A recording stand-in for the task store — the wave loop's only Forge write path
    in these tests. ``writes`` captures every status update as (task, status, notes, mode)."""

    def __init__(self) -> None:
        self.writes: list[tuple[str, str, str, str | None]] = []

    def update_status(self, task, status, notes="", execution_mode=None):
        self.writes.append((task, status, notes, execution_mode))


@pytest.fixture
def wired(monkeypatch, tmp_path):
    """Happy single-wave world; tests override individual boundaries."""
    monkeypatch.setattr(orc.settings, "runs_dir", tmp_path)
    monkeypatch.setattr(orc.settings, "wave_size", 4)
    monkeypatch.setattr(orc, "require_approved_framing", lambda run_dir: _framing())
    monkeypatch.setattr(orc, "_load_tree", lambda run_dir: [_leaf()])
    monkeypatch.setattr(orc, "reconcile", lambda feature: [])
    monkeypatch.setattr(orc, "fetch_epic_rows", lambda *a, **k: [])
    monkeypatch.setattr(orc, "get_changed_files", lambda repo: [])
    monkeypatch.setattr(orc, "wave_start_rev", lambda repo: "c0")

    plans = iter([_plan("leaf-a"), _plan(done=1)])  # one wave, then dry
    monkeypatch.setattr(orc, "plan_wave", lambda *a, **k: next(plans))
    monkeypatch.setattr(orc, "run_wave", lambda plan, repo, **k: [_done(t) for t in plan.dispatch])
    monkeypatch.setattr(orc, "verify_wave", lambda repo, *, wave, from_change, **k: _report(wave))
    monkeypatch.setattr(orc, "replan", lambda f, t, r, a, **kw: [])
    monkeypatch.setattr(orc, "get_task_store", lambda: _FakeStore())
    monkeypatch.setattr(orc, "emit_fixup", lambda *a, **k: None)
    monkeypatch.setattr(orc, "ensure_epic_bookmark", lambda repo, slug, log: "pipeline/toy-epic")
    monkeypatch.setattr(orc, "update_epic_bookmark", lambda repo, slug, log: "pipeline/toy-epic")
    monkeypatch.setattr(orc, "mirror_run_dir", lambda *a, **k: "mirror-commit")
    monkeypatch.setattr(orc, "mirror_framing", lambda *a, **k: "framing-commit")
    monkeypatch.setattr(orc, "hydrate_run_dir", lambda *a, **k: False)
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
        orc, "replan", lambda f, t, r, a, **kw: [HaltAction(reason="framing invalidated")]
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
    from forge.coding_pipeline.architect import FramingNotApprovedError

    def refuse(run_dir):
        raise FramingNotApprovedError("not approved")

    monkeypatch.setattr(orc, "require_approved_framing", refuse)
    with pytest.raises(FramingNotApprovedError):
        _run()


def test_epic_rows_shared_between_planner_and_context_builder(wired, monkeypatch):
    """One Forge read per wave: the same rows feed plan_wave AND the sibling list,
    and dispatch receives a live preamble builder."""
    from forge.coding_pipeline.models import LeafRow

    rows_sentinel = [LeafRow(task="done leaf", status="Done")]
    monkeypatch.setattr(orc, "fetch_epic_rows", lambda *a, **k: rows_sentinel)
    seen = {}

    plans = iter([_plan("leaf-a"), _plan(done=1)])
    monkeypatch.setattr(
        orc, "plan_wave", lambda *a, **k: seen.update(plan_rows=k.get("rows")) or next(plans)
    )

    def fake_run_wave(plan, repo, **k):
        seen["preamble_for"] = k.get("preamble_for")
        return [_done(t) for t in plan.dispatch]

    monkeypatch.setattr(orc, "run_wave", fake_run_wave)
    _run()
    assert seen["plan_rows"] is rows_sentinel
    assert callable(seen["preamble_for"])


# --- the replan wiring ---------------------------------------------------------------


def test_suite_red_report_reaches_replan_and_loop_continues(wired, monkeypatch):
    seen_reports = []
    monkeypatch.setattr(
        orc, "verify_wave", lambda repo, *, wave, from_change, **k: _report(wave, passed=False)
    )
    monkeypatch.setattr(orc, "replan", lambda f, t, r, a, **kw: seen_reports.append(r) or [])
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
    monkeypatch.setattr(orc, "replan", lambda f, t, r, a, **kw: seen.update(a) or [])
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

    def boom(f, t, r, a, **kw):
        raise orc.ArchitectError(
            "replan produced no usable actions: output failed validation",
            raw='{"actions": [{"kind": "fixup", "oops": true}]}',
        )

    monkeypatch.setattr(orc, "replan", boom)
    store = _FakeStore()
    monkeypatch.setattr(orc, "get_task_store", lambda: store)

    result = _run()

    # escalation survived the degrade
    assert ("leaf-a", "Spec Needed") in [(w[0], w[1]) for w in store.writes]
    assert (wired / "toy-epic" / "wave-0001.json").exists()  # record persisted
    journal = (wired / "toy-epic" / "journal.jsonl").read_text()
    assert "replan-degraded" in journal
    assert "oops" in journal  # the model's raw output is captured (JSON-escaped) for diagnosis
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

    def boom(f, t, r, a, **kw):
        raise orc.ArchitectError("output failed validation")

    monkeypatch.setattr(orc, "replan", boom)
    result = _run()
    assert result.status == "dry"
    assert (wired / "toy-epic" / "wave-0001.json").exists()


def test_escalation_action_flips_task_to_spec_needed_and_manual(wired, monkeypatch):
    store = _FakeStore()
    monkeypatch.setattr(orc, "get_task_store", lambda: store)
    monkeypatch.setattr(
        orc,
        "replan",
        lambda f, t, r, a, **kw: [EscalateAction(leaf_title="leaf-a", diagnostics="boom")],
    )
    _run()
    # Spec Needed removes worker eligibility; Manual keeps a human re-arm from
    # silently re-entering the auto pool (design doc: "Spec Needed + Manual").
    assert ("leaf-a", "Spec Needed", "Manual") in [(w[0], w[1], w[3]) for w in store.writes]


def test_fixup_action_emits_idempotently(wired, monkeypatch):
    emitted = []

    class _Outcome:
        external_ref = "pipeline:toy-epic:fix:x"
        action = "created"

    monkeypatch.setattr(
        orc,
        "emit_fixup",
        lambda leaf, **k: emitted.append((leaf.title, k.get("finding_slug"))) or _Outcome(),
    )
    monkeypatch.setattr(
        orc,
        "replan",
        lambda f, t, r, a, **kw: [FixupAction(finding_slug="x", leaf=_leaf("fix the thing"))],
    )
    _run()
    # the finding's slug reaches emission — it keys the cross-replan-stable ref
    assert emitted == [("fix the thing", "x")]


def test_respec_action_reopens_with_revised_spec(wired, monkeypatch):
    store = _FakeStore()
    monkeypatch.setattr(orc, "get_task_store", lambda: store)
    monkeypatch.setattr(
        orc,
        "replan",
        lambda f, t, r, a, **kw: [
            RespecAction(leaf_title="leaf-a", revised=_leaf("leaf-a"), rationale="shrink")
        ],
    )
    _run()
    task, status, notes, _mode = store.writes[0]
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
    from forge.coding_pipeline.journal import persist_wave

    persist_wave(
        wired,
        "toy-epic",
        __import__("forge.coding_pipeline.models", fromlist=["WaveRecord"]).WaveRecord(
            wave=2, report=WaveReport(wave=2)
        ),
    )

    result = _run()
    assert result.waves_run == 1
    assert (wired / "toy-epic" / "wave-0003.json").exists()


# --- landed-noop settlement + fixup title dedup (deps-v2, live) --------------------


def test_settle_landed_noops_marks_done_and_skips(tmp_path):
    store = _FakeStore()
    outcomes = [
        LeafOutcome(leaf="hero", status="failed", reason="no file changes produced"),
        LeafOutcome(leaf="fresh", status="failed", reason="tests failed: boom"),
    ]
    orc._settle_landed_noops(outcomes, {"hero"}, store=store, run_dir=tmp_path, log=lambda m: None)
    assert outcomes[0].status == "skipped"  # neither landed nor failed for replan
    assert outcomes[1].status == "failed"  # genuine failures untouched
    assert store.writes and store.writes[0][:2] == ("hero", "Done")


def test_settle_landed_noops_ignores_real_failures_of_landed_leaves(tmp_path):
    store = _FakeStore()
    outcomes = [LeafOutcome(leaf="hero", status="failed", reason="tests failed: boom")]
    orc._settle_landed_noops(outcomes, {"hero"}, store=store, run_dir=tmp_path, log=lambda m: None)
    assert outcomes[0].status == "failed"
    assert store.writes == []


def test_apply_actions_skips_duplicate_title_fixup(tmp_path, monkeypatch):
    emitted: list[str] = []
    monkeypatch.setattr(orc, "emit_fixup", lambda leaf, **k: emitted.append(leaf.title))
    actions = [orc.FixupAction(finding_slug="s", leaf=_leaf("Fix The Thing"))]
    halted = orc._apply_actions(
        actions,
        project="Meta",
        epic_slug="e",
        run_dir=tmp_path,
        store=_FakeStore(),
        log=lambda m: None,
        existing_titles={"fix the thing"},
    )
    assert halted is False
    assert emitted == []


# --- token budget guard ---------------------------------------------------------------


def test_budget_exhaustion_parks_the_run_before_dispatch(wired, monkeypatch):
    monkeypatch.setattr(orc.settings, "epic_token_budget", 100)
    run_dir = wired / "toy-epic"
    run_dir.mkdir(parents=True, exist_ok=True)
    # prior waves/runs already spent past the budget — the ledger resumes that total on load
    (run_dir / "usage.json").write_text('{"input_tokens": 80, "output_tokens": 40, "calls": 5}')

    result = _run()
    assert result.status == "budget-exhausted"
    assert result.waves_run == 0  # parked at the wave boundary, nothing dispatched
    assert result.total_tokens == 120
    assert any("budget exhausted" in n for n in result.notes)
    # fail-closed marker in the journal, resumable (raise the budget and re-run)
    journal = (run_dir / "journal.jsonl").read_text()
    assert '"event": "budget_exhausted"' in journal
    assert '"parked": true' in journal


def test_under_budget_runs_normally(wired, monkeypatch):
    monkeypatch.setattr(orc.settings, "epic_token_budget", 10_000)
    result = _run()
    assert result.status == "dry"  # the normal one-wave-then-dry path is untouched
    assert result.waves_run == 1


def test_spend_is_reported_at_completion(wired, monkeypatch):
    from forge.shared import usage as usage_mod

    def dispatch_and_spend(plan, repo, **k):
        usage_mod.record_usage(600, 150)  # a wave's ensemble spend hits the ambient ledger
        return [_done(t) for t in plan.dispatch]

    monkeypatch.setattr(orc, "run_wave", dispatch_and_spend)
    result = _run()
    assert result.status == "dry"
    assert result.total_tokens == 750
    assert any("pipeline API spend: 750 tokens" in n for n in result.notes)
