"""Tests for the refactor ensemble's Forge-task mapping (forge_emit mocked)."""

from __future__ import annotations

from forge.refactor_ensemble import main as main_mod
from forge.refactor_ensemble.emit import external_ref, plan_to_specs
from forge.refactor_ensemble.models import (
    CanonicalSmell,
    RefactorPlan,
    ScoredSmell,
    Verdict,
)


def _scored(loc, typ, impact, *, status="confirmed", effort="small") -> ScoredSmell:
    smell = CanonicalSmell(id="RF-x", location=loc, smell_type=typ, effort=effort, impact=impact)
    verdict = Verdict(status=status, votes_real=2, votes_total=2, impact=impact, reasonings=["why"])
    return ScoredSmell(smell=smell, verdict=verdict)


def _plan(confirmed=(), tentative=(), rejected=()) -> RefactorPlan:
    return RefactorPlan(
        focus="maintainability",
        files=["a.py"],
        raw_count=0,
        canonical_count=0,
        dedup_ok=True,
        confirmed=list(confirmed),
        tentative=list(tentative),
        rejected=list(rejected),
    )


def test_external_ref_is_stable_and_run_independent():
    a = _scored("mod::foo", "Duplication", "high")
    b = _scored("  mod::foo ", "duplication", "high")  # whitespace / case noise
    # ids differ run-to-run, but the ref keys only on location + smell_type
    assert external_ref(a) == external_ref(b) == "refactor:mod::foo:duplication"


def test_plan_to_specs_only_confirmed():
    plan = _plan(
        confirmed=[_scored("m::a", "complexity", "high")],
        tentative=[_scored("m::b", "naming", "medium")],
        rejected=[_scored("m::c", "idiom", "low")],
    )
    specs = plan_to_specs(plan)
    assert len(specs) == 1
    spec = specs[0]
    assert spec.external_ref == "refactor:m::a:complexity"
    assert spec.task_type == "refactor"
    assert spec.estimate == "s"  # effort small -> s
    assert "Refactor m::a [complexity]" == spec.title
    assert "Skeptic panel" in spec.content  # verification surfaced for the reviewer


def test_plan_to_specs_min_impact_filter():
    plan = _plan(
        confirmed=[
            _scored("m::hi", "complexity", "high"),
            _scored("m::mid", "naming", "medium"),
            _scored("m::lo", "idiom", "low"),
        ]
    )
    refs = {s.external_ref for s in plan_to_specs(plan, min_impact="medium")}
    assert refs == {"refactor:m::hi:complexity", "refactor:m::mid:naming"}


def test_effort_maps_to_estimate():
    plan = _plan(confirmed=[_scored("m::a", "dead-code", "high", effort="large")])
    assert plan_to_specs(plan)[0].estimate == "l"


def test_main_emit_requires_project(monkeypatch, capsys):
    # --emit-tasks without --project is rejected before any run
    monkeypatch.setattr(
        main_mod, "run_refactor", lambda *a, **k: (_ for _ in ()).throw(AssertionError("ran"))
    )
    rc = main_mod.main(["some.py", "--emit-tasks"])
    assert rc == 2
    assert "requires --project" in capsys.readouterr().err


def test_main_emits_confirmed_plan(monkeypatch):
    plan = _plan(confirmed=[_scored("m::a", "complexity", "high")])
    monkeypatch.setattr(main_mod, "run_refactor", lambda *a, **k: plan)

    captured = {}

    def fake_emit_plan(p, *, project, min_impact, dry_run, log):
        captured.update(project=project, min_impact=min_impact, dry_run=dry_run, plan=p)
        from forge.shared.forge_emit import EmitSummary

        return EmitSummary(project=project)

    monkeypatch.setattr("forge.refactor_ensemble.emit.emit_plan", fake_emit_plan)
    rc = main_mod.main(
        ["some.py", "--emit-tasks", "--project", "Meta", "--min-impact", "high", "--dry-run-emit"]
    )
    assert rc == 0
    assert captured["project"] == "Meta"
    assert captured["min_impact"] == "high"
    assert captured["dry_run"] is True
    assert captured["plan"] is plan
