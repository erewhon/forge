"""Tests for the testing ensemble's Forge-task mapping (forge_emit mocked)."""

from __future__ import annotations

from forge.testing_ensemble import main as main_mod
from forge.testing_ensemble.emit import external_ref, report_to_specs
from forge.testing_ensemble.models import CanonicalGap, ScoredGap, Verdict

# Aliased: pytest would otherwise try to collect the ``TestReport`` dataclass as a test class.
from forge.testing_ensemble.models import TestReport as _TestReport


def _scored(target, typ, sev, *, status="confirmed") -> ScoredGap:
    gap = CanonicalGap(
        id="TG-x",
        target=target,
        gap_type=typ,
        severity=sev,
        suggested_test="assert f() raises on bad input",
        why_it_matters="an untested error path",
    )
    verdict = Verdict(status=status, votes_real=2, votes_total=2, severity=sev, reasonings=["why"])
    return ScoredGap(gap=gap, verdict=verdict)


def _report(confirmed=(), tentative=(), rejected=()) -> _TestReport:
    return _TestReport(
        focus="coverage",
        source_files=["a.py"],
        test_files=["test_a.py"],
        raw_count=0,
        canonical_count=0,
        dedup_ok=True,
        confirmed=list(confirmed),
        tentative=list(tentative),
        rejected=list(rejected),
    )


def test_external_ref_is_stable_and_run_independent():
    a = _scored("mod::foo", "Error-Path", "high")
    b = _scored("  mod::foo ", "error-path", "high")  # whitespace / case noise
    assert external_ref(a) == external_ref(b) == "test:mod::foo:error-path"


def test_report_to_specs_only_confirmed():
    report = _report(
        confirmed=[_scored("m::a", "coverage", "high")],
        tentative=[_scored("m::b", "edge-case", "medium")],
        rejected=[_scored("m::c", "regression", "low")],
    )
    specs = report_to_specs(report)
    assert len(specs) == 1
    spec = specs[0]
    assert spec.external_ref == "test:m::a:coverage"
    assert spec.task_type == "test"
    assert spec.estimate == "s"
    assert spec.title == "Add test: m::a [coverage]"
    assert "Skeptic panel" in spec.content


def test_report_to_specs_min_severity_filter():
    report = _report(
        confirmed=[
            _scored("m::crit", "coverage", "critical"),
            _scored("m::hi", "error-path", "high"),
            _scored("m::lo", "edge-case", "low"),
        ]
    )
    refs = {s.external_ref for s in report_to_specs(report, min_severity="high")}
    assert refs == {"test:m::crit:coverage", "test:m::hi:error-path"}


def test_main_emit_requires_project(monkeypatch, capsys):
    monkeypatch.setattr(
        main_mod, "run_review", lambda *a, **k: (_ for _ in ()).throw(AssertionError("ran"))
    )
    rc = main_mod.main(["some.py", "--emit-tasks"])
    assert rc == 2
    assert "requires --project" in capsys.readouterr().err


def test_main_emits_confirmed_report(monkeypatch):
    report = _report(confirmed=[_scored("m::a", "coverage", "high")])
    monkeypatch.setattr(main_mod, "run_review", lambda *a, **k: report)

    captured = {}

    def fake_emit_report(r, *, project, min_severity, dry_run, log):
        captured.update(project=project, min_severity=min_severity, dry_run=dry_run, report=r)
        from forge.shared.forge_emit import EmitSummary

        return EmitSummary(project=project)

    monkeypatch.setattr("forge.testing_ensemble.emit.emit_report", fake_emit_report)
    rc = main_mod.main(
        ["some.py", "--emit-tasks", "--project", "Meta", "--min-severity", "high", "--dry-run-emit"]
    )
    assert rc == 0
    assert captured["project"] == "Meta"
    assert captured["min_severity"] == "high"
    assert captured["dry_run"] is True
    assert captured["report"] is report
