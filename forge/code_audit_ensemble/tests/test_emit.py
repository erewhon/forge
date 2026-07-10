"""Tests for the code-audit ensemble's Forge-task mapping (forge_emit mocked)."""

from __future__ import annotations

from forge.code_audit_ensemble import main as main_mod
from forge.code_audit_ensemble.emit import external_ref, report_to_specs
from forge.code_audit_ensemble.models import (
    AuditReport,
    CanonicalFinding,
    ScoredFinding,
    Verdict,
)


def _scored(title, file, sev, *, line="", status="confirmed") -> ScoredFinding:
    finding = CanonicalFinding(
        id="CA-x",
        title=title,
        file=file,
        line=line,
        severity=sev,
        scenario="the handle leaks on the error path",
        suggestion="use a context manager",
    )
    verdict = Verdict(status=status, votes_real=2, votes_total=2, severity=sev, reasonings=["why"])
    return ScoredFinding(finding=finding, verdict=verdict)


def _report(confirmed=(), tentative=(), rejected=()) -> AuditReport:
    return AuditReport(
        focus="correctness",
        files=["a.py"],
        raw_count=0,
        canonical_count=0,
        dedup_ok=True,
        confirmed=list(confirmed),
        tentative=list(tentative),
        rejected=list(rejected),
    )


def test_external_ref_slugs_file_and_title_stably():
    a = _scored("Null deref in parse()", "src/parse.py", "high")
    b = _scored("null deref in   PARSE()", "src/parse.py", "high")  # phrasing-stable noise
    assert external_ref(a) == external_ref(b) == "audit:src-parse-py:null-deref-in-parse"


def test_report_to_specs_only_confirmed():
    report = _report(
        confirmed=[_scored("Leak", "io.py", "high", line="42")],
        tentative=[_scored("Maybe slow", "io.py", "medium")],
        rejected=[_scored("Nit", "io.py", "low")],
    )
    specs = report_to_specs(report)
    assert len(specs) == 1
    spec = specs[0]
    assert spec.external_ref == "audit:io-py:leak"
    assert spec.task_type == "bug-fix"
    assert spec.title == "Fix: Leak (io.py)"
    assert "io.py:42" in spec.content  # location with line surfaced
    assert "Skeptic panel" in spec.content


def test_report_to_specs_min_severity_filter():
    report = _report(
        confirmed=[
            _scored("Crash", "a.py", "critical"),
            _scored("Race", "b.py", "high"),
            _scored("Typo", "c.py", "low"),
        ]
    )
    refs = {s.external_ref for s in report_to_specs(report, min_severity="high")}
    assert refs == {"audit:a-py:crash", "audit:b-py:race"}


def test_main_emit_requires_project(monkeypatch, capsys):
    monkeypatch.setattr(
        main_mod, "run_audit", lambda *a, **k: (_ for _ in ()).throw(AssertionError("ran"))
    )
    rc = main_mod.main(["some.py", "--emit-tasks"])
    assert rc == 2
    assert "requires --project" in capsys.readouterr().err


def test_main_emits_confirmed_report(monkeypatch):
    report = _report(confirmed=[_scored("Leak", "io.py", "high")])
    monkeypatch.setattr(main_mod, "run_audit", lambda *a, **k: report)

    captured = {}

    def fake_emit_report(r, *, project, min_severity, dry_run, log):
        captured.update(project=project, min_severity=min_severity, dry_run=dry_run, report=r)
        from forge.shared.forge_emit import EmitSummary

        return EmitSummary(project=project)

    monkeypatch.setattr("forge.code_audit_ensemble.emit.emit_report", fake_emit_report)
    rc = main_mod.main(
        ["some.py", "--emit-tasks", "--project", "Meta", "--min-severity", "high", "--dry-run-emit"]
    )
    assert rc == 0
    assert captured["project"] == "Meta"
    assert captured["min_severity"] == "high"
    assert captured["dry_run"] is True
    assert captured["report"] is report
