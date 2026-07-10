"""Tests for the code-audit ensemble (recipe mocked — no network)."""

from __future__ import annotations

from forge.code_audit_ensemble import audit as audit_mod
from forge.code_audit_ensemble.audit import _vote, collect_code, render, run_audit
from forge.code_audit_ensemble.models import CanonicalFinding, Verdict
from forge.shared.panel import ItemVerdict, PanelResult
from forge.shared.recipe import RecipeResult


class _Panel:
    def __init__(self, responses):
        self.responses = responses


def test_collect_code_reads_files_and_caps(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("print('hi')\n")
    ctx, files = collect_code([str(f)], max_chars=10_000)
    assert "print('hi')" in ctx
    assert files == [str(f)]


def test_collect_code_empty_for_missing_paths(tmp_path):
    ctx, files = collect_code([str(tmp_path / "nope.py")], max_chars=10_000)
    assert ctx == ""
    assert files == []


def test_vote_confirms_on_unanimous_real_and_bumps_severity():
    finding = CanonicalFinding(id="CA-1", title="t", severity="low")
    panel = _Panel(
        [
            {"real": True, "confidence": "high", "severity": "critical", "reasoning": "r1"},
            {"real": True, "confidence": "low", "severity": "medium", "reasoning": "r2"},
        ]
    )
    v = _vote(finding, panel)
    assert v.status == "confirmed"
    assert v.severity == "critical"  # max adjusted among the real votes
    assert (v.votes_real, v.votes_total) == (2, 2)


def test_vote_rejects_when_no_real_votes():
    v = _vote(CanonicalFinding(id="x", title="t"), _Panel([{"real": False, "reasoning": "no"}]))
    assert v.status == "rejected"
    assert v.votes_real == 0


def test_vote_tentative_on_split():
    panel = _Panel([{"real": True, "confidence": "low"}, {"real": False}])
    v = _vote(CanonicalFinding(id="x", title="t", severity="medium"), panel)
    assert v.status == "tentative"


def test_run_audit_assembles_and_sorts(monkeypatch, tmp_path):
    f = tmp_path / "a.py"
    f.write_text("x = 1\n")
    hi = CanonicalFinding(id="CA-1", title="high one", severity="high")
    lo = CanonicalFinding(id="CA-2", title="low one", severity="low")
    rej = CanonicalFinding(id="CA-3", title="rejected one", severity="medium")
    verdicts = [
        ItemVerdict(item=lo, panel=PanelResult(), verdict=Verdict("confirmed", 2, 2, "low", ["r"])),
        ItemVerdict(
            item=hi, panel=PanelResult(), verdict=Verdict("confirmed", 2, 2, "high", ["r"])
        ),
        ItemVerdict(item=rej, panel=PanelResult(), verdict=Verdict("rejected", 0, 2, "medium", [])),
    ]

    def fake_recipe(**_kwargs):
        return RecipeResult(raw=[1, 2], canonical=[hi, lo, rej], verdicts=verdicts, dedup_ok=True)

    monkeypatch.setattr(audit_mod, "discover_dedup_verify", fake_recipe)
    report = run_audit([str(f)], "correctness")

    assert [s.finding.id for s in report.confirmed] == ["CA-1", "CA-2"]  # severity desc
    assert [s.finding.id for s in report.rejected] == ["CA-3"]
    md = render(report)
    assert "high one" in md
    assert "2 confirmed" in md
    assert "Rejected" in md


def test_run_audit_raises_without_code(tmp_path):
    import pytest

    with pytest.raises(ValueError, match="no readable source"):
        run_audit([str(tmp_path / "missing.py")], "correctness")
