"""Wave-verification tests — tester, roster, and panels mocked; the gate logic is under test."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from forge.coding_pipeline import verify as v
from forge.coding_pipeline.models import ReviewFinding
from forge.shared.panel import PanelResult


def _member(label: str) -> SimpleNamespace:
    return SimpleNamespace(executor=object(), system="SYS", label=label)


def _panel(responses: list[dict], attempted: int | None = None) -> PanelResult:
    return PanelResult(
        responses=responses,
        member_labels=[f"m{i}" for i in range(len(responses))],
        attempted=attempted if attempted is not None else len(responses),
        quorum_met=True,
    )


def _finding(summary: str = "off-by-one in pager", file: str = "src/pager.py") -> ReviewFinding:
    return ReviewFinding(slug=v.stable_slug(file, summary), summary=summary, file=file)


@pytest.fixture
def roster(monkeypatch):
    monkeypatch.setattr(v, "_roster_members", lambda system: [_member("a"), _member("b")])


# --- slug stability ---------------------------------------------------------------


def test_stable_slug_is_deterministic_and_ref_safe():
    a = v.stable_slug("src/pager.py", "Off-by-one in page window")
    b = v.stable_slug("src/pager.py", "Off-by-one in page window")
    assert a == b
    assert "," not in a and " " not in a  # safe inside pipeline:{epic}:fix:{slug}


# --- collect_findings ---------------------------------------------------------------


def test_collect_flattens_dedups_and_ranks(roster, monkeypatch):
    responses = [
        {
            "findings": [
                {"summary": "Off-by-one in pager", "file": "src/pager.py", "severity": "high"},
                {"summary": "sleep in test", "file": "tests/test_a.py", "severity": "low"},
            ]
        },
        {  # second provider re-reports the same pager bug with identical wording
            "findings": [
                {"summary": "Off-by-one in pager", "file": "src/pager.py", "severity": "high"},
            ]
        },
    ]
    monkeypatch.setattr(v, "run_member_panel", lambda **kw: _panel(responses))
    out = v.collect_findings("DIFF")
    assert len(out) == 2  # slug dedup collapsed the duplicate
    assert out[0].severity == "high"  # ranked most-severe first
    assert all(not f.confirmed for f in out)


def test_collect_drops_malformed_envelopes_and_caps(roster, monkeypatch):
    from forge.coding_pipeline.config import settings

    monkeypatch.setattr(settings, "review_max_findings", 2)
    responses = [
        {"nonsense": True},  # malformed envelope: dropped, not fatal
        {
            "findings": [
                {"summary": f"bug {i}", "file": f"f{i}.py", "severity": "weird-sev"}
                for i in range(5)
            ]
        },
    ]
    monkeypatch.setattr(v, "run_member_panel", lambda **kw: _panel(responses))
    out = v.collect_findings("DIFF")
    assert len(out) == 2  # capped
    assert all(f.severity == "medium" for f in out)  # unknown severity normalized


def test_collect_with_no_active_roster_is_empty(monkeypatch):
    monkeypatch.setattr(v, "_roster_members", lambda system: [])
    assert v.collect_findings("DIFF") == []


# --- confirm vote ---------------------------------------------------------------


def test_majority_of_responders_confirms():
    assert v._majority_real(_finding(), _panel([{"real": True}, {"real": True}, {"real": False}]))
    assert not v._majority_real(_finding(), _panel([{"real": True}, {"real": False}]))  # tie
    assert not v._majority_real(_finding(), _panel([]))  # zero responders fails closed


def test_confirm_sets_flags_via_vote(roster, monkeypatch):
    findings = [_finding("bug A", "a.py"), _finding("bug B", "b.py")]

    def fake_verify_each(items, **kwargs):
        # first finding voted real, second refuted
        votes = [
            _panel([{"real": True}, {"real": True}]),
            _panel([{"real": False}, {"real": False}]),
        ]
        return [
            SimpleNamespace(item=item, panel=p, verdict=kwargs["aggregate"](item, p))
            for item, p in zip(items, votes)
        ]

    monkeypatch.setattr(v, "verify_each", fake_verify_each)
    out = v.confirm_findings("DIFF", findings)
    assert [f.confirmed for f in out] == [True, False]


def test_confirm_without_roster_leaves_all_unconfirmed(monkeypatch):
    monkeypatch.setattr(v, "_roster_members", lambda system: [])
    out = v.confirm_findings("DIFF", [_finding()])
    assert out and all(not f.confirmed for f in out)


# --- verify_wave ---------------------------------------------------------------


@pytest.fixture
def wired(monkeypatch):
    monkeypatch.setattr(v, "run_tests", lambda repo: (True, "all green"))
    monkeypatch.setattr(v, "wave_diff", lambda repo, frm: "diff --git a/x b/x\n+new\n-old\n")
    monkeypatch.setattr(v, "collect_findings", lambda diff: [_finding()])
    confirmed = _finding()
    confirmed.confirmed = True
    monkeypatch.setattr(v, "confirm_findings", lambda diff, f: [confirmed])


def test_verify_wave_green_path(wired):
    report = v.verify_wave(Path("/repo"), wave=1, from_change="abc")
    assert report.suite_green
    assert report.findings and report.findings[0].confirmed
    assert report.diff_stat == "1 file(s), +1/-1"


def test_verify_wave_red_suite_is_hard_fail_but_still_reviews(wired, monkeypatch):
    monkeypatch.setattr(v, "run_tests", lambda repo: (False, "boom " * 600))
    report = v.verify_wave(Path("/repo"), wave=2, from_change="abc")
    assert not report.suite_green
    assert len(report.suite.output_tail) <= 2000  # tail capped for the journal
    assert report.findings  # review still ran — replan sees both signals


def test_verify_wave_empty_diff_skips_review(wired, monkeypatch):
    monkeypatch.setattr(v, "wave_diff", lambda repo, frm: "")
    called = []
    monkeypatch.setattr(v, "collect_findings", lambda diff: called.append(1))
    report = v.verify_wave(Path("/repo"), wave=3, from_change="abc")
    assert called == []
    assert report.findings == []
    assert report.diff_stat == "empty diff"


def test_verify_wave_skip_review_flag(wired):
    report = v.verify_wave(Path("/repo"), wave=4, from_change="abc", skip_review=True)
    assert report.findings == []


# --- vcs helpers against a real jj-less temp dir ----------------------------------


def test_vcs_helpers_raise_without_vcs(tmp_path):
    with pytest.raises(Exception, match="No VCS"):
        v.wave_start_rev(tmp_path)
    with pytest.raises(Exception, match="No VCS"):
        v.wave_diff(tmp_path, "abc")


# --- wave-start basis against a real jj repo ---------------------------------------
# Regression for the e2e dry-run finding: recording @'s change id made every wave
# diff empty, because the worker's describe-in-place commit turns @ itself into the
# landed commit. The basis must be the pre-wave tip (@-).


def _jj(tmp_path, *args):
    import os
    import subprocess

    env = {**os.environ, "JJ_USER": "test", "JJ_EMAIL": "test@example.com"}
    res = subprocess.run(
        ["jj", *args], cwd=tmp_path, capture_output=True, text=True, timeout=30, env=env
    )
    assert res.returncode == 0, f"jj {' '.join(args)} failed: {res.stderr}"
    return res.stdout


@pytest.mark.skipif(__import__("shutil").which("jj") is None, reason="jj not installed")
def test_wave_start_rev_sees_worker_style_commits(tmp_path):
    _jj(tmp_path, "git", "init")
    (tmp_path / "a.txt").write_text("one\n")
    # Land the pre-wave state the way the worker does: describe @ in place, then new.
    _jj(tmp_path, "describe", "-m", "pre-wave")
    _jj(tmp_path, "new")

    start = v.wave_start_rev(tmp_path)

    # The "wave": a leaf edits a file and the worker commits describe-in-place.
    (tmp_path / "b.txt").write_text("two\n")
    _jj(tmp_path, "describe", "-m", "auto: leaf lands")
    _jj(tmp_path, "new")

    diff = v.wave_diff(tmp_path, start)
    assert "b.txt" in diff, f"wave diff must contain the landed change, got: {diff!r}"
    assert "a.txt" not in diff  # pre-wave state is the basis, not part of the diff


# --- consolidation (the recipe's dedup stage) ---------------------------------------


def _env(findings=(), covered=()):
    return v.ConsolidationEnvelope(
        findings=[v.CanonicalFinding(**f) for f in findings],
        covered=[v.CoveredFinding(**c) for c in covered],
    )


def _structured_returning(monkeypatch, value):
    calls = []

    def fake_structured(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(value=value, error=None if value else "boom")

    monkeypatch.setattr(v, "structured", fake_structured)
    monkeypatch.setattr(v, "rotation_pool", lambda slots, role: object())
    return calls


def test_consolidate_merges_paraphrase_twins(monkeypatch):
    twins = [
        _finding("list-units crashes on empty domain table", "src/cli.py"),
        _finding("list-units crashes when the domain table is empty", "src/cli.py"),
        _finding("list-units crashes on empty domain table", None),  # file omitted
    ]
    _structured_returning(
        monkeypatch,
        _env(
            findings=[
                {
                    "summary": "list-units crashes on empty domain table",
                    "file": "src/cli.py",
                    "severity": "high",
                    "merged_slugs": [f.slug for f in twins],
                }
            ]
        ),
    )
    canonical, ok, dropped = v.consolidate_findings(twins)
    assert ok and dropped == []
    assert len(canonical) == 1  # three phrasings -> one canonical finding
    assert canonical[0].severity == "high"
    assert canonical[0].slug == v.stable_slug("src/cli.py", canonical[0].summary)


def test_consolidate_drops_candidates_covered_by_open_fixups(monkeypatch):
    finding = _finding("temperature offset applied twice", "src/temp.py")
    _structured_returning(
        monkeypatch,
        _env(covered=[{"slug": finding.slug, "by_task": "Fix double offset in K->F"}]),
    )
    canonical, ok, dropped = v.consolidate_findings(
        [finding, _finding()], ["Fix double offset in K->F"]
    )
    assert ok
    assert dropped == [f"{finding.slug} (covered by: Fix double offset in K->F)"]


def test_consolidate_fails_open_on_llm_failure(monkeypatch):
    findings = [_finding("a", "a.py"), _finding("b", "b.py")]
    _structured_returning(monkeypatch, None)
    canonical, ok, dropped = v.consolidate_findings(findings)
    assert not ok
    assert canonical == findings  # raw passthrough, wave never blocked


def test_consolidate_fails_open_when_consolidator_invents_findings(monkeypatch):
    findings = [_finding("only one", "a.py")]
    _structured_returning(
        monkeypatch,
        _env(
            findings=[
                {"summary": "only one", "file": "a.py", "severity": "low", "merged_slugs": []},
                {
                    "summary": "invented extra",
                    "file": "b.py",
                    "severity": "high",
                    "merged_slugs": [],
                },
            ]
        ),
    )
    canonical, ok, _ = v.consolidate_findings(findings, ["some open fixup"])
    assert not ok
    assert canonical == findings  # a consolidator may reduce work, never invent it


def test_consolidate_skips_llm_when_nothing_to_merge(monkeypatch):
    calls = _structured_returning(monkeypatch, _env())
    single = [_finding()]
    canonical, ok, dropped = v.consolidate_findings(single)
    assert (canonical, ok, dropped) == (single, True, [])
    assert calls == []  # no candidates to merge, no open fixups -> no call
    assert v.consolidate_findings([]) == ([], True, [])


def test_consolidate_normalizes_unknown_severity(monkeypatch):
    _structured_returning(
        monkeypatch,
        _env(
            findings=[
                {"summary": "s", "file": None, "severity": "catastrophic", "merged_slugs": []}
            ]
        ),
    )
    canonical, ok, _ = v.consolidate_findings([_finding("s1"), _finding("s2")])
    assert ok and canonical[0].severity == "medium"


def test_verify_wave_routes_findings_through_consolidation(wired, monkeypatch):
    seen = {}

    def fake_consolidate(findings, existing=None):
        seen["raw"] = list(findings)
        seen["existing"] = existing
        return [findings[0]], True, ["dead-slug (covered by: Open fixup)"]

    monkeypatch.setattr(v, "consolidate_findings", fake_consolidate)
    monkeypatch.setattr(v, "collect_findings", lambda diff: [_finding("x"), _finding("y")])
    report = v.verify_wave(Path("/repo"), wave=9, from_change="abc", existing_fixups=["Open fixup"])
    assert seen["existing"] == ["Open fixup"]
    assert report.raw_findings == 2
    assert report.consolidation_ok is True
    assert report.dropped_covered == ["dead-slug (covered by: Open fixup)"]
    assert len(report.findings) == 1  # confirm ran on the canonical set
