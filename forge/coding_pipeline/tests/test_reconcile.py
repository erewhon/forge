"""Reconcile barrier tests — scripted-jj unit cases plus a real-jj integration proof.

The unit cases pin argv shapes and the per-leaf fail-closed decisions; the integration
test (skipped without jj on PATH) proves the load-bearing claim: two green-in-isolation
commits touching the same line produce a first-class conflicted commit that the barrier
detects, abandons, and demotes — while disjoint commits chain linearly.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from forge.coding_pipeline import reconcile as rc
from forge.coding_pipeline.models import LeafOutcome
from forge.shared.workspaces import JJError


def _outcome(leaf: str) -> LeafOutcome:
    return LeafOutcome(leaf=leaf, status="done", commit_id=f"cid-{leaf}", changed_files=["f"])


class _Proc:
    def __init__(self, stdout: str = "") -> None:
        self.stdout = stdout
        self.returncode = 0


class ScriptedJJ:
    """A fake _run_jj: records every argv; answers change-id and conflict queries from
    the script; raises JJError where told to."""

    def __init__(self, conflicted: set[str] | None = None, fail_rebase_of: set[str] | None = None):
        self.calls: list[list[str]] = []
        self.conflicted = conflicted or set()
        self.fail_rebase_of = fail_rebase_of or set()

    def __call__(self, args: list[str], *, cwd: Path, check: bool = True) -> _Proc:
        self.calls.append(args)
        if args[0] == "log" and "change_id.short()" in args:
            return _Proc(stdout=args[2].replace("rev-", "chg-"))  # rev-a -> chg-a
        if args[0] == "log":  # conflict template query
            return _Proc(stdout="true" if args[2] in self.conflicted else "false")
        if args[0] == "rebase":
            if args[2] in self.fail_rebase_of:
                raise JJError(f"rebase of {args[2]} exploded")
            return _Proc()
        if args[0] == "resolve":
            return _Proc(stdout="forge/x.py    2-sided conflict")
        return _Proc()


@pytest.fixture
def jj(monkeypatch):
    scripted = ScriptedJJ()
    monkeypatch.setattr(rc, "_run_jj", scripted)
    return scripted


def test_clean_chain_lands_in_dispatch_order(jj, tmp_path):
    demotes: list = []
    results = rc.reconcile_wave(
        tmp_path,
        "base0",
        [(_outcome("A"), "rev-a"), (_outcome("B"), "rev-b")],
        on_demote=lambda t, n: demotes.append(t),
        log=lambda m: None,
    )
    assert [r.status for r in results] == ["done", "done"]
    assert demotes == []
    rebases = [c for c in jj.calls if c[0] == "rebase"]
    # each leaf rebases onto the ACCUMULATING head, addressed by stable change id
    assert rebases == [
        ["rebase", "-r", "chg-a", "-d", "base0"],
        ["rebase", "-r", "chg-b", "-d", "chg-a"],
    ]
    assert jj.calls[-1] == ["new", "chg-b"]  # working copy repositioned on the final head


def test_conflicted_leaf_is_abandoned_and_demoted(jj, tmp_path):
    jj.conflicted = {"chg-b"}
    demotes: list[tuple[str, str]] = []
    results = rc.reconcile_wave(
        tmp_path,
        "base0",
        [(_outcome("A"), "rev-a"), (_outcome("B"), "rev-b"), (_outcome("C"), "rev-c")],
        on_demote=lambda t, n: demotes.append((t, n)),
        log=lambda m: None,
    )
    assert [r.status for r in results] == ["done", "failed", "done"]
    assert results[1].commit_id is None  # the commit is gone; never report a stale id
    assert "integration conflict" in results[1].reason
    assert demotes and demotes[0][0] == "B"
    assert "forge/x.py" in demotes[0][1]  # the resolve --list detail made it into the note
    # the conflicted-file read happens BEFORE the abandon destroys the evidence
    resolve_i = jj.calls.index(["resolve", "--list", "-r", "chg-b"])
    abandon_i = jj.calls.index(["abandon", "chg-b"])
    assert resolve_i < abandon_i
    # C lands on A's head — the conflicted B never became the head
    assert ["rebase", "-r", "chg-c", "-d", "chg-a"] in jj.calls
    assert jj.calls[-1] == ["new", "chg-c"]


def test_jj_failure_demotes_without_abandoning(jj, tmp_path):
    jj.fail_rebase_of = {"chg-a"}
    demotes: list[tuple[str, str]] = []
    results = rc.reconcile_wave(
        tmp_path,
        "base0",
        [(_outcome("A"), "rev-a"), (_outcome("B"), "rev-b")],
        on_demote=lambda t, n: demotes.append((t, n)),
        log=lambda m: None,
    )
    assert results[0].status == "failed"
    assert "reconcile jj failure" in results[0].reason
    assert not any(c[0] == "abandon" for c in jj.calls)  # infra error never destroys work
    assert "NOT abandoned" in demotes[0][1]
    # B still integrates, onto the untouched base
    assert ["rebase", "-r", "chg-b", "-d", "base0"] in jj.calls
    assert results[1].status == "done"


def test_failing_demotion_write_keeps_batch_alive_and_shouts(jj, tmp_path):
    jj.conflicted = {"chg-a"}

    def bad_demote(title, note):
        raise RuntimeError("forge down")

    results = rc.reconcile_wave(
        tmp_path,
        "base0",
        [(_outcome("A"), "rev-a"), (_outcome("B"), "rev-b")],
        on_demote=bad_demote,
        log=lambda m: None,
    )
    assert results[0].status == "failed"
    assert "DEMOTION WRITE FAILED" in results[0].reason  # the Forge mismatch is legible
    assert results[1].status == "done"  # batch-mates unaffected


def test_all_demoted_leaves_working_copy_alone(jj, tmp_path):
    jj.conflicted = {"chg-a", "chg-b"}
    rc.reconcile_wave(
        tmp_path,
        "base0",
        [(_outcome("A"), "rev-a"), (_outcome("B"), "rev-b")],
        on_demote=lambda t, n: None,
        log=lambda m: None,
    )
    assert not any(c[0] == "new" for c in jj.calls)  # head never advanced — nothing to park on


def test_empty_landed_is_a_no_op(jj, tmp_path):
    assert (
        rc.reconcile_wave(tmp_path, "base0", [], on_demote=lambda t, n: None, log=lambda m: None)
        == []
    )
    assert jj.calls == []


def test_reposition_failure_is_repo_level(jj, tmp_path, monkeypatch):
    real = jj.__call__

    def flaky(args, *, cwd, check=True):
        if args[0] == "new":
            raise JJError("workspace stale")
        return real(args, cwd=cwd, check=check)

    monkeypatch.setattr(rc, "_run_jj", flaky)
    with pytest.raises(rc.ReconcileError, match="repositioning"):
        rc.reconcile_wave(
            tmp_path,
            "base0",
            [(_outcome("A"), "rev-a")],
            on_demote=lambda t, n: None,
            log=lambda m: None,
        )


# --- bisect-on-red -------------------------------------------------------------------


def _suite_script(red_after: set[str]):
    """A run_suite fake keyed on the LAST `jj new <rev>` the ScriptedJJ saw: red when the
    working copy sits on (a child of) a rev in `red_after`."""

    def make(jj: ScriptedJJ):
        def run_suite():
            positioned = [c[1] for c in jj.calls if c[0] == "new"]
            here = positioned[-1] if positioned else "base0"
            return (here not in red_after), f"suite at {here}"

        return run_suite

    return make


def _chain3():
    return [("A", "chg-a"), ("B", "chg-b"), ("C", "chg-c")]


def test_bisect_attributes_first_red_and_repairs(jj, tmp_path):
    # green at A, red at B (and would be red at C — never tested: walk stops at first red)
    run_suite = _suite_script({"chg-b"})(jj)
    demotes: list[tuple[str, str]] = []
    result = rc.bisect_red(
        tmp_path, "base0", _chain3(), run_suite, on_demote=lambda t, n: demotes.append((t, n)),
        log=lambda m: None,
    )
    assert result is not None
    assert result.offender == "B"
    assert result.repaired_green is True  # C on A is green in this script
    # the rest of the chain moved off B before B was abandoned, then @ went to the tip
    rebase_i = jj.calls.index(["rebase", "-s", "chg-c", "-d", "chg-a"])
    abandon_i = jj.calls.index(["abandon", "chg-b"])
    assert rebase_i < abandon_i
    assert ["new", "chg-c"] in jj.calls
    assert demotes and demotes[0][0] == "B"
    assert "semantic conflict" in demotes[0][1]


def test_bisect_offender_first_rebases_rest_onto_base(jj, tmp_path):
    run_suite = _suite_script({"chg-a"})(jj)
    result = rc.bisect_red(
        tmp_path, "base0", _chain3(), run_suite, on_demote=lambda t, n: None, log=lambda m: None
    )
    assert result is not None and result.offender == "A"
    assert ["rebase", "-s", "chg-b", "-d", "base0"] in jj.calls


def test_bisect_offender_last_needs_no_rebase(jj, tmp_path):
    run_suite = _suite_script({"chg-c"})(jj)
    result = rc.bisect_red(
        tmp_path, "base0", _chain3(), run_suite, on_demote=lambda t, n: None, log=lambda m: None
    )
    assert result is not None and result.offender == "C"
    assert not any(c[:2] == ["rebase", "-s"] for c in jj.calls)
    assert ["abandon", "chg-c"] in jj.calls
    assert ["new", "chg-b"] in jj.calls  # tip is now the previous leaf


def test_bisect_all_green_returns_none_and_touches_nothing(jj, tmp_path):
    run_suite = _suite_script(set())(jj)  # flake: red wave, green walk
    result = rc.bisect_red(
        tmp_path, "base0", _chain3(), run_suite, on_demote=lambda t, n: None, log=lambda m: None
    )
    assert result is None
    assert not any(c[0] in ("abandon", "rebase") for c in jj.calls)


def test_bisect_still_red_after_repair_demotes_anyway(jj, tmp_path):
    # B is the first red point; backing it out does NOT green the suite (C×A conflict too).
    run_suite = _suite_script({"chg-b", "chg-c"})(jj)
    demotes: list[str] = []
    result = rc.bisect_red(
        tmp_path, "base0", _chain3(), run_suite, on_demote=lambda t, n: demotes.append(t),
        log=lambda m: None,
    )
    assert result is not None
    assert result.offender == "B"  # first-red evidence stands
    assert result.repaired_green is False  # the remaining red flows to replan
    assert demotes == ["B"]


def test_bisect_jj_failure_fails_open(jj, tmp_path):
    jj.fail_rebase_of = {"chg-c"}  # the back-out rebase explodes
    run_suite = _suite_script({"chg-b"})(jj)
    demotes: list[str] = []
    result = rc.bisect_red(
        tmp_path, "base0", _chain3(), run_suite, on_demote=lambda t, n: demotes.append(t),
        log=lambda m: None,
    )
    assert result is None
    assert demotes == []  # no attribution claimed when the surgery failed


# --- apply_bisect (the orchestrator hook) ---------------------------------------------


def _report(*outcomes: LeafOutcome, green: bool):
    from forge.coding_pipeline.models import SuiteResult, WaveReport

    return WaveReport(
        wave=1, outcomes=list(outcomes), suite=SuiteResult(passed=green, output_tail="t")
    )


def _done_leaf(title: str) -> LeafOutcome:
    return LeafOutcome(leaf=title, status="done", commit_id=f"cid-{title}")


def test_apply_bisect_noop_when_suite_green(jj, tmp_path):
    report = _report(_done_leaf("A"), _done_leaf("B"), green=True)
    out = rc.apply_bisect(
        report, repo=tmp_path, base_rev="base0", concurrency=3,
        run_suite=lambda: (True, ""), on_demote=lambda t, n: None, log=lambda m: None,
    )
    assert out is None
    assert jj.calls == []


def test_apply_bisect_noop_for_serial_waves(jj, tmp_path):
    report = _report(_done_leaf("A"), _done_leaf("B"), green=False)
    out = rc.apply_bisect(
        report, repo=tmp_path, base_rev="base0", concurrency=1,
        run_suite=lambda: (True, ""), on_demote=lambda t, n: None, log=lambda m: None,
    )
    assert out is None
    assert jj.calls == []


def test_apply_bisect_noop_for_single_landed_leaf(jj, tmp_path):
    failed = LeafOutcome(leaf="B", status="failed", reason="x")
    report = _report(_done_leaf("A"), failed, green=False)
    out = rc.apply_bisect(
        report, repo=tmp_path, base_rev="base0", concurrency=3,
        run_suite=lambda: (True, ""), on_demote=lambda t, n: None, log=lambda m: None,
    )
    assert out is None  # one landed leaf IS the attribution already


def test_apply_bisect_rewrites_offender_and_suite(jj, tmp_path):
    report = _report(_done_leaf("A"), _done_leaf("B"), green=False)
    # apply_bisect builds the chain from commit_ids (cid-A, cid-B); ScriptedJJ echoes
    # them back as their own change ids, so "red at cid-B" scripts the walk directly

    def run_suite():
        news = [c[1] for c in jj.calls if c[0] == "new"]
        here = news[-1] if news else "base0"
        return (here != "cid-B"), "tail-text"

    demotes: list[str] = []
    out = rc.apply_bisect(
        report, repo=tmp_path, base_rev="base0", concurrency=3,
        run_suite=run_suite, on_demote=lambda t, n: demotes.append(t), log=lambda m: None,
    )
    assert out is not None and out.offender == "B"
    by_leaf = {o.leaf: o for o in report.outcomes}
    assert by_leaf["B"].status == "failed"
    assert by_leaf["B"].commit_id is None
    assert "bisect" in by_leaf["B"].reason
    assert by_leaf["A"].status == "done"
    assert report.suite is not None and report.suite.passed is True  # repaired verdict
    assert demotes == ["B"]


# --- the real thing -----------------------------------------------------------------


needs_jj = pytest.mark.skipif(shutil.which("jj") is None, reason="jj not on PATH")


def _jj(args: list[str], cwd: Path) -> str:
    env = os.environ | {"JJ_USER": "test", "JJ_EMAIL": "test@test.invalid"}
    proc = subprocess.run(["jj", *args], cwd=cwd, capture_output=True, text=True, env=env)
    assert proc.returncode == 0, f"jj {args}: {proc.stderr}"
    return proc.stdout.strip()


def _make_sibling(repo: Path, base: str, path: str, content: str, msg: str) -> str:
    """A described commit editing one file as a direct child of base; @ parked back off it."""
    _jj(["new", base], cwd=repo)
    (repo / path).write_text(content)
    _jj(["describe", "-m", msg], cwd=repo)
    change = _jj(["log", "-r", "@", "--no-graph", "-T", "change_id.short()"], cwd=repo)
    _jj(["new", base], cwd=repo)  # park @ elsewhere so the sibling is a bare head
    return change


@needs_jj
def test_real_jj_conflict_detected_and_disjoint_chain_lands(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _jj(["git", "init"], cwd=repo)
    (repo / "f.txt").write_text("line1\nline2\n")
    _jj(["describe", "-m", "base"], cwd=repo)
    base = _jj(["log", "-r", "@", "--no-graph", "-T", "commit_id"], cwd=repo)

    # A and B both rewrite f.txt line 1 (collision); C touches its own file (disjoint).
    chg_a = _make_sibling(repo, base, "f.txt", "AAA\nline2\n", "A")
    chg_b = _make_sibling(repo, base, "f.txt", "BBB\nline2\n", "B")
    chg_c = _make_sibling(repo, base, "g.txt", "gee\n", "C")

    demotes: list[tuple[str, str]] = []
    results = rc.reconcile_wave(
        repo,
        base,
        [(_outcome("A"), chg_a), (_outcome("B"), chg_b), (_outcome("C"), chg_c)],
        on_demote=lambda t, n: demotes.append((t, n)),
        log=lambda m: None,
    )

    assert [r.status for r in results] == ["done", "failed", "done"]
    assert [t for t, _ in demotes] == ["B"]
    assert "f.txt" in demotes[0][1]  # the conflicted path is named in the note
    # the surviving chain is linear: base <- A <- C <- @, and B is gone
    parent_of_c = _jj(["log", "-r", chg_c + "-", "--no-graph", "-T", "change_id.short()"], cwd=repo)
    assert parent_of_c == chg_a
    at_parent = _jj(["log", "-r", "@-", "--no-graph", "-T", "change_id.short()"], cwd=repo)
    assert at_parent == chg_c
    desc_tpl = 'description.first_line() ++ "|"'
    all_descs = _jj(["log", "-r", "::", "--no-graph", "-T", desc_tpl], cwd=repo)
    assert "B|" not in all_descs  # abandoned, not lurking
    # nothing in the surviving graph is conflicted
    conflicted = _jj(["log", "-r", "::", "--no-graph", "-T", 'if(conflict, "C", "")'], cwd=repo)
    assert conflicted == ""
    # the working copy holds the integrated content of BOTH survivors
    assert (repo / "f.txt").read_text() == "AAA\nline2\n"
    assert (repo / "g.txt").read_text() == "gee\n"
