"""Epic branch + gate tests — bookmark lifecycle on a real temp git repo (the automerge test
precedent); the sign-off wiring with fake seats and a mocked quorum."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from forge.coding_pipeline import vcs_epic as ve
from forge.coding_pipeline.models import FramingProposal
from forge.shared.ensemble import ExecResult, ExecStatus, FailureClass, Prompt
from forge.shared.signoff import SeatVerdict, SignoffResult, SignoffSeat


def _git(repo: Path, *args: str) -> str:
    res = subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", *args],
        capture_output=True,
        text=True,
        cwd=repo,
    )
    assert res.returncode == 0, res.stderr
    return res.stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "-b", "main")
    (tmp_path / "a.txt").write_text("one\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-m", "c1")
    return tmp_path


def _framing() -> FramingProposal:
    return FramingProposal(
        goal_as_stated="g",
        restated_goal="restated goal",
        recommendation="the approved recommendation",
        value_ordering=["first slice ships value", "second slice polishes"],
        epic_slug="toy",
        approved=True,
    )


# --- bookmark lifecycle -------------------------------------------------------------


def test_ensure_creates_branch_at_tip_once(repo):
    branch = ve.ensure_epic_bookmark(repo, "toy", push=False, log=lambda m: None)
    assert branch == "pipeline/toy"
    first = _git(repo, "rev-parse", branch)
    assert first == _git(repo, "rev-parse", "HEAD")

    # a new commit lands; ensure must NOT move the existing bookmark
    (repo / "a.txt").write_text("two\n")
    _git(repo, "commit", "-am", "c2")
    ve.ensure_epic_bookmark(repo, "toy", push=False, log=lambda m: None)
    assert _git(repo, "rev-parse", branch) == first


def test_update_advances_branch_to_tip(repo):
    ve.ensure_epic_bookmark(repo, "toy", push=False, log=lambda m: None)
    (repo / "a.txt").write_text("two\n")
    _git(repo, "commit", "-am", "c2")
    ve.update_epic_bookmark(repo, "toy", push=False, log=lambda m: None)
    assert _git(repo, "rev-parse", "pipeline/toy") == _git(repo, "rev-parse", "HEAD")


def test_push_failure_is_a_warning_not_an_error(repo):
    # no remote configured: the push fails, the call still succeeds
    warnings: list[str] = []
    ve.ensure_epic_bookmark(repo, "toy", push=True, log=warnings.append)
    assert any("push" in w for w in warnings)


def test_epic_diff_is_merge_base_to_tip(repo):
    # epic branch gets a commit main doesn't have
    _git(repo, "checkout", "-b", "pipeline/toy")
    (repo / "b.txt").write_text("epic work\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "epic c2")
    _git(repo, "checkout", "main")

    diff = ve.epic_diff(repo, "toy")
    assert "epic work" in diff
    assert "b.txt" in diff


def test_no_vcs_raises(tmp_path):
    with pytest.raises(Exception, match="No VCS"):
        ve.ensure_epic_bookmark(tmp_path, "toy", push=False, log=lambda m: None)


# --- the gate -----------------------------------------------------------------------


def test_gate_passes_framing_context_and_fails_closed(monkeypatch, tmp_path):
    calls: dict = {}

    def fake_signoff(diff, **kwargs):
        calls["diff"] = diff
        calls.update(kwargs)
        return SignoffResult(approved=True, attempted=2, approvals=2, providers=["a", "b"])

    monkeypatch.setattr(ve, "epic_diff", lambda repo, slug, main="main": "THE-EPIC-DIFF")
    monkeypatch.setattr(ve, "full_quorum_signoff", fake_signoff)

    result = ve.run_epic_gate(tmp_path, "toy", _framing(), seats=[object(), object()])
    assert result.approved
    assert calls["diff"] == "THE-EPIC-DIFF"
    assert calls["ref"] == "pipeline/toy"
    # The gate judges scope against the WHOLE approved framing, not the goal line alone —
    # a goal emphasizing the first slice must not turn the rest of the scope into "drift".
    assert "restated goal" in calls["context"]
    assert "the approved recommendation" in calls["context"]
    assert "1. first slice ships value" in calls["context"]
    assert "2. second slice polishes" in calls["context"]
    assert calls["system"] == ve.EPIC_SIGNOFF_SYSTEM


def test_gate_blocks_empty_diff_without_any_llm_call(monkeypatch, tmp_path):
    def boom(*a, **k):
        raise AssertionError("no sign-off call for an empty diff")

    monkeypatch.setattr(ve, "epic_diff", lambda repo, slug, main="main": "")
    monkeypatch.setattr(ve, "full_quorum_signoff", boom)
    result = ve.run_epic_gate(tmp_path, "toy", _framing(), seats=[])
    assert not result.approved
    assert "empty epic diff" in result.reason


# --- the map-reduce path (oversized diffs) --------------------------------------------


class _SeatExec:
    """A fake seat executor: fixed OK summary, or a terminal failure with a reason."""

    def __init__(self, label: str, *, output: str = "No red flags.", ok: bool = True) -> None:
        self.label = label
        self._output = output
        self._ok = ok
        self.calls = 0

    async def run(self, prompt: Prompt, *, timeout: float) -> ExecResult:
        self.calls += 1
        if self._ok:
            return ExecResult(executor=self.label, status=ExecStatus.OK, output=self._output)
        return ExecResult(
            executor=self.label,
            status=ExecStatus.ERROR,
            error="boom: provider down",
            failure_class=FailureClass.TERMINAL,
        )


def _fake_diff(n_files: int, body_chars: int) -> str:
    """A unified diff with n single-hunk files, each body ~body_chars wide."""
    return "".join(
        f"diff --git a/f{i}.py b/f{i}.py\n--- a/f{i}.py\n+++ b/f{i}.py\n"
        f"@@ -1 +1 @@\n+{'x' * body_chars}\n"
        for i in range(n_files)
    )


def _shrink_gate_thresholds(monkeypatch) -> None:
    # 2 files x ~215 chars: over the 100-char single-pass ceiling, packs into 2 x 300-char slices.
    monkeypatch.setattr(ve.settings, "epic_gate_max_diff_chars", 100)
    monkeypatch.setattr(ve.settings, "epic_gate_chunk_chars", 300)


def test_oversized_diff_gates_via_map_reduce(monkeypatch, tmp_path):
    monkeypatch.setattr(ve, "epic_diff", lambda repo, slug, main="main": _fake_diff(2, 150))
    _shrink_gate_thresholds(monkeypatch)
    calls: dict = {}

    def fake_signoff(diff, **kwargs):
        calls["diff"] = diff
        calls.update(kwargs)
        return SignoffResult(approved=True, attempted=2, approvals=2, providers=["a", "b"])

    monkeypatch.setattr(ve, "full_quorum_signoff", fake_signoff)
    first = _SeatExec("s0", output="SUMMARY-OF-SLICE. No red flags.")
    seats = [
        SignoffSeat(provider="a", executor=first),
        SignoffSeat(provider="b", executor=_SeatExec("s1")),
    ]

    result = ve.run_epic_gate(tmp_path, "toy", _framing(), seats=seats)
    assert result.approved
    assert "map-reduce over 2 slice(s)" in result.strategy
    assert calls["system"] == ve.EPIC_REDUCE_SYSTEM
    assert "SUMMARY-OF-SLICE" in calls["diff"]  # the reduce verdict judges slice summaries...
    assert "xxxx" not in calls["diff"]  # ...never the raw oversized diff
    # the full framing (goal + recommendation + value ordering) reaches the reduce
    assert "restated goal" in calls["context"]
    assert "the approved recommendation" in calls["context"]
    assert "first slice ships value" in calls["context"]
    assert first.calls == 2  # failover map pool: the first healthy seat summarized both slices


def test_map_pool_prefers_the_cheap_seat_but_reduce_gets_the_full_roster(monkeypatch, tmp_path):
    monkeypatch.setattr(ve, "epic_diff", lambda repo, slug, main="main": _fake_diff(2, 150))
    _shrink_gate_thresholds(monkeypatch)
    monkeypatch.setattr(ve.settings, "epic_gate_map_preferred", "local")
    calls: dict = {}

    def fake_signoff(diff, **kwargs):
        calls.update(kwargs)
        return SignoffResult(approved=True, attempted=2, approvals=2, providers=["a", "l"])

    monkeypatch.setattr(ve, "full_quorum_signoff", fake_signoff)
    metered = _SeatExec("anthropic-exec")
    cheap = _SeatExec("local-exec")
    seats = [
        SignoffSeat(provider="anthropic", executor=metered),
        SignoffSeat(provider="local", executor=cheap),
    ]

    result = ve.run_epic_gate(tmp_path, "toy", _framing(), seats=seats)
    assert result.approved
    assert cheap.calls == 2 and metered.calls == 0  # map rides the local seat
    assert calls["seats"] == seats  # the reduce quorum still seats everyone, roster order


def test_failed_slice_summary_fails_closed_before_reduce(monkeypatch, tmp_path):
    monkeypatch.setattr(ve, "epic_diff", lambda repo, slug, main="main": _fake_diff(2, 150))
    _shrink_gate_thresholds(monkeypatch)

    def boom(*a, **k):
        raise AssertionError("no reduce verdict over an incomplete map")

    monkeypatch.setattr(ve, "full_quorum_signoff", boom)
    seats = [
        SignoffSeat(provider="a", executor=_SeatExec("s0", ok=False)),
        SignoffSeat(provider="b", executor=_SeatExec("s1", ok=False)),
    ]

    result = ve.run_epic_gate(tmp_path, "toy", _framing(), seats=seats)
    assert not result.approved
    assert "map stage incomplete" in result.reason
    assert "boom: provider down" in result.reason  # the WHY is in the verdict


def test_over_cap_split_blocks_instead_of_dropping_slices(monkeypatch, tmp_path):
    monkeypatch.setattr(ve, "epic_diff", lambda repo, slug, main="main": _fake_diff(2, 150))
    _shrink_gate_thresholds(monkeypatch)
    monkeypatch.setattr(ve.settings, "epic_gate_max_chunks", 1)

    class _Untouchable:
        label = "never"

        async def run(self, prompt: Prompt, *, timeout: float) -> ExecResult:
            raise AssertionError("no LLM call when the gate refuses the split")

    def no_reduce(*a, **k):
        raise AssertionError("no reduce verdict when the gate refuses the split")

    monkeypatch.setattr(ve, "full_quorum_signoff", no_reduce)
    seats = [SignoffSeat(provider="a", executor=_Untouchable())]

    result = ve.run_epic_gate(tmp_path, "toy", _framing(), seats=seats)
    assert not result.approved
    assert "refuses to drop slices" in result.reason


# --- rendering ------------------------------------------------------------------------


def test_render_approved_instructs_human_merge_only():
    result = SignoffResult(approved=True, attempted=3, approvals=3, providers=["a", "b", "c"])
    out = ve.render_epic_gate(result, "toy", tip="abc123")
    assert "APPROVED" in out
    assert "HUMAN merge" in out
    assert "pipeline/toy" in out and "abc123" in out
    assert "advance" not in out.lower() or "never advances" in out  # no auto-merge language
    # the EXACT merge command a human runs — the render is the pipeline's terminal action
    assert "jj bookmark set main -r pipeline/toy && jj git push --bookmark main" in out
    assert "3/3" in out and "a, b, c" in out  # quorum provenance visible


def test_approved_gate_never_touches_vcs(tmp_path, monkeypatch):
    """Approval RENDERS, never merges: an approved run_epic_gate must issue only
    read commands (diff) — no bookmark moves, no pushes, no commits (dry-run Q5)."""
    commands: list[list[str]] = []

    def spy_run(cmd, cwd):
        commands.append(cmd)
        return SimpleNamespace(returncode=0, stdout="diff --git a/x b/x\n+y\n", stderr="")

    monkeypatch.setattr(ve, "_run", spy_run)
    monkeypatch.setattr(ve, "detect_vcs", lambda repo: "jj")
    approved = SignoffResult(approved=True, attempted=2, approvals=2, providers=["x", "y"])
    monkeypatch.setattr(ve, "full_quorum_signoff", lambda *a, **k: approved)
    result = ve.run_epic_gate(tmp_path, "toy", _framing())
    assert result.approved
    mutating = ("bookmark", "push", "commit", "describe", "new", "set")
    for cmd in commands:
        assert not any(m in cmd for m in mutating), f"gate issued a mutating command: {cmd}"


def test_render_blocked_lists_blockers():
    result = SignoffResult(
        approved=False,
        attempted=3,
        approvals=2,
        reason="quorum 3/3, approvals 2/3",
        blockers=["test weakened in leaf 2"],
    )
    out = ve.render_epic_gate(result, "toy")
    assert "BLOCKED" in out
    assert "quorum 3/3" in out
    assert "test weakened" in out


def test_render_distinguishes_no_verdict_from_rejection():
    # The distill-evals failure mode: 0/2 responders must NOT read as a unanimous rejection.
    result = SignoffResult(
        approved=False,
        attempted=2,
        approvals=0,
        reason="quorum 0/2, approvals 0/2",
        seats=[
            SeatVerdict(provider="anthropic", reason="AuthenticationError: missing api key"),
            SeatVerdict(provider="local", approve=False),
        ],
        strategy="map-reduce over 12 slice(s) (diff 1500000 chars)",
    )
    out = ve.render_epic_gate(result, "toy")
    assert "Strategy: map-reduce over 12 slice(s)" in out
    assert "anthropic: NO VERDICT (AuthenticationError: missing api key)" in out
    assert "local: responded — did NOT approve" in out


def test_render_approved_lists_seat_verdicts():
    result = SignoffResult(
        approved=True,
        attempted=2,
        approvals=2,
        providers=["a", "b"],
        seats=[
            SeatVerdict(provider="a", approve=True, reason="clean"),
            SeatVerdict(provider="b", approve=True),
        ],
    )
    out = ve.render_epic_gate(result, "toy")
    assert "a: approved — clean" in out
    assert "b: approved" in out


def test_jj_push_argv_has_no_allow_new(monkeypatch):
    """jj 0.42 dropped --allow-new (new bookmarks push by default). Verify the argv."""
    captured: list[list[str]] = []

    def fake_run(cmd, cwd=None):
        captured.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(ve, "_run", fake_run)

    ve._push_branch(Path("/tmp"), "jj", "pipeline/toy", lambda m: None)

    # The command should contain --bookmark but NOT --allow-new
    cmd = captured[0]
    assert "--bookmark" in cmd
    assert "--allow-new" not in cmd


# --- diff-literacy manifest (gate-local-seat-diff-literacy) -------------------------


def test_diff_manifest_counts_per_file():
    diff = (
        "diff --git a/pkg/mod.py b/pkg/mod.py\n"
        "--- a/pkg/mod.py\n"
        "+++ b/pkg/mod.py\n"
        "@@ -1,2 +1,3 @@\n"
        " keep\n"
        "+added one\n"
        "+added two\n"
        "-removed\n"
        "diff --git a/pkg/tests/test_mod.py b/pkg/tests/test_mod.py\n"
        "--- /dev/null\n"
        "+++ b/pkg/tests/test_mod.py\n"
        "@@ -0,0 +1,1 @@\n"
        "+def test_x(): ...\n"
    )
    manifest = ve.diff_manifest(diff)
    assert "2 file(s)" in manifest
    assert "- pkg/mod.py: +2/-1" in manifest
    assert "- pkg/tests/test_mod.py: +1/-0" in manifest
    assert "unchanged, not missing" in manifest


def test_diff_manifest_empty_for_empty_diff():
    assert ve.diff_manifest("") == ""


def test_single_pass_gate_prepends_manifest(monkeypatch, tmp_path):
    seen = {}

    def fake_signoff(diff_text, **kwargs):
        seen["body"] = diff_text
        return SignoffResult(approved=True, attempted=2, approvals=2)

    monkeypatch.setattr(ve, "full_quorum_signoff", fake_signoff)
    monkeypatch.setattr(
        ve,
        "epic_diff",
        lambda repo, slug, main="main": (
            "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a\n+b\n"
        ),
    )
    ve.run_epic_gate(tmp_path, "toy", _framing(), seats=[object(), object()])
    assert seen["body"].startswith("## File manifest")
    assert "- x.py: +1/-1" in seen["body"]
    assert "diff --git a/x.py" in seen["body"]  # the diff itself still follows


def test_signoff_prompt_carries_diff_literacy_rules():
    for prompt in (ve.EPIC_SIGNOFF_SYSTEM, ve.EPIC_MAP_SYSTEM):
        assert "omits unchanged lines" in prompt
    assert "DIFF LITERACY" in ve.EPIC_SIGNOFF_SYSTEM
    assert "manifest" in ve.EPIC_SIGNOFF_SYSTEM
