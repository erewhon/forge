"""Framing-stage tests: mocked pool (no LLM), persistence guards, and the approval gate."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from forge.coding_pipeline import architect as arch
from forge.coding_pipeline.architect import (
    ArchitectError,
    FramingExistsError,
    FramingNotApprovedError,
    approve_framing,
    load_framing,
    persist_framing,
    propose_framing,
    render_framing,
    require_approved_framing,
)
from forge.coding_pipeline.models import (
    EscalateAction,
    FramingProposal,
    GoalSpec,
    Inventory,
    LeafOutcome,
    LeafSpec,
    ReviewFinding,
    SuiteResult,
    TaskTree,
    WaveReport,
)


def _proposal(**overrides) -> FramingProposal:
    base = dict(
        goal_as_stated="build web parity",
        restated_goal="serve the desktop frontend from the daemon",
        rescoped=True,
        recommendation="platform-shim approach",
        epic_slug="web-shim",
        risks=["scope creep"],
        value_ordering=["read-only view first"],
    )
    base.update(overrides)
    return FramingProposal.model_validate(base)


def _goal(**overrides) -> GoalSpec:
    base = dict(goal="build web parity", project="Nous")
    base.update(overrides)
    return GoalSpec.model_validate(base)


def _inventory() -> Inventory:
    return Inventory(project="Nous", repo="/repo", tree="src/\n  app.rs")


def _mock_structured(monkeypatch, value, error=None, raw=""):
    calls: list[dict] = []

    def fake(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(value=value, error=error, ok=value is not None, raw=raw)

    monkeypatch.setattr(arch, "structured", fake)
    return calls


# --- propose_framing -----------------------------------------------------------


def test_propose_framing_forces_approved_false(monkeypatch):
    # even a model that claims approval gets it stripped
    _mock_structured(monkeypatch, _proposal(approved=True))
    out = propose_framing(_goal(), _inventory())
    assert out.approved is False


def test_propose_framing_prompt_carries_goal_and_inventory(monkeypatch):
    calls = _mock_structured(monkeypatch, _proposal())
    propose_framing(_goal(context="daemon exists", value_hints=["viewer first"]), _inventory())
    user = calls[0]["user"]
    assert "build web parity" in user
    assert "daemon exists" in user
    assert "- viewer first" in user
    assert "app.rs" in user  # rendered inventory included
    assert calls[0]["schema"] is FramingProposal


def test_propose_framing_human_slug_beats_model_slug(monkeypatch):
    _mock_structured(monkeypatch, _proposal(epic_slug="model-idea"))
    out = propose_framing(_goal(epic_slug="human-choice"), _inventory())
    assert out.epic_slug == "human-choice"


def test_propose_framing_pool_exhaustion_raises(monkeypatch):
    _mock_structured(monkeypatch, None, error="pool exhausted")
    with pytest.raises(ArchitectError, match="pool exhausted"):
        propose_framing(_goal(), _inventory())


# --- persistence + approval gate -------------------------------------------------


def test_persist_refuses_overwrite_without_force(tmp_path):
    persist_framing(_proposal(), tmp_path)
    with pytest.raises(FramingExistsError):
        persist_framing(_proposal(recommendation="new idea"), tmp_path)
    # force overwrites
    persist_framing(_proposal(recommendation="new idea"), tmp_path, force=True)
    assert load_framing(tmp_path).recommendation == "new idea"


def test_persist_writes_json_and_md(tmp_path):
    persist_framing(_proposal(), tmp_path)
    assert (tmp_path / "framing.json").exists()
    md = (tmp_path / "framing.md").read_text()
    assert "Architect push-back" in md  # rescoped framing is called out
    assert "NO — review and approve" in md


def test_gate_refuses_missing_and_unapproved_framing(tmp_path):
    with pytest.raises(ArchitectError, match="no framing.json"):
        require_approved_framing(tmp_path)
    persist_framing(_proposal(), tmp_path)
    with pytest.raises(FramingNotApprovedError):
        require_approved_framing(tmp_path)


def test_approve_flips_gate_open(tmp_path):
    persist_framing(_proposal(), tmp_path)
    approved = approve_framing(tmp_path)
    assert approved.approved is True
    # gate now passes, and the persisted md reflects approval
    assert require_approved_framing(tmp_path).approved is True
    assert "Approved:** yes" in (tmp_path / "framing.md").read_text()


def test_approve_without_framing_raises(tmp_path):
    with pytest.raises(ArchitectError, match="no framing.json"):
        approve_framing(tmp_path)


def test_render_framing_unrescoped_has_no_pushback_banner():
    md = render_framing(_proposal(rescoped=False))
    assert "Architect push-back" not in md
    assert "## Restated goal" in md


# --- A2: decomposition ------------------------------------------------------


def _leaf(title: str, **overrides) -> LeafSpec:
    base = dict(title=title, content="spec", feature="Web Shim", estimate="s")
    base.update(overrides)
    return LeafSpec.model_validate(base)


def _shaped(title: str, **flags) -> arch.LeafBoundedness:
    base = dict(
        leaf_title=title,
        single_concern=True,
        bounded_diff=True,
        small_estimate=True,
        testable_acceptance=True,
        files_named=True,
    )
    base.update(flags)
    return arch.LeafBoundedness.model_validate(base)


def _mock_decompose(monkeypatch, leaves, verdicts=None):
    """Wire structured() to return a TaskTree and discover() to return boundedness verdicts.
    verdicts=None means 'all leaves pass'."""
    calls = _mock_structured(monkeypatch, TaskTree(leaves=leaves) if leaves else None)
    if verdicts is None:
        verdicts = [_shaped(leaf.title) for leaf in leaves or []]
    monkeypatch.setattr(arch, "discover", lambda finders, **kw: verdicts)
    return calls


def test_decompose_refuses_unapproved_framing(monkeypatch):
    _mock_decompose(monkeypatch, [_leaf("a")])
    with pytest.raises(FramingNotApprovedError):
        arch.decompose(_proposal(approved=False), _inventory())


def test_decompose_applies_conservative_floor(monkeypatch):
    leaves = [
        _leaf("auto leaf", execution_mode="Auto-OK", requires_tests=False, max_files=None),
        _leaf("novel leaf", execution_mode="Auto-OK", complexity="novel"),
        _leaf("specless leaf", execution_mode="Auto-Preferred", status="Spec Needed"),
        _leaf("blank feature", feature=" "),
    ]
    _mock_decompose(monkeypatch, leaves)
    out = arch.decompose(_proposal(approved=True), _inventory())
    auto = next(leaf for leaf in out if leaf.title == "auto leaf")
    assert auto.requires_tests is True  # auto always tests
    assert auto.max_files == 5  # auto always capped
    assert auto.model_tier == "coder"  # bare/unset tier floors to a tool-capable one
    assert next(le for le in out if le.title == "novel leaf").execution_mode == "Manual"
    assert next(le for le in out if le.title == "specless leaf").execution_mode == "Manual"
    assert next(le for le in out if le.title == "blank feature").feature == "Web Shim"


def test_model_tier_floor_respects_explicit_tiers_and_manual(monkeypatch):
    leaves = [
        _leaf("bare auto", execution_mode="Auto-OK", model_tier="auto"),
        _leaf("cloud pool", execution_mode="Auto-OK", model_tier="auto-full"),
        _leaf("manual leaf", execution_mode="Manual", model_tier="auto"),
    ]
    _mock_decompose(monkeypatch, leaves)
    out = arch.decompose(_proposal(approved=True), _inventory())
    assert next(le for le in out if le.title == "bare auto").model_tier == "coder"
    # explicit larger pools stand — the floor only replaces unset/"auto"
    assert next(le for le in out if le.title == "cloud pool").model_tier == "auto-full"
    # Manual leaves are human-owned; their tier is not the pipeline's business
    assert next(le for le in out if le.title == "manual leaf").model_tier == "auto"


def test_requires_tests_floors_max_files_to_3(monkeypatch):
    """Auto-OK leaves with requires_tests must have max_files >= 3 (impl + test + incidental)."""
    leaves = [
        _leaf("needs tests low cap", execution_mode="Auto-OK", requires_tests=True, max_files=1),
        _leaf("needs tests exact", execution_mode="Auto-OK", requires_tests=True, max_files=2),
        _leaf("needs tests ok", execution_mode="Auto-OK", requires_tests=True, max_files=3),
        _leaf("no tests", execution_mode="Auto-OK", requires_tests=False, max_files=1),
    ]
    _mock_decompose(monkeypatch, leaves)
    out = arch.decompose(_proposal(approved=True), _inventory())
    by_title = {le.title: le for le in out}
    assert by_title["needs tests low cap"].max_files == 3
    assert by_title["needs tests exact"].max_files == 3
    assert by_title["needs tests ok"].max_files == 3
    # Confirmed wave finding (pipeline:build wave 2): a requires_tests=False Auto-OK
    # leaf is force-flipped to requires_tests=True by the conservative tags, so the
    # floor applies to it too — assert it, don't just feed it in.
    assert by_title["no tests"].requires_tests is True
    assert by_title["no tests"].max_files == 3
    # Manual leaves get the floor too: Manual-authored leaves are routinely re-armed
    # to Auto-OK later (deps-v2, live) and carry their tight caps with them.
    manual = _leaf("manual needs tests", execution_mode="Manual", requires_tests=True, max_files=1)
    _mock_decompose(monkeypatch, [manual])
    out2 = arch.decompose(_proposal(approved=True), _inventory())
    assert out2[0].max_files == 3


def test_decompose_rejects_unknown_dep(monkeypatch):
    _mock_decompose(monkeypatch, [_leaf("a", depends_on=["ghost"])])
    with pytest.raises(ArchitectError, match="unknown titles"):
        arch.decompose(_proposal(approved=True), _inventory())


def test_decompose_rejects_dependency_cycle(monkeypatch):
    _mock_decompose(monkeypatch, [_leaf("a", depends_on=["b"]), _leaf("b", depends_on=["a"])])
    with pytest.raises(ArchitectError, match="cycle"):
        arch.decompose(_proposal(approved=True), _inventory())


def test_boundedness_failure_demotes_auto_leaf(monkeypatch):
    leaves = [_leaf("wobbly", execution_mode="Auto-OK")]
    _mock_decompose(
        monkeypatch, leaves, verdicts=[_shaped("wobbly", files_named=False, notes="no files")]
    )
    out = arch.decompose(_proposal(approved=True), _inventory())
    assert out[0].execution_mode == "Manual"
    assert out[0].complexity == "novel"
    assert out[0].boundedness is not None and not out[0].boundedness.worker_shaped


def test_missing_boundedness_verdict_demotes_fail_closed(monkeypatch):
    leaves = [_leaf("unchecked", execution_mode="Auto-OK")]
    _mock_decompose(monkeypatch, leaves, verdicts=[])  # finder dropped, no verdict
    out = arch.decompose(_proposal(approved=True), _inventory())
    assert out[0].execution_mode == "Manual"
    assert "unavailable" in out[0].boundedness.notes


def test_manual_leaf_keeps_mode_despite_failing_boundedness(monkeypatch):
    leaves = [_leaf("big design", execution_mode="Manual", complexity="novel")]
    _mock_decompose(monkeypatch, leaves, verdicts=[_shaped("big design", single_concern=False)])
    out = arch.decompose(_proposal(approved=True), _inventory())
    assert out[0].execution_mode == "Manual"  # terminal state, not an error


def test_decompose_pool_exhaustion_raises(monkeypatch):
    _mock_decompose(monkeypatch, None)
    with pytest.raises(ArchitectError, match="no usable tree"):
        arch.decompose(_proposal(approved=True), _inventory())


def test_persist_and_load_tree_round_trip(tmp_path, monkeypatch):
    leaves = [_leaf("a"), _leaf("b", depends_on=["a"])]
    arch.persist_tree(leaves, tmp_path)
    assert arch.load_tree(tmp_path) == leaves
    md = (tmp_path / "tree.md").read_text()
    assert "**a**" in md and "← a" in md


# --- A4: replan --------------------------------------------------------------


def _report(**overrides) -> WaveReport:
    base = dict(wave=1, suite=SuiteResult(passed=True))
    base.update(overrides)
    return WaveReport.model_validate(base)


def _failed_leaf(title: str, reason: str = "tests red") -> LeafOutcome:
    return LeafOutcome(leaf=title, status="failed", reason=reason)


def _landed_leaf(title: str) -> LeafOutcome:
    return LeafOutcome(leaf=title, status="done", commit_id="abc")


def _finding(slug: str, confirmed: bool = True) -> ReviewFinding:
    return ReviewFinding(slug=slug, summary=f"issue {slug}", confirmed=confirmed)


def _envelope(monkeypatch, actions):
    """Wire structured() to return a ReplanEnvelope; returns the call-recorder."""
    return _mock_structured(monkeypatch, arch.ReplanEnvelope(actions=actions))


def _structured_spy(monkeypatch):
    calls = []
    monkeypatch.setattr(
        arch, "structured", lambda **kw: calls.append(kw) or (_ for _ in ()).throw(AssertionError)
    )
    return calls


def test_attempt_cap_escalation_is_deterministic_no_llm(monkeypatch):
    calls = _structured_spy(monkeypatch)  # any LLM call would raise
    report = _report(outcomes=[_failed_leaf("wobbly")])
    actions = arch.replan(_proposal(approved=True), [_leaf("wobbly")], report, {"wobbly": 2})
    assert calls == []
    assert len(actions) == 1
    assert isinstance(actions[0], EscalateAction)
    assert actions[0].leaf_title == "wobbly"
    assert "tests red" in actions[0].diagnostics


def test_clean_wave_returns_no_actions_without_llm(monkeypatch):
    _structured_spy(monkeypatch)
    report = _report(outcomes=[_landed_leaf("fine")])
    assert arch.replan(_proposal(approved=True), [_leaf("fine")], report, {}) == []


def test_under_cap_failure_goes_to_llm_for_respec(monkeypatch):
    revised = _leaf("wobbly", execution_mode="Auto-OK", max_files=None, requires_tests=False)
    _envelope(
        monkeypatch,
        [arch.RespecAction(leaf_title="wobbly", revised=revised, rationale="shrink scope")],
    )
    report = _report(outcomes=[_failed_leaf("wobbly")])
    actions = arch.replan(_proposal(approved=True), [_leaf("wobbly")], report, {"wobbly": 1})
    assert len(actions) == 1
    respec = actions[0]
    assert isinstance(respec, arch.RespecAction)
    # conservative floor applied to the revised leaf: auto implies tests + a cap
    assert respec.revised.requires_tests is True
    assert respec.revised.max_files == 5


def test_confirmed_findings_become_fixups(monkeypatch):
    fixup_leaf = _leaf("fix dangling ref")
    _envelope(monkeypatch, [arch.FixupAction(finding_slug="dangling-ref", leaf=fixup_leaf)])
    report = _report(
        outcomes=[_landed_leaf("a")],
        findings=[_finding("dangling-ref"), _finding("noise", confirmed=False)],
    )
    actions = arch.replan(_proposal(approved=True), [_leaf("a")], report, {})
    assert len(actions) == 1
    assert isinstance(actions[0], arch.FixupAction)


def test_integration_red_reaches_llm(monkeypatch):
    fix_leaf = _leaf("untangle interaction", complexity="novel")
    calls = _envelope(
        monkeypatch, [arch.IntegrationFixAction(leaf=fix_leaf, rationale="a+b clash")]
    )
    report = _report(
        outcomes=[_landed_leaf("a"), _landed_leaf("b")],
        suite=SuiteResult(passed=False, output_tail="boom"),
    )
    actions = arch.replan(_proposal(approved=True), [_leaf("a"), _leaf("b")], report, {})
    assert isinstance(actions[0], arch.IntegrationFixAction)
    assert "RED" in calls[0]["user"]


def test_llm_actions_on_escalated_leaves_are_dropped(monkeypatch):
    _envelope(
        monkeypatch,
        [arch.RespecAction(leaf_title="capped", revised=_leaf("capped"), rationale="")],
    )
    report = _report(
        outcomes=[_failed_leaf("capped"), _failed_leaf("retryable")],
    )
    actions = arch.replan(
        _proposal(approved=True),
        [_leaf("capped"), _leaf("retryable")],
        report,
        {"capped": 2, "retryable": 0},
    )
    kinds = [a.kind for a in actions]
    assert kinds == ["escalate"]  # model's respec of the escalated leaf was discarded


def test_replan_over_emission_cap_halts(monkeypatch):
    from forge.shared.forge_emit import settings as emit_settings

    monkeypatch.setattr(emit_settings, "max_per_run", 1)
    _envelope(
        monkeypatch,
        [
            arch.FixupAction(finding_slug="f1", leaf=_leaf("fix one")),
            arch.FixupAction(finding_slug="f2", leaf=_leaf("fix two")),
        ],
    )
    report = _report(outcomes=[_landed_leaf("a")], findings=[_finding("f1"), _finding("f2")])
    actions = arch.replan(_proposal(approved=True), [_leaf("a")], report, {})
    assert len(actions) == 1
    assert isinstance(actions[0], arch.HaltAction)
    assert "emission cap" in actions[0].reason


def test_replan_pool_exhaustion_raises(monkeypatch):
    _mock_structured(monkeypatch, None, error="pool exhausted")
    report = _report(outcomes=[_failed_leaf("wobbly")])
    with pytest.raises(ArchitectError, match="no usable actions"):
        arch.replan(_proposal(approved=True), [_leaf("wobbly")], report, {"wobbly": 0})


def test_replan_prompt_enumerates_strict_leafspec_enums():
    # Regression for pipeline:build:fix:replan-validation. A live coder replan was discarded
    # because REPLAN_SYSTEM declared task_type a free `str` while LeafSpec enforces a 6-value
    # Literal, so the model guessed "implementation". The prompt must spell out every strictly
    # validated enum (task_type is the one that bit) — and must not label it a bare str.
    import typing

    from forge.coding_pipeline.models import LeafSpec

    for value in typing.get_args(LeafSpec.model_fields["task_type"].annotation):
        assert f'"{value}"' in arch.REPLAN_SYSTEM, f"task_type {value!r} missing from REPLAN_SYSTEM"
    assert '"task_type": str' not in arch.REPLAN_SYSTEM


def test_replan_failure_carries_model_raw_output(monkeypatch):
    # The whole point of pipeline:build:fix:replan-validation: when the coder's replan JSON
    # never validates, the raw payload must survive on the exception so a human can see it.
    bad_json = '{"actions": [{"kind": "fixup", "leaf": "not a LeafSpec"}]}'
    _mock_structured(monkeypatch, None, error="output failed validation", raw=bad_json)
    report = _report(outcomes=[_failed_leaf("wobbly")])
    with pytest.raises(ArchitectError) as exc:
        arch.replan(_proposal(approved=True), [_leaf("wobbly")], report, {"wobbly": 0})
    assert exc.value.raw == bad_json


# --- replan: landed leaves are terminal (deps-v2 waves 10-11) ---------------------


def test_replan_drops_respec_of_leaf_landed_this_wave(monkeypatch):
    _envelope(
        monkeypatch,
        [arch.RespecAction(leaf_title="hero", revised=_leaf("hero"), rationale="tweak")],
    )
    # a confirmed finding forces the model consult; the respec target landed THIS wave
    report = _report(outcomes=[_landed_leaf("hero")], findings=[_finding("real-one")])
    actions = arch.replan(_proposal(approved=True), [_leaf("hero")], report, {})
    assert actions == []


def test_replan_drops_respec_of_historically_landed_leaf(monkeypatch):
    _envelope(
        monkeypatch,
        [arch.RespecAction(leaf_title="old-hero", revised=_leaf("old-hero"), rationale="r")],
    )
    report = _report(outcomes=[_failed_leaf("other")])
    actions = arch.replan(
        _proposal(approved=True),
        [_leaf("old-hero"), _leaf("other")],
        report,
        {"other": 1},
        landed_titles={"old-hero"},
    )
    assert actions == []


def test_replan_drops_fixup_with_unconfirmed_slug(monkeypatch):
    # Fixups may only fix findings confirmed THIS wave — an invented slug defeats
    # the ref-keyed dedup and refiles the same phantom under a new ref (deps-v2
    # waves 17-18, live).
    _envelope(monkeypatch, [arch.FixupAction(finding_slug="invented", leaf=_leaf("phantom"))])
    report = _report(outcomes=[_landed_leaf("a")], findings=[_finding("real-one")])
    actions = arch.replan(_proposal(approved=True), [_leaf("a")], report, {})
    assert actions == []


def test_replan_drops_integration_fix_when_suite_green(monkeypatch):
    _envelope(monkeypatch, [arch.IntegrationFixAction(leaf=_leaf("untangle"))])
    report = _report(outcomes=[_landed_leaf("a")], findings=[_finding("real-one")])
    actions = arch.replan(_proposal(approved=True), [_leaf("a")], report, {})
    assert actions == []


# --- conservative tags: requires_tests headroom (deps-v2 waves 1-3) ---------------


def test_requires_tests_leaf_gets_file_scope_headroom():
    leaf = _leaf(
        "tight",
        requires_tests=True,
        max_files=4,
        file_scope=["a.py", "b.py", "c.py", "d.py", "tests/test_a.py"],
    )
    arch._apply_conservative_tags([leaf], _proposal(approved=True))
    assert leaf.max_files == 6  # len(file_scope) + 1


def test_requires_tests_headroom_leaves_generous_budgets_alone():
    leaf = _leaf("roomy", requires_tests=True, max_files=8, file_scope=["a.py"])
    arch._apply_conservative_tags([leaf], _proposal(approved=True))
    assert leaf.max_files == 8


# --- no-progress (Ralph-loop) guard ---------------------------------------------------


def test_no_progress_leaf_escalates_under_cap_without_llm(monkeypatch):
    calls = _structured_spy(monkeypatch)  # any LLM call would raise
    report = _report(outcomes=[_failed_leaf("wobbly", reason="assert 1 == 2")])
    # UNDER the attempt cap (1 < 2) but stuck: the last two attempts failed identically, so the
    # guard escalates now instead of burning the remaining attempt on the same mistake.
    actions = arch.replan(
        _proposal(approved=True), [_leaf("wobbly")], report, {"wobbly": 1}, stuck={"wobbly"}
    )
    assert calls == []  # escalated deterministically; never reached the model
    assert len(actions) == 1 and isinstance(actions[0], EscalateAction)
    assert actions[0].leaf_title == "wobbly"
    assert "no-progress" in actions[0].diagnostics
    assert "assert 1 == 2" in actions[0].diagnostics  # the last failure is preserved for the human


def test_differing_failures_under_cap_still_go_to_the_model(monkeypatch):
    # Not stuck (empty set): existing behavior — an under-cap failure is still sent for respec.
    revised = _leaf("wobbly", execution_mode="Auto-OK", max_files=None, requires_tests=False)
    _envelope(monkeypatch, [arch.RespecAction(leaf_title="wobbly", revised=revised)])
    report = _report(outcomes=[_failed_leaf("wobbly")])
    actions = arch.replan(
        _proposal(approved=True), [_leaf("wobbly")], report, {"wobbly": 1}, stuck=set()
    )
    assert len(actions) == 1 and isinstance(actions[0], arch.RespecAction)


def test_deterministic_escalations_distinguishes_cap_from_no_progress():
    report = _report(
        outcomes=[_failed_leaf("capped", "boom"), _failed_leaf("looping", "same error each time")]
    )
    esc = arch.deterministic_escalations(report, {"capped": 2, "looping": 1}, stuck={"looping"})
    by = {e.leaf_title: e.diagnostics for e in esc}
    assert set(by) == {"capped", "looping"}
    assert "no-progress" in by["looping"]  # stuck-under-cap gets the Ralph-loop framing
    assert by["capped"] == "boom"  # capped-only keeps its raw reason, no no-progress tag
    assert "no-progress" not in by["capped"]
