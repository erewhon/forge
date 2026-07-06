"""Tests for the shared structured-output panel (fake executors, no network)."""

from __future__ import annotations

import json

from pydantic import BaseModel

from agents.shared.ensemble import ExecResult, ExecStatus, FailureClass, Pool, Prompt
from agents.shared.panel import (
    Finder,
    PanelMember,
    build_lens_members,
    discover,
    run_member_panel,
    run_panel,
    structured,
    verify_each,
)


class FakeExec:
    def __init__(
        self, label: str, *, output: str = "{}", status: ExecStatus = ExecStatus.OK
    ) -> None:
        self.label = label
        self._out = output
        self._status = status

    async def run(self, prompt: Prompt, *, timeout: float) -> ExecResult:
        fc = FailureClass.NONE if self._status == ExecStatus.OK else FailureClass.TERMINAL
        return ExecResult(
            executor=self.label,
            status=self._status,
            output=self._out,
            failure_class=fc,
            latency_ms=1,
        )


def _j(d: dict) -> str:
    return json.dumps(d)


def test_collects_parsed_responses():
    execs = [FakeExec("a", output=_j({"x": 1})), FakeExec("b", output=_j({"x": 2}))]
    res = run_panel(executors=execs, system="s", user="u", floor=2)
    assert res.attempted == 2
    assert res.responses == [{"x": 1}, {"x": 2}]
    assert res.member_labels == ["a", "b"]
    assert res.quorum_met


def test_drops_unparseable_and_errored_members():
    execs = [
        FakeExec("a", output=_j({"x": 1})),
        FakeExec("b", output="not json at all"),  # unparseable -> dropped
        FakeExec("c", status=ExecStatus.ERROR, output=""),  # errored -> dropped
    ]
    res = run_panel(executors=execs, system="s", user="u", floor=2)
    assert res.responses == [{"x": 1}]
    assert res.attempted == 3
    assert not res.quorum_met  # only 1 usable < floor 2


def test_parse_miss_is_retried_before_costing_the_seat():
    class FlakyExec:
        """Prose on the first call, valid JSON on the second — a stochastic parse miss."""

        def __init__(self) -> None:
            self.label = "flaky"
            self.calls = 0

        async def run(self, prompt: Prompt, *, timeout: float) -> ExecResult:
            self.calls += 1
            out = "sure! here's my verdict:" if self.calls == 1 else _j({"x": 7})
            return ExecResult(executor=self.label, status=ExecStatus.OK, output=out)

    flaky = FlakyExec()
    res = run_panel(executors=[flaky], system="s", user="u", floor=1)
    assert flaky.calls == 2
    assert res.responses == [{"x": 7}]  # the retry saved the seat
    assert res.failures == []


def test_dropped_members_are_recorded_as_failures_with_reasons():
    execs = [
        FakeExec("a", output=_j({"x": 1})),
        FakeExec("b", output="not json at all"),
        FakeExec("c", status=ExecStatus.ERROR, output=""),
    ]
    res = run_panel(executors=execs, system="s", user="u", floor=2)
    failures = dict(res.failures)
    assert set(failures) == {"b", "c"}  # the successful member is not a failure
    assert "no parseable JSON" in failures["b"]  # transport OK, payload unusable
    assert failures["c"]  # errored member carries a reason (falls back to "no response")


def test_quorum_met_with_floor_one():
    execs = [FakeExec("a", output=_j({"x": 1})), FakeExec("b", status=ExecStatus.ERROR)]
    res = run_panel(executors=execs, system="s", user="u", floor=1)
    assert res.quorum_met
    assert len(res.responses) == 1


def test_extracts_json_from_code_fence():
    execs = [FakeExec("a", output="```json\n" + _j({"x": 9}) + "\n```")]
    res = run_panel(executors=execs, system="s", user="u", floor=1)
    assert res.responses == [{"x": 9}]


# --- perspective-diverse path (run_member_panel + build_lens_members) ---


class CapturingExec:
    """A fake that echoes the system prompt it was handed, so we can assert each member ran its
    own lens system prompt rather than a shared one."""

    def __init__(self, label: str) -> None:
        self.label = label
        self.seen_system: str | None = None

    async def run(self, prompt: Prompt, *, timeout: float) -> ExecResult:
        self.seen_system = prompt.system
        return ExecResult(
            executor=self.label,
            status=ExecStatus.OK,
            output=_j({"lens": prompt.system}),
            failure_class=FailureClass.NONE,
            latency_ms=1,
        )


def test_member_panel_runs_each_members_own_system_prompt():
    members = [
        PanelMember(executor=CapturingExec("a"), system="LENS-A", label="m/a"),
        PanelMember(executor=CapturingExec("b"), system="LENS-B", label="m/b"),
    ]
    res = run_member_panel(members=members, user="u", floor=2)
    assert res.attempted == 2
    assert res.member_labels == ["m/a", "m/b"]  # member labels, not executor labels
    assert {r["lens"] for r in res.responses} == {"LENS-A", "LENS-B"}
    assert res.quorum_met


def test_build_lens_members_round_robins_models_and_prepends_base():
    lenses = [
        ("source", "look at sources"),
        ("depth", "look at depth"),
        ("claims", "look at claims"),
    ]
    members = build_lens_members(
        lenses, ["m1", "m2"], base_url="http://x/v1", api_key="k", base_system="BASE"
    )
    assert len(members) == 3
    # models cycle m1, m2, m1
    assert [m.executor.model for m in members] == ["m1", "m2", "m1"]
    # label encodes model + lens name; system is base + the lens directive
    assert members[0].label == "m1/source"
    assert members[0].system == "BASE\n\nlook at sources"
    assert members[2].label == "m1/claims"


def test_build_lens_members_empty_models_is_empty():
    members = build_lens_members(
        [("x", "y")], [], base_url="http://x/v1", api_key="k", base_system="BASE"
    )
    assert members == []


# --- single-pool structured output (structured + StructuredResult) ---


class _Pick(BaseModel):
    winner: int
    note: str = ""


def _pool(*execs, backoff: float = 0.0) -> Pool:
    return Pool(role="t", executors=list(execs), retry_backoff_s=backoff)


def test_structured_parses_into_model():
    res = structured(
        pool=_pool(FakeExec("a", output=_j({"winner": 0, "note": "x"}))),
        schema=_Pick,
        system="s",
        user="u",
    )
    assert res.ok
    assert res.value == _Pick(winner=0, note="x")


def test_structured_fails_over_on_unparseable_output():
    # First executor's output never validates → demoted to transient, retried, then failed over.
    res = structured(
        pool=_pool(FakeExec("a", output="not json"), FakeExec("b", output=_j({"winner": 2}))),
        schema=_Pick,
        system="s",
        user="u",
    )
    assert res.ok
    assert res.value is not None
    assert res.value.winner == 2


def test_structured_none_when_pool_exhausted():
    res = structured(
        pool=_pool(FakeExec("a", output="not the json we asked for")),
        schema=_Pick,
        system="s",
        user="u",
    )
    assert not res.ok
    assert res.value is None
    assert res.error is not None  # carries the underlying failure for the caller to log
    # ...and the model's actual (unparseable) output survives exhaustion, for diagnosis.
    assert res.raw == "not the json we asked for"


def test_structured_predicate_rejects_then_fails_over():
    # winner=9 fails the in-range predicate (schema alone can't express it) → fail over to winner=1.
    res = structured(
        pool=_pool(
            FakeExec("a", output=_j({"winner": 9})),
            FakeExec("b", output=_j({"winner": 1})),
        ),
        schema=_Pick,
        system="s",
        user="u",
        predicate=lambda p: p.winner < 2,
    )
    assert res.ok
    assert res.value is not None
    assert res.value.winner == 1


# --- per-item adversarial verification (verify_each) ---


class _EchoUserExec:
    """Echoes back the user prompt it was handed (plus a fixed score), so a test can assert each
    item ran its own make_user message through the panel."""

    def __init__(self, label: str, score: int) -> None:
        self.label = label
        self._score = score

    async def run(self, prompt: Prompt, *, timeout: float) -> ExecResult:
        return ExecResult(
            executor=self.label,
            status=ExecStatus.OK,
            output=_j({"user": prompt.user, "score": self._score}),
            failure_class=FailureClass.NONE,
            latency_ms=1,
        )


def test_verify_each_runs_a_panel_per_item_and_aggregates():
    members = [
        PanelMember(executor=_EchoUserExec("a", 1), system="LENS-A", label="m/a"),
        PanelMember(executor=_EchoUserExec("b", 3), system="LENS-B", label="m/b"),
    ]
    items = ["finding-1", "finding-2"]

    def aggregate(item, panel):
        return {
            "item": item,
            "total": sum(r["score"] for r in panel.responses),
            "seen_user": panel.responses[0]["user"],
        }

    verdicts = verify_each(
        items,
        members=members,
        make_user=lambda i: f"verify: {i}",
        aggregate=aggregate,
        floor=2,
        concurrency=2,
    )
    assert [v.item for v in verdicts] == items  # order preserved
    assert all(v.panel.quorum_met for v in verdicts)
    assert [v.verdict["total"] for v in verdicts] == [4, 4]  # 1 + 3 from the two members
    # each item's OWN make_user message reached its panel
    assert [v.verdict["seen_user"] for v in verdicts] == ["verify: finding-1", "verify: finding-2"]


def test_verify_each_drops_unusable_members_before_aggregate():
    members = [
        PanelMember(executor=FakeExec("a", output=_j({"score": 5})), system="A", label="m/a"),
        PanelMember(executor=FakeExec("b", output="junk"), system="B", label="m/b"),  # unparseable
    ]

    def aggregate(_item, panel):
        return {"usable": len(panel.responses), "quorum": panel.quorum_met}

    out = verify_each(
        ["x"], members=members, make_user=lambda i: i, aggregate=aggregate, floor=2, concurrency=1
    )
    # the junk member is dropped; only 1 usable response, below the floor of 2 → degraded
    assert out[0].verdict == {"usable": 1, "quorum": False}


# --- typed multi-finder discovery (discover) ---


class _FItem(BaseModel):
    title: str


class _FEnv(BaseModel):
    findings: list[_FItem] = []


def test_discover_returns_one_envelope_per_finder():
    finders = [Finder(label=f"f{i}", system="S", user="u") for i in range(3)]
    pool = _pool(FakeExec("m", output=_j({"findings": [{"title": "x"}]})))
    out = discover(finders, pool=pool, schema=_FEnv, concurrency=2)
    assert len(out) == 3  # one envelope per finder
    assert all(e.findings[0].title == "x" for e in out)


def test_discover_drops_finders_with_no_usable_output():
    finders = [Finder(label="f", system="S", user="u")]
    pool = _pool(FakeExec("m", output="junk"), backoff=0)
    assert discover(finders, pool=pool, schema=_FEnv, concurrency=1) == []
