"""Tests for the discover→dedup→verify recipe (fake executors, no network)."""

from __future__ import annotations

import json

from pydantic import BaseModel

from agents.shared.ensemble import ExecResult, ExecStatus, FailureClass, Pool
from agents.shared.ensemble.models import Prompt
from agents.shared.panel import Finder, PanelMember
from agents.shared.recipe import discover_dedup_verify


class FakeExec:
    def __init__(self, label: str, output: str) -> None:
        self.label = label
        self._out = output

    async def run(self, prompt: Prompt, *, timeout: float) -> ExecResult:
        return ExecResult(
            executor=self.label,
            status=ExecStatus.OK,
            output=self._out,
            failure_class=FailureClass.NONE,
            latency_ms=1,
        )


class _Item(BaseModel):
    id: str = ""
    title: str


class _Env(BaseModel):
    findings: list[_Item] = []


def _j(d: dict) -> str:
    return json.dumps(d)


def _confirm(item: _Item, panel) -> dict:
    reals = sum(1 for r in panel.responses if r.get("real") is True)
    return {"id": item.id, "confirmed": reals >= 1}


def test_recipe_discovers_dedups_and_verifies():
    finders = [
        Finder(label="f1", system="A", user="code"),
        Finder(label="f2", system="B", user="code"),
    ]
    finder_pool = Pool(
        role="find", executors=[FakeExec("m", _j({"findings": [{"title": "leak"}]}))]
    )
    dedup_pool = Pool(
        role="dedup", executors=[FakeExec("m", _j({"findings": [{"id": "C1", "title": "leak"}]}))]
    )
    verify_members = [
        PanelMember(executor=FakeExec("a", _j({"real": True})), system="L1", label="m/a"),
        PanelMember(executor=FakeExec("b", _j({"real": True})), system="L2", label="m/b"),
    ]

    result = discover_dedup_verify(
        finders=finders,
        finder_pool=finder_pool,
        finding_schema=_Env,
        findings_of=lambda e: e.findings,
        dedup_pool=dedup_pool,
        dedup_system="merge",
        build_dedup_user=lambda raw: f"{len(raw)} findings",
        canonical_schema=_Env,
        canonical_of=lambda e: e.findings,
        verify_members=verify_members,
        verify_make_user=lambda f: f.title,
        verify_aggregate=_confirm,
        verify_floor=1,
        concurrency=2,
    )
    assert len(result.raw) == 2  # two finders, one finding each
    assert result.dedup_ok
    assert [c.id for c in result.canonical] == ["C1"]  # collapsed to one canonical finding
    assert len(result.verdicts) == 1
    assert result.verdicts[0].verdict == {"id": "C1", "confirmed": True}


def test_recipe_no_findings_short_circuits():
    finders = [Finder(label="f1", system="A", user="code")]
    finder_pool = Pool(role="find", executors=[FakeExec("m", _j({"findings": []}))])

    def _boom(_raw):
        raise AssertionError("dedup must not run when there are no findings")

    result = discover_dedup_verify(
        finders=finders,
        finder_pool=finder_pool,
        finding_schema=_Env,
        findings_of=lambda e: e.findings,
        dedup_pool=Pool(role="dedup", executors=[FakeExec("m", "unused")]),
        dedup_system="merge",
        build_dedup_user=_boom,
        canonical_schema=_Env,
        canonical_of=lambda e: e.findings,
        verify_members=[],
        verify_make_user=lambda f: "",
        verify_aggregate=lambda f, p: None,
    )
    assert result.raw == []
    assert result.canonical == []
    assert result.verdicts == []
    assert not result.dedup_ok


def test_recipe_falls_back_to_raw_when_dedup_fails():
    finders = [Finder(label="f1", system="A", user="code")]
    finder_pool = Pool(
        role="find", executors=[FakeExec("m", _j({"findings": [{"id": "R1", "title": "x"}]}))]
    )
    dedup_pool = Pool(role="dedup", executors=[FakeExec("m", "not json")], retry_backoff_s=0)
    verify_members = [
        PanelMember(executor=FakeExec("a", _j({"real": False})), system="L", label="m/a")
    ]

    result = discover_dedup_verify(
        finders=finders,
        finder_pool=finder_pool,
        finding_schema=_Env,
        findings_of=lambda e: e.findings,
        dedup_pool=dedup_pool,
        dedup_system="merge",
        build_dedup_user=lambda raw: "x",
        canonical_schema=_Env,
        canonical_of=lambda e: e.findings,
        verify_members=verify_members,
        verify_make_user=lambda f: f.title,
        verify_aggregate=lambda f, p: {"id": f.id, "usable": len(p.responses)},
        verify_floor=1,
    )
    assert not result.dedup_ok
    assert [c.id for c in result.canonical] == ["R1"]  # consolidator down → verify the raw findings
    assert result.verdicts[0].verdict == {"id": "R1", "usable": 1}
