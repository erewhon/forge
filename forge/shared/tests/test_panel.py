"""Tests for the shared structured-output panel (fake executors, no network)."""

from __future__ import annotations

import json

from agents.shared.ensemble import ExecResult, ExecStatus, FailureClass, Prompt
from agents.shared.panel import run_panel


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


def test_quorum_met_with_floor_one():
    execs = [FakeExec("a", output=_j({"x": 1})), FakeExec("b", status=ExecStatus.ERROR)]
    res = run_panel(executors=execs, system="s", user="u", floor=1)
    assert res.quorum_met
    assert len(res.responses) == 1


def test_extracts_json_from_code_fence():
    execs = [FakeExec("a", output="```json\n" + _j({"x": 9}) + "\n```")]
    res = run_panel(executors=execs, system="s", user="u", floor=1)
    assert res.responses == [{"x": 9}]
