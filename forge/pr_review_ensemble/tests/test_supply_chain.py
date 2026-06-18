"""Unit tests for the supply-chain audit pass (pre-scan + focused ensemble audit)."""

from __future__ import annotations

import asyncio
import json

from agents.pr_review_ensemble.logger import log_supply_chain
from agents.pr_review_ensemble.prompts import SUPPLY_CHAIN_SYSTEM_PROMPT
from agents.pr_review_ensemble.providers import ReviewerSlot
from agents.pr_review_ensemble.renderer import render_supply_chain
from agents.pr_review_ensemble.supply_chain import run_supply_chain_audit
from agents.shared.ensemble import CombineResult, ExecResult, ExecStatus, Pool, Prompt


def _file(path: str, added: list[str]) -> str:
    body = "".join(f"+{ln}\n" for ln in added)
    return (
        f"diff --git a/{path} b/{path}\nindex 0..1 100644\n"
        f"--- a/{path}\n+++ b/{path}\n@@ -0,0 +1,{len(added)} @@\n{body}"
    )


class FakeExec:
    def __init__(self, label: str, *, output: str = "VERDICT: CLEAR", boom: bool = False) -> None:
        self.label = label
        self._output = output
        self._boom = boom
        self.prompts: list[Prompt] = []

    async def run(self, prompt: Prompt, *, timeout: float) -> ExecResult:
        self.prompts.append(prompt)
        if self._boom:
            raise AssertionError("executor should not have been called")
        return ExecResult(
            executor=self.label, status=ExecStatus.OK, output=self._output, latency_ms=1
        )


class FakeCombiner:
    def __init__(self, *, text: str = "VERDICT: NEEDS REVIEW") -> None:
        self._text = text
        self.called = False

    async def combine(self, inputs):
        self.called = True
        return CombineResult(text=self._text, combiner="fake:agg")


def _slot(provider: str, exec_: FakeExec) -> ReviewerSlot:
    return ReviewerSlot(
        provider=provider, model="m", pool=Pool(role=f"review:{provider}", executors=[exec_])
    )


def _audit(diff, slots, aggregator=None):
    return asyncio.run(
        run_supply_chain_audit(diff_text=diff, pr_ref="PR", slots=slots, aggregator=aggregator)
    )


def test_no_signals_short_circuits_with_no_model_calls():
    clean = _file("src/util.py", ["def add(a, b):", "    return a + b"])
    booms = [_slot("local", FakeExec("l", boom=True)), _slot("anthropic", FakeExec("a", boom=True))]
    res = _audit(clean, booms, FakeCombiner())
    assert res.ensemble is None
    assert not res.scan.has_signals
    assert all(s.pool.executors[0].prompts == [] for s in booms)  # never called


def test_with_signals_runs_focused_audit():
    diff = _file("src/util.py", ["x = 1"]) + _file("package.json", ['  "postinstall": "node x.js"'])
    fakes = [FakeExec("local:coder"), FakeExec("zen:kimi")]
    slots = [_slot("local", fakes[0]), _slot("opencode_zen", fakes[1])]
    agg = FakeCombiner(text="VERDICT: SUSPICIOUS — postinstall runs arbitrary code")
    res = _audit(diff, slots, agg)

    assert res.scan.has_signals
    assert res.ensemble is not None
    assert res.ensemble.quorum_state == "full"
    assert res.ensemble.aggregated_review == "VERDICT: SUSPICIOUS — postinstall runs arbitrary code"

    # auditors got the supply-chain prompt, the signals preamble, and ONLY the flagged file
    p = fakes[0].prompts[0]
    assert p.system == SUPPLY_CHAIN_SYSTEM_PROMPT
    assert "pre-scan signals" in p.user.lower()
    assert "package.json" in p.user
    assert "src/util.py" not in p.user  # the clean file is excluded from the focused audit


def test_render_no_signals_clear(tmp_path):
    clean = _file("src/util.py", ["x = 1"])
    booms = [_slot("local", FakeExec("l", boom=True)), _slot("anthropic", FakeExec("a", boom=True))]
    res = _audit(clean, booms, FakeCombiner())
    md = render_supply_chain(res)
    assert "Verdict: CLEAR (deterministic)" in md

    path = tmp_path / "runs.jsonl"
    log_supply_chain(res, log_path=path)
    rec = json.loads(path.read_text(encoding="utf-8").strip())
    assert rec["pass"] == "supply-chain"
    assert rec["audited"] is False
    assert rec["signal_count"] == 0


def test_render_with_signals_includes_table_and_verdict():
    diff = _file("package.json", ['  "postinstall": "node x.js"'])
    fakes = [FakeExec("local:coder"), FakeExec("zen:kimi")]
    slots = [_slot("local", fakes[0]), _slot("opencode_zen", fakes[1])]
    res = _audit(diff, slots, FakeCombiner(text="VERDICT: NEEDS REVIEW"))
    md = render_supply_chain(res)
    assert "## Pre-scan signals" in md
    assert "install-hook" in md
    assert "Synthesized audit" in md
    assert "VERDICT: NEEDS REVIEW" in md
    assert "## Raw audits" in md
