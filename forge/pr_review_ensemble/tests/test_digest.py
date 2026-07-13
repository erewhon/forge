"""Unit tests for the size-guarded hybrid digest pass.

No network: the digest pool is FakeExec-backed (prompt-aware, so map vs reduce vs single are
distinguishable), so these pin the wiring deterministically — single-pass, the map-reduce path
(split → summarize each chunk → synthesize), failed-chunk tolerance, reduce concat fallback,
rotation order, and the render/log shapes.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime

from forge.pr_review_ensemble.config import settings
from forge.pr_review_ensemble.digest import build_digest_pool, run_digest
from forge.pr_review_ensemble.logger import log_digest
from forge.pr_review_ensemble.models import DigestResult
from forge.pr_review_ensemble.prompts import (
    DIGEST_MAP_SYSTEM_PROMPT,
    DIGEST_REDUCE_SYSTEM_PROMPT,
)
from forge.pr_review_ensemble.providers import ReviewerSlot, SkipExecutor
from forge.pr_review_ensemble.renderer import render_digest
from forge.shared.ensemble import ExecResult, ExecStatus, FailureClass, Pool, Prompt


def _ok(label: str, output: str) -> ExecResult:
    return ExecResult(executor=label, status=ExecStatus.OK, output=output, latency_ms=1)


def _err(label: str) -> ExecResult:
    return ExecResult(
        executor=label,
        status=ExecStatus.ERROR,
        output="",
        error="down",
        failure_class=FailureClass.TERMINAL,
        latency_ms=1,
    )


class FakeExec:
    """Prompt-aware fake: returns by system prompt (map/reduce/single), with optional per-call
    failure when the user message contains a marker (to simulate one bad chunk)."""

    def __init__(
        self,
        label: str,
        *,
        default: ExecResult | None = None,
        router: dict[str, ExecResult] | None = None,
        boom: bool = False,
        fail_if_user_contains: str | None = None,
    ) -> None:
        self.label = label
        self._default = default or _ok(label, "# Digest\nbody")
        self._router = router or {}
        self._boom = boom
        self._fail_sub = fail_if_user_contains
        self.calls: list[str] = []  # system prompts seen

    async def run(self, prompt: Prompt, *, timeout: float) -> ExecResult:
        self.calls.append(prompt.system)
        if self._boom:
            raise AssertionError("executor should not have been called")
        if self._fail_sub and self._fail_sub in prompt.user:
            return _err(self.label)
        return self._router.get(prompt.system, self._default).model_copy()


def _pool(*execs: FakeExec) -> Pool:
    return Pool(role="digest", executors=list(execs))


def _slot(provider: str, *, active: bool = True) -> ReviewerSlot:
    ex = FakeExec(f"{provider}:m") if active else SkipExecutor(f"{provider}:m", "skip")
    return ReviewerSlot(
        provider=provider,
        model="m",
        pool=Pool(role=f"review:{provider}", executors=[ex]),
        skipped_reason=None if active else "skip",
    )


def _run(diff="a\nb\n", pr="PR", pool=None, slots=None):
    return asyncio.run(run_digest(diff_text=diff, pr_ref=pr, pool=pool, slots=slots))


def _file_diff(path: str, body: str = "+x\n") -> str:
    return (
        f"diff --git a/{path} b/{path}\nindex 0..1 100644\n"
        f"--- a/{path}\n+++ b/{path}\n@@ -1 +1 @@\n{body}"
    )


# --- single-pass -------------------------------------------------------------


def test_single_pass_success():
    res = _run(pool=_pool(FakeExec("local:coder", default=_ok("local:coder", "# Digest\nhello"))))
    assert res.strategy == "single"
    assert res.chunks == 0
    assert res.digest == "# Digest\nhello"
    assert res.model == "local:coder"
    assert res.error is None


def test_single_pass_fails_over_to_next():
    first = FakeExec("anthropic:x", default=_err("anthropic:x"))
    second = FakeExec("local:coder", default=_ok("local:coder", "# Digest\nok"))
    res = _run(pool=_pool(first, second))
    assert res.digest == "# Digest\nok"
    assert res.model == "local:coder"  # failed over


def test_single_pass_all_fail():
    res = _run(pool=_pool(FakeExec("a:x", default=_err("a:x"))))
    assert res.digest is None
    assert res.error is not None


def test_empty_pool():
    res = _run(pool=Pool(role="digest", executors=[]))
    assert res.digest is None
    assert "no active providers" in (res.error or "")


# --- map-reduce (over budget) ------------------------------------------------


def _map_reduce_router(label: str) -> dict[str, ExecResult]:
    return {
        DIGEST_MAP_SYSTEM_PROMPT: _ok(label, "- a thing changed"),
        DIGEST_REDUCE_SYSTEM_PROMPT: _ok(label, "# Digest\nSYNTHESIZED"),
    }


def test_map_reduce_triggers_over_budget(monkeypatch):
    monkeypatch.setattr(settings, "digest_max_diff_chars", 50)  # force the map-reduce path
    monkeypatch.setattr(settings, "digest_chunk_chars", 100)  # one chunk per small file
    diff = _file_diff("a.py") + _file_diff("b.py")
    fake = FakeExec("local:coder", router=_map_reduce_router("local:coder"))
    res = _run(diff=diff, pool=_pool(fake))

    assert res.strategy == "map_reduce"
    assert res.chunks == 2  # two files, each its own chunk at this budget
    assert res.digest == "# Digest\nSYNTHESIZED"
    assert fake.calls.count(DIGEST_MAP_SYSTEM_PROMPT) == 2  # one map call per chunk
    assert fake.calls.count(DIGEST_REDUCE_SYSTEM_PROMPT) == 1  # one reduce


def test_map_reduce_tolerates_one_failed_chunk(monkeypatch):
    monkeypatch.setattr(settings, "digest_max_diff_chars", 50)
    monkeypatch.setattr(settings, "digest_chunk_chars", 100)
    # Marker lives in one file's raw diff body — so it only appears in that chunk's MAP input,
    # not in the reduce input (which sees only the map summaries).
    diff = _file_diff("good.py") + _file_diff("bad.py", body="+ZZMARKER\n")
    fake = FakeExec(
        "local:coder", router=_map_reduce_router("local:coder"), fail_if_user_contains="ZZMARKER"
    )
    res = _run(diff=diff, pool=_pool(fake))
    assert res.strategy == "map_reduce"
    assert res.digest == "# Digest\nSYNTHESIZED"  # reduce still runs since one map succeeded
    assert fake.calls.count(DIGEST_MAP_SYSTEM_PROMPT) == 2  # both chunks attempted


def test_map_reduce_all_chunks_fail(monkeypatch):
    monkeypatch.setattr(settings, "digest_max_diff_chars", 50)
    monkeypatch.setattr(settings, "digest_chunk_chars", 100)
    diff = _file_diff("a.py") + _file_diff("b.py")
    fake = FakeExec("local:coder", default=_err("local:coder"))  # every call fails
    res = _run(diff=diff, pool=_pool(fake))
    assert res.strategy == "map_reduce"
    assert res.digest is None
    assert "chunk summaries failed" in (res.error or "")


def test_map_reduce_reduce_failure_concat_fallback(monkeypatch):
    monkeypatch.setattr(settings, "digest_max_diff_chars", 50)
    monkeypatch.setattr(settings, "digest_chunk_chars", 100)
    diff = _file_diff("a.py") + _file_diff("b.py")
    router = {
        DIGEST_MAP_SYSTEM_PROMPT: _ok("local:coder", "- summary of a.py"),
        DIGEST_REDUCE_SYSTEM_PROMPT: _err("local:coder"),  # reduce is down
    }
    fake = FakeExec("local:coder", router=router)
    res = _run(diff=diff, pool=_pool(fake))
    assert res.strategy == "map_reduce"
    assert res.model == "fallback:concat"
    assert "showing per-file summaries" in (res.digest or "")
    assert "summary of a.py" in (res.digest or "")  # the per-file summaries survive


def test_map_reduce_caps_chunks(monkeypatch):
    monkeypatch.setattr(settings, "digest_max_diff_chars", 1)
    monkeypatch.setattr(settings, "digest_chunk_chars", 10)  # each file its own chunk
    monkeypatch.setattr(settings, "digest_max_chunks", 2)
    diff = "".join(_file_diff(f"f{i}.py") for i in range(5))  # 5 chunks, cap 2
    fake = FakeExec("local:coder", router=_map_reduce_router("local:coder"))
    res = _run(diff=diff, pool=_pool(fake))
    assert res.chunks == 2
    assert res.chunks_dropped == 3
    assert fake.calls.count(DIGEST_MAP_SYSTEM_PROMPT) == 2  # only the kept chunks are summarized


# --- rotation, render, log ---------------------------------------------------


def test_build_digest_pool_rotation_excludes_inactive(monkeypatch):
    monkeypatch.setattr(settings, "aggregator_provider", "sonnet-5")  # preferred, inactive here
    slots = [_slot("sonnet-5", active=False), _slot("glm"), _slot("m3")]
    pool = build_digest_pool(slots)
    assert [e.label for e in pool.executors] == ["glm:m", "m3:m"]


def _result(**kw) -> DigestResult:
    base = dict(pr_ref="PR", timestamp=datetime.now(UTC), diff_lines=10, diff_chars=100)
    base.update(kw)
    return DigestResult(**base)


def test_render_digest_single():
    md = render_digest(_result(digest="# D\nbody", model="local:coder"))
    assert "# PR Digest: PR" in md
    assert "**Generated by:** local:coder (single-pass)" in md
    assert "body" in md


def test_render_digest_map_reduce():
    md = render_digest(
        _result(
            digest="# D",
            model="opencode_zen:kimi",
            strategy="map_reduce",
            chunks=7,
            chunks_dropped=2,
        )
    )
    assert "map-reduce over 7 chunk(s)" in md
    assert "2 dropped" in md


def test_render_digest_failed():
    md = render_digest(_result(error="everything was down"))
    assert "Digest failed" in md
    assert "everything was down" in md


def test_log_digest(tmp_path):
    res = _result(digest="x" * 20, model="opencode_zen:kimi", strategy="map_reduce", chunks=3)
    path = tmp_path / "runs.jsonl"
    log_digest(res, log_path=path)
    rec = json.loads(path.read_text(encoding="utf-8").strip())
    assert rec["pass"] == "digest"
    assert rec["ok"] is True
    assert rec["strategy"] == "map_reduce"
    assert rec["chunks"] == 3
    assert rec["digest_chars"] == 20
