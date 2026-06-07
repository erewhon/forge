"""Combiners reduce many ExecResults into one — and never become a hard single point of failure.

AggregateCombiner synthesizes N successful outputs via a pooled aggregator role (with failover);
if every aggregator model is down it falls back to deterministic concatenation of the raw
outputs, so the ensemble still returns something useful. JudgeCombiner and GateCombiner (the
adversarial / pass-fail shapes) implement the same protocol and land with parallel_edit and (c).
"""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel

from agents.shared.ensemble.models import ExecResult, Prompt
from agents.shared.ensemble.pool import Pool


class CombineResult(BaseModel):
    text: str
    combiner: str  # executor label that combined, or "fallback:concat"
    used_fallback: bool = False
    error: str | None = None


class Combiner(Protocol):
    async def combine(self, inputs: list[ExecResult]) -> CombineResult: ...


def _concat(results: list[ExecResult]) -> str:
    return "\n\n".join(f"### {r.executor}\n\n{r.output}" for r in results)


class AggregateCombiner:
    """Synthesize results; fall back to concatenation when the pool is unavailable."""

    def __init__(
        self,
        *,
        pool: Pool,
        system: str,
        header: str = "",
        timeout: float = 120.0,
        max_tokens: int = 4096,
    ) -> None:
        self.pool = pool
        self.system = system
        self.header = header
        self.timeout = timeout
        self.max_tokens = max_tokens

    async def combine(self, inputs: list[ExecResult]) -> CombineResult:
        successful = [r for r in inputs if r.ok]
        body = _concat(successful)
        user = f"{self.header}\n\n{body}".strip()
        prompt = Prompt(system=self.system, user=user, max_tokens=self.max_tokens)

        result = await self.pool.run(prompt, timeout=self.timeout)
        if result.ok:
            return CombineResult(text=result.output, combiner=result.executor)

        fallback = (
            f"_(combiner unavailable: {result.error or 'unknown'}; showing raw outputs)_\n\n{body}"
        )
        return CombineResult(
            text=fallback,
            combiner="fallback:concat",
            used_fallback=True,
            error=result.error,
        )
