"""Shadow comparison: the production roster and the all-local roster review the same diff.

The point is a generate-and-compare trial of the full-local ensemble (Lightning/gpt-oss/Coder-Next)
against the sonnet-anchored production roster, on real diffs, before deciding whether the frontier
seat can be dropped. Both runs are logged like normal ensemble runs (pr_ref suffixed
``[baseline]`` / ``[local]``), and the rendered output puts the two advisories side by side under a
mechanical comparison header — the judgment stays human.
"""

from __future__ import annotations

from dataclasses import dataclass

from forge.pr_review_ensemble.models import EnsembleResult
from forge.pr_review_ensemble.providers import build_local_reviewer_slots, build_reviewer_slots
from forge.pr_review_ensemble.renderer import render_markdown
from forge.pr_review_ensemble.runner import run_ensemble


@dataclass
class ShadowResult:
    baseline: EnsembleResult
    local: EnsembleResult


async def run_shadow(*, diff_text: str, pr_ref: str) -> ShadowResult:
    # Sequential on purpose: lightning sits in BOTH rosters and the talos B70 serves one model,
    # so concurrent runs would double-load the same card and skew the latency comparison.
    baseline = await run_ensemble(
        diff_text=diff_text, pr_ref=f"{pr_ref} [baseline]", slots=build_reviewer_slots()
    )
    local = await run_ensemble(
        diff_text=diff_text, pr_ref=f"{pr_ref} [local]", slots=build_local_reviewer_slots()
    )
    return ShadowResult(baseline=baseline, local=local)


def _mean_ok_latency_ms(result: EnsembleResult) -> int | None:
    latencies = [r.latency_ms for r in result.reviews if r.status == "ok" and r.latency_ms]
    return round(sum(latencies) / len(latencies)) if latencies else None


def _summary_cell(result: EnsembleResult) -> tuple[str, str, str, str]:
    seats = ", ".join(result.providers_attempted)
    quorum = (
        f"{result.quorum_state} "
        f"({len(result.providers_succeeded)}/{len(result.providers_attempted)})"
    )
    if result.aggregated_review is None:
        aggregator = "—"
    elif result.aggregator_used_fallback or not result.aggregator_provider:
        aggregator = "(concat fallback)"
    else:
        aggregator = result.aggregator_provider
    latency = f"{ms}ms" if (ms := _mean_ok_latency_ms(result)) is not None else "—"
    return seats, quorum, aggregator, latency


def render_shadow(shadow: ShadowResult) -> str:
    base = _summary_cell(shadow.baseline)
    local = _summary_cell(shadow.local)
    rows = ["Seats", "Quorum", "Aggregator", "Mean seat latency"]
    table = "\n".join(
        f"| {name} | {bv} | {lv} |" for name, bv, lv in zip(rows, base, local, strict=True)
    )
    return (
        f"# Shadow review comparison — {shadow.baseline.pr_ref.removesuffix(' [baseline]')}\n\n"
        f"| | production (baseline) | all-local |\n|---|---|---|\n{table}\n\n"
        f"## Production advisory (baseline)\n\n{render_markdown(shadow.baseline)}\n\n"
        f"## All-local advisory\n\n{render_markdown(shadow.local)}\n"
    )
