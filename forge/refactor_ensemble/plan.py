"""Refactoring ensemble: discover → dedup → verify code smells, then assemble a refactoring plan.

Reads the target source into one code context (executors have no file tools, so finders read
*provided* code, like the PR-review ensemble reads a diff), fans out smell-angle finders,
consolidates overlaps, then runs a perspective-diverse skeptic panel per suggestion whose two lenses
— "safety" (behavior-preserving?) and "worth-it" (not bikeshedding?) — keep the plan high-signal.
The orchestration is the shared ``discover_dedup_verify`` recipe; this module supplies the code
context, prompts, schemas, and the vote. Output is an advisory plan (Nous-task emission is a
follow-on).
"""

from __future__ import annotations

import json
from pathlib import Path

from forge.refactor_ensemble.config import settings
from forge.refactor_ensemble.models import (
    IMPACT_RANK,
    CanonicalEnvelope,
    CanonicalSmell,
    RefactorPlan,
    ScoredSmell,
    Smell,
    SmellsEnvelope,
    Verdict,
)
from forge.refactor_ensemble.prompts import (
    DEDUP_SYSTEM,
    FINDER_ANGLES,
    SKEPTIC_BASE,
    SKEPTIC_LENSES,
    build_dedup_user,
    finder_system,
    verify_user,
)
from forge.shared.ensemble import ApiExecutor, Pool
from forge.shared.panel import Finder, PanelResult, build_lens_members
from forge.shared.recipe import discover_dedup_verify

_CODE_EXTS = {
    ".py",
    ".rs",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".go",
    ".java",
    ".rb",
    ".c",
    ".h",
    ".cpp",
    ".sh",
}


def collect_code(paths: list[str], max_chars: int) -> tuple[str, list[str]]:
    """Read the targets (a directory is walked for source files) into one labelled code context,
    capped at ``max_chars``. Returns (context, included_files)."""
    targets: list[Path] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            targets.extend(
                sorted(f for f in path.rglob("*") if f.is_file() and f.suffix in _CODE_EXTS)
            )
        elif path.is_file():
            targets.append(path)

    blocks: list[str] = []
    included: list[str] = []
    total = 0
    for f in targets:
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        block = f"### {f}\n```\n{text}\n```\n"
        if total + len(block) > max_chars:
            room = max_chars - total
            if room > 400:
                blocks.append(f"### {f} (truncated)\n```\n{text[:room]}\n```\n")
                included.append(str(f))
            break
        blocks.append(block)
        included.append(str(f))
        total += len(block)
    return "\n".join(blocks), included


def _router_pool(role: str, models: list[str]) -> Pool:
    return Pool(
        role=role,
        executors=[
            ApiExecutor(
                label=f"router:{m}",
                kind="openai",
                model=m,
                base_url=settings.openai_base_url,
                api_key=settings.openai_api_key,
            )
            for m in models
        ],
    )


def _smells_json(smells: list[Smell]) -> str:
    return json.dumps([s.model_dump() for s in smells], indent=2)


def _vote(smell: CanonicalSmell, panel: PanelResult) -> Verdict:
    """The verify_each aggregate: vote the skeptic panel into a verdict for one suggestion."""
    votes = panel.responses
    reals = [v for v in votes if v.get("real") is True]
    n = len(votes)
    if not reals:
        status = "rejected"
    elif any(str(v.get("confidence")).lower() == "high" for v in reals) or (n and len(reals) == n):
        status = "confirmed"
    else:
        status = "tentative"
    impact = smell.impact
    for v in reals:
        adjusted = str(v.get("impact", "")).lower()
        if IMPACT_RANK.get(adjusted, 0) > IMPACT_RANK.get(impact, 0):
            impact = adjusted
    reasonings = [str(v.get("reasoning", "")).strip() for v in votes if v.get("reasoning")]
    return Verdict(
        status=status,
        votes_real=len(reals),
        votes_total=n,
        impact=impact,
        reasonings=reasonings,
    )


def run_refactor(paths: list[str], focus: str) -> RefactorPlan:
    code_context, files = collect_code(paths, settings.max_code_chars)
    if not code_context.strip():
        raise ValueError("no readable source files in the given paths")

    finders = [
        Finder(label=name, system=finder_system(focus, directive), user=code_context)
        for name, directive in FINDER_ANGLES
    ]
    members = build_lens_members(
        SKEPTIC_LENSES,
        settings.verify_models,
        base_url=settings.openai_base_url,
        api_key=settings.openai_api_key,
        base_system=SKEPTIC_BASE,
    )

    result = discover_dedup_verify(
        finders=finders,
        finder_pool=_router_pool("refactor-find", settings.finder_models),
        finding_schema=SmellsEnvelope,
        findings_of=lambda e: e.smells,
        dedup_pool=_router_pool("refactor-dedup", settings.dedup_models),
        dedup_system=DEDUP_SYSTEM,
        build_dedup_user=lambda raw: build_dedup_user(_smells_json(raw)),
        canonical_schema=CanonicalEnvelope,
        canonical_of=lambda e: e.smells,
        verify_members=members,
        verify_make_user=lambda s: verify_user(code_context, s.model_dump_json()),
        verify_aggregate=_vote,
        verify_floor=settings.verify_floor,
        concurrency=settings.concurrency,
        max_tokens=settings.max_tokens,
        timeout=settings.per_call_timeout,
        log=lambda m: print(f"  {m}"),
    )

    scored = [ScoredSmell(smell=v.item, verdict=v.verdict) for v in result.verdicts]

    def by_impact(s: ScoredSmell) -> int:
        return IMPACT_RANK.get(s.verdict.impact, 0)

    return RefactorPlan(
        focus=focus,
        files=files,
        raw_count=len(result.raw),
        canonical_count=len(result.canonical),
        dedup_ok=result.dedup_ok,
        confirmed=sorted(
            (s for s in scored if s.verdict.status == "confirmed"), key=by_impact, reverse=True
        ),
        tentative=sorted(
            (s for s in scored if s.verdict.status == "tentative"), key=by_impact, reverse=True
        ),
        rejected=[s for s in scored if s.verdict.status == "rejected"],
    )


def render(plan: RefactorPlan) -> str:
    lines = [
        f"# Refactoring plan — {plan.focus}",
        "",
        f"Files: {len(plan.files)} | raw suggestions: {plan.raw_count} → "
        f"{plan.canonical_count} canonical "
        f"(dedup {'ok' if plan.dedup_ok else 'unavailable — raw suggestions verified'})",
        f"Verdicts: **{len(plan.confirmed)} confirmed**, {len(plan.tentative)} tentative, "
        f"{len(plan.rejected)} rejected",
        "",
    ]
    for heading, group in (("Confirmed", plan.confirmed), ("Tentative", plan.tentative)):
        if not group:
            continue
        lines.append(f"## {heading}")
        for s in group:
            sm, v = s.smell, s.verdict
            kind = f" [{sm.smell_type}]" if sm.smell_type else ""
            lines.append(
                f"### [{v.impact.upper()} impact / {sm.effort} effort] {sm.location}{kind}"
                f" — {v.votes_real}/{v.votes_total} skeptics"
            )
            if sm.proposed_refactor:
                lines.append(f"- **Refactor:** {sm.proposed_refactor}")
            if sm.benefit:
                lines.append(f"- **Benefit:** {sm.benefit}")
            if sm.risk and sm.risk.strip().lower() not in ("none", ""):
                lines.append(f"- **Risk:** {sm.risk}")
            if v.reasonings:
                lines.append(f"- **Panel:** {' | '.join(v.reasonings[:2])}")
            lines.append("")
    if plan.rejected:
        lines.append(f"## Rejected ({len(plan.rejected)})")
        lines.extend(f"- ~~{s.smell.location}~~" for s in plan.rejected)
        lines.append("")
    if not plan.confirmed and not plan.tentative:
        lines.append("_No confirmed or tentative refactoring suggestions._")
    return "\n".join(lines)
