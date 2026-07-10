"""Testing-review ensemble: discover → dedup → verify test-coverage gaps, then assemble a report.

Reads the SOURCE and its EXISTING TESTS into one labelled context (executors have no file tools, so
finders read *provided* code — and they need the tests to know what's already covered), fans out
gap-angle finders, consolidates overlaps, then runs a perspective-diverse skeptic panel per gap
whose load-bearing lens is "is this case already covered?". The orchestration is the shared
``discover_dedup_verify`` recipe — this module supplies the context, prompts, schemas, and the vote.
"""

from __future__ import annotations

import json
from pathlib import Path

from forge.shared.ensemble import ApiExecutor, Pool
from forge.shared.panel import Finder, PanelResult, build_lens_members
from forge.shared.recipe import discover_dedup_verify
from forge.testing_ensemble.config import settings
from forge.testing_ensemble.models import (
    SEVERITY_RANK,
    CanonicalEnvelope,
    CanonicalGap,
    ScoredGap,
    TestGap,
    TestGapsEnvelope,
    TestReport,
    Verdict,
)
from forge.testing_ensemble.prompts import (
    DEDUP_SYSTEM,
    FINDER_ANGLES,
    SKEPTIC_BASE,
    SKEPTIC_LENSES,
    build_dedup_user,
    finder_system,
    verify_user,
)

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


def _is_test(path: Path) -> bool:
    name = path.name
    if name.startswith("test_") or name.endswith("_test.py") or name == "conftest.py":
        return True
    return any(part.lower() in ("tests", "test") for part in path.parts)


def _gather(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            out.extend(sorted(f for f in path.rglob("*") if f.is_file() and f.suffix in _CODE_EXTS))
        elif path.is_file():
            out.append(path)
    return out


def _read_section(files: list[Path], budget: int) -> tuple[str, list[str]]:
    blocks: list[str] = []
    included: list[str] = []
    total = 0
    for f in files:
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        block = f"### {f}\n```\n{text}\n```\n"
        if total + len(block) > budget:
            room = budget - total
            if room > 400:
                blocks.append(f"### {f} (truncated)\n```\n{text[:room]}\n```\n")
                included.append(str(f))
            break
        blocks.append(block)
        included.append(str(f))
        total += len(block)
    return "\n".join(blocks), included


def collect_context(paths: list[str]) -> tuple[str, list[str], list[str]]:
    """Read the targets into a SOURCE section and an EXISTING TESTS section (files are classified by
    name/path), each budgeted so neither crowds the other out. Returns (context, sources, tests)."""
    files = _gather(paths)
    sources = [f for f in files if not _is_test(f)]
    tests = [f for f in files if _is_test(f)]

    budget = settings.max_context_chars
    source_budget = int(budget * settings.source_fraction) if tests else budget
    source_text, source_files = _read_section(sources, source_budget)
    test_text, test_files = _read_section(tests, budget - len(source_text))

    context = f"## SOURCE UNDER TEST\n\n{source_text}\n\n## EXISTING TESTS\n\n"
    context += (
        test_text
        if test_text.strip()
        else "(No existing tests found — propose a starting suite.)\n"
    )
    return context, source_files, test_files


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


def _gaps_json(gaps: list[TestGap]) -> str:
    return json.dumps([g.model_dump() for g in gaps], indent=2)


def _vote(gap: CanonicalGap, panel: PanelResult) -> Verdict:
    """The verify_each aggregate: turn the skeptic panel's votes into a verdict for one gap."""
    votes = panel.responses
    reals = [v for v in votes if v.get("real") is True]
    n = len(votes)
    if not reals:
        status = "rejected"
    elif any(str(v.get("confidence")).lower() == "high" for v in reals) or (n and len(reals) == n):
        status = "confirmed"
    else:
        status = "tentative"
    severity = gap.severity
    for v in reals:
        adjusted = str(v.get("severity", "")).lower()
        if SEVERITY_RANK.get(adjusted, 0) > SEVERITY_RANK.get(severity, 0):
            severity = adjusted
    reasonings = [str(v.get("reasoning", "")).strip() for v in votes if v.get("reasoning")]
    return Verdict(
        status=status,
        votes_real=len(reals),
        votes_total=n,
        severity=severity,
        reasonings=reasonings,
    )


def run_review(paths: list[str], focus: str) -> TestReport:
    context, source_files, test_files = collect_context(paths)
    if not source_files:
        raise ValueError("no readable source files in the given paths")

    finders = [
        Finder(label=name, system=finder_system(focus, directive), user=context)
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
        finder_pool=_router_pool("test-find", settings.finder_models),
        finding_schema=TestGapsEnvelope,
        findings_of=lambda e: e.gaps,
        dedup_pool=_router_pool("test-dedup", settings.dedup_models),
        dedup_system=DEDUP_SYSTEM,
        build_dedup_user=lambda raw: build_dedup_user(_gaps_json(raw)),
        canonical_schema=CanonicalEnvelope,
        canonical_of=lambda e: e.gaps,
        verify_members=members,
        verify_make_user=lambda g: verify_user(context, g.model_dump_json()),
        verify_aggregate=_vote,
        verify_floor=settings.verify_floor,
        concurrency=settings.concurrency,
        max_tokens=settings.max_tokens,
        timeout=settings.per_call_timeout,
        log=lambda m: print(f"  {m}"),
    )

    scored = [ScoredGap(gap=v.item, verdict=v.verdict) for v in result.verdicts]

    def by_severity(s: ScoredGap) -> int:
        return SEVERITY_RANK.get(s.verdict.severity, 0)

    return TestReport(
        focus=focus,
        source_files=source_files,
        test_files=test_files,
        raw_count=len(result.raw),
        canonical_count=len(result.canonical),
        dedup_ok=result.dedup_ok,
        confirmed=sorted(
            (s for s in scored if s.verdict.status == "confirmed"), key=by_severity, reverse=True
        ),
        tentative=sorted(
            (s for s in scored if s.verdict.status == "tentative"), key=by_severity, reverse=True
        ),
        rejected=[s for s in scored if s.verdict.status == "rejected"],
    )


def render(report: TestReport) -> str:
    lines = [
        f"# Test-coverage review — {report.focus}",
        "",
        f"Source files: {len(report.source_files)} | test files: {len(report.test_files)} | "
        f"raw gaps: {report.raw_count} → {report.canonical_count} canonical "
        f"(dedup {'ok' if report.dedup_ok else 'unavailable — raw gaps verified'})",
        f"Verdicts: **{len(report.confirmed)} confirmed**, {len(report.tentative)} tentative, "
        f"{len(report.rejected)} rejected",
        "",
    ]
    if not report.test_files:
        lines.append("> No existing tests were found — gaps below describe a starting suite.\n")
    for heading, group in (
        ("Confirmed gaps", report.confirmed),
        ("Tentative gaps", report.tentative),
    ):
        if not group:
            continue
        lines.append(f"## {heading}")
        for s in group:
            g, v = s.gap, s.verdict
            kind = f" [{g.gap_type}]" if g.gap_type else ""
            lines.append(
                f"### [{v.severity.upper()}] {g.target}{kind}"
                f" — {v.votes_real}/{v.votes_total} skeptics"
            )
            if g.why_it_matters:
                lines.append(f"- **Why:** {g.why_it_matters}")
            if g.suggested_test:
                lines.append(f"- **Suggested test:** {g.suggested_test}")
            if v.reasonings:
                lines.append(f"- **Panel:** {' | '.join(v.reasonings[:2])}")
            lines.append("")
    if report.rejected:
        lines.append(f"## Rejected ({len(report.rejected)})")
        lines.extend(f"- ~~{s.gap.target}~~" for s in report.rejected)
        lines.append("")
    if not report.confirmed and not report.tentative:
        lines.append("_No confirmed or tentative coverage gaps._")
    return "\n".join(lines)
