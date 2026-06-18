"""The code-audit ensemble: discover → dedup → verify over provided code, then assemble a report.

Reads the target source into one code context (the harness executors have no file tools, so finders
read *provided* code, like the PR-review ensemble reads a diff), fans out angle-specialized finders,
consolidates overlaps, then runs a perspective-diverse skeptic panel per finding and votes each to
confirmed / tentative / rejected. The whole orchestration is the shared ``discover_dedup_verify``
recipe — this module just supplies the prompts, schemas, code context, and the vote.
"""

from __future__ import annotations

import json
from pathlib import Path

from agents.code_audit_ensemble.config import settings
from agents.code_audit_ensemble.models import (
    SEVERITY_RANK,
    AuditReport,
    CanonicalEnvelope,
    CanonicalFinding,
    Finding,
    FindingsEnvelope,
    ScoredFinding,
    Verdict,
)
from agents.code_audit_ensemble.prompts import (
    DEDUP_SYSTEM,
    FINDER_ANGLES,
    SKEPTIC_BASE,
    SKEPTIC_LENSES,
    build_dedup_user,
    finder_system,
    verify_user,
)
from agents.shared.ensemble import ApiExecutor, Pool
from agents.shared.panel import Finder, PanelResult, build_lens_members
from agents.shared.recipe import discover_dedup_verify

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
            if room > 400:  # squeeze in a truncated tail rather than nothing
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


def _findings_json(findings: list[Finding]) -> str:
    return json.dumps([f.model_dump() for f in findings], indent=2)


def _vote(finding: CanonicalFinding, panel: PanelResult) -> Verdict:
    """The verify_each aggregate: turn the skeptic panel's votes into a verdict for one finding."""
    votes = panel.responses
    reals = [v for v in votes if v.get("real") is True]
    n = len(votes)
    if not reals:
        status = "rejected"
    elif any(str(v.get("confidence")).lower() == "high" for v in reals) or (n and len(reals) == n):
        status = "confirmed"
    else:
        status = "tentative"
    severity = finding.severity
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


def run_audit(paths: list[str], focus: str) -> AuditReport:
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
        finder_pool=_router_pool("audit-find", settings.finder_models),
        finding_schema=FindingsEnvelope,
        findings_of=lambda e: e.findings,
        dedup_pool=_router_pool("audit-dedup", settings.dedup_models),
        dedup_system=DEDUP_SYSTEM,
        build_dedup_user=lambda raw: build_dedup_user(_findings_json(raw)),
        canonical_schema=CanonicalEnvelope,
        canonical_of=lambda e: e.findings,
        verify_members=members,
        verify_make_user=lambda f: verify_user(code_context, f.model_dump_json()),
        verify_aggregate=_vote,
        verify_floor=settings.verify_floor,
        concurrency=settings.concurrency,
        max_tokens=settings.max_tokens,
        timeout=settings.per_call_timeout,
        log=lambda m: print(f"  {m}"),
    )

    scored = [ScoredFinding(finding=v.item, verdict=v.verdict) for v in result.verdicts]

    def by_severity(s: ScoredFinding) -> int:
        return SEVERITY_RANK.get(s.verdict.severity, 0)

    return AuditReport(
        focus=focus,
        files=files,
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


def render(report: AuditReport) -> str:
    lines = [
        f"# Code audit — {report.focus}",
        "",
        f"Files: {len(report.files)} | raw findings: {report.raw_count} → "
        f"{report.canonical_count} canonical "
        f"(dedup {'ok' if report.dedup_ok else 'unavailable — raw findings verified'})",
        f"Verdicts: **{len(report.confirmed)} confirmed**, {len(report.tentative)} tentative, "
        f"{len(report.rejected)} rejected",
        "",
    ]
    for heading, group in (("Confirmed", report.confirmed), ("Tentative", report.tentative)):
        if not group:
            continue
        lines.append(f"## {heading}")
        for s in group:
            f, v = s.finding, s.verdict
            loc = f" (`{f.file}{':' + f.line if f.line else ''}`)" if f.file else ""
            lines.append(
                f"### [{v.severity.upper()}] {f.title}{loc}"
                f" — {v.votes_real}/{v.votes_total} skeptics"
            )
            if f.scenario:
                lines.append(f"- **Scenario:** {f.scenario}")
            if f.suggestion:
                lines.append(f"- **Fix:** {f.suggestion}")
            if v.reasonings:
                lines.append(f"- **Panel:** {' | '.join(v.reasonings[:2])}")
            lines.append("")
    if report.rejected:
        lines.append(f"## Rejected ({len(report.rejected)})")
        lines.extend(f"- ~~{s.finding.title}~~" for s in report.rejected)
        lines.append("")
    if not report.confirmed and not report.tentative:
        lines.append("_No confirmed or tentative issues._")
    return "\n".join(lines)
