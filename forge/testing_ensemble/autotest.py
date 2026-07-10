"""The testing ensemble's auto-merge loop: generate tests for confirmed gaps, gate them hard, then
either push a branch (default) or advance ``main`` (``--auto-merge``).

The loop closes what the analysis half leaves open. Every gate is **fail-closed** — a miss reverts
the working copy and falls back to the existing review-then-implement Forge emit, so the worst case
is "no auto-merge, tasks filed for a human" rather than a bad merge:

    review → generate → [tests-only] → [green] → [full-quorum sign-off] → push branch (→ merge)
                              │ any gate fails → revert working copy → emit Forge tasks → stop

The tests-only classifier, the branch/merge VCS actions, and the decision log live in
``forge.shared.automerge``; the full-quorum sign-off gate lives in ``forge.shared.signoff``
(Dependabot and the coding pipeline's epic gate reuse both). The sign-off panel is seated from
pr_review's cross-family provider roster. Only the test *generation* is testing-specific.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from forge.pr_review_ensemble.providers import build_reviewer_slots
from forge.shared.automerge import (
    advance_main,
    classify_tests_only,
    log_decision,
    push_branch,
    slugify,
    working_diff,
)
from forge.shared.forge_emit import EmitSummary
from forge.shared.signoff import SignoffResult, SignoffSeat, full_quorum_signoff
from forge.task_worker.tester import run_tests
from forge.task_worker.vcs import VCSError, detect_vcs, revert_changes
from forge.testing_ensemble.config import settings
from forge.testing_ensemble.emit import emit_report
from forge.testing_ensemble.generate import apply_generated, generate_tests
from forge.testing_ensemble.models import SEVERITY_RANK, ScoredGap, TestReport
from forge.testing_ensemble.review import collect_context, run_review

_SIGNOFF_SYSTEM = """You are a merge gatekeeper for an automated, TESTS-ONLY change.

The diff should contain only new/edited test files. Approve it for automatic merge ONLY if ALL hold:
- The tests are correct and actually exercise the behavior they claim to.
- They are deterministic — no dependence on real network, wall-clock time, randomness, or sleeps.
- They do NOT weaken, delete, rename, or skip existing tests, and don't lower coverage.
- They add NO new external/third-party dependencies.
- Every changed file is a test file (no source, config, CI, or manifest changes).

Be strict: this merges with no human review. If anything is uncertain, do NOT approve.

Respond with ONLY a JSON object: {"approve": true|false, "blockers": ["..."], "notes": "..."}"""


def _signoff(diff_text: str, *, pr_ref: str) -> SignoffResult:
    """Seat the shared full-quorum gate from pr_review's active provider roster."""
    seats = [
        SignoffSeat(provider=s.provider, executor=s.pool.executors[0])
        for s in build_reviewer_slots()
        if s.active
    ]
    return full_quorum_signoff(
        diff_text,
        seats=seats,
        system=_SIGNOFF_SYSTEM,
        ref=pr_ref,
        context="This change must contain ONLY test files.",
        max_tokens=settings.signoff_max_tokens,
        timeout=settings.signoff_timeout,
    )


AutoStatus = Literal["merged", "branched", "blocked", "no-gaps", "planned", "error"]


@dataclass
class AutoTestResult:
    status: AutoStatus
    reason: str = ""
    branch: str | None = None
    change_id: str | None = None
    merged_to_main: bool = False
    gaps_targeted: int = 0
    generated_files: list[str] = field(default_factory=list)
    tests_passed: bool | None = None
    signoff: SignoffResult | None = None
    emitted: EmitSummary | None = None


def _confirmed_at(report: TestReport, min_severity: str, limit: int) -> list[ScoredGap]:
    floor = SEVERITY_RANK.get(min_severity, 1)
    eligible = [g for g in report.confirmed if SEVERITY_RANK.get(g.verdict.severity, 0) >= floor]
    return eligible[:limit]


def auto_test(
    paths: list[str],
    *,
    repo_path: Path,
    focus: str = "test coverage and robustness",
    project: str | None = None,
    auto_merge: bool = False,
    max_gaps: int | None = None,
    min_severity: str = "low",
    branch_prefix: str = "auto-tests",
    dry_run: bool = False,
    log: Callable[[str], None] = print,
) -> AutoTestResult:
    """Run the generate→gate→act loop over ``paths`` in ``repo_path``.

    Default action pushes a ``{branch_prefix}/<slug>`` branch; ``auto_merge`` also advances main.
    On any gate failure the working copy is reverted and (if ``project`` is set) the confirmed gaps
    are emitted as review-then-implement Forge tasks. ``dry_run`` stops after planning (no LLM test
    generation, no writes).
    """
    vcs = detect_vcs(repo_path)
    if vcs not in ("jj", "git"):
        return AutoTestResult(status="error", reason=f"no jj/git repo at {repo_path}")

    limit = max_gaps if max_gaps is not None else settings.auto_max_gaps
    report = run_review(paths, focus)
    gaps = _confirmed_at(report, min_severity, limit)
    if not gaps:
        log("no confirmed gaps at or above the severity floor — nothing to auto-test")
        return AutoTestResult(status="no-gaps")

    slug = slugify(gaps[0].gap.target)
    branch = f"{branch_prefix}/{slug}"

    if dry_run:
        log(f"[dry-run] would generate {len(gaps)} test(s), gate, and push {branch}")
        return AutoTestResult(status="planned", branch=branch, gaps_targeted=len(gaps))

    def _fallback(
        reason: str, *, passed: bool | None = None, so: SignoffResult | None = None
    ) -> AutoTestResult:
        log(f"blocked: {reason} — reverting and falling back to Forge emit")
        try:
            revert_changes(repo_path)
        except VCSError as e:
            log(f"warning: revert failed: {e}")
        emitted = None
        if project:
            emitted = emit_report(report, project=project, min_severity=min_severity, log=log)
        _log_decision(repo_path, "blocked", reason, branch, gaps, passed, so, auto_merge)
        return AutoTestResult(
            status="blocked",
            reason=reason,
            branch=branch,
            gaps_targeted=len(gaps),
            tests_passed=passed,
            signoff=so,
            emitted=emitted,
        )

    # 1. Generate + apply
    context, _sources, _tests = collect_context(paths)
    env = generate_tests(context, gaps, log=log)
    written = apply_generated(repo_path, env, log=log)
    if not written:
        return _fallback("generator produced no applicable test files")
    log(f"generated {len(written)} test file(s): {', '.join(written)}")

    # 2. Gate — tests-only
    verdict = classify_tests_only(repo_path)
    if not verdict.ok:
        return _fallback(f"tests-only gate: {verdict.reason}")

    # 3. Gate — green suite
    passed, output = run_tests(repo_path)
    if not passed:
        return _fallback(f"green-suite gate failed:\n{output[-800:]}", passed=False)

    # 4. Gate — full-quorum sign-off
    diff_text = working_diff(repo_path)
    so = _signoff(diff_text, pr_ref=branch)
    if not so.approved:
        detail = so.reason + (f"; blockers: {'; '.join(so.blockers[:3])}" if so.blockers else "")
        return _fallback(f"sign-off gate: {detail}", passed=True, so=so)

    # 5. Action — push branch, optionally advance main
    push = push_branch(repo_path, branch, _commit_message(gaps, written))
    status: AutoStatus = "branched"
    merged = False
    if auto_merge:
        advance_main(repo_path, push.change_id)
        status, merged = "merged", True

    _log_decision(
        repo_path, status, "", branch, gaps, True, so, auto_merge, change_id=push.change_id
    )
    log(f"{status}: {branch} @ {push.change_id}" + (" (merged to main)" if merged else ""))
    return AutoTestResult(
        status=status,
        branch=branch,
        change_id=push.change_id,
        merged_to_main=merged,
        gaps_targeted=len(gaps),
        generated_files=written,
        tests_passed=True,
        signoff=so,
    )


def _commit_message(gaps: list[ScoredGap], written: list[str]) -> str:
    targets = ", ".join(g.gap.target for g in gaps[:3])
    body = (
        "Auto-generated by `meta testing --auto`.\n"
        f"Closes {len(gaps)} confirmed coverage gap(s); files: {', '.join(written)}.\n\n"
        "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
    )
    return f"test: cover {targets}\n\n{body}"


def _log_decision(
    repo_path: Path,
    status: str,
    reason: str,
    branch: str,
    gaps: list[ScoredGap],
    passed: bool | None,
    so: SignoffResult | None,
    auto_merge: bool,
    *,
    change_id: str | None = None,
) -> None:
    try:
        log_decision(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "agent": "testing-automerge",
                "repo": str(repo_path),
                "status": status,
                "reason": reason,
                "branch": branch,
                "change_id": change_id,
                "auto_merge": auto_merge,
                "gaps_targeted": len(gaps),
                "targets": [g.gap.target for g in gaps],
                "tests_passed": passed,
                "signoff": None
                if so is None
                else {
                    "approved": so.approved,
                    "approvals": so.approvals,
                    "attempted": so.attempted,
                    "providers": so.providers,
                },
            },
            settings.auto_log_path,
        )
    except OSError:
        pass  # a logging failure must never block or crash the loop


def render_auto(result: AutoTestResult) -> str:
    """One-screen human summary of an auto-test run."""
    lines = [f"# meta testing --auto — {result.status}"]
    if result.reason:
        lines.append(f"\n{result.reason}")
    if result.gaps_targeted:
        lines.append(f"\n- Gaps targeted: {result.gaps_targeted}")
    if result.generated_files:
        lines.append(f"- Generated: {', '.join(result.generated_files)}")
    if result.tests_passed is not None:
        lines.append(f"- Suite: {'green' if result.tests_passed else 'red'}")
    if result.signoff is not None:
        so = result.signoff
        provs = ", ".join(so.providers)
        lines.append(f"- Sign-off: {so.approvals}/{so.attempted} approved ({provs})")
    if result.branch:
        merged = " → merged to main" if result.merged_to_main else ""
        lines.append(f"- Branch: {result.branch}{merged}")
    if result.emitted is not None:
        lines.append(f"- Fallback: {result.emitted.line()}")
    return "\n".join(lines)
