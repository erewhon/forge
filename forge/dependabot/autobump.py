"""The dependabot bumper's scan → bump → gate → act loop (mirrors testing's autotest.py).

    scan → pick → apply bump → [manifest-only] → [green suite] → [supply-chain sign-off]
         → push branch (→ advance main with --auto-merge)
      │ policy-ineligible or any gate miss → push advisory branch → Forge task → stop

Fail-closed is inherited from the testing loop: the loop can END at advisory (branch pushed,
task filed, nothing merged) but a gate miss can never fall THROUGH to a merge. One bump per
branch, one candidate per run — the v1 policy from the approved framing. The manifest-only
classifier, VCS actions, and decision log live in ``forge.shared.automerge``; the full-quorum
sign-off in ``forge.shared.signoff``, seated from pr_review's cross-family roster.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from forge.dependabot.audit import run_audit
from forge.dependabot.bump import apply_bump, lockfile_delta
from forge.dependabot.config import settings
from forge.dependabot.emit import emit_advisory
from forge.dependabot.models import (
    BumpCandidate,
    BumpResult,
    EvidenceBundle,
)
from forge.dependabot.policy import auto_eligible
from forge.dependabot.prompts import SUPPLY_CHAIN_SIGNOFF, render_evidence
from forge.dependabot.scan import scan_outdated
from forge.dependabot.supply_chain import collect_evidence
from forge.shared.automerge import (
    advance_main,
    classify_manifest_only,
    log_decision,
    push_branch,
    repark_working_copy,
    slugify,
    working_copy_base,
    working_diff,
)
from forge.shared.signoff import SignoffResult, SignoffSeat, full_quorum_signoff
from forge.task_worker.tester import run_tests
from forge.task_worker.vcs import VCSError, detect_vcs, get_changed_files, revert_changes


def _signoff(diff_text: str, *, pr_ref: str, context: str) -> SignoffResult:
    """Seat the shared full-quorum gate from pr_review's active provider roster."""
    from forge.pr_review_ensemble.providers import build_reviewer_slots

    seats = [
        SignoffSeat(provider=s.provider, executor=s.pool.executors[0])
        for s in build_reviewer_slots()
        if s.active
    ]
    return full_quorum_signoff(
        diff_text,
        seats=seats,
        system=SUPPLY_CHAIN_SIGNOFF,
        ref=pr_ref,
        context=context,
        max_tokens=settings.signoff_max_tokens,
        timeout=settings.signoff_timeout,
    )


def _branch_for(candidate: BumpCandidate) -> str:
    return f"{settings.branch_prefix}/{slugify(candidate.name)}-{slugify(candidate.latest)}"


def _pick(candidates: list[BumpCandidate]) -> tuple[BumpCandidate, bool]:
    """(candidate, pre_eligible): the first patch/minor candidate rides the auto track; when
    none qualifies even pre-evidence, the top candidate goes straight to advisory."""
    for c in candidates:
        if c.delta in ("patch", "minor"):
            return c, True
    return candidates[0], False


def auto_bump(
    repo_path: Path,
    *,
    project: str | None = None,
    auto_merge: bool = False,
    dry_run: bool = False,
    log: Callable[[str], None] = print,
) -> BumpResult:
    """Run the scan → bump → gate → act loop once (one candidate, one branch).

    Default action pushes a ``deps/<name>-<version>`` branch; ``auto_merge`` also advances
    main WHEN every gate approves. Policy-ineligible candidates and gate misses end in the
    advisory action: the bump is pushed as a branch, a Forge task is filed (when ``project``
    is set), and nothing merges. ``dry_run`` stops after candidate selection.
    """
    vcs = detect_vcs(repo_path)
    if vcs not in ("jj", "git"):
        return BumpResult(status="error", reason=f"no jj/git repo at {repo_path}")

    candidates = scan_outdated(repo_path)
    if not candidates:
        log("no outdated direct dependencies — nothing to bump")
        return BumpResult(status="no-candidates")
    findings = run_audit(repo_path)
    log(f"{len(candidates)} outdated candidate(s); {len(findings)} audit finding(s) repo-wide")

    candidate, pre_eligible = _pick(candidates)
    branch = _branch_for(candidate)

    if dry_run:
        track = "auto" if pre_eligible else "advisory (delta not patch/minor)"
        log(
            f"[dry-run] would bump {candidate.name} {candidate.current} -> "
            f"{candidate.latest} ({candidate.delta}) on {branch}; track: {track}"
        )
        return BumpResult(status="planned", candidate=candidate, branch=branch)

    # Clean-WC guard (same rule as the task worker): push_branch commits the WHOLE working
    # copy, so running over uncommitted work would scoop it into the bump branch — the exact
    # incident that added this guard (a worker's half-finished CLI files rode a real advisory
    # branch). Dry runs above are read-only and exempt.
    dirty = [f for f in get_changed_files(repo_path) if f.strip()]
    if dirty:
        return BumpResult(
            status="error",
            reason=(
                f"working copy not clean ({len(dirty)} changed file(s): "
                f"{', '.join(dirty[:5])}) — the bumper owns the whole working copy; "
                "commit or revert first"
            ),
        )

    # Where the working copy sits NOW — after any branch push that doesn't advance main, the
    # loop reparks here so the bump branch stays a side head, never the mainline's parent.
    base = working_copy_base(repo_path)

    # 1. Apply the bump (manifest+lockfile-only by construction of `uv lock -P`).
    changed = apply_bump(repo_path, candidate)
    if not changed:
        log(f"{candidate.name}: constraints already pin it — nothing to bump")
        return BumpResult(
            status="no-candidates",
            reason=f"{candidate.name} is constraint-pinned; lock unchanged",
            candidate=candidate,
        )

    # 2. Evidence + policy — the deterministic dial runs before any LLM.
    evidence = collect_evidence(candidate, findings, lockfile_delta(repo_path))

    def _advisory(reason: str, *, passed: bool | None = None, so=None) -> BumpResult:
        log(f"advisory: {reason}")
        try:
            push_branch(repo_path, branch, _commit_message(candidate, evidence))
            repark_working_copy(repo_path, base)
        except VCSError as e:
            # Same cleanup contract as the post-gate action: revert whatever is uncommitted,
            # then best-effort repark — push_branch may have already committed the bump, and
            # without the repark that commit becomes the mainline's parent (live finding).
            try:
                revert_changes(repo_path)
                repark_working_copy(repo_path, base)
            except VCSError as cleanup_err:
                log(f"warning: cleanup after failed advisory push also failed: {cleanup_err}")
            _log(repo_path, "error", f"{reason}; push failed: {e}", candidate, evidence, so=so)
            return BumpResult(
                status="error", reason=f"{reason}; push failed: {e}", candidate=candidate
            )
        emitted = emit_advisory(
            candidate, evidence, reason, project=project, branch=branch, log=log
        )
        if emitted is not None:
            log(f"advisory task: {emitted.line()}")
        _log(repo_path, "advisory", reason, candidate, evidence, passed=passed, so=so)
        return BumpResult(
            status="advisory",
            reason=reason,
            candidate=candidate,
            branch=branch,
            evidence=evidence,
            tests_passed=passed,
        )

    eligible, why_not = auto_eligible(evidence, require_attestation=settings.require_attestation)
    if not eligible:
        return _advisory(why_not)

    # 3. Gate — manifest-only diff.
    verdict = classify_manifest_only(repo_path)
    if not verdict.ok:
        return _advisory(f"manifest-only gate: {verdict.reason}")

    # 4. Gate — green suite (uv run pytest re-syncs the env to the bumped lock).
    passed, output = run_tests(repo_path)
    if not passed:
        return _advisory(f"green-suite gate failed:\n{output[-800:]}", passed=False)

    # 5. Gate — full-quorum supply-chain sign-off over the diff + evidence.
    so = _signoff(working_diff(repo_path), pr_ref=branch, context=render_evidence(evidence))
    if not so.approved:
        detail = so.reason + (f"; blockers: {'; '.join(so.blockers[:3])}" if so.blockers else "")
        return _advisory(f"sign-off gate: {detail}", passed=True, so=so)

    # 6. Action — push the branch; advance main only under the explicit flag. A merge leaves
    # the working copy on the bump commit (that IS main now); a plain branch reparks so the
    # bump stays a side head. A VCS failure here is still fail-closed: revert, best-effort
    # repark, log, error status — never a traceback and never a half-acted merge.
    try:
        push = push_branch(repo_path, branch, _commit_message(candidate, evidence))
        status, merged = "branched", False
        if auto_merge:
            advance_main(repo_path, push.change_id)
            status, merged = "merged", True
        else:
            repark_working_copy(repo_path, base)
    except VCSError as e:
        log(f"error: VCS action failed after all gates passed: {e}")
        try:
            revert_changes(repo_path)
            repark_working_copy(repo_path, base)
        except VCSError as cleanup_err:
            log(f"warning: cleanup after failed VCS action also failed: {cleanup_err}")
        _log(repo_path, "error", f"VCS action failed: {e}", candidate, evidence, so=so)
        return BumpResult(
            status="error",
            reason=f"VCS action failed after all gates passed: {e}",
            candidate=candidate,
            evidence=evidence,
            tests_passed=True,
        )

    _log(repo_path, status, "", candidate, evidence, passed=True, so=so, change_id=push.change_id)
    log(f"{status}: {branch} @ {push.change_id}" + (" (merged to main)" if merged else ""))
    return BumpResult(
        status=status,  # type: ignore[arg-type]
        candidate=candidate,
        branch=branch,
        change_id=push.change_id,
        merged_to_main=merged,
        evidence=evidence,
        tests_passed=True,
    )


def _commit_message(candidate: BumpCandidate, evidence: EvidenceBundle) -> str:
    fixed = ", ".join(f.vuln_id for f in evidence.findings_current)
    fix_line = f"Fixes {fixed}.\n" if fixed else ""
    return (
        f"deps: bump {candidate.name} {candidate.current} -> {candidate.latest}\n\n"
        f"Auto-generated by `meta deps` ({candidate.delta} bump; evidence "
        f"{'complete' if evidence.complete else 'INCOMPLETE'}).\n{fix_line}\n"
        "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
    )


def _log(
    repo_path: Path,
    status: str,
    reason: str,
    candidate: BumpCandidate,
    evidence: EvidenceBundle | None,
    *,
    passed: bool | None = None,
    so: SignoffResult | None = None,
    change_id: str | None = None,
) -> None:
    try:
        log_decision(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "agent": "dependabot-bumper",
                "repo": str(repo_path),
                "status": status,
                "reason": reason,
                "candidate": candidate.model_dump(),
                "evidence_complete": None if evidence is None else evidence.complete,
                "change_id": change_id,
                "tests_passed": passed,
                "signoff": None
                if so is None
                else {
                    "approved": so.approved,
                    "approvals": so.approvals,
                    "attempted": so.attempted,
                    "providers": so.providers,
                    "seats": [
                        {"provider": s.provider, "approve": s.approve, "reason": s.reason}
                        for s in so.seats
                    ],
                },
            },
            settings.auto_log_path,
        )
    except OSError:
        pass  # a logging failure must never block or crash the loop


def render_bump(result: BumpResult) -> str:
    """One-screen human summary of a bumper run. A merged security fix appends the human-gated
    release proposal (rendered commands only — the bumper never cuts releases)."""
    from forge.dependabot.release import render_release_proposal, should_propose_release

    lines = [f"# meta deps — {result.status}"]
    if result.reason:
        lines.append(f"\n{result.reason}")
    if result.candidate is not None:
        c = result.candidate
        lines.append(f"\n- Bump: {c.name} {c.current} -> {c.latest} ({c.delta})")
    if result.evidence is not None:
        lines.append(f"- Evidence complete: {'yes' if result.evidence.complete else 'NO'}")
        fixed = ", ".join(f.vuln_id for f in result.evidence.findings_current)
        if fixed:
            lines.append(f"- Fixes: {fixed}")
    if result.tests_passed is not None:
        lines.append(f"- Suite: {'green' if result.tests_passed else 'red'}")
    if result.branch:
        merged = " → merged to main" if result.merged_to_main else ""
        lines.append(f"- Branch: {result.branch}{merged}")
    if (
        result.merged_to_main
        and result.evidence is not None
        and result.change_id
        and should_propose_release(result.evidence)
    ):
        lines += [
            "",
            render_release_proposal(
                result.evidence,
                merged_change_id=result.change_id,
                on=datetime.now(UTC).date(),
            ),
        ]
    return "\n".join(lines)
