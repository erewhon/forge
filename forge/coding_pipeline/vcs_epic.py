"""Epic integration branch + the final sign-off gate (design: "VCS & blast radius").

All pipeline work lands on ``{branch_prefix}/{epic_slug}``; ``main`` moves only after the epic
gate — a full-quorum, cross-family sign-off on the whole epic diff — and then only by a HUMAN.
There is deliberately no ``advance_main`` call anywhere in this module: the pipeline's terminal
action is a rendered "ready for human merge" summary, never a merge.

Bookmark lifecycle: ``ensure_epic_bookmark`` creates the bookmark at the current tip when absent
(and never moves an existing one); ``update_epic_bookmark`` advances it to the tip after a wave
checkpoint. Pushes ride git.example.com's push-to-create; a push failure is a warning, not a wave
killer — the bookmark is local-first and re-pushed at the next checkpoint.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path

from agents.coding_pipeline.config import settings
from agents.coding_pipeline.models import FramingProposal
from agents.shared.signoff import SignoffResult, SignoffSeat, full_quorum_signoff
from agents.task_worker.vcs import VCSError, detect_vcs

_TIMEOUT = 30

EPIC_SIGNOFF_SYSTEM = """You are the FINAL merge gatekeeper for an epic built by an automated \
coding pipeline. The diff is the epic's ENTIRE accumulated change set; per-leaf tests and \
per-wave reviews already ran. Approve for handoff to a human merger ONLY if ALL hold:
- The changes are correct and coherent as a whole — the pieces fit together, no half-migrations,
  no contradictory edits between leaves.
- The work matches the stated epic framing (provided as context), not some drifted goal.
- No surprising dependency-manifest or lockfile changes (new dependencies need human eyes).
- Tests accompany the behavior they claim to cover and do not weaken existing coverage.

Be strict: this is the last automated check before a human merges to main. If anything is
uncertain, do NOT approve.

Respond with ONLY a JSON object: {"approve": true|false, "blockers": ["..."], "notes": "..."}"""


def epic_branch(epic_slug: str) -> str:
    return f"{settings.branch_prefix}/{epic_slug}"


def _run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=_TIMEOUT, cwd=cwd)


def _branch_exists(repo: Path, vcs: str, branch: str) -> bool:
    if vcs == "jj":
        return _run(["jj", "log", "--no-graph", "-r", branch, "-T", '""'], repo).returncode == 0
    return _run(["git", "rev-parse", "--verify", "--quiet", branch], repo).returncode == 0


def _set_branch_to_tip(repo: Path, vcs: str, branch: str) -> None:
    if vcs == "jj":
        res = _run(["jj", "bookmark", "set", branch, "-r", "@-"], repo)
    elif _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], repo).stdout.strip() == branch:
        return  # checked out: it already rides HEAD
    else:
        res = _run(["git", "branch", "-f", branch, "HEAD"], repo)
    if res.returncode != 0:
        raise VCSError(f"setting {branch} failed: {res.stderr.strip()}")


def _push_branch(repo: Path, vcs: str, branch: str, log: Callable[[str], None]) -> bool:
    """Best-effort push (git.example.com is push-to-create). Returns True when pushed."""
    if vcs == "jj":
        res = _run(["jj", "git", "push", "--bookmark", branch], repo)
    else:
        res = _run(["git", "push", "-u", "origin", branch], repo)
    if res.returncode != 0:
        log(f"warning: push of {branch} failed (will retry next checkpoint): {res.stderr.strip()}")
        return False
    return True


def ensure_epic_bookmark(
    repo: Path, epic_slug: str, *, push: bool = True, log: Callable[[str], None] = print
) -> str:
    """Create ``pipeline/<epic-slug>`` at the current tip if absent; NEVER move an existing one
    (a human may have positioned it). Returns the branch name."""
    vcs = detect_vcs(repo)
    if vcs not in ("jj", "git"):
        raise VCSError(f"No VCS detected in {repo}")
    branch = epic_branch(epic_slug)
    if not _branch_exists(repo, vcs, branch):
        _set_branch_to_tip(repo, vcs, branch)
        log(f"created epic bookmark {branch}")
        if push:
            _push_branch(repo, vcs, branch, log)
    return branch


def update_epic_bookmark(
    repo: Path, epic_slug: str, *, push: bool = True, log: Callable[[str], None] = print
) -> str:
    """Advance the epic bookmark to the current tip (the wave checkpoint) and re-push."""
    vcs = detect_vcs(repo)
    if vcs not in ("jj", "git"):
        raise VCSError(f"No VCS detected in {repo}")
    branch = epic_branch(epic_slug)
    _set_branch_to_tip(repo, vcs, branch)
    if push:
        _push_branch(repo, vcs, branch, log)
    return branch


def epic_diff(repo: Path, epic_slug: str, *, main: str = "main") -> str:
    """The epic's entire accumulated diff: merge-base of ``main`` and the epic branch → tip."""
    vcs = detect_vcs(repo)
    branch = epic_branch(epic_slug)
    if vcs == "jj":
        res = _run(["jj", "diff", "--no-pager", "--git", "--from", main, "--to", branch], repo)
    elif vcs == "git":
        res = _run(["git", "diff", f"{main}...{branch}"], repo)
    else:
        raise VCSError(f"No VCS detected in {repo}")
    if res.returncode != 0:
        raise VCSError(f"epic diff failed: {res.stderr.strip()}")
    return res.stdout


def _default_seats() -> list[SignoffSeat]:
    from agents.pr_review_ensemble.providers import build_reviewer_slots

    return [
        SignoffSeat(provider=slot.provider, executor=slot.pool.executors[0])
        for slot in build_reviewer_slots()
        if slot.active
    ]


def run_epic_gate(
    repo: Path,
    epic_slug: str,
    framing: FramingProposal,
    *,
    seats: list[SignoffSeat] | None = None,
    main: str = "main",
) -> SignoffResult:
    """The final automated check: full-quorum cross-family sign-off on the whole epic diff.

    Fail-closed exactly like the auto-merge gate (degraded quorum, one dissent, or an
    unparseable verdict = blocked). Approval means "ready for a HUMAN to merge" — nothing
    here or downstream advances ``main``.
    """
    diff = epic_diff(repo, epic_slug, main=main)
    if not diff.strip():
        return SignoffResult(
            approved=False, attempted=0, approvals=0, reason="empty epic diff — nothing to gate"
        )
    return full_quorum_signoff(
        diff,
        seats=seats if seats is not None else _default_seats(),
        system=EPIC_SIGNOFF_SYSTEM,
        ref=epic_branch(epic_slug),
        context=f"Epic framing (approved by a human): {framing.restated_goal}",
        max_tokens=settings.review_max_tokens,
        timeout=settings.review_timeout,
    )


def render_epic_gate(result: SignoffResult, epic_slug: str, *, tip: str = "") -> str:
    """The human-facing verdict — what `meta build gate` prints."""
    branch = epic_branch(epic_slug)
    lines = [f"# Epic gate — {branch}" + (f" @ {tip}" if tip else "")]
    if result.approved:
        lines.append(
            f"\nAPPROVED by full quorum ({result.approvals}/{result.attempted}: "
            f"{', '.join(result.providers)})."
        )
        lines.append("\nReady for HUMAN merge — the pipeline never advances main.")
        lines.append(f"  jj bookmark set main -r {branch} && jj git push --bookmark main")
    else:
        lines.append(f"\nBLOCKED: {result.reason}")
        for blocker in result.blockers[:5]:
            lines.append(f"  - {blocker}")
    return "\n".join(lines)
