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

import asyncio
import subprocess
from collections.abc import Callable
from pathlib import Path

from forge.coding_pipeline.config import settings
from forge.coding_pipeline.models import FramingProposal
from forge.shared.ensemble import Pool, Prompt, map_items
from forge.shared.signoff import SignoffResult, SignoffSeat, full_quorum_signoff
from forge.task_worker.vcs import VCSError, detect_vcs

_TIMEOUT = 30

EPIC_SIGNOFF_SYSTEM = """You are the FINAL merge gatekeeper for an epic built by an automated \
coding pipeline. The diff is the epic's ENTIRE accumulated change set; per-leaf tests and \
per-wave reviews already ran. Approve for handoff to a human merger ONLY if ALL hold:
- The changes are correct and coherent as a whole — the pieces fit together, no half-migrations,
  no contradictory edits between leaves.
- The work matches the approved epic framing (provided as context), not some drifted goal. The
  framing's value ordering is the build sequence WITHIN this epic — the diff is the finished
  epic, so later ordering items being present is completed scope, not drift. Drift means work
  the framing never asked for.
- No surprising dependency-manifest or lockfile changes (new dependencies need human eyes).
- Tests accompany the behavior they claim to cover and do not weaken existing coverage.

Be strict: this is the last automated check before a human merges to main. If anything is
uncertain, do NOT approve.

Respond with ONLY a JSON object: {"approve": true|false, "blockers": ["..."], "notes": "..."}"""

EPIC_MAP_SYSTEM = """You are summarizing ONE slice of a large epic diff for a merge gatekeeper \
who cannot read the whole diff. Report, factually and compactly:
- What this slice changes (files, behavior) in a few bullets.
- RED FLAGS the gatekeeper must know about: dependency-manifest or lockfile changes, deleted or \
weakened tests, half-migrations (old and new paths both live), contradictory edits, debug \
leftovers, hardcoded secrets, or anything else suspicious.
- If there is nothing suspicious, end with exactly: No red flags.

Do NOT approve or reject — you see only a slice; the verdict happens elsewhere. Plain markdown,
under 300 words."""

EPIC_REDUCE_SYSTEM = """You are the FINAL merge gatekeeper for an epic built by an automated \
coding pipeline. The epic diff was too large to read whole, so you are judging per-slice \
gatekeeper summaries that cover the ENTIRE diff (a missing slice fails the gate before it \
reaches you). Per-leaf tests and per-wave reviews already ran. Approve for handoff to a human \
merger ONLY if ALL hold:
- The slices are coherent as a whole — the pieces fit together, no half-migrations, no
  contradictory edits between slices.
- The work matches the approved epic framing (provided as context), not some drifted goal. The
  framing's value ordering is the build sequence WITHIN this epic — the diff is the finished
  epic, so later ordering items being present is completed scope, not drift. Drift means work
  the framing never asked for.
- No slice reports dependency-manifest or lockfile changes (new dependencies need human eyes).
- No slice reports deleted or weakened test coverage.
- No slice reports a red flag the other slices don't resolve.

Be strict: this is the last automated check before a human merges to main, and you are judging
summaries rather than code — uncertainty weighs AGAINST approval. If anything is unclear, do
NOT approve.

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
    from forge.pr_review_ensemble.providers import build_reviewer_slots

    return [
        SignoffSeat(provider=slot.provider, executor=slot.pool.executors[0])
        for slot in build_reviewer_slots()
        if slot.active
    ]


def _gate_context(framing: FramingProposal) -> str:
    """The framing context every gate seat judges scope against. The restated goal alone is not
    enough: distill-evals' goal line emphasized the first vertical slice while the recommendation
    and value ordering carried the full approved scope, and the gate blocked the rest as drift.
    Pass everything the human approved about WHAT to build; inventory/options stay out."""
    lines = [
        "Epic framing (approved by a human) — judge scope against ALL of it, not the goal alone:",
        f"Goal: {framing.restated_goal}",
        f"Approved recommendation: {framing.recommendation}",
    ]
    if framing.value_ordering:
        lines.append(
            "Approved value ordering — the build sequence within this epic; ALL items are in "
            "scope and expected to be present in the finished diff:"
        )
        lines.extend(f"  {i}. {v}" for i, v in enumerate(framing.value_ordering, 1))
    return "\n".join(lines)


def _map_reduce_gate(
    diff: str, epic_slug: str, framing: FramingProposal, seats: list[SignoffSeat]
) -> SignoffResult:
    """Gate an oversized epic diff: deterministic per-file split → one gatekeeper summary per
    slice (failover pool over the seat executors, bounded concurrency) → full-quorum verdict
    over the slice summaries. A gate must never judge code it hasn't seen, so a failed slice
    summary or an over-cap split fails closed — no silent truncation."""
    from forge.pr_review_ensemble.diffsplit import split_diff

    chunks = split_diff(diff, chunk_chars=settings.epic_gate_chunk_chars)
    strategy = f"map-reduce over {len(chunks)} slice(s) (diff {len(diff)} chars)"
    if len(chunks) > settings.epic_gate_max_chunks:
        return SignoffResult(
            approved=False,
            attempted=0,
            approvals=0,
            strategy=strategy,
            reason=(
                f"epic diff splits into {len(chunks)} slices, over the "
                f"{settings.epic_gate_max_chunks}-slice cap — the gate refuses to drop slices; "
                "gate sub-epics separately or raise CODING_PIPELINE_EPIC_GATE_MAX_CHUNKS"
            ),
        )

    # Stable sort: the preferred (cheap) seat leads the failover order, the rest keep roster order.
    map_seats = sorted(seats, key=lambda s: s.provider != settings.epic_gate_map_preferred)
    pool = Pool(role="epic-gate-map", executors=[s.executor for s in map_seats])

    async def _summaries():
        async def one(chunk):
            files = ", ".join(chunk.files) or "chunk"
            user = f"Epic: {epic_branch(epic_slug)}\nFiles in this slice: {files}\n\n{chunk.text}"
            prompt = Prompt(
                system=EPIC_MAP_SYSTEM, user=user, max_tokens=settings.epic_gate_map_max_tokens
            )
            return await pool.run(prompt, timeout=settings.review_timeout)

        return await map_items(chunks, one, concurrency=settings.epic_gate_map_concurrency)

    results = asyncio.run(_summaries())
    failed = [(c, r) for c, r in zip(chunks, results) if not r.ok]
    if failed:
        names = "; ".join(", ".join(c.files) or "chunk" for c, _ in failed[:5])
        return SignoffResult(
            approved=False,
            attempted=0,
            approvals=0,
            strategy=strategy,
            reason=(
                f"map stage incomplete: {len(failed)}/{len(chunks)} slice summaries failed "
                f"({names}; first error: {failed[0][1].error}) — the gate must not judge "
                "unseen code"
            ),
        )

    body = "\n\n".join(
        f"### Slice {i + 1}/{len(chunks)}: {', '.join(c.files) or 'chunk'}\n\n{r.output}"
        for i, (c, r) in enumerate(zip(chunks, results))
    )
    result = full_quorum_signoff(
        f"(per-slice gatekeeper summaries of the full epic diff — {len(chunks)} slices, "
        f"all present)\n\n{body}",
        seats=seats,
        system=EPIC_REDUCE_SYSTEM,
        ref=epic_branch(epic_slug),
        context=_gate_context(framing),
        max_tokens=settings.epic_gate_signoff_max_tokens,
        timeout=settings.review_timeout,
    )
    result.strategy = strategy
    return result


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
    unparseable verdict = blocked). A diff past ``epic_gate_max_diff_chars`` gates via
    map-reduce (per-slice summaries, then the quorum verdict over the summaries) instead of
    one oversized call per seat. Approval means "ready for a HUMAN to merge" — nothing here
    or downstream advances ``main``.
    """
    diff = epic_diff(repo, epic_slug, main=main)
    if not diff.strip():
        return SignoffResult(
            approved=False, attempted=0, approvals=0, reason="empty epic diff — nothing to gate"
        )
    resolved = seats if seats is not None else _default_seats()
    if len(diff) > settings.epic_gate_max_diff_chars:
        return _map_reduce_gate(diff, epic_slug, framing, resolved)
    result = full_quorum_signoff(
        diff,
        seats=resolved,
        system=EPIC_SIGNOFF_SYSTEM,
        ref=epic_branch(epic_slug),
        context=_gate_context(framing),
        max_tokens=settings.epic_gate_signoff_max_tokens,
        timeout=settings.review_timeout,
    )
    result.strategy = f"single pass (diff {len(diff)} chars)"
    return result


def render_epic_gate(result: SignoffResult, epic_slug: str, *, tip: str = "") -> str:
    """The human-facing verdict — what `meta build gate` prints. Per-seat lines make a quorum
    miss legible: "no verdict (timed out)" is a different failure from "responded and blocked"."""
    branch = epic_branch(epic_slug)
    lines = [f"# Epic gate — {branch}" + (f" @ {tip}" if tip else "")]
    if result.strategy:
        lines.append(f"Strategy: {result.strategy}")
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
    if result.seats:
        lines.append("\nSeats:")
        for seat in result.seats:
            if seat.approve is True:
                verdict = "approved"
            elif seat.approve is False:
                verdict = "responded — did NOT approve"
            else:
                verdict = f"NO VERDICT ({seat.reason or 'no response'})"
            note = f" — {seat.reason}" if seat.approve is not None and seat.reason else ""
            lines.append(f"  - {seat.provider}: {verdict}{note}")
    return "\n".join(lines)
