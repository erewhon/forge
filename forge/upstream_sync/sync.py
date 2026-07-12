"""The upstream sync loop: fetch → compare → merge in a worktree → gate → push, fail-closed.

    fetch upstream → behind? → worktree merge → [green suite] → [collision seat]
         │ up to date → stop                        │ any miss → push branch → task → stop
         │ textual conflict → task → stop           │ all green → push branch
                                                       (→ advance main with --auto-merge)

Differences from the bumper's loop, both deliberate:

- **No clean-WC guard needed** — all merge work happens in a temporary ``git worktree``;
  the caller's checkout (which jj may own — sprinkles is jj-colocated) is never touched.
  The repo gains only refs: the fetched remote refs and the sync branch.
- **Gate misses still push the branch** (like the bumper's advisory path): a red suite or
  a collision verdict is exactly what the reviewing human needs to see on a branch. Only
  a textual conflict files a task without a branch — there is no mergeable state to push.

``--auto-merge`` advances the remote default branch only when: tests pass, the collision
seat affirmatively says no collision (None = unknown blocks — fail-closed), and the local
default branch matches origin's (otherwise someone else moved it; a human reconciles).
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from forge.shared.automerge import log_decision
from forge.shared.gitops import GitError, detect_branch, git, git_ok, temporary_worktree
from forge.task_worker.tester import run_tests
from forge.upstream_sync.config import settings
from forge.upstream_sync.emit import emit_advisory
from forge.upstream_sync.layer import compute_layer
from forge.upstream_sync.models import SyncResult
from forge.upstream_sync.seat import collision_verdict


def sync_upstream(
    repo_path: Path,
    *,
    project: str | None = None,
    auto_merge: bool = False,
    dry_run: bool = False,
    log: Callable[[str], None] = print,
) -> SyncResult:
    """Run the sync once. ``dry_run`` stops after the comparison (no worktree, no writes)."""
    repo_path = repo_path.resolve()
    if not (repo_path / ".git").exists():
        return _err(
            repo_path,
            log,
            f"{repo_path} is not a git-backed repo (v1 supports git; "
            "a jj-colocated repo works — a pure jj repo does not)",
        )

    remote = settings.remote
    if not git_ok(repo_path, "remote", "get-url", remote):
        return _err(
            repo_path,
            log,
            f"remote {remote!r} is not configured — "
            f"`git remote add {remote} <url>` (or set UPSTREAM_SYNC_REMOTE)",
        )

    log(f"fetching {remote}...")
    try:
        git(repo_path, "fetch", remote, timeout=settings.fetch_timeout)
        ubranch = settings.upstream_branch or detect_branch(repo_path, f"refs/remotes/{remote}")
        lbranch = settings.local_branch or detect_branch(repo_path, "refs/heads")
        upstream_tip = git(repo_path, "rev-parse", f"refs/remotes/{remote}/{ubranch}")
        local_tip = git(repo_path, "rev-parse", f"refs/heads/{lbranch}")
        merge_base = git(repo_path, "merge-base", local_tip, upstream_tip)
    except GitError as e:
        return _err(repo_path, log, str(e))

    if merge_base == upstream_tip:
        log(f"up to date with {remote}/{ubranch}")
        result = SyncResult(status="up-to-date", upstream_tip=upstream_tip, merge_base=merge_base)
        _log(repo_path, result)
        return result

    behind = int(git(repo_path, "rev-list", "--count", f"{merge_base}..{upstream_tip}"))
    upstream_range = f"{merge_base}..{upstream_tip}"
    upstream_files = [
        f for f in git(repo_path, "diff", "--name-only", upstream_range).splitlines() if f
    ]
    upstream_log = git(
        repo_path, "log", "--oneline", "--no-decorate", "-n", str(settings.log_cap), upstream_range
    )
    layer = compute_layer(repo_path, merge_base, local_tip)
    overlap = sorted(set(upstream_files) & (set(layer.modified) | set(layer.added)))
    branch = f"{settings.branch_prefix}/{_today()}-{upstream_tip[:8]}"

    log(
        f"{behind} upstream commit(s) since merge-base; layer: {len(layer.added)} added / "
        f"{len(layer.modified)} modified; overlap: {len(overlap)} file(s)"
    )

    if dry_run:
        log(
            f"[dry-run] would merge {remote}/{ubranch} ({upstream_tip[:8]}) into {lbranch} "
            f"on {branch}"
        )
        return SyncResult(
            status="planned",
            branch=branch,
            upstream_tip=upstream_tip,
            merge_base=merge_base,
            commits_behind=behind,
            layer=layer,
            overlap=overlap,
        )

    repo_name = repo_path.name
    base = SyncResult(
        status="error",
        branch=branch,
        upstream_tip=upstream_tip,
        merge_base=merge_base,
        commits_behind=behind,
        layer=layer,
        overlap=overlap,
    )

    try:
        with temporary_worktree(repo_path, branch, local_tip) as wt:
            # 1. The merge — a textual conflict ends the run with a task, never a push.
            try:
                git(
                    wt,
                    "merge",
                    "--no-ff",
                    "--no-edit",
                    "-m",
                    _merge_message(remote, ubranch, behind, upstream_tip),
                    upstream_tip,
                    timeout=settings.merge_timeout,
                )
            except GitError:
                conflicted = [
                    f for f in git(wt, "diff", "--name-only", "--diff-filter=U").splitlines() if f
                ]
                git(wt, "merge", "--abort", check=False)
                result = base.model_copy(
                    update={
                        "status": "conflict",
                        "branch": None,
                        "conflicted": conflicted,
                        "reason": f"textual merge conflict in {len(conflicted)} file(s): "
                        f"{', '.join(conflicted[:10])}",
                    }
                )
                log(f"conflict: {result.reason}")
                _emit(repo_name, result, upstream_log, project, log)
                _log(repo_path, result)
                return result

            # 2. Gates. Both run to completion — the reviewing human wants both answers.
            tests_passed, test_output = run_tests(wt)
            verdict = collision_verdict(
                layer=layer,
                upstream_files=upstream_files,
                upstream_log=upstream_log,
                upstream_stat=git(repo_path, "diff", "--stat", upstream_range),
                overlap=overlap,
                overlap_diff=_overlap_diff(repo_path, upstream_range, overlap),
            )

            # 3. The branch pushes regardless of gate outcomes (advisory = reviewable).
            git(wt, "push", "--force", "-u", "origin", branch, timeout=settings.fetch_timeout)

            blockers = []
            if not tests_passed:
                blockers.append(f"green-suite gate failed:\n{test_output[-800:]}")
            if verdict.collision:
                cited = "; ".join(f"{f.file}: {f.reason}" for f in verdict.findings[:5])
                blockers.append(f"collision seat blocked: {cited}")

            if blockers:
                result = base.model_copy(
                    update={
                        "status": "advisory",
                        "reason": " | ".join(blockers),
                        "tests_passed": tests_passed,
                        "collision": verdict,
                    }
                )
                log(f"advisory: {result.reason[:300]}")
                _emit(repo_name, result, upstream_log, project, log)
                _log(repo_path, result)
                return result

            # 4. All green. Default parks at the branch; --auto-merge advances the remote
            # default branch — fail-closed on unknown collision and on a moved origin.
            status, merged, reason = "branched", False, ""
            if auto_merge:
                if verdict.collision is None:
                    reason = (
                        f"not advancing {lbranch}: collision verdict unknown "
                        f"({verdict.notes or 'no seat output'})"
                    )
                elif not _origin_matches(repo_path, lbranch, local_tip):
                    reason = (
                        f"not advancing {lbranch}: local {lbranch} != origin/{lbranch} "
                        "— someone else moved it; reconcile first"
                    )
                else:
                    git(wt, "push", "origin", f"HEAD:{lbranch}", timeout=settings.fetch_timeout)
                    status, merged = "merged", True
                    reason = (
                        f"origin/{lbranch} advanced to the sync merge; local {lbranch} "
                        "is now behind — pull / jj git fetch to catch up"
                    )
            result = base.model_copy(
                update={
                    "status": status,
                    "reason": reason,
                    "tests_passed": tests_passed,
                    "collision": verdict,
                    "merged_to_main": merged,
                }
            )
            log(f"{status}: {branch} @ {upstream_tip[:8]}" + (f" — {reason}" if reason else ""))
            _log(repo_path, result)
            return result
    except GitError as e:
        result = base.model_copy(update={"status": "error", "reason": f"VCS action failed: {e}"})
        log(f"error: {result.reason}")
        _log(repo_path, result)
        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _today() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d")


def _merge_message(remote: str, ubranch: str, behind: int, upstream_tip: str) -> str:
    return (
        f"merge {remote}/{ubranch} ({behind} commits, tip {upstream_tip[:8]})\n\n"
        "Auto-generated by `forge upstream`.\n\n"
        "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
    )


def _overlap_diff(repo: Path, upstream_range: str, overlap: list[str]) -> str:
    if not overlap:
        return ""
    diff = git(repo, "diff", upstream_range, "--", *overlap[:50])
    cap = settings.diff_cap
    return diff if len(diff) <= cap else diff[:cap] + "\n... [diff capped]"


def _origin_matches(repo: Path, branch: str, local_tip: str) -> bool:
    """True when origin's copy of *branch* is exactly at *local_tip* (post-fetch view)."""
    if not git_ok(repo, "fetch", "origin", branch):
        return False
    try:
        return git(repo, "rev-parse", f"refs/remotes/origin/{branch}") == local_tip
    except GitError:
        return False


def _emit(
    repo_name: str,
    result: SyncResult,
    upstream_log: str,
    project: str | None,
    log: Callable[[str], None],
) -> None:
    try:
        summary = emit_advisory(repo_name, result, upstream_log, project=project, log=log)
        if summary is not None:
            log(f"advisory task: {summary.line()}")
    except Exception as e:  # noqa: BLE001 — tracker outage must not turn a sync into a crash
        log(f"warning: advisory task emission failed: {e}")


def _err(repo_path: Path, log: Callable[[str], None], reason: str) -> SyncResult:
    log(f"error: {reason}")
    result = SyncResult(status="error", reason=reason)
    _log(repo_path, result)
    return result


def _log(repo_path: Path, result: SyncResult) -> None:
    try:
        log_decision(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "agent": "upstream-sync",
                "repo": str(repo_path),
                **result.model_dump(),
            },
            settings.auto_log_path,
        )
    except OSError:
        pass  # a logging failure must never block or crash the loop
