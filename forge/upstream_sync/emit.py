"""Advisory task emission for upstream syncs that need human eyes.

A task is filed only on the advisory paths (merge conflict, gate miss, collision verdict)
— a green branch push is a report, not a task, mirroring the bumper. ``external_ref`` is
keyed by the upstream tip, so re-running the sync against the same upstream state dedupes
instead of stacking tasks.
"""

from __future__ import annotations

from collections.abc import Callable

from forge.shared.forge_emit import EmitSpec, EmitSummary
from forge.shared.task_store import get_task_store
from forge.upstream_sync.models import SyncResult


def external_ref(repo_name: str, upstream_tip: str) -> str:
    return f"upstream:{repo_name}:{upstream_tip[:12]}"


def _title(repo_name: str, result: SyncResult) -> str:
    # Comma-free (Forge Depends On cells split on commas).
    what = "merge conflict" if result.status == "conflict" else "review needed"
    return f"Upstream sync {what} for {repo_name} ({result.commits_behind} commits behind)"


def _content(repo_name: str, result: SyncResult, upstream_log: str) -> str:
    lines = [
        f"`forge upstream` could not cleanly land the upstream sync for `{repo_name}`.",
        "",
        f"**Reason:** {result.reason}",
        "",
        f"**Upstream:** {result.commits_behind} commit(s) ahead "
        f"(tip `{(result.upstream_tip or '')[:12]}` from "
        f"merge-base `{(result.merge_base or '')[:12]}`)",
    ]
    if result.branch and result.status != "conflict":
        lines += [
            "",
            f"**Branch:** `{result.branch}` (the merge is applied there — review and merge "
            "or discard)",
        ]
    if result.conflicted:
        lines += ["", "**Conflicted files:**"] + [f"- `{f}`" for f in result.conflicted[:40]]
    if result.collision and result.collision.findings:
        lines += ["", "**Collision findings:**"] + [
            f"- `{f.file}` — {f.reason}" for f in result.collision.findings[:20]
        ]
    if result.collision and result.collision.notes:
        lines += ["", f"**Seat notes:** {result.collision.notes}"]
    if result.layer:
        lines += [
            "",
            f"**Additive layer:** {len(result.layer.added)} fork-added file(s), "
            f"{len(result.layer.modified)} fork-modified file(s); "
            f"overlap with this sync: {', '.join(result.overlap[:15]) or 'none'}",
        ]
    if upstream_log:
        capped = upstream_log.splitlines()[:40]
        lines += ["", "**Upstream commits:**", "```", *capped, "```"]
    lines += ["", "---", "_Auto-proposed by `forge upstream`. Review before merging._"]
    return "\n".join(lines)


def emit_advisory(
    repo_name: str,
    result: SyncResult,
    upstream_log: str,
    *,
    project: str | None,
    log: Callable[[str], None] = print,
) -> EmitSummary | None:
    """File the advisory task (when *project* is set). Never raises past the caller's log —
    a tracker outage must not turn a completed sync into a crash."""
    if not project:
        return None
    spec = EmitSpec(
        title=_title(repo_name, result),
        content=_content(repo_name, result, upstream_log),
        external_ref=external_ref(repo_name, result.upstream_tip or "unknown"),
        task_type="chore",
        priority=4 if result.status == "conflict" else 5,
    )
    return get_task_store().emit([spec], project=project, status="Ready", log=log)
