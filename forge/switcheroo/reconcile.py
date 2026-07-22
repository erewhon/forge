"""Switch-back: turn a finished (or interrupted) failover window back into a resumable briefing.

Coming back from an outage is a **diff read, not a teleport**. Three sources combine:

- the **baton** — where the session was (goal, plan, decisions, pre-failover anchor);
- the **failover journal** — what the fleet drained, across every repo, with per-leaf commit ids;
- the **home-repo diff since the anchor** — what actually changed on disk here (a leaf that landed
  in the home repo, plus any work the session left mid-flight at switchover).

This module owns only the *reading* half: computing the home-repo delta and rendering the combined
"while you were away" briefing. Consuming the window (baton re-anchor, archival) is the CLI's job in
:mod:`forge.switcheroo.main`, so the render stays a pure function that's trivial to test.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from pydantic import BaseModel, Field

from forge.shared.baton import Baton
from forge.switcheroo.journal import render_failover_summary
from forge.switcheroo.models import FailoverLog

_TIMEOUT = 15


def _work_files(raw: str) -> list[str]:
    """Changed files, minus switcheroo's own ``.forge/`` machinery (baton, journal) — the returning
    session reconciles *work*, not the handoff bookkeeping this tool wrote as it ran."""
    return [
        f.strip() for f in raw.splitlines() if f.strip() and not f.strip().startswith(".forge/")
    ]


class HomeCommit(BaseModel):
    change_id: str
    description: str


class HomeChanges(BaseModel):
    """The home repo's delta since the failover anchor — best-effort, never fatal."""

    vcs: str | None = None
    anchor: str | None = None
    commits: list[HomeCommit] = Field(default_factory=list)
    changed_files: list[str] = Field(default_factory=list)
    note: str = ""  #: Why the delta is empty/unavailable (no anchor, not a repo, probe failed).


def _run(args: list[str], repo: Path) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(args, cwd=repo, capture_output=True, text=True, timeout=_TIMEOUT)
    except (OSError, subprocess.SubprocessError):
        return None


def _jj_changes(home: Path, anchor: str) -> HomeChanges:
    log = _run(
        [
            "jj",
            "--no-pager",
            "log",
            "-r",
            f"{anchor}..@",
            "--no-graph",
            "-T",
            'change_id.short() ++ "\t" ++ description.first_line() ++ "\n"',
        ],
        home,
    )
    diff = _run(["jj", "--no-pager", "diff", "--from", anchor, "--name-only"], home)
    if log is None or log.returncode != 0 or diff is None or diff.returncode != 0:
        return HomeChanges(vcs="jj", anchor=anchor, note="jj could not resolve the anchor")
    commits: list[HomeCommit] = []
    for line in log.stdout.splitlines():
        cid, _, desc = line.partition("\t")
        if cid.strip() and desc.strip():  # drop the empty working-copy commit at the tip
            commits.append(HomeCommit(change_id=cid.strip(), description=desc.strip()))
    return HomeChanges(
        vcs="jj", anchor=anchor, commits=commits, changed_files=_work_files(diff.stdout)
    )


def _git_changes(home: Path, anchor: str) -> HomeChanges:
    log = _run(["git", "log", "--oneline", "--no-decorate", f"{anchor}..HEAD"], home)
    diff = _run(["git", "diff", "--name-only", f"{anchor}..HEAD"], home)
    if log is None or log.returncode != 0 or diff is None or diff.returncode != 0:
        return HomeChanges(vcs="git", anchor=anchor, note="git could not resolve the anchor")
    commits = [
        HomeCommit(change_id=cid, description=desc)
        for cid, _, desc in (ln.partition(" ") for ln in log.stdout.splitlines())
        if cid.strip()
    ]
    return HomeChanges(
        vcs="git", anchor=anchor, commits=commits, changed_files=_work_files(diff.stdout)
    )


def home_repo_changes(home: Path, anchor: str | None) -> HomeChanges:
    """The home repo's changes since *anchor* — the local half of "what changed while I was away".
    Fully best-effort: an unversioned baton, a non-repo, or a probe failure yields an empty result
    with a ``note``, never an exception (switch-back must always produce a briefing)."""
    from forge.task_worker.vcs import detect_vcs

    if not anchor:
        return HomeChanges(note="baton carried no VCS anchor; home-repo delta unavailable")
    try:
        vcs = detect_vcs(home)
    except Exception:  # noqa: BLE001 — detection is advisory here
        vcs = ""
    if vcs == "jj":
        return _jj_changes(home, anchor)
    if vcs == "git":
        return _git_changes(home, anchor)
    return HomeChanges(vcs=vcs or None, anchor=anchor, note="home is not a jj/git repo")


def render_switchback(baton: Baton | None, log: FailoverLog | None, changes: HomeChanges) -> str:
    """The consolidated switch-back briefing: where we were, what the fleet did, what changed here,
    and how to resume. A pure function of its three inputs."""
    lines = ["=" * 60, "SWITCH-BACK — resume briefing", "=" * 60, ""]

    if baton is not None:
        lines.append(f"You were working on: {baton.goal or '(no goal recorded)'}")
        if baton.next_action:
            lines.append(f"  pre-failover next action: {baton.next_action}")
        if baton.plan:
            lines.append("  remaining plan:")
            lines += [f"    - {step}" for step in baton.plan]
        if baton.decisions:
            lines.append("  decisions in force:")
            lines += [f"    - {d}" for d in baton.decisions]
        lines.append("")

    lines.append("While you were away:")
    if log is not None:
        lines.append(render_failover_summary(log))
    else:
        lines.append("  (no failover window on record)")
    lines.append("")

    lines.append(f"Home repo delta since anchor {changes.anchor or '(none)'}:")
    if changes.note:
        lines.append(f"  {changes.note}")
    elif not changes.commits and not changes.changed_files:
        lines.append(
            "  no home-repo changes (the fleet worked other repos — see the commits above)"
        )
    else:
        for c in changes.commits:
            lines.append(f"  · {c.change_id}  {c.description}")
        if changes.changed_files:
            lines.append(f"  files: {', '.join(changes.changed_files)}")
    lines.append("")

    lines += [
        "Resume:",
        "  1. Reconcile each landed commit above (each in its own repo).",
        "  2. Review the home-repo delta; fold anything useful into your plan.",
        "  3. Continue — the baton has been re-anchored to the post-failover state.",
    ]
    return "\n".join(lines)
