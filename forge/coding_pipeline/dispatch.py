"""Serial dispatcher — run a planned wave through ``task_worker.run_one``, one leaf at a time.

Serial on purpose (design: "Dispatch — serial MVP"): the worker demands a clean working copy
and commits on the host, so N=1 in flight is trivially safe; parallel jj-workspace dispatch is
explicitly deferred. Two hard lessons from the dogfood are encoded here:

- **The repo lock.** Two staggered `meta task` invocations once selected sibling leaves against
  one working copy — the In Progress gate narrows that race but doesn't close it. A wave takes
  a per-repo lockfile (``.task_worker/dispatch.lock``, pid inside) before dispatching; a live
  holder aborts the wave, a dead holder's lock is stolen with a note.
- **A leaf failure does not abort the wave** — the worker already reverted and re-opened the
  task; the remaining leaves still get their shot. Only a *preflight* failure aborts, with
  nothing dispatched.

Working-copy positioning (the epic bookmark) is the orchestrator's job — the dispatcher never
moves VCS state; per-leaf safety (fresh gate re-check, clean-WC guard, max_files, tests,
revert-on-fail) all live in ``run_one`` and are not duplicated here.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

from agents.coding_pipeline.journal import append_leaf_context, append_leaf_outcome
from agents.coding_pipeline.models import LeafOutcome, WavePlan
from agents.task_worker.main import run_one
from agents.task_worker.models import RunOutcome, TaskInfo
from agents.task_worker.nous_client import find_task, get_task_spec
from agents.task_worker.sandbox import make_sandbox
from agents.task_worker.vcs import detect_vcs

_LOCK_RELPATH = Path(".task_worker") / "dispatch.lock"


class DispatchError(RuntimeError):
    """The wave could not start (preflight or lock) — nothing was dispatched."""


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by someone else
    return True


@contextmanager
def repo_lock(repo: Path) -> Iterator[Path]:
    """Hold the per-repo dispatch lock for the duration of a wave.

    The lock file carries the holder's pid. A live holder raises DispatchError
    (another dispatch owns the repo); a dead holder's lock is stale and stolen.
    """
    lock_path = repo / _LOCK_RELPATH
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            holder = int(lock_path.read_text().strip())
        except (OSError, ValueError):
            holder = -1
        if holder > 0 and _pid_alive(holder):
            raise DispatchError(
                f"another dispatch holds {lock_path} (pid {holder}, alive) — refusing to race it"
            ) from None
        # stale lock from a dead process: steal it
        lock_path.write_text(str(os.getpid()))
    else:
        with os.fdopen(fd, "w") as fh:
            fh.write(str(os.getpid()))
    try:
        yield lock_path
    finally:
        try:
            lock_path.unlink(missing_ok=True)
        except OSError:
            pass  # a leftover lock is stale-detected next time; never mask the wave's outcome


def _preflight(repo: Path) -> str:
    """'' when the wave may start, else the reason it must not."""
    if not detect_vcs(repo):
        return f"no jj/git repo at {repo}"
    ready, status = make_sandbox(repo).preflight()
    if not ready:
        return f"sandbox not ready: {status}"
    return ""


def _to_leaf_outcome(title: str, outcome: RunOutcome) -> LeafOutcome:
    return LeafOutcome(
        leaf=title,
        status=outcome.status,
        reason=outcome.reason,
        commit_id=outcome.commit_id,
        changed_files=outcome.changed_files,
        duration_s=outcome.duration_s,
    )


def _augmented_spec(
    task: TaskInfo,
    preamble_for: Callable[[TaskInfo], str],
    journal_dir: Path | None,
    fetch_spec: Callable[[str], str],
    log: Callable[[str], None],
) -> str | None:
    """Preamble + spec for the leaf, or ``None`` to dispatch plain.

    Context injection must NEVER break dispatch: any failure (preamble build, spec
    fetch) logs, journals the error, and falls back to the worker's own spec fetch.
    """
    try:
        preamble = preamble_for(task)
    except Exception as e:  # noqa: BLE001 — degrade path by design
        log(f"  epic-context preamble failed ({e}) — dispatching plain")
        if journal_dir is not None:
            append_leaf_context(journal_dir, task.task, deps_landed=[], chars=0, error=str(e))
        return None
    if not preamble:
        return None
    try:
        spec = fetch_spec(task.task)
    except Exception as e:  # noqa: BLE001 — degrade path by design
        log(f"  spec fetch for context injection failed ({e}) — dispatching plain")
        if journal_dir is not None:
            append_leaf_context(journal_dir, task.task, deps_landed=[], chars=0, error=str(e))
        return None
    if journal_dir is not None:
        deps_landed = [d for d in task.deps if f'"{d}"' in preamble]
        append_leaf_context(journal_dir, task.task, deps_landed=deps_landed, chars=len(preamble))
    return f"{preamble}\n\n---\n\n{spec}"


def run_wave(
    plan: WavePlan,
    repo: Path,
    *,
    journal_dir: Path | None = None,
    run_leaf: Callable[..., RunOutcome] = run_one,
    find: Callable[[str], TaskInfo | None] = find_task,
    preamble_for: Callable[[TaskInfo], str] | None = None,
    fetch_spec: Callable[[str], str] = get_task_spec,
    log: Callable[[str], None] = print,
) -> list[LeafOutcome]:
    """Dispatch ``plan.dispatch`` serially; return one ``LeafOutcome`` per leaf, in order.

    Each outcome is journaled as it lands (when ``journal_dir`` is given), so a crash
    mid-wave loses nothing. Raises :class:`DispatchError` if preflight or the repo lock
    fails — in that case nothing was dispatched and Forge state is untouched.

    ``preamble_for`` (the epic-context builder) prepends sibling-contract context to
    the spec passed into ``run_leaf``; when absent, empty, or failing, the leaf runs
    plain and the worker fetches its own spec — injection can never block a wave.
    """
    reason = _preflight(repo)
    if reason:
        raise DispatchError(f"wave preflight failed: {reason}")

    outcomes: list[LeafOutcome] = []
    with repo_lock(repo):
        for title in plan.dispatch:
            task = find(title)
            if task is None:
                log(f"leaf not found in Forge, skipping: {title}")
                outcome = LeafOutcome(
                    leaf=title, status="skipped", reason="task not found in Forge"
                )
            else:
                log(f"dispatching leaf: {title}")
                spec = None
                if preamble_for is not None:
                    spec = _augmented_spec(task, preamble_for, journal_dir, fetch_spec, log)
                if spec is None:
                    outcome = _to_leaf_outcome(title, run_leaf(task))
                else:
                    outcome = _to_leaf_outcome(title, run_leaf(task, spec=spec))
            outcomes.append(outcome)
            if journal_dir is not None:
                append_leaf_outcome(journal_dir, title, outcome)
            log(f"  -> {outcome.status}" + (f" ({outcome.reason})" if outcome.reason else ""))
    return outcomes
