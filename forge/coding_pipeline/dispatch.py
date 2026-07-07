"""The wave dispatcher — serial by default, workspace-concurrent behind ``dispatch_concurrency``.

Serial (N=1) is today's proven path: the worker demands a clean working copy and commits on
the host, so one leaf in flight is trivially safe. Concurrency > 1 fans the wave's leaves out
into per-leaf jj workspaces (each under an ephemeral ``gaol run-once`` sandbox — the dx
container is path-bound and can't serve workspace paths), then integrates through the SERIAL
reconcile barrier: rebase each landed commit onto the accumulating epic head in dispatch
order, detect jj's first-class conflicts, demote offenders back to Ready for replan.
Correctness rests on the barrier, never on scheduling. Hard lessons encoded here:

- **The repo lock.** Two staggered `meta task` invocations once selected sibling leaves against
  one working copy — the In Progress gate narrows that race but doesn't close it. A wave takes
  a per-repo lockfile (``.task_worker/dispatch.lock``, pid inside) before dispatching; a live
  holder aborts the wave, a dead holder's lock is stolen with a note.
- **A leaf failure does not abort the wave** — the worker already reverted and re-opened the
  task; the remaining leaves still get their shot (concurrent: a crashed worker thread is one
  failed leaf, its batch-mates run on). Only a *preflight* failure aborts, with nothing
  dispatched.
- **Journal writes stay in one thread.** Worker outcomes are journaled from the event loop as
  each future completes — append-only writes never interleave. A reconcile demotion journals
  its own ``reconcile_demotion`` event rather than a second ``leaf_dispatch``: one dispatch is
  ONE attempt, however it ends, or the escalation cap double-counts.

Working-copy positioning (the epic bookmark) is the orchestrator's job — the serial dispatcher
never moves VCS state, and the concurrent path only advances it through the reconcile barrier;
per-leaf safety (fresh gate re-check, clean-WC guard, max_files, tests, revert-on-fail) all
live in ``run_one`` and are not duplicated here.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

from agents.coding_pipeline.config import settings
from agents.coding_pipeline.journal import (
    _journal_path,
    append_leaf_context,
    append_leaf_outcome,
)
from agents.coding_pipeline.models import LeafOutcome, WavePlan
from agents.shared.automerge import log_decision
from agents.task_worker.main import run_one
from agents.task_worker.models import RunOutcome, TaskInfo
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


def _preflight(repo: Path, sandbox_kind: str | None = None) -> str:
    """'' when the wave may start, else the reason it must not.

    ``sandbox_kind`` is the kind the wave will actually run under — the concurrent path
    probes run-once (its workspaces can't use the path-bound dx container)."""
    if not detect_vcs(repo):
        return f"no jj/git repo at {repo}"
    ready, status = make_sandbox(repo, kind=sandbox_kind).preflight()
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
    find: Callable[[str], TaskInfo | None] | None = None,
    preamble_for: Callable[[TaskInfo], str] | None = None,
    fetch_spec: Callable[[str], str] | None = None,
    concurrency: int | None = None,
    log: Callable[[str], None] = print,
) -> list[LeafOutcome]:
    """Dispatch ``plan.dispatch``; return one ``LeafOutcome`` per leaf, in plan order.

    ``concurrency`` (default: ``settings.dispatch_concurrency``) of 1 is the serial path,
    byte-for-byte the pre-concurrency behavior; above 1, leaves fan out into per-leaf jj
    workspaces under run-once sandboxes and integrate through the serial reconcile barrier.

    Each outcome is journaled as it lands (when ``journal_dir`` is given), so a crash
    mid-wave loses nothing. Raises :class:`DispatchError` if preflight or the repo lock
    fails — in that case nothing was dispatched and Forge state is untouched.

    ``find`` and ``fetch_spec`` default to the configured task store (injectable for
    tests). ``preamble_for`` (the epic-context builder) prepends sibling-contract context
    to the spec passed into ``run_leaf``; when absent, empty, or failing, the leaf runs
    plain and the worker fetches its own spec — injection can never block a wave.
    """
    if find is None or fetch_spec is None:
        from agents.shared.task_store import get_task_store

        store = get_task_store()
        find = find or store.find_task
        fetch_spec = fetch_spec or store.get_spec

    cap = concurrency if concurrency is not None else settings.dispatch_concurrency
    if cap <= 1:
        reason = _preflight(repo)
        if reason:
            raise DispatchError(f"wave preflight failed: {reason}")
        with repo_lock(repo):
            return _run_serial(
                plan,
                repo,
                journal_dir=journal_dir,
                run_leaf=run_leaf,
                find=find,
                preamble_for=preamble_for,
                fetch_spec=fetch_spec,
                log=log,
            )

    reason = _preflight(repo, "gaol-run-once")
    if reason:
        raise DispatchError(f"wave preflight failed: {reason}")
    with repo_lock(repo):
        return _run_concurrent(
            plan,
            repo,
            journal_dir=journal_dir,
            run_leaf=run_leaf,
            find=find,
            preamble_for=preamble_for,
            fetch_spec=fetch_spec,
            cap=cap,
            log=log,
        )


def _run_serial(
    plan: WavePlan,
    repo: Path,
    *,
    journal_dir: Path | None,
    run_leaf: Callable[..., RunOutcome],
    find: Callable[[str], TaskInfo | None],
    preamble_for: Callable[[TaskInfo], str] | None,
    fetch_spec: Callable[[str], str],
    log: Callable[[str], None],
) -> list[LeafOutcome]:
    """Today's proven path, unchanged: one leaf at a time against the main working copy."""
    outcomes: list[LeafOutcome] = []
    for title in plan.dispatch:
        task = find(title)
        if task is None:
            log(f"leaf not found in Forge, skipping: {title}")
            outcome = LeafOutcome(leaf=title, status="skipped", reason="task not found in Forge")
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


def _slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:30] or "leaf"


def _run_concurrent(
    plan: WavePlan,
    repo: Path,
    *,
    journal_dir: Path | None,
    run_leaf: Callable[..., RunOutcome],
    find: Callable[[str], TaskInfo | None],
    preamble_for: Callable[[TaskInfo], str] | None,
    fetch_spec: Callable[[str], str],
    cap: int,
    log: Callable[[str], None],
) -> list[LeafOutcome]:
    """Fan the wave out into per-leaf jj workspaces, then reconcile at the serial barrier.

    Forge IO (find/spec/preamble) happens up front in this thread; workspaces are created
    serially at the shared base rev; workers run in threads (``run_one`` is synchronous and
    subprocess-heavy) bounded by ``cap``; journal/log writes happen on the event loop as
    each future completes. The barrier: every landed commit rebases onto the accumulating
    head in DISPATCH order, conflicts demote to Ready (their attempt already journaled —
    the demotion is a separate event, never a second ``leaf_dispatch``). Workspaces are
    forgotten in a finally — success, conflict, or crash.
    """
    import asyncio

    from agents.coding_pipeline.reconcile import ReconcileError, reconcile_wave
    from agents.shared.ensemble.pool import map_items
    from agents.shared.workspaces import (
        JJError,
        create_workspace,
        forget_workspace,
        resolve_base_rev,
        workspace_destination,
    )

    base = resolve_base_rev(repo, "@-")
    log(f"concurrent dispatch: {len(plan.dispatch)} leaf(s), cap {cap}, base {base[:12]}")

    prepared: list[tuple[str, TaskInfo | None, str | None]] = []
    for title in plan.dispatch:
        task = find(title)
        if task is None:
            log(f"leaf not found in Forge, skipping: {title}")
            prepared.append((title, None, None))
            continue
        spec = None
        if preamble_for is not None:
            spec = _augmented_spec(task, preamble_for, journal_dir, fetch_spec, log)
        prepared.append((title, task, spec))

    workspaces: dict[str, Path] = {}
    try:
        for title, task, _ in prepared:
            if task is None:
                continue
            dest = workspace_destination(repo, _slug(title), prefix="cw")
            create_workspace(repo, dest, base_rev=base)
            workspaces[title] = dest
    except JJError as e:
        for ws in workspaces.values():
            forget_workspace(repo, ws)
        raise DispatchError(f"workspace setup failed, nothing dispatched: {e}") from e

    def _worker(item: tuple[str, TaskInfo | None, str | None]) -> LeafOutcome:
        title, task, spec = item
        if task is None:
            return LeafOutcome(leaf=title, status="skipped", reason="task not found in Forge")
        try:
            kwargs: dict = {"repo": workspaces[title], "sandbox_kind": "gaol-run-once"}
            if spec is not None:
                kwargs["spec"] = spec
            return _to_leaf_outcome(title, run_leaf(task, **kwargs))
        except Exception as e:  # noqa: BLE001 — a dead worker is one failed leaf, never the batch
            return LeafOutcome(leaf=title, status="failed", reason=f"worker crashed: {e}")

    async def _fan_out() -> list[LeafOutcome]:
        async def one(item: tuple[str, TaskInfo | None, str | None]) -> LeafOutcome:
            log(f"dispatching leaf (workspace): {item[0]}")
            outcome = await asyncio.to_thread(_worker, item)
            # Journal from the event-loop thread — append-only writes never interleave.
            if journal_dir is not None:
                append_leaf_outcome(journal_dir, outcome.leaf, outcome)
            log(
                f"  -> {outcome.leaf}: {outcome.status}"
                + (f" ({outcome.reason})" if outcome.reason else "")
            )
            return outcome

        return await map_items(prepared, one, concurrency=cap)

    try:
        outcomes = asyncio.run(_fan_out())

        # map_items preserves input order, so `landed` is already dispatch order.
        landed = [(o, o.commit_id) for o in outcomes if o.status == "done" and o.commit_id]

        from agents.shared.task_store import get_task_store

        store = get_task_store()

        def _demote(title: str, note: str) -> None:
            store.update_status(title, "Ready", notes=note)
            if journal_dir is not None:
                log_decision(
                    {"event": "reconcile_demotion", "leaf": title},
                    _journal_path(journal_dir),
                )

        try:
            reconciled = {
                o.leaf: o
                for o in reconcile_wave(repo, base, landed, on_demote=_demote, log=log)
            }
        except ReconcileError as e:
            # Leaves DID run and Forge was updated — but the working copy never advanced,
            # so verify/replan would judge a stale state. Abort the wave loudly instead.
            raise DispatchError(
                f"wave dispatched but the reconcile barrier failed to reposition the "
                f"working copy: {e}"
            ) from e
        return [reconciled.get(o.leaf, o) for o in outcomes]
    finally:
        for ws in workspaces.values():
            forget_workspace(repo, ws)
