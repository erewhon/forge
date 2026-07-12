"""Autonomous task worker — pick up a worker-ready task and execute it.

Flow (one-shot MVP):
  1. Query Nous for the highest-priority worker-ready task.
  2. Fetch its full spec as markdown.
  3. Verify the target repo has a supported VCS.
  4. Preflight: project's `gaol dx` container must be running.
  5. Verify a clean working copy, mark the task In Progress.
  6. Run OpenCode inside the dx container against the local LLM router.
  7. Sanity-check the diff (max_files guardrail).
  8. Lint the changed files inside the dx container (autofix-then-recheck).
  9. Run tests inside the dx container if required.
 10. Commit the change (on host) and mark the task Done.

Any failure path reverts the working copy (host) and flips the task back to
Ready with a diagnostic note. Execution is sandboxed to the dx container;
VCS operations stay on the host.

``run_one`` is the callable per-task API (the coding pipeline's dispatcher
consumes it); ``run`` is the pick-next CLI wrapper around it. ``run_one``
re-checks the worker gate (Ready AND Auto AND unblocked) against a fresh
Nous read before touching anything, fail-closed — however the caller chose
the task.
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

from forge.shared.task_store import get_task_store
from forge.task_worker.config import settings
from forge.task_worker.executor import execute_task_with_opencode
from forge.task_worker.linter import run_lint
from forge.task_worker.models import RunOutcome, TaskInfo
from forge.task_worker.sandbox import make_sandbox
from forge.task_worker.tester import run_build, run_tests
from forge.task_worker.vcs import (
    VCSError,
    commit,
    detect_vcs,
    get_changed_files,
    revert_changes,
)


def _tail(s: str, n: int = 500) -> str:
    """Last ~n chars, trimmed forward to a line boundary so notes never open mid-line
    (dry-run finding: truncated tails like 'ined: 31.25E' read as garbage in Forge)."""
    if len(s) <= n:
        return s
    cut = s[-n:]
    nl = cut.find("\n")
    if 0 <= nl < len(cut) - 1:
        return cut[nl + 1 :]
    return cut  # one long line: better mid-line than empty


# A BLOCKED marker is a LINE starting with "BLOCKED:" (the model protocol) or the guardrail
# "> **Blocked:**" that get_task_spec injects. Line-anchored on purpose: a spec that merely
# *mentions* the protocol ("print a line starting with BLOCKED:") must not trip it.
_BLOCKED_LINE_RE = re.compile(r"(?im)^\s*(?:>\s*)?(?:\*+)?blocked(?:\*+)?:")


def _has_blocked_marker(spec: str) -> bool:
    return _BLOCKED_LINE_RE.search(spec) is not None


def _safe_revert(repo_path, label: str) -> None:
    """Revert and log. Never raise; a failure here is surfaced but not fatal."""
    try:
        revert_changes(repo_path)
    except VCSError as e:
        print(f"[{label}] Revert FAILED: {e}")
    except Exception as e:  # noqa: BLE001 — worker must never crash here
        print(f"[{label}] Revert raised unexpectedly: {e}")


def _safe_status(task_name: str, status: str, notes: str = "") -> bool:
    """Update task status, logging but swallowing errors. Returns True if the write landed."""
    if settings.dry_run:
        print(f"[DRY RUN] Would set '{task_name}' -> {status}")
        return False
    try:
        get_task_store().update_status(task_name, status, notes=notes)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"Failed to update task status to {status}: {e}")
        return False


def run_one(
    task: TaskInfo,
    *,
    spec: str | None = None,
    repo: Path | None = None,
    sandbox_kind: str | None = None,
) -> RunOutcome:
    """Execute one specific task through the full safety path; return a structured outcome.

    Pass ``spec`` to skip the Nous spec fetch (the CLI path fetches it here). ``repo``
    replaces the conventional-checkout resolution entirely — the concurrent dispatcher runs
    each leaf inside its own jj workspace, and the clean-WC guard, revert-on-fail, sandbox,
    and commit must ALL act on that same directory (a mixed state would corrupt the host
    checkout). ``sandbox_kind`` selects the sandbox implementation for ``make_sandbox``;
    None keeps the env-keyed default. Every failure path maps to a distinct
    ``RunOutcome.reason``; any revert happens before the task's status is written back.
    """
    start = time.time()

    def _outcome(
        status: str,
        reason: str = "",
        *,
        commit_id: str | None = None,
        changed: list[str] | None = None,
        notes_written: bool = False,
    ) -> RunOutcome:
        return RunOutcome(
            task=task.task,
            project=task.project,
            status=status,
            reason=reason,
            commit_id=commit_id,
            changed_files=changed or [],
            duration_s=time.time() - start,
            notes_written=notes_written,
        )

    # 0. Gate re-check against a fresh read — fail closed on any error
    try:
        gate = get_task_store().worker_gate(task.task)
    except Exception as e:  # noqa: BLE001
        gate = f"gate check failed: {e}"
    if gate:
        print(f"Refusing task: {gate}")
        return _outcome("skipped", f"worker gate: {gate}")

    # 1. Get full spec (unless the caller already has it)
    if spec is None:
        try:
            spec = get_task_store().get_spec(task.task)
        except Exception as e:  # noqa: BLE001
            print(f"Failed to fetch task spec: {e}")
            return _outcome("skipped", f"spec fetch failed: {e}")

    if _has_blocked_marker(spec):
        print("Spec contains BLOCKED marker, skipping")
        return _outcome("skipped", "spec contains BLOCKED marker")

    if repo is not None:
        project_dir = repo
        if not project_dir.is_dir():
            print(f"Repo override not found: {project_dir}")
            return _outcome("skipped", f"repo override not found: {project_dir}")
    else:
        # Forge project names are Title Case ("Meta"); checkouts are conventionally lowercase.
        project_dir = settings.projects_dir / task.project
        if not project_dir.is_dir():
            lowered = settings.projects_dir / task.project.lower()
            if lowered.is_dir():
                project_dir = lowered
        if not project_dir.is_dir():
            print(f"Project dir not found: {project_dir}")
            return _outcome("skipped", f"project dir not found: {project_dir}")

    vcs = detect_vcs(project_dir)
    if not vcs:
        print(f"No VCS detected in {project_dir}, skipping")
        return _outcome("skipped", f"no VCS detected in {project_dir}")
    print(f"  repo={project_dir} vcs={vcs}")

    # 3. Preflight: require a ready sandbox (a running gaol dx container by default).
    # The kind pass-through stays out of the default call so None is byte-for-byte today's
    # env-keyed behavior; the factory grows its ``kind`` parameter in the run-once leaf.
    if sandbox_kind is None:
        sandbox = make_sandbox(project_dir)
    else:
        sandbox = make_sandbox(project_dir, kind=sandbox_kind)
    dx_ready, dx_status = sandbox.preflight()
    if not dx_ready:
        print(
            f"dx container not ready for {task.project} ({dx_status}). "
            f"Skipping. Run `cd {project_dir} && gaol dx shell` to set it up."
        )
        return _outcome("skipped", f"dx container not ready ({dx_status})")
    print(f"  dx: {dx_status}")

    # 4. Verify clean working copy before starting
    try:
        existing_changes = get_changed_files(project_dir)
    except VCSError as e:
        print(f"VCS inspection failed: {e}")
        return _outcome("skipped", f"VCS inspection failed: {e}")

    if existing_changes:
        print(
            f"Working copy not clean ({len(existing_changes)} files changed). "
            f"Skipping to avoid conflicts."
        )
        return _outcome(
            "skipped", f"working copy not clean ({len(existing_changes)} files changed)"
        )

    # 5. Mark in progress
    _safe_status(task.task, "In Progress", notes="Autonomous worker started")

    # 6. Execute via OpenCode (inside dx container). A degenerate session — over in
    # seconds with zero file changes (an empty generation from a router hiccup;
    # observed live: a 2.8s zero-tool-call session) — retries in-process so it never
    # burns a journal-counted attempt.
    model = task.model_tier or settings.model_tier_default
    print(f"Executing with model llm/{model}...")
    retries = max(0, settings.degenerate_retries)
    for exec_attempt in range(1 + retries):
        exec_start = time.time()
        try:
            success, stdout_tail, blocked = execute_task_with_opencode(
                task, spec, project_dir, model, settings.task_timeout_seconds, sandbox=sandbox
            )
        except Exception as e:  # noqa: BLE001
            success = False
            stdout_tail = f"executor raised: {e}"
            blocked = False

        duration = time.time() - exec_start
        print(f"OpenCode finished in {duration:.1f}s (success={success})")

        if blocked or exec_attempt >= retries:
            break
        if duration >= settings.degenerate_session_seconds:
            break
        try:
            probe = get_changed_files(project_dir)
        except VCSError:
            probe = []
        if probe:
            break
        print(
            f"Degenerate session ({duration:.1f}s, no file changes) — retrying in-process "
            f"({exec_attempt + 1}/{retries})..."
        )

    if blocked:
        print(f"Model reported BLOCKED, reverting. Tail: {_tail(stdout_tail, 200)}")
        _safe_revert(project_dir, "model-blocked")
        nw = _safe_status(
            task.task,
            "Ready",
            notes=f"Worker model reported BLOCKED:\n\n{_tail(stdout_tail, 500)}",
        )
        return _outcome(
            "failed", f"model reported BLOCKED: {_tail(stdout_tail, 200)}", notes_written=nw
        )

    if not success:
        # A bare non-zero exit is advisory: session-end plugin crashes (observed:
        # open-mem missing its in-container API key) fail the process after the model
        # finished. If a diff exists, the gates below are the real arbiter — a
        # truncated or bad diff fails scope/tests and reverts there. No diff = a
        # genuine startup/run failure.
        try:
            leftover = get_changed_files(project_dir)
        except VCSError:
            leftover = []
        if not leftover:
            print(f"OpenCode failed, reverting. Tail: {_tail(stdout_tail, 200)}")
            _safe_revert(project_dir, "post-opencode-failure")
            nw = _safe_status(
                task.task,
                "Ready",
                notes=f"Autonomous worker failed:\n\n{_tail(stdout_tail, 500)}",
            )
            return _outcome(
                "failed", f"opencode failed: {_tail(stdout_tail, 200)}", notes_written=nw
            )
        print(
            f"OpenCode exited non-zero but left {len(leftover)} changed file(s) — "
            f"proceeding to the gates (exit code is advisory; gates decide)."
        )

    # 7. Check scope
    try:
        changed = get_changed_files(project_dir)
    except VCSError as e:
        print(f"VCS inspection failed after execution: {e}")
        _safe_revert(project_dir, "post-exec-vcs-fail")
        nw = _safe_status(task.task, "Ready", notes=f"Worker VCS inspection failed: {e}")
        return _outcome("failed", f"post-exec VCS inspection failed: {e}", notes_written=nw)

    max_files = task.max_files if task.max_files is not None else settings.default_max_files
    if len(changed) > max_files:
        print(f"Changed {len(changed)} files, max allowed {max_files}. Reverting.")
        _safe_revert(project_dir, "max-files-exceeded")
        nw = _safe_status(
            task.task,
            "Ready",
            notes=(
                f"Worker exceeded max_files ({len(changed)} > {max_files}). "
                f"Files: {', '.join(changed)}"
            ),
        )
        return _outcome(
            "failed",
            f"max_files exceeded ({len(changed)} > {max_files})",
            changed=changed,
            notes_written=nw,
        )

    if not changed:
        print("No files changed. Reverting to clean state and marking Ready.")
        _safe_revert(project_dir, "no-changes")
        nw = _safe_status(
            task.task,
            "Ready",
            notes=(
                "Worker produced no file changes; nothing to commit.\n\n"
                f"Session tail:\n\n{_tail(stdout_tail, 500)}"
            ),
        )
        return _outcome("failed", "no file changes produced", notes_written=nw)

    # 7.5 Static-check gate — ALWAYS runs, independent of requires_tests. Statically
    # broken code (doesn't compile, doesn't type-check, syntax errors) must never
    # land, even for tasks that don't require tests. Checks are additive per
    # detected language; a detected language with its tool missing from the
    # sandbox fails closed rather than passing silent.
    print("Static checks...")
    try:
        build_ok, build_output, build_ran = run_build(
            project_dir, sandbox=sandbox, changed_files=changed
        )
    except Exception as e:  # noqa: BLE001
        build_ok, build_output, build_ran = False, f"builder raised: {e}", True
    if build_ran and not build_ok:
        print(f"Static checks failed. Reverting. Tail: {_tail(build_output, 200)}")
        _safe_revert(project_dir, "build-failed")
        nw = _safe_status(
            task.task,
            "Ready",
            notes=f"Worker static checks failed:\n\n{_tail(build_output, 500)}",
        )
        return _outcome(
            "failed",
            f"static checks failed: {_tail(build_output, 200)}",
            changed=changed,
            notes_written=nw,
        )
    if build_ran:
        print("Static checks clean")

    # 8. Lint gate (inside dx container) — changed files only, autofix-then-recheck.
    # Runs BEFORE tests so a single test run validates the final (possibly autofixed)
    # state. Only violations that survive autofix fail the leaf.
    if task.requires_tests:
        print("Linting changed files...")
        try:
            lint_ok, lint_output, lint_fixed = run_lint(project_dir, changed, sandbox=sandbox)
        except Exception as e:  # noqa: BLE001
            lint_ok, lint_output, lint_fixed = False, f"linter raised: {e}", False
        if lint_fixed:
            print("Lint autofix applied to changed files.")
        if not lint_ok:
            print(f"Lint failed after autofix. Reverting. Tail: {_tail(lint_output, 200)}")
            _safe_revert(project_dir, "lint-failed")
            nw = _safe_status(
                task.task,
                "Ready",
                notes=f"Worker lint failed (after autofix attempt):\n\n{_tail(lint_output, 500)}",
            )
            return _outcome(
                "failed",
                f"lint failed: {_tail(lint_output, 200)}",
                changed=changed,
                notes_written=nw,
            )
        print("Lint clean")

    # 9. Run tests if required (inside dx container)
    if task.requires_tests:
        print("Running tests...")
        try:
            tests_passed, test_output = run_tests(
                project_dir, sandbox=sandbox, changed_files=changed
            )
        except Exception as e:  # noqa: BLE001
            tests_passed = False
            test_output = f"tester raised: {e}"

        if not tests_passed:
            print(f"Tests failed. Reverting. Tail: {_tail(test_output, 200)}")
            _safe_revert(project_dir, "tests-failed")
            nw = _safe_status(
                task.task,
                "Ready",
                notes=f"Worker tests failed:\n\n{_tail(test_output, 500)}",
            )
            return _outcome(
                "failed",
                f"tests failed: {_tail(test_output, 200)}",
                changed=changed,
                notes_written=nw,
            )
        print("Tests passed")

    # 10. Commit (on host)
    if settings.dry_run:
        print(f"[DRY RUN] Would commit {len(changed)} files: {changed}")
        print("[DRY RUN] Reverting to leave repo clean.")
        _safe_revert(project_dir, "dry-run-cleanup")
        return _outcome(
            "skipped", "dry-run: executed and reverted, nothing committed", changed=changed
        )

    commit_msg = f"{settings.commit_prefix}{task.task}\n\n(Autonomous worker: task {task.id})"
    try:
        commit_id = commit(project_dir, commit_msg)
    except VCSError as e:
        print(f"Commit failed: {e}. Reverting.")
        _safe_revert(project_dir, "commit-failed")
        nw = _safe_status(task.task, "Ready", notes=f"Worker commit failed: {e}")
        return _outcome("failed", f"commit failed: {e}", changed=changed, notes_written=nw)

    print(f"Committed: {commit_id}")

    # 10. Mark done
    nw = _safe_status(
        task.task,
        "Done",
        notes=(
            "Completed by autonomous worker.\n"
            f"- Changed files: {', '.join(changed)}\n"
            f"- Commit: {commit_id}\n"
            f"- Duration: {duration:.1f}s"
        ),
    )
    print(f"Task '{task.task}' completed")
    return _outcome("done", commit_id=commit_id, changed=changed, notes_written=nw)


def run(*, project_filter: str | None = None) -> None:
    """Pick and execute one task (MVP). Future: loop mode."""
    print(f"Task Worker starting (dry_run={settings.dry_run})")

    # 1. Find next worker-ready task
    allowed = list(settings.allowed_projects)
    if project_filter:
        allowed = [project_filter]

    try:
        task = get_task_store().next_ready(allowed)
    except Exception as e:  # noqa: BLE001
        print(f"Failed to query Nous for tasks: {e}")
        return

    if task is None:
        print("No worker-ready tasks found")
        return

    print(f"Selected task: [{task.project}] {task.task}")
    print(
        f"  priority={task.priority} mode={task.execution_mode} "
        f"tier={task.model_tier or settings.model_tier_default}"
    )
    run_one(task)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Autonomous task worker")
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="Only pick tasks from this project",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run OpenCode but don't commit or update Nous",
    )
    args = parser.parse_args(argv)
    if args.dry_run:
        settings.dry_run = True
    run(project_filter=args.project)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
