"""Autonomous task worker — pick up a worker-ready task and execute it.

Flow (one-shot MVP):
  1. Query Nous for the highest-priority worker-ready task.
  2. Fetch its full spec as markdown.
  3. Verify the target repo has a supported VCS.
  4. Preflight: project's `gaol dx` container must be running.
  5. Verify a clean working copy, mark the task In Progress.
  6. Run OpenCode inside the dx container against the local LLM router.
  7. Sanity-check the diff (max_files guardrail).
  8. Run tests inside the dx container if required.
  9. Commit the change (on host) and mark the task Done.

Any failure path reverts the working copy (host) and flips the task back to
Ready with a diagnostic note. Execution is sandboxed to the dx container;
VCS operations stay on the host.
"""

from __future__ import annotations

import argparse
import time

from agents.task_worker.config import settings
from agents.task_worker.dx import check_dx_ready
from agents.task_worker.executor import execute_task_with_opencode
from agents.task_worker.nous_client import (
    find_next_task,
    get_task_spec,
    update_task_status,
)
from agents.task_worker.tester import run_tests
from agents.task_worker.vcs import (
    VCSError,
    commit,
    detect_vcs,
    get_changed_files,
    revert_changes,
)


def _tail(s: str, n: int = 500) -> str:
    return s if len(s) <= n else s[-n:]


def _safe_revert(repo_path, label: str) -> None:
    """Revert and log. Never raise; a failure here is surfaced but not fatal."""
    try:
        revert_changes(repo_path)
    except VCSError as e:
        print(f"[{label}] Revert FAILED: {e}")
    except Exception as e:  # noqa: BLE001 — worker must never crash here
        print(f"[{label}] Revert raised unexpectedly: {e}")


def _safe_status(task_name: str, status: str, notes: str = "") -> None:
    """Update task status, logging but swallowing errors."""
    if settings.dry_run:
        print(f"[DRY RUN] Would set '{task_name}' -> {status}")
        return
    try:
        update_task_status(task_name, status, notes=notes)
    except Exception as e:  # noqa: BLE001
        print(f"Failed to update task status to {status}: {e}")


def run(*, project_filter: str | None = None) -> None:
    """Pick and execute one task (MVP). Future: loop mode."""
    print(f"Task Worker starting (dry_run={settings.dry_run})")

    # 1. Find next worker-ready task
    allowed = list(settings.allowed_projects)
    if project_filter:
        allowed = [project_filter]

    try:
        task = find_next_task(allowed)
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

    # 2. Get full spec
    try:
        spec = get_task_spec(task.task)
    except Exception as e:  # noqa: BLE001
        print(f"Failed to fetch task spec: {e}")
        return

    if "BLOCKED:" in spec.upper():
        print("Spec contains BLOCKED marker, skipping")
        return

    project_dir = settings.projects_dir / task.project
    if not project_dir.is_dir():
        print(f"Project dir not found: {project_dir}")
        return

    vcs = detect_vcs(project_dir)
    if not vcs:
        print(f"No VCS detected in {project_dir}, skipping")
        return
    print(f"  repo={project_dir} vcs={vcs}")

    # 3. Preflight: require a running gaol dx container for this project
    dx_ready, dx_status = check_dx_ready(project_dir)
    if not dx_ready:
        print(
            f"dx container not ready for {task.project} ({dx_status}). "
            f"Skipping. Run `cd {project_dir} && gaol dx shell` to set it up."
        )
        return
    print(f"  dx: {dx_status}")

    # 4. Verify clean working copy before starting
    try:
        existing_changes = get_changed_files(project_dir)
    except VCSError as e:
        print(f"VCS inspection failed: {e}")
        return

    if existing_changes:
        print(
            f"Working copy not clean ({len(existing_changes)} files changed). "
            f"Skipping to avoid conflicts."
        )
        return

    # 5. Mark in progress
    _safe_status(task.task, "In Progress", notes="Autonomous worker started")

    # 6. Execute via OpenCode (inside dx container)
    start = time.time()
    model = task.model_tier or settings.model_tier_default
    print(f"Executing with model llm/{model}...")
    try:
        success, stdout_tail = execute_task_with_opencode(
            task, spec, project_dir, model, settings.task_timeout_seconds
        )
    except Exception as e:  # noqa: BLE001
        success = False
        stdout_tail = f"executor raised: {e}"

    duration = time.time() - start
    print(f"OpenCode finished in {duration:.1f}s (success={success})")

    if not success:
        print(f"OpenCode failed, reverting. Tail: {_tail(stdout_tail, 200)}")
        _safe_revert(project_dir, "post-opencode-failure")
        _safe_status(
            task.task,
            "Ready",
            notes=f"Autonomous worker failed:\n\n{_tail(stdout_tail, 500)}",
        )
        return

    # 7. Check scope
    try:
        changed = get_changed_files(project_dir)
    except VCSError as e:
        print(f"VCS inspection failed after execution: {e}")
        _safe_revert(project_dir, "post-exec-vcs-fail")
        _safe_status(task.task, "Ready", notes=f"Worker VCS inspection failed: {e}")
        return

    max_files = task.max_files if task.max_files is not None else settings.default_max_files
    if len(changed) > max_files:
        print(
            f"Changed {len(changed)} files, max allowed {max_files}. Reverting."
        )
        _safe_revert(project_dir, "max-files-exceeded")
        _safe_status(
            task.task,
            "Ready",
            notes=(
                f"Worker exceeded max_files ({len(changed)} > {max_files}). "
                f"Files: {', '.join(changed)}"
            ),
        )
        return

    if not changed:
        print("No files changed. Reverting to clean state and marking Ready.")
        _safe_revert(project_dir, "no-changes")
        _safe_status(
            task.task,
            "Ready",
            notes="Worker produced no file changes; nothing to commit.",
        )
        return

    # 8. Run tests if required (inside dx container)
    if task.requires_tests:
        print("Running tests...")
        try:
            tests_passed, test_output = run_tests(project_dir)
        except Exception as e:  # noqa: BLE001
            tests_passed = False
            test_output = f"tester raised: {e}"

        if not tests_passed:
            print(f"Tests failed. Reverting. Tail: {_tail(test_output, 200)}")
            _safe_revert(project_dir, "tests-failed")
            _safe_status(
                task.task,
                "Ready",
                notes=f"Worker tests failed:\n\n{_tail(test_output, 500)}",
            )
            return
        print("Tests passed")

    # 9. Commit (on host)
    if settings.dry_run:
        print(f"[DRY RUN] Would commit {len(changed)} files: {changed}")
        print("[DRY RUN] Reverting to leave repo clean.")
        _safe_revert(project_dir, "dry-run-cleanup")
        return

    commit_msg = (
        f"{settings.commit_prefix}{task.task}\n\n"
        f"(Autonomous worker: task {task.id})"
    )
    try:
        commit_id = commit(project_dir, commit_msg)
    except VCSError as e:
        print(f"Commit failed: {e}. Reverting.")
        _safe_revert(project_dir, "commit-failed")
        _safe_status(task.task, "Ready", notes=f"Worker commit failed: {e}")
        return

    print(f"Committed: {commit_id}")

    # 10. Mark done
    _safe_status(
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


def main() -> None:
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
    args = parser.parse_args()
    if args.dry_run:
        settings.dry_run = True
    run(project_filter=args.project)


if __name__ == "__main__":
    main()
