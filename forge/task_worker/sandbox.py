"""The sandbox seam: where task execution and test runs actually happen.

The worker's safety story is "execution is sandboxed; VCS is host-only" — but *which* sandbox is
environment-specific. The local-lab uses `gaol dx` containers; the work environment (no gaol) will
need its own implementation. This module is the seam: a small ``Sandbox`` protocol, the
``GaolDxSandbox`` implementation (delegating to the existing ``dx``/``tester`` code, behavior
unchanged), and a ``make_sandbox`` factory keyed by ``TASK_WORKER_SANDBOX`` (only ``gaol-dx``
exists today; the work implementation belongs to the work-harness task).

``run`` deliberately returns the raw ``CompletedProcess`` and lets ``TimeoutExpired`` /
``FileNotFoundError`` propagate — the callers (executor, tester) already own that error handling
and its exact user-facing messages, and moving it here would change behavior.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Protocol, runtime_checkable

from agents.task_worker.config import settings
from agents.task_worker.dx import check_dx_ready, dx_run


@runtime_checkable
class Sandbox(Protocol):
    """Where the worker executes commands and tests for one repo."""

    repo: Path

    def preflight(self) -> tuple[bool, str]:
        """Is the sandbox ready to run? Returns (ready, status_line)."""
        ...

    def run(self, cmd: list[str], *, timeout: int) -> subprocess.CompletedProcess[str]:
        """Run *cmd* inside the sandbox. May raise TimeoutExpired/FileNotFoundError.

        Implementations MUST ensure the command cannot outlive this call inside the
        sandbox — an orphaned process keeps writing into the shared repo (see
        GaolDxSandbox.run for the incident that proved it)."""
        ...

    def run_tests(self) -> tuple[bool, str]:
        """Run the repo's test suite inside the sandbox. Returns (passed, output_tail)."""
        ...


class GaolDxSandbox:
    """The local-lab sandbox: a running `gaol dx` container with the repo bind-mounted."""

    # Host-side grace beyond the container-side timeout, and the SIGKILL follow-up
    # after the container-side SIGTERM.
    _HOST_GRACE_S = 60
    _KILL_AFTER_S = 30

    def __init__(self, repo: Path) -> None:
        self.repo = repo

    def preflight(self) -> tuple[bool, str]:
        return check_dx_ready(self.repo)

    def run(self, cmd: list[str], *, timeout: int) -> subprocess.CompletedProcess[str]:
        # The timeout must die INSIDE the container: killing the host-side gaol client
        # (subprocess timeout) orphans the in-container process, which keeps running and
        # keeps writing into the bind-mounted repo — a zombie OpenCode overwrote the
        # working copy half an hour after its worker run was reverted (dogfood find).
        # coreutils `timeout` enforces the budget in-container; the host-side kill is a
        # delayed backstop that should never fire first.
        guarded = ["timeout", f"--kill-after={self._KILL_AFTER_S}", f"{timeout}s", *cmd]
        return dx_run(self.repo, guarded, timeout=timeout + self._HOST_GRACE_S)

    def run_tests(self) -> tuple[bool, str]:
        from agents.task_worker import tester  # local import: tester uses make_sandbox

        return tester.run_tests(self.repo, sandbox=self)


def make_sandbox(repo: Path) -> Sandbox:
    """Build the configured sandbox for *repo*. Unknown kinds raise — never guess a sandbox."""
    kind = settings.sandbox
    if kind == "gaol-dx":
        return GaolDxSandbox(repo)
    raise ValueError(f"unknown TASK_WORKER_SANDBOX {kind!r} (only 'gaol-dx' is implemented)")
