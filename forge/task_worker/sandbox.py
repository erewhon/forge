"""The sandbox seam: where task execution and test runs actually happen.

The worker's safety story is "execution is sandboxed; VCS is host-only" — but *which* sandbox is
environment-specific. The local-lab uses `gaol dx` containers; the work environment (no gaol) will
need its own implementation. This module is the seam: a small ``Sandbox`` protocol, the
``GaolDxSandbox`` implementation (delegating to the existing ``dx``/``tester`` code, behavior
unchanged), the ``GaolRunOnceSandbox`` implementation (ephemeral containers for repos without a
dx container — the concurrent dispatcher's jj workspaces), and a ``make_sandbox`` factory keyed
by ``TASK_WORKER_SANDBOX`` with a per-call ``kind`` override.

``run`` deliberately returns the raw ``CompletedProcess`` and lets ``TimeoutExpired`` /
``FileNotFoundError`` propagate — the callers (executor, tester) already own that error handling
and its exact user-facing messages, and moving it here would change behavior.
"""

from __future__ import annotations

import shutil
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


class GaolRunOnceSandbox:
    """Ephemeral sandboxes for repos without a dx container — the dispatcher's jj workspaces.

    Every ``run()`` is its own ``gaol run-once`` container with the repo bind-mounted writable
    at the SAME absolute path (opencode derives its project root from PWD, so cwd semantics
    must match the host). opencode's host config dir is mounted shared; a per-sandbox data dir
    seeded with only ``auth.json`` gives each concurrent worker a private ``opencode.db``
    (concurrent sessions corrupt the shared sqlite WAL — parallel_edit finding). That state
    dir lives under the repo's self-ignored ``.task_worker/``, so it dies with the workspace
    and never dirties the diff. Per-sandbox memory/CPU caps keep fan-out from exhausting the
    host.

    run-once's own ``--timeout`` does the deterministic in-container kill + teardown (the
    coreutils-timeout wrapper GaolDxSandbox needs is built into run-once); the host-side
    subprocess timeout is a delayed backstop that should never fire first.
    """

    _HOST_GRACE_S = 60

    def __init__(self, repo: Path) -> None:
        self.repo = repo

    def preflight(self) -> tuple[bool, str]:
        if shutil.which(settings.gaol_binary) is None:
            return False, f"gaol binary {settings.gaol_binary!r} not on PATH"
        try:
            probe = subprocess.run(
                [settings.gaol_binary, "run-once", "--help"],
                capture_output=True,
                text=True,
                timeout=15,
            )
        except (subprocess.TimeoutExpired, OSError) as e:
            return False, f"gaol run-once probe failed: {e}"
        if probe.returncode != 0:
            return False, f"gaol run-once probe exited {probe.returncode}"
        return True, "gaol run-once (ephemeral per command)"

    def _opencode_mounts(self) -> list[tuple[str, str]]:
        """Host opencode config (shared, read-mostly) + a per-sandbox data dir seeded with
        only auth.json. Empty when opencode isn't set up on the host — plain commands
        (pytest, ruff) don't need it."""
        mounts: list[tuple[str, str]] = []
        home = settings.runonce_home
        host_cfg = Path.home() / ".config" / "opencode"
        if host_cfg.is_dir():
            mounts.append((str(host_cfg), f"{home}/.config/opencode"))
        host_auth = Path.home() / ".local" / "share" / "opencode" / "auth.json"
        if host_auth.is_file():
            state = self.repo / ".task_worker" / "opencode-state"
            state.mkdir(parents=True, exist_ok=True)
            gitignore = state.parent / ".gitignore"  # executor's self-ignore convention
            if not gitignore.exists():
                gitignore.write_text("*\n", encoding="utf-8")
            target = state / "auth.json"
            if not target.exists():
                shutil.copy2(host_auth, target)
            mounts.append((str(state), f"{home}/.local/share/opencode"))
        return mounts

    def run(self, cmd: list[str], *, timeout: int) -> subprocess.CompletedProcess[str]:
        # KEEP ARGS PLAIN: gaol re-joins argv into an unquoted shell string, so a
        # metacharacter in any element (backticks!) would be shell-interpreted in the
        # container. Elements pass through verbatim; nothing here may quote or join them.
        ws = str(self.repo)
        args = [
            settings.gaol_binary,
            "run-once",
            "--runtime",
            settings.runonce_runtime,
            "--image",
            settings.runonce_image,
            "--mount",
            f"{ws}:{ws}",
        ]
        for host, container in self._opencode_mounts():
            args += ["--mount", f"{host}:{container}"]
        if settings.runonce_memory:
            args += ["--memory", settings.runonce_memory]
        if settings.runonce_cpus:
            args += ["--cpus", str(settings.runonce_cpus)]
        args += [
            "--workdir",
            ws,
            "--env",
            f"PWD={ws}",
            "--env",
            f"HOME={settings.runonce_home}",
            "--timeout",
            str(max(1, timeout)),
            "--",
            *cmd,
        ]
        return subprocess.run(
            args,
            cwd=self.repo,
            capture_output=True,
            text=True,
            timeout=timeout + self._HOST_GRACE_S,
        )

    def run_tests(self) -> tuple[bool, str]:
        from agents.task_worker import tester  # local import: tester uses make_sandbox

        return tester.run_tests(self.repo, sandbox=self)


def make_sandbox(repo: Path, kind: str | None = None) -> Sandbox:
    """Build the configured sandbox for *repo*. ``kind`` overrides the env-keyed default —
    the concurrent dispatcher selects ``gaol-run-once`` for jj workspaces, which have no dx
    container. Unknown kinds raise — never guess a sandbox."""
    resolved = kind or settings.sandbox
    if resolved == "gaol-dx":
        return GaolDxSandbox(repo)
    if resolved == "gaol-run-once":
        return GaolRunOnceSandbox(repo)
    raise ValueError(
        f"unknown sandbox kind {resolved!r} (implemented: 'gaol-dx', 'gaol-run-once')"
    )
