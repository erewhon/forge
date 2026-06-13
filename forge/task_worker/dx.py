"""Helpers for running commands inside a `gaol dx` container.

The worker requires every target project to have a running dx container.
Execution and tests both run inside the container so the LLM can't touch
anything outside the bind-mounted repo. VCS operations stay on the host.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


class DXNotReadyError(RuntimeError):
    """Raised when the project's dx container isn't set up or running."""


def check_dx_ready(project_dir: Path) -> tuple[bool, str]:
    """Return (ready, status_line) for the project's dx container.

    Ready means `gaol dx info` reports Status: running. Anything else
    (not created, stopped, unknown) is not ready.
    """
    try:
        result = subprocess.run(
            ["gaol", "dx", "info"],
            capture_output=True,
            text=True,
            timeout=15,
            cwd=project_dir,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        return False, f"gaol dx info failed: {e}"

    if result.returncode != 0:
        return False, f"gaol dx info exit {result.returncode}: {result.stderr.strip()}"

    # Parse "Status: running" / "Status: stopped" / "Status: not created" etc.
    # Strip ANSI color codes — gaol dx uses them even in non-TTY output.
    import re

    ansi_re = re.compile(r"\x1b\[[0-9;]*m")
    status = ""
    for line in result.stdout.splitlines():
        clean = ansi_re.sub("", line).strip()
        if clean.lower().startswith("status:"):
            status = clean.split(":", 1)[1].strip().lower()
            break

    if not status:
        return False, "Could not determine dx container status"

    return status == "running", f"dx status: {status}"


def dx_run(
    project_dir: Path,
    cmd: list[str],
    timeout: int = 60,
) -> subprocess.CompletedProcess[str]:
    """Run a command inside the project's dx container.

    Example: dx_run(repo, ["uv", "run", "pytest"])
    """
    full_cmd = ["gaol", "dx", "run", "--"] + cmd
    return subprocess.run(
        full_cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=project_dir,
    )
