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
        return False, (
            f"gaol dx info failed — the gaol daemon is unreachable or not installed; "
            f"ensure gaol is running and on PATH. Detail: {e}"
        )

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


def _parse_df_avail_kb(df_output: str) -> int | None:
    """Available 1K-blocks from ``df -Pk`` output — the last data row (POSIX format keeps
    each filesystem on one line). Skips the header and the sandbox's ``[dx] Running:``
    banner by requiring numeric size/available columns."""
    for line in reversed([ln for ln in df_output.splitlines() if ln.strip()]):
        parts = line.split()
        if len(parts) >= 4 and parts[1].isdigit() and parts[3].isdigit():
            return int(parts[3])
    return None


def check_disk_free(project_dir: Path, min_free_mb: int) -> tuple[bool, str]:
    """Return (ok, status_line) for free space on the dx container's filesystem.

    A near-full container silently kills builds: ``go build`` writes thousands of object
    files into GOCACHE, and on a near-full COW btrfs that thrashes until the gate's
    ``timeout`` SIGKILLs it — a non-zero exit with no diagnostics, misread as a compile
    failure (the Observinator escape, 2026-07-13, when the shared nspawn pool hit 100%).
    Refusing here turns a doomed, misattributed run into a clean skip that names the cause.

    Fail-open: any probe error (df missing, timeout, unparseable) returns ok=True so a
    flaky probe never blocks all work — the gate's own diagnostics stay the backstop.
    """
    try:
        result = subprocess.run(
            ["gaol", "dx", "run", "--", "df", "-Pk", "/"],
            capture_output=True,
            text=True,
            timeout=20,
            cwd=project_dir,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        return True, f"disk check skipped ({e})"
    if result.returncode != 0:
        return True, f"disk check skipped (df exit {result.returncode})"
    avail_kb = _parse_df_avail_kb(result.stdout)
    if avail_kb is None:
        return True, "disk check skipped (unparseable df)"
    free_mb = avail_kb // 1024
    if free_mb < min_free_mb:
        return False, f"container disk low: {free_mb} MiB free (< {min_free_mb} MiB floor)"
    return True, f"disk ok: {free_mb} MiB free"
