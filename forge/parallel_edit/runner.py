"""Run candidate edits in parallel jj workspaces, one per candidate (claude or opencode)."""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

from agents.parallel_edit.config import settings
from agents.parallel_edit.models import CandidateSpec, DiffStat, EditRun
from agents.parallel_edit.workspaces import (
    JJError,
    collect_diff,
    create_workspace,
    ensure_git_marker,
    forget_workspace,
    workspace_destination,
)

_TAIL_CHARS = 2000


def _tail(data: bytes, limit: int = _TAIL_CHARS) -> str:
    decoded = data.decode("utf-8", errors="replace")
    return decoded[-limit:] if len(decoded) > limit else decoded


def _partial_diff(workspace: Path, base_rev: str) -> tuple[str, DiffStat]:
    """Best-effort diff capture for non-success exits (timeout / nonzero return).

    A thorough-but-slow candidate that gets killed at the timeout, or one that exits nonzero
    after writing real edits, has still produced work on disk. Capture it so the judge can
    weigh it instead of discarding it. Returns ("", DiffStat()) if the diff can't be read.
    """
    try:
        return collect_diff(workspace, base_rev)
    except JJError:
        return "", DiffStat()


def _build_cmd(spec: CandidateSpec, prompt: str) -> list[str]:
    """Construct the agent CLI invocation for a candidate.

    claude:   ``claude -p PROMPT --model M --permission-mode … --output-format text``
    opencode: ``opencode run -m M --dangerously-skip-permissions PROMPT`` (M is an opencode
              model ref such as ``llm/glm-5.1``, routed through the local LLM router).

    Both run with cwd set to the candidate's isolated jj workspace and edit files in place; the
    diff is collected from the workspace afterward, so neither tool's stdout format matters.
    """
    if spec.kind == "opencode":
        return [
            settings.opencode_binary,
            "run",
            "-m",
            spec.model,
            "--dangerously-skip-permissions",
            prompt,
        ]
    return [
        settings.claude_binary,
        "-p",
        prompt,
        "--model",
        spec.model,
        "--permission-mode",
        settings.permission_mode,
        "--output-format",
        settings.output_format,
    ]


async def _exec(
    cmd: list[str], *, workspace: Path, timeout: float
) -> tuple[int | None, bytes, bytes, str | None]:
    """Run a candidate CLI in workspace. Returns (returncode, stdout, stderr, error_message)."""
    # Override PWD too: changing cwd alone leaves the inherited PWD pointing at the parent's
    # directory, and opencode derives its project root from PWD — so it would otherwise edit
    # files in the launching process's directory instead of the candidate workspace.
    env = {**os.environ, "PWD": str(workspace)}
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=workspace,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError as e:
        return None, b"", b"", f"binary not found ({cmd[0]}): {e}"

    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        return proc.returncode, stdout, stderr, None
    except TimeoutError:
        proc.kill()
        try:
            stdout, stderr = await proc.communicate()
        except Exception:
            stdout, stderr = b"", b""
        return None, stdout, stderr, f"Timed out after {timeout:.1f}s"


async def run_candidate(
    *,
    prompt: str,
    spec: CandidateSpec,
    repo: Path,
    base_rev: str,
    timeout: float | None = None,
) -> EditRun:
    """Provision a workspace, run the candidate agent in it, return an EditRun with its diff."""
    workspace = workspace_destination(repo, spec.label)
    try:
        create_workspace(repo, workspace, base_rev=base_rev)
    except JJError as e:
        return EditRun(
            label=spec.label,
            model=spec.display,
            workspace_path=workspace,
            status="error",
            error_message=f"workspace setup failed: {e}",
        )

    # opencode finds the project via a .git marker; a jj workspace has only .jj.
    if spec.kind == "opencode":
        ensure_git_marker(workspace)

    start = time.monotonic()
    effective_timeout = timeout if timeout is not None else settings.per_run_timeout_seconds
    returncode, stdout, stderr, timeout_msg = await _exec(
        _build_cmd(spec, prompt), workspace=workspace, timeout=effective_timeout
    )
    latency_ms = int((time.monotonic() - start) * 1000)

    stdout_tail = _tail(stdout)
    stderr_tail = _tail(stderr)

    if timeout_msg is not None:
        diff_text, diff_stat = _partial_diff(workspace, base_rev)
        return EditRun(
            label=spec.label,
            model=spec.display,
            workspace_path=workspace,
            status="timeout",
            diff_text=diff_text,
            diff_stat=diff_stat,
            latency_ms=latency_ms,
            stdout_tail=stdout_tail,
            stderr_tail=stderr_tail,
            error_message=timeout_msg,
        )

    if returncode != 0:
        diff_text, diff_stat = _partial_diff(workspace, base_rev)
        return EditRun(
            label=spec.label,
            model=spec.display,
            workspace_path=workspace,
            status="error",
            diff_text=diff_text,
            diff_stat=diff_stat,
            latency_ms=latency_ms,
            returncode=returncode,
            stdout_tail=stdout_tail,
            stderr_tail=stderr_tail,
            error_message=f"{spec.kind} exited with code {returncode}",
        )

    try:
        diff_text, diff_stat = collect_diff(workspace, base_rev)
    except JJError as e:
        return EditRun(
            label=spec.label,
            model=spec.display,
            workspace_path=workspace,
            status="error",
            latency_ms=latency_ms,
            returncode=returncode,
            stdout_tail=stdout_tail,
            stderr_tail=stderr_tail,
            error_message=f"diff collection failed: {e}",
        )

    status = "no_changes" if not diff_text.strip() else "ok"
    return EditRun(
        label=spec.label,
        model=spec.display,
        workspace_path=workspace,
        status=status,
        diff_text=diff_text,
        diff_stat=diff_stat,
        latency_ms=latency_ms,
        returncode=returncode,
        stdout_tail=stdout_tail,
        stderr_tail=stderr_tail,
    )


async def run_all(
    *,
    prompt: str,
    candidates: list[CandidateSpec],
    repo: Path,
    base_rev: str,
) -> list[EditRun]:
    """Run every candidate concurrently. Workspaces persist for caller cleanup."""
    coros = [
        run_candidate(prompt=prompt, spec=spec, repo=repo, base_rev=base_rev) for spec in candidates
    ]
    return await asyncio.gather(*coros)


def cleanup_runs_selective(repo: Path, runs: list[EditRun]) -> list[Path]:
    """Cleanup based on settings; return paths kept on disk for the human to inspect."""
    kept: list[Path] = []
    for run in runs:
        succeeded = run.status in ("ok", "no_changes")
        should_clean = settings.cleanup_on_success if succeeded else settings.cleanup_on_failure
        if should_clean and run.workspace_path.exists():
            try:
                forget_workspace(repo, run.workspace_path)
            except Exception:
                kept.append(run.workspace_path)
        elif run.workspace_path.exists():
            kept.append(run.workspace_path)
    return kept
