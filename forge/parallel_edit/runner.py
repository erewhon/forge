"""Run candidate edits in parallel jj workspaces, one per candidate (claude or opencode)."""

from __future__ import annotations

import asyncio
import math
import os
import shutil
import tempfile
import time
from pathlib import Path

from forge.parallel_edit.config import settings
from forge.parallel_edit.models import CandidateSpec, DiffStat, EditRun
from forge.parallel_edit.workspaces import (
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
            # --pure disables external opencode plugins (open-mem, TTS hooks). Without it the
            # open-mem plugin injects an auto-context block into AGENTS.md, a note into .gitignore,
            # and a .open-mem/ cache on every run — cruft that pollutes the candidate diff and gets
            # mis-scored as the model's scope sprawl. The `llm/` provider is from auth.json (not a
            # plugin), so it still resolves under --pure.
            "--pure",
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


def _opencode_mounts() -> tuple[list[tuple[str, str]], Path | None]:
    """Mounts that let ``opencode run -m llm/…`` reach the router inside the sandbox.

    - Host ``~/.config/opencode`` is mounted shared (read-mostly: provider/router config).
    - A fresh PER-CANDIDATE ``~/.local/share/opencode`` is created seeded with only the host's
      ``auth.json``, then mounted — so each candidate gets its own ``opencode.db`` and concurrent
      candidates don't corrupt the host's shared WAL sqlite. Returns ``(mounts, state_dir)``; the
      caller removes ``state_dir`` after the run (None if opencode isn't set up on the host).
    """
    mounts: list[tuple[str, str]] = []
    home = settings.sandbox_home
    host_cfg = Path.home() / ".config" / "opencode"
    if host_cfg.is_dir():
        mounts.append((str(host_cfg), f"{home}/.config/opencode"))
    state: Path | None = None
    host_auth = Path.home() / ".local" / "share" / "opencode" / "auth.json"
    if host_auth.is_file():
        state = Path(tempfile.mkdtemp(prefix="pe-oc-"))
        shutil.copy2(host_auth, state / "auth.json")
        mounts.append((str(state), f"{home}/.local/share/opencode"))
    return mounts, state


def _opencode_host_state() -> tuple[dict[str, str], Path | None]:
    """Per-candidate opencode data dir for LOOSE (non-sandboxed) host runs.

    Concurrent ``opencode run`` processes share ``~/.local/share/opencode/opencode.db`` and collide
    ("database is locked") under fan-out. Point each candidate at its own ``XDG_DATA_HOME`` seeded
    with only the host's ``auth.json`` (which the ``llm/`` provider needs), so every opencode
    process gets a private ``opencode.db``. Config (``~/.config/opencode``: provider/router setup)
    is left on the host path, so ``XDG_CONFIG_HOME`` is deliberately not overridden. Returns
    ``(env_overrides, state_dir)``; the caller removes ``state_dir`` after the run. Returns
    ``({}, None)`` when opencode auth isn't set up on the host (then env is left untouched).
    """
    host_auth = Path.home() / ".local" / "share" / "opencode" / "auth.json"
    if not host_auth.is_file():
        return {}, None
    state = Path(tempfile.mkdtemp(prefix="pe-oc-host-"))
    data_dir = state / "opencode"
    data_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(host_auth, data_dir / "auth.json")
    return {"XDG_DATA_HOME": str(state)}, state


def _wrap_sandbox(
    cmd: list[str],
    workspace: Path,
    timeout: float,
    extra_mounts: list[tuple[str, str]] | None = None,
) -> list[str]:
    """Wrap a candidate CLI argv to run inside an ephemeral ``gaol run-once`` sandbox.

    The jj workspace is bind-mounted writable at the *same* path inside the sandbox so cwd/PWD
    semantics (opencode derives its project root from PWD) are identical to the host path. The
    command runs as the host workspace owner, so edits persist to the workspace owned correctly.
    ``extra_mounts`` (e.g. the opencode config/auth) are added after the workspace; per-sandbox
    resource caps (``sandbox_memory``/``sandbox_cpus``) keep concurrent fan-out from exhausting the
    host. ``HOME`` is set so opencode finds its mounted config.

    run-once's own ``--timeout`` does the deterministic in-sandbox kill + teardown; a SIGKILL from
    the asyncio backstop would bypass that guard, so the backstop is set ``sandbox_grace_seconds``
    beyond this value by the caller. No ``--json``: run-once streams the inner command's
    stdout/stderr through and propagates its exit code, which is exactly what ``_exec`` consumes.
    """
    ws = str(workspace)
    args = [
        settings.gaol_binary,
        "run-once",
        "--runtime",
        settings.sandbox_runtime,
        "--image",
        settings.sandbox_image,
        "--mount",
        f"{ws}:{ws}",
    ]
    for host, container in extra_mounts or []:
        args += ["--mount", f"{host}:{container}"]
    if settings.sandbox_memory:
        args += ["--memory", settings.sandbox_memory]
    if settings.sandbox_cpus:
        args += ["--cpus", str(settings.sandbox_cpus)]
    args += [
        "--workdir",
        ws,
        "--env",
        f"PWD={ws}",
        "--env",
        f"HOME={settings.sandbox_home}",
        "--timeout",
        str(max(1, math.ceil(timeout))),
        "--",
        *cmd,
    ]
    return args


def _should_sandbox(spec: CandidateSpec) -> bool:
    """Per-kind sandboxing: only untrusted candidate kinds run inside a gaol sandbox.

    ``claude`` is trusted and authenticates via the host's OAuth (which isn't mounted into the
    sandbox), so it runs loose on the host even when ``sandbox`` is enabled; the open-fleet kinds
    (opencode → router models) are the ones isolated. Controlled by ``sandbox_exempt_kinds``.
    """
    return settings.sandbox and spec.kind not in settings.sandbox_exempt_kinds


async def _exec(
    cmd: list[str],
    *,
    workspace: Path,
    timeout: float,
    env_extra: dict[str, str] | None = None,
) -> tuple[int | None, bytes, bytes, str | None]:
    """Run a candidate CLI in workspace. Returns (returncode, stdout, stderr, error_message)."""
    # Override PWD too: changing cwd alone leaves the inherited PWD pointing at the parent's
    # directory, and opencode derives its project root from PWD — so it would otherwise edit
    # files in the launching process's directory instead of the candidate workspace.
    env = {**os.environ, "PWD": str(workspace)}
    if env_extra:
        env.update(env_extra)
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
    candidate_cmd = _build_cmd(spec, prompt)
    oc_state: Path | None = None
    host_env: dict[str, str] = {}
    sandboxed = _should_sandbox(spec)
    if sandboxed:
        extra_mounts: list[tuple[str, str]] = []
        if settings.sandbox_mount_opencode and spec.kind == "opencode":
            extra_mounts, oc_state = _opencode_mounts()
        exec_cmd = _wrap_sandbox(candidate_cmd, workspace, effective_timeout, extra_mounts)
        exec_timeout = effective_timeout + settings.sandbox_grace_seconds
    else:
        exec_cmd = candidate_cmd
        exec_timeout = effective_timeout
        # Loose host fan-out: give each opencode candidate a private data dir so concurrent
        # processes don't collide on the shared opencode.db ("database is locked").
        if spec.kind == "opencode":
            host_env, oc_state = _opencode_host_state()
    returncode, stdout, stderr, timeout_msg = await _exec(
        exec_cmd, workspace=workspace, timeout=exec_timeout, env_extra=host_env
    )
    # Drop the per-candidate opencode state (sandbox mount seed, or the loose-host private data
    # dir) now that the process has exited — neither outlives the run.
    if oc_state is not None:
        shutil.rmtree(oc_state, ignore_errors=True)
    # run-once kills an over-budget candidate internally and exits 124; surface that as a timeout
    # (with partial diff) rather than a generic nonzero error, matching the host-path semantics.
    if sandboxed and timeout_msg is None and returncode == 124:
        returncode = None
        timeout_msg = f"sandbox timed out after {effective_timeout:.0f}s"
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
