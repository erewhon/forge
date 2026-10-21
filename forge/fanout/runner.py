"""Run analysis seats concurrently against one repo (read-only) and capture each one's findings."""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import signal
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from forge.fanout.config import settings
from forge.fanout.models import SeatRun, SeatSpec
from forge.fanout.profiles import Profile, load_profile

_TAIL_CHARS = 2000
_DRAIN_SECONDS = 10.0
SeatCallback = Callable[[SeatRun], None]
# pi's turn-stats extension prints "⏱ turn 0 · …" lines; telemetry, not findings.
_PI_STATS_LINE = re.compile(r"^\s*⏱.*$", re.MULTILINE)


def _tail(data: bytes, limit: int = _TAIL_CHARS) -> str:
    decoded = data.decode("utf-8", errors="replace")
    return decoded[-limit:] if len(decoded) > limit else decoded


def build_cmd(spec: SeatSpec, prompt: str, profile: Profile | None = None) -> list[str]:
    """Construct the seat CLI invocation; the profile supplies the read-only tool allowlists.

    claude: ``claude -p PROMPT --model M [--system-prompt S] --allowedTools … --disallowedTools …
            --output-format text``
    pi:     ``pi -p --no-session --no-skills --provider P --model M --tools … [--system-prompt S]
            -- PROMPT``

    Both run with cwd set to the target and may only read; findings are whatever they print.
    """
    profile = profile or load_profile(None)
    system = ["--system-prompt", profile.system_prompt] if profile.system_prompt is not None else []
    if spec.kind == "pi":
        return [
            settings.pi_binary,
            "-p",
            "--no-session",
            "--no-skills",
            "--provider",
            settings.pi_provider,
            "--model",
            spec.model,
            "--tools",
            ",".join(profile.pi_tools),
            *system,
            "--",
            prompt,
        ]
    cmd = [settings.claude_binary, "-p", prompt, "--model", spec.model, *system]
    if profile.claude_allowed_tools:
        cmd += ["--allowedTools", *profile.claude_allowed_tools]
    if profile.claude_disallowed_tools:
        cmd += ["--disallowedTools", *profile.claude_disallowed_tools]
    cmd += ["--output-format", "text"]
    return cmd


def findings_filename(spec: SeatSpec) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", spec.display).strip("-")
    return f"{spec.label}-{slug}.md"


def clean_output(kind: str, stdout: str) -> str:
    if kind == "pi":
        stdout = _PI_STATS_LINE.sub("", stdout)
    return stdout.strip() + "\n" if stdout.strip() else ""


async def _pump(stream: asyncio.StreamReader | None, buf: bytearray) -> None:
    """Copy a child's pipe into buf as it arrives, so output read before a kill is not lost."""
    if stream is None:
        return
    while chunk := await stream.read(65536):
        buf.extend(chunk)


def _killpg(proc: asyncio.subprocess.Process, sig: int) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(proc.pid, sig)


@dataclass
class ExecResult:
    returncode: int | None
    stdout: bytes
    stderr: bytes
    error: str | None = None
    cancelled: bool = False


def seat_env(spec: SeatSpec, cwd: Path) -> dict[str, str]:
    """Environment for a seat process: inherited, PWD pinned to the repo, pi's tool budget."""
    env = {**os.environ, "PWD": str(cwd)}
    if spec.kind == "pi":
        env["PI_GUARD_BUDGET"] = str(settings.pi_tool_budget)
    return env


async def _exec(
    cmd: list[str], *, cwd: Path, timeout: float, env: dict[str, str] | None = None
) -> ExecResult:
    """Run a seat CLI and capture what it printed.

    The child gets its own session so a timeout (or Ctrl-C) kills the whole process group — a
    seat's MCP servers and tool shells would otherwise outlive it holding the pipes open. Output
    is pumped into buffers continuously, so a timed-out seat keeps what it printed before the
    kill; the post-kill drain is bounded so a stubborn descendant cannot hang the sweep. On
    cancellation (Ctrl-C) the group is killed and the buffered output is still returned, flagged
    ``cancelled``, so the caller can save the partial findings before the cancellation resumes.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=cwd,
            env=env if env is not None else {**os.environ, "PWD": str(cwd)},
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
    except OSError as e:
        return ExecResult(None, b"", b"", f"could not start {cmd[0]}: {e}")

    out, err = bytearray(), bytearray()
    pumps = asyncio.gather(_pump(proc.stdout, out), _pump(proc.stderr, err))
    error: str | None = None
    cancelled = False
    try:
        await asyncio.wait_for(proc.wait(), timeout=timeout)
    except TimeoutError:
        error = f"timed out after {timeout:.0f}s"
    except asyncio.CancelledError:
        error, cancelled = "cancelled", True
    # Always sweep the group: a seat that exited cleanly can still leave an MCP server or tool
    # shell behind, and a timed-out / cancelled one certainly will.
    _killpg(proc, signal.SIGKILL)
    with contextlib.suppress(Exception):
        await asyncio.shield(asyncio.wait_for(proc.wait(), timeout=_DRAIN_SECONDS))
    with contextlib.suppress(Exception):
        await asyncio.shield(asyncio.wait_for(pumps, timeout=_DRAIN_SECONDS))
    pumps.cancel()
    returncode = None if error else proc.returncode
    return ExecResult(returncode, bytes(out), bytes(err), error, cancelled)


async def run_seat(
    *,
    spec: SeatSpec,
    prompt: str,
    repo: Path,
    findings_dir: Path,
    timeout: float | None = None,
    on_done: SeatCallback | None = None,
    profile: Profile | None = None,
) -> SeatRun:
    """Run one seat in the repo and write whatever it printed to findings_dir.

    ``on_done`` sees the SeatRun before it is returned — including a ``cancelled`` one, which is
    reported and then re-raised as CancelledError — so a caller can checkpoint incrementally.
    """
    effective_timeout = timeout if timeout is not None else settings.per_seat_timeout_seconds
    cmd = build_cmd(spec, prompt, profile)
    start = time.monotonic()
    res = await _exec(cmd, cwd=repo, timeout=effective_timeout, env=seat_env(spec, repo))
    run = _classify(spec, cmd, res, findings_dir, latency_ms=int((time.monotonic() - start) * 1000))
    if on_done is not None:
        on_done(run)
    if res.cancelled:
        raise asyncio.CancelledError
    return run


def _classify(
    spec: SeatSpec, cmd: list[str], res: ExecResult, findings_dir: Path, *, latency_ms: int
) -> SeatRun:
    """Turn an exec result into a SeatRun, saving any printed output as the findings file."""
    returncode, stdout, stderr, fail_msg = res.returncode, res.stdout, res.stderr, res.error
    text = clean_output(spec.kind, stdout.decode("utf-8", errors="replace"))
    findings_path: Path | None = None
    write_error: str | None = None
    if text:
        # Partial output from a timed-out or crashed seat is still evidence; keep it, labelled.
        findings_path = findings_dir / findings_filename(spec)
        try:
            findings_dir.mkdir(parents=True, exist_ok=True)
            findings_path.write_text(text, encoding="utf-8")
        except OSError as e:
            findings_path, write_error = None, f"could not write findings: {e}"

    base = dict(
        label=spec.label,
        kind=spec.kind,
        model=spec.model,
        display=spec.display,
        cmd=cmd,
        findings_path=findings_path,
        findings_chars=len(text),
        latency_ms=latency_ms,
        returncode=returncode,
        stdout_tail=_tail(stdout),
        stderr_tail=_tail(stderr),
    )
    if write_error is not None:
        return SeatRun(status="error", error_message=write_error, **base)
    if res.cancelled:
        return SeatRun(status="cancelled", error_message="sweep interrupted", **base)
    if fail_msg is not None:
        status = "timeout" if fail_msg.startswith("timed out") else "error"
        return SeatRun(status=status, error_message=fail_msg, **base)
    if returncode != 0:
        return SeatRun(
            status="error", error_message=f"{spec.kind} exited with code {returncode}", **base
        )
    if not text:
        return SeatRun(status="empty", error_message="seat printed nothing", **base)
    return SeatRun(status="ok", **base)


def _failed_run(spec: SeatSpec, exc: BaseException) -> SeatRun:
    return SeatRun(
        label=spec.label,
        kind=spec.kind,
        model=spec.model,
        display=spec.display,
        status="error",
        cmd=[],
        error_message=f"seat crashed: {type(exc).__name__}: {exc}",
    )


async def run_all(
    *,
    specs: list[SeatSpec],
    prompt: str,
    repo: Path,
    findings_dir: Path,
    timeout: float | None,
    on_done: SeatCallback | None = None,
    profile: Profile | None = None,
) -> list[SeatRun]:
    """Run every seat concurrently. One seat blowing up never loses the others' results."""
    findings_dir.mkdir(parents=True, exist_ok=True)
    results = await asyncio.gather(
        *(
            run_seat(
                spec=s,
                prompt=prompt,
                repo=repo,
                findings_dir=findings_dir,
                timeout=timeout,
                on_done=on_done,
                profile=profile,
            )
            for s in specs
        ),
        return_exceptions=True,
    )
    runs: list[SeatRun] = []
    for spec, res in zip(specs, results, strict=True):
        if isinstance(res, asyncio.CancelledError):
            raise res
        runs.append(_failed_run(spec, res) if isinstance(res, BaseException) else res)
    return runs
