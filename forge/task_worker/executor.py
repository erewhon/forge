"""Shell out to OpenCode (inside the project's dx container) to execute a task spec.

The worker writes the full task spec to a gitignored file inside the
bind-mounted repo so we don't have to pass multi-KB markdown through shell
quoting. OpenCode inside the dx container reads the file and executes.
"""

from __future__ import annotations

import re
import subprocess
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from agents.task_worker.models import TaskInfo

if TYPE_CHECKING:
    from agents.task_worker.sandbox import Sandbox

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
# The refusal protocol is "print a line starting with BLOCKED: and STOP" — so a genuine refusal
# sits at the END of the transcript. Only the tail window is scanned (line-anchored, ANSI
# stripped): earlier occurrences are quoted material (the transcript includes the model reading
# the spec file, whose rules text mentions the marker and can wrap to a line start).
_BLOCKED_LINE_RE = re.compile(r"(?im)^\s*blocked:")
_BLOCKED_TAIL_LINES = 15


def _model_refused(combined: str) -> bool:
    clean = _ANSI_RE.sub("", combined)
    tail_lines = [line for line in clean.splitlines() if line.strip()][-_BLOCKED_TAIL_LINES:]
    return _BLOCKED_LINE_RE.search("\n".join(tail_lines)) is not None


_STDOUT_TAIL = 500
_SPEC_DIR = ".task_worker"

_SPEC_HEADER = """# Task Spec for Autonomous Worker

You are an autonomous task worker executing inside a gaol dx container.
The repository is bind-mounted; files outside this directory are not
accessible. Execute the task described below.

## Rules

- Make the minimum changes needed to complete the task.
- Do NOT commit or push; the harness will commit after inspecting the diff.
- Do NOT modify files unrelated to this task.
- Do NOT touch the `.task_worker/` directory (that's me talking to myself).
- If you cannot proceed for any reason, print a line starting with
  `BLOCKED:` and stop. The harness will treat that as a failure and revert.

## Task

"""


def _tail(text: str, n: int = _STDOUT_TAIL) -> str:
    if len(text) <= n:
        return text
    return text[-n:]


def _write_spec(project_dir: Path, task: TaskInfo, spec: str) -> Path:
    """Write the task spec to <project>/.task_worker/spec-<id>.md.

    Ensures the spec directory exists and is gitignored/jj-ignored.
    Returns the relative path (from project_dir) for the opencode prompt.
    """
    spec_dir = project_dir / _SPEC_DIR
    spec_dir.mkdir(exist_ok=True)

    # Belt-and-braces: make sure this dir won't get committed.
    gitignore = spec_dir / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text("*\n", encoding="utf-8")

    # Unique filename so concurrent workers don't stomp (future-proofing).
    fname = f"spec-{task.id}-{uuid.uuid4().hex[:8]}.md"
    spec_path = spec_dir / fname
    spec_path.write_text(_SPEC_HEADER + spec, encoding="utf-8")
    return spec_path


def _cleanup_spec(spec_path: Path) -> None:
    """Remove the spec file. Swallows errors — we don't want cleanup to mask failures."""
    try:
        spec_path.unlink(missing_ok=True)
    except OSError:
        pass


def execute_task_with_opencode(
    task: TaskInfo,
    spec: str,
    project_dir: Path,
    model: str,
    timeout: int,
    sandbox: Sandbox | None = None,
) -> tuple[bool, str, bool]:
    """Run the task inside the project's sandbox. Returns (success, stdout_tail, blocked).

    ``blocked`` is True only for an explicit model refusal (a BLOCKED: line) — the caller
    must revert. A bare non-zero exit is advisory: plugins crashing at session end (observed:
    open-mem with no in-container API key) fail the process after the model finished, so the
    caller decides based on whether a diff was left behind.

    The spec is written to `<project>/.task_worker/spec-<id>-<uuid>.md`; the
    opencode prompt tells the model to read that file. File is deleted on the
    way out (success or failure).
    """
    if sandbox is None:
        from agents.task_worker.sandbox import make_sandbox

        sandbox = make_sandbox(project_dir)
    spec_path = _write_spec(project_dir, task, spec)
    rel_spec = spec_path.relative_to(project_dir).as_posix()

    # No backticks/quotes/metacharacters in this prompt: gaol dx run re-joins argv
    # into an unquoted shell string (gaol-cli dx.rs), so backticks would be executed
    # as command substitution and the spec path silently deleted from the prompt
    # (observed: workers glob-hunting for the spec and failing leaves).
    prompt = (
        f"Read the task spec at {rel_spec} and execute the task it describes. "
        f"Follow the rules stated in the spec file. Do not modify the spec file "
        f"itself or anything under .task_worker/ directory."
    )

    # Inside the container: `opencode run -m llm/<tier> --dangerously-skip-permissions <prompt>`
    # No --dir needed because gaol dx already drops us into the bind-mounted repo.
    cmd = [
        "opencode",
        "run",
        "-m",
        f"llm/{model}",
        "--dangerously-skip-permissions",
        prompt,
    ]

    try:
        result = sandbox.run(cmd, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        out = e.stdout if isinstance(e.stdout, str) else ""
        err = e.stderr if isinstance(e.stderr, str) else ""
        _cleanup_spec(spec_path)
        return False, _tail(f"TIMEOUT after {timeout}s\n{out}\n{err}"), False
    except FileNotFoundError:
        _cleanup_spec(spec_path)
        return False, "gaol binary not found on PATH", False
    except Exception as e:  # noqa: BLE001
        _cleanup_spec(spec_path)
        return False, f"dx_run raised: {e}", False

    combined = (result.stdout or "") + "\n" + (result.stderr or "")
    tail = _tail(combined)

    _cleanup_spec(spec_path)

    # An explicit BLOCKED marker at the end of the transcript means the model refused.
    if _model_refused(combined):
        return False, tail, True

    return result.returncode == 0, tail, False
