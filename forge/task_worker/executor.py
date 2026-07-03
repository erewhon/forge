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
# The refusal protocol is a LINE starting with "BLOCKED:". Line-anchored (after ANSI strip) so
# model narration that merely mentions the marker mid-sentence doesn't read as a refusal.
_BLOCKED_LINE_RE = re.compile(r"(?im)^\s*blocked:")

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
) -> tuple[bool, str]:
    """Run the task inside the project's sandbox. Returns (success, stdout_tail).

    The spec is written to `<project>/.task_worker/spec-<id>-<uuid>.md`; the
    opencode prompt tells the model to read that file. File is deleted on the
    way out (success or failure).
    """
    if sandbox is None:
        from agents.task_worker.sandbox import make_sandbox

        sandbox = make_sandbox(project_dir)
    spec_path = _write_spec(project_dir, task, spec)
    rel_spec = spec_path.relative_to(project_dir).as_posix()

    prompt = (
        f"Read the task spec at `{rel_spec}` and execute the task it describes. "
        f"Follow the rules stated in the spec file. Do not modify the spec file "
        f"itself or anything under `.task_worker/`."
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
        return False, _tail(f"TIMEOUT after {timeout}s\n{out}\n{err}")
    except FileNotFoundError:
        _cleanup_spec(spec_path)
        return False, "gaol binary not found on PATH"
    except Exception as e:  # noqa: BLE001
        _cleanup_spec(spec_path)
        return False, f"dx_run raised: {e}"

    combined = (result.stdout or "") + "\n" + (result.stderr or "")
    tail = _tail(combined)

    _cleanup_spec(spec_path)

    # An explicit BLOCKED marker means the model chose not to proceed.
    if _BLOCKED_LINE_RE.search(_ANSI_RE.sub("", combined)):
        return False, tail

    return result.returncode == 0, tail
