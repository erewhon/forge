"""Run one model edit via OpenCode — directly, no container.

The wave-loop worker runs OpenCode inside a ``gaol dx`` container; grind runs on the work machine
where there is no gaol (see ``task_worker/sandbox.py`` — "the work environment (no gaol) will need
its own implementation"). So grind shells out to OpenCode **directly** in the repo. The spec is
written to a jj-ignored file the model reads (avoids multi-KB shell quoting), and the BLOCKED
refusal protocol is shared with the task worker so a refusal is detected identically.
"""

from __future__ import annotations

import subprocess
import uuid
from pathlib import Path

# Reuse the task worker's refusal detector so BLOCKED means the same thing everywhere.
from forge.task_worker.executor import _model_refused, _tail

_SPEC_DIR = ".forge/grind"


def _write_spec(repo: Path, spec_text: str) -> Path:
    spec_dir = repo / _SPEC_DIR
    spec_dir.mkdir(parents=True, exist_ok=True)
    gitignore = spec_dir / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text("*\n", encoding="utf-8")
    spec_path = spec_dir / f"spec-{uuid.uuid4().hex[:8]}.md"
    spec_path.write_text(spec_text, encoding="utf-8")
    return spec_path


def run_opencode_edit(
    repo: Path, spec_text: str, model: str, timeout: int
) -> tuple[bool, str, bool]:
    """Run one OpenCode edit turn. Returns ``(ok, output_tail, blocked)``.

    ``blocked`` is True only for an explicit ``BLOCKED:`` refusal (the caller should stop). A bare
    non-zero exit is advisory — plugins can crash at session end after the model finished — so the
    caller decides based on whether the edit actually changed files.
    """
    spec_path = _write_spec(repo, spec_text)
    rel_spec = spec_path.relative_to(repo).as_posix()

    # No backticks/quotes/metacharacters in the prompt (the task-worker lesson): some runners
    # re-join argv into an unquoted shell string, so a backtick would execute as substitution.
    prompt = (
        f"Read the instructions at {rel_spec} and follow them exactly. "
        f"Do not modify that file or anything under {_SPEC_DIR} and do not commit."
    )
    cmd = ["opencode", "run", "-m", model, "--dangerously-skip-permissions", prompt]

    try:
        result = subprocess.run(cmd, cwd=repo, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        out = e.stdout if isinstance(e.stdout, str) else ""
        err = e.stderr if isinstance(e.stderr, str) else ""
        _cleanup(spec_path)
        return False, _tail(f"TIMEOUT after {timeout}s\n{out}\n{err}"), False
    except FileNotFoundError:
        _cleanup(spec_path)
        return False, "opencode not found on PATH", False
    except Exception as e:  # noqa: BLE001 — a broken run is one failed turn, never a crash
        _cleanup(spec_path)
        return False, f"opencode run raised: {e}", False

    combined = (result.stdout or "") + "\n" + (result.stderr or "")
    tail = _tail(combined)
    _cleanup(spec_path)
    if _model_refused(combined):
        return False, tail, True
    return result.returncode == 0, tail, False


def _cleanup(spec_path: Path) -> None:
    try:
        spec_path.unlink(missing_ok=True)
    except OSError:
        pass
