"""Lint the leaf's changed files inside the sandbox — the quality gate beside tests.

Dogfood findings (pipeline:build waves): worker leaves passed the test gate but rode
in with ruff violations a human had to clean up at the wave gate. Two scoping lessons
shape this gate:

- **Changed files only.** A repo-wide check fails every leaf on pre-existing debt the
  leaf never touched (meta's own tree has known violations outside the worker's
  blast radius). The gate judges the leaf's work, not the repo's history.
- **Autofix, then judge.** A plain revert-on-lint-red gate would have thrown away
  otherwise-green leaves over line-length nits — both dogfooded leaves landed correct
  code with E501s. ``ruff check --fix`` + ``ruff format`` on the changed files runs
  first; only violations that SURVIVE autofix fail the gate (revert-on-fail upstream).
  The gate runs before tests so a single test run validates the final, fixed state.

Scope: Python/ruff only, and only when the repo shows ruff intent (``ruff`` appears in
pyproject.toml — config section or dependency). Repo-scoped linters (``pnpm lint``,
``cargo clippy``) can't be cheaply confined to changed files and would hit the same
pre-existing-debt wall, so non-Python changes pass vacuously for now.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from forge.task_worker.sandbox import Sandbox

_LINT_TIMEOUT = 120
_OUTPUT_TAIL = 1000


def _tail(text: str, n: int = _OUTPUT_TAIL) -> str:
    if len(text) <= n:
        return text
    cut = text[-n:]
    nl = cut.find("\n")
    if 0 <= nl < len(cut) - 1:
        return cut[nl + 1 :]
    return cut


def _ruff_cmd(repo_path: Path) -> list[str] | None:
    """The ruff invocation for this repo, or None when the repo shows no ruff intent.

    ``uv run ruff`` when ruff is a project dependency (respects the repo's pinned
    version); ``uvx ruff`` when only a ``[tool.ruff]`` config section exists (e.g.
    fixture repos that configure ruff without depending on it).
    """
    path = repo_path / "pyproject.toml"
    if not path.exists():
        return None
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    if "ruff" not in content:
        return None
    config_only = "[tool.ruff" in content and "ruff" not in content.replace("[tool.ruff", "")
    if config_only:
        return ["uvx", "ruff"]
    return ["uv", "run", "ruff"]


def _run(sandbox: Sandbox, cmd: list[str]) -> tuple[int, str]:
    try:
        result = sandbox.run(cmd, timeout=_LINT_TIMEOUT)
    except subprocess.TimeoutExpired:
        return 1, f"TIMEOUT after {_LINT_TIMEOUT}s: {' '.join(cmd)}"
    except FileNotFoundError as e:
        return 1, f"gaol binary not found: {e}"
    except Exception as e:  # noqa: BLE001
        return 1, f"sandbox run raised: {e}"
    return result.returncode, (result.stdout or "") + "\n" + (result.stderr or "")


def run_lint(
    repo_path: Path,
    changed_files: list[str],
    sandbox: Sandbox | None = None,
) -> tuple[bool, str, bool]:
    """Lint the leaf's changed Python files. Returns (passed, output_tail, fixed).

    ``fixed`` is True when the autofix pass ran (the working copy may differ from
    what the model wrote — callers should test AFTER this gate). Vacuous passes:
    no Python files changed, or the repo shows no ruff intent.
    """
    py_files = [f for f in changed_files if f.endswith(".py")]
    if not py_files:
        return True, "no lintable files changed", False
    base = _ruff_cmd(repo_path)
    if base is None:
        return True, "no linter configured (no ruff in pyproject)", False

    if sandbox is None:
        from forge.task_worker.sandbox import make_sandbox

        sandbox = make_sandbox(repo_path)

    def check() -> tuple[int, str]:
        rc_check, out_check = _run(sandbox, [*base, "check", *py_files])
        rc_fmt, out_fmt = _run(sandbox, [*base, "format", "--check", *py_files])
        return rc_check or rc_fmt, f"{out_check}\n{out_fmt}"

    rc, out = check()
    if rc == 0:
        return True, "lint clean", False

    # Autofix on the changed files only, then re-judge: only violations that
    # survive the fix fail the leaf.
    _run(sandbox, [*base, "check", "--fix", *py_files])
    _run(sandbox, [*base, "format", *py_files])
    rc, out = check()
    if rc == 0:
        return True, "lint clean after autofix", True
    return False, _tail(out), True
