"""Run the project's tests inside its sandbox (gaol dx by default).

Detection runs on the host (file existence checks), but the actual test
commands run inside the sandbox so they see the right toolchain.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from forge.task_worker.sandbox import Sandbox

_TEST_TIMEOUT = 300  # 5 min
_OUTPUT_TAIL = 1000


def _has_justfile_test_recipe(repo_path: Path) -> bool:
    """Return True if Justfile exists with a `test` recipe."""
    for name in ("Justfile", "justfile", ".justfile"):
        path = repo_path / name
        if not path.exists():
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for line in content.splitlines():
            stripped = line.lstrip()
            if stripped.startswith("test:") or stripped.startswith("test "):
                return True
    return False


def _pyproject_has_pytest(repo_path: Path) -> bool:
    """Rough check for pytest configuration or dep in pyproject.toml."""
    path = repo_path / "pyproject.toml"
    if not path.exists():
        return False
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return "[tool.pytest" in content or "pytest" in content


def _package_json_has_test_script(repo_path: Path) -> bool:
    path = repo_path / "package.json"
    if not path.exists():
        return False
    try:
        import json

        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return False
    scripts = data.get("scripts", {})
    return isinstance(scripts, dict) and "test" in scripts


def _tail(text: str, n: int = _OUTPUT_TAIL) -> str:
    if len(text) <= n:
        return text
    return text[-n:]


def _run_cmds(sandbox: Sandbox, cmds: list[list[str]]) -> tuple[bool, str]:
    """Run each command in sequence, stopping at the first failure.

    Returns (passed, output_tail). A non-zero exit, timeout, or missing
    toolchain all fail closed.
    """
    outputs: list[str] = []
    for cmd in cmds:
        try:
            result = sandbox.run(cmd, timeout=_TEST_TIMEOUT)
        except subprocess.TimeoutExpired as e:
            out = (e.stdout or "") if isinstance(e.stdout, str) else ""
            err = (e.stderr or "") if isinstance(e.stderr, str) else ""
            outputs.append(f"$ {' '.join(cmd)}\nTIMEOUT after {_TEST_TIMEOUT}s\n{out}\n{err}")
            return False, _tail("\n".join(outputs))
        except FileNotFoundError as e:
            return False, f"gaol binary not found: {e}"
        except Exception as e:  # noqa: BLE001
            return False, f"dx_run raised: {e}"

        combined = (result.stdout or "") + "\n" + (result.stderr or "")
        outputs.append(f"$ {' '.join(cmd)}\n{combined}")
        if result.returncode != 0:
            return False, _tail("\n".join(outputs))

    return True, _tail("\n".join(outputs))


def run_tests(repo_path: Path, sandbox: Sandbox | None = None) -> tuple[bool, str]:
    """Run the project's tests inside its sandbox. Returns (passed, output_tail).

    Detection order (first match wins):
      1. Justfile with test recipe -> `just test`
      2. go.mod -> `go build ./...` then `go test ./...`
      3. pyproject.toml with pytest -> `uv run pytest`
      4. package.json with scripts.test -> `pnpm test`
      5. Cargo.toml -> `cargo test`
      6. Nothing found -> (True, "no tests configured")

    The go.mod branch is deliberately ahead of the package.json branch: a Go
    repo commonly also carries a package.json (web assets) without a `test`
    script, which previously fell through to "no tests configured" so the Go
    code was never compiled or tested. `go test ./...` compiles production and
    test code, catching build errors as well as failures.
    """
    if _has_justfile_test_recipe(repo_path):
        cmds = [["just", "test"]]
    elif (repo_path / "go.mod").exists():
        cmds = [["go", "build", "./..."], ["go", "test", "./..."]]
    elif _pyproject_has_pytest(repo_path):
        cmds = [["uv", "run", "pytest"]]
    elif _package_json_has_test_script(repo_path):
        cmds = [["pnpm", "test"]]
    elif (repo_path / "Cargo.toml").exists():
        cmds = [["cargo", "test"]]
    else:
        return True, "no tests configured"

    if sandbox is None:
        from forge.task_worker.sandbox import make_sandbox

        sandbox = make_sandbox(repo_path)

    return _run_cmds(sandbox, cmds)


def run_build(repo_path: Path, sandbox: Sandbox | None = None) -> tuple[bool, str, bool]:
    """Compile-check the project, independent of whether the task requires tests.

    Returns (ok, output_tail, ran). ``ran`` is False for languages without a
    cheap standalone compile step (Python/JS), where correctness is deferred to
    the test gate. For compiled languages (Go, Rust) a failure here means the
    change does not build and must block the leaf — this is the gate that makes
    landing non-compiling code impossible regardless of requires_tests.
    """
    if (repo_path / "go.mod").exists():
        cmds = [["go", "build", "./..."]]
    elif (repo_path / "Cargo.toml").exists():
        cmds = [["cargo", "build"]]
    else:
        return True, "no compile gate for this project", False

    if sandbox is None:
        from forge.task_worker.sandbox import make_sandbox

        sandbox = make_sandbox(repo_path)

    ok, output = _run_cmds(sandbox, cmds)
    return ok, output, True
