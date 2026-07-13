"""Run the project's tests inside its sandbox (gaol dx by default).

Detection runs on the host (file existence checks), but the actual test
commands run inside the sandbox so they see the right toolchain.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from forge.task_worker.sandbox import Sandbox

_TEST_TIMEOUT = 300  # 5 min
_OUTPUT_TAIL = 1000

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

# Appended to a failure note when a command was killed with no diagnostics — so an
# infra kill (timeout/OOM/full disk) is never misread as the leaf's code failing to
# build (the Observinator escape, 2026-07-13: a full nspawn pool made `go build` overrun
# the 300s gate and the note showed only the sandbox banner).
_KILLED_HINT = (
    "\n\n(No diagnostics captured — the command was KILLED before it could report, NOT a "
    "compile/test failure. Suspect the sandbox: build timeout, OOM, or a full container "
    "disk. Investigate container health, not the leaf's code.)"
)


def _describe_exit(code: int) -> str:
    """Label a non-zero exit, naming the timeout/kill signatures that otherwise reach the
    failure note as an empty body. A ``timeout``-wrapped command that overruns exits 124
    (SIGTERM) or 137 (128+SIGKILL) with NO diagnostics — indistinguishable from a compile
    failure unless the code is named."""
    if code == 124:
        return "exit 124 — KILLED (command timed out)"
    if code == 137:
        return "exit 137 — KILLED (SIGKILL: timeout --kill-after, or OOM)"
    if code == 143:
        return "exit 143 — KILLED (SIGTERM)"
    if code > 128:
        return f"exit {code} — killed by signal {code - 128}"
    return f"exit {code}"


def _is_kill(code: int) -> bool:
    # coreutils `timeout`: 124 timed out, 137 = 128+SIGKILL (--kill-after), 143 = 128+SIGTERM.
    return code in (124, 137, 143) or code > 128


def _no_meaningful_output(combined: str) -> bool:
    """True when a command emitted nothing but the sandbox's own ``[dx] Running:`` banner —
    the fingerprint of a process killed before it could report (no compiler/test output)."""
    for line in _ANSI_RE.sub("", combined).splitlines():
        s = line.strip()
        if s and not s.startswith("[dx]"):
            return False
    return True


# A failing test's location detail: `foo_test.go:42:`, `foo.rs:42:`, `test_x.py::test_y`.
_FAIL_LOC_RE = re.compile(r"_test\.(?:go|py|rs):\d+|::[\w\[\]-]+\s")


def _failure_digest(text: str, max_lines: int = 25, max_chars: int = 800) -> str:
    """High-signal 'what failed' lines gathered from ANYWHERE in the output, so the failing
    test name survives even when a plain tail would drop it. ``go test ./...`` prints a
    failing package's ``--- FAIL: Name`` early, then streams trailing ``ok`` lines from
    slower passing packages — the tail keeps the ``ok``s, not the FAIL (the Observinator
    test-gate escape, 2026-07-13). Empty when no failure markers match (e.g. a compile
    error, whose diagnostics the tail already carries)."""
    picked: list[str] = []
    for raw in _ANSI_RE.sub("", text).splitlines():
        line = raw.rstrip()
        s = line.strip()
        if not s or s.startswith("[dx]"):
            continue
        if (
            s.startswith(("--- FAIL", "=== FAIL", "--- ERROR", "panic:", "thread '", "E "))
            or "FAILED" in s
            or "panicked" in s
            or _FAIL_LOC_RE.search(s)
        ):
            picked.append(line)
            if len(picked) >= max_lines:
                break
    return "\n".join(picked)[:max_chars]


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

    Returns (passed, output_tail). A non-zero exit, timeout, or missing toolchain all
    fail closed. Every failure records the interpreted exit code, and a kill with no
    output appends an explicit hint — so an infra kill is never mistaken for a code
    failure in the task note (the Observinator escape).
    """
    outputs: list[str] = []
    for cmd in cmds:
        try:
            result = sandbox.run(cmd, timeout=_TEST_TIMEOUT)
        except subprocess.TimeoutExpired as e:
            out = (e.stdout or "") if isinstance(e.stdout, str) else ""
            err = (e.stderr or "") if isinstance(e.stderr, str) else ""
            outputs.append(f"$ {' '.join(cmd)}\nTIMEOUT after {_TEST_TIMEOUT}s\n{out}\n{err}")
            return False, _tail("\n".join(outputs)) + _KILLED_HINT
        except FileNotFoundError as e:
            return False, f"gaol binary not found: {e}"
        except Exception as e:  # noqa: BLE001
            return False, f"dx_run raised: {e}"

        combined = (result.stdout or "") + "\n" + (result.stderr or "")
        outputs.append(f"$ {' '.join(cmd)}\n{combined}")
        if result.returncode != 0:
            # Surface the failing test/assertion lines (which `go test ./...` prints mid-
            # stream) at the END, so main's tail keeps them instead of trailing `ok` lines.
            digest = _failure_digest(combined)
            if digest:
                outputs.append(f"--- failing ---\n{digest}")
            outputs.append(f"[{_describe_exit(result.returncode)}]")
            note = _tail("\n".join(outputs))
            if _is_kill(result.returncode) or _no_meaningful_output(combined):
                note += _KILLED_HINT
            return False, note

    return True, _tail("\n".join(outputs))


def _changed_existing(repo_path: Path, changed_files: list[str] | None, suffix: str) -> list[str]:
    """Changed files with ``suffix`` that still exist (deletions carry nothing to check)."""
    if not changed_files:
        return []
    return sorted(f for f in changed_files if f.endswith(suffix) and (repo_path / f).is_file())


# The compileall exclude keeps vendored/venv trees out of the Python syntax floor.
_COMPILEALL_EXCLUDE = r"(\.venv|node_modules|\.git|vendor)"

# What run_tests probes for, in order — named in the fall-through reason string so a
# pass without a runner is a disclosed decision, never a silent one.
_TEST_PROBES = (
    "justfile test recipe",
    "go.mod",
    "pyproject.toml [pytest]",
    "package.json scripts.test",
    "Cargo.toml",
    "pyproject.toml (syntax floor)",
)


def run_tests(
    repo_path: Path,
    sandbox: Sandbox | None = None,
    changed_files: list[str] | None = None,
) -> tuple[bool, str]:
    """Run the project's tests inside its sandbox. Returns (passed, output_tail).

    Detection order (first match wins):
      1. Justfile with test recipe -> `just test`
      2. go.mod -> `go build ./...` then `go test ./...`
      3. pyproject.toml with pytest -> `uv run pytest`
      4. package.json with scripts.test -> `pnpm test`
      5. Cargo.toml -> `cargo test`
      6. pyproject.toml WITHOUT pytest -> Python syntax floor (py_compile the
         changed .py files; whole-tree compileall when no change list is given)
      7. Nothing found -> (True, reason) where the reason names every probe —
         a pass without a runner must be a disclosed decision, not a silent one.

    The go.mod branch is deliberately ahead of the package.json branch: a Go
    repo commonly also carries a package.json (web assets) without a `test`
    script, which previously fell through to "no tests configured" so the Go
    code was never compiled or tested (the Observinator escape). `go test ./...`
    compiles production and test code, catching build errors as well as failures.

    Tool availability is enforced by execution, not host-side probing: commands
    run inside the sandbox, so a missing runner exits non-zero ("command not
    found") and the gate fails closed naming the tool.
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
    elif (repo_path / "pyproject.toml").exists():
        # Python project with no pytest: a syntax floor beats a silent pass.
        changed_py = _changed_existing(repo_path, changed_files, ".py")
        if changed_py:
            cmds = [["python3", "-m", "py_compile", *changed_py]]
        else:
            cmds = [["python3", "-m", "compileall", "-q", "-x", _COMPILEALL_EXCLUDE, "."]]
    else:
        return True, f"no test runner configured (probed: {', '.join(_TEST_PROBES)})"

    if sandbox is None:
        from forge.task_worker.sandbox import make_sandbox

        sandbox = make_sandbox(repo_path)

    return _run_cmds(sandbox, cmds)


def _static_checks(
    repo_path: Path, changed_files: list[str] | None
) -> list[tuple[str, list[list[str]]]]:
    """All applicable always-on static checks, labeled. Additive, not first-match:
    a repo can be Go + TypeScript + shell at once (Observinator is)."""
    checks: list[tuple[str, list[list[str]]]] = []
    if (repo_path / "go.mod").exists():
        checks.append(("go", [["go", "build", "./..."]]))
    if (repo_path / "Cargo.toml").exists():
        checks.append(("rust", [["cargo", "build"]]))
    if (repo_path / "tsconfig.json").exists():
        checks.append(("typescript", [["pnpm", "exec", "tsc", "--noEmit"]]))
    changed_sh = _changed_existing(repo_path, changed_files, ".sh")
    if changed_sh:
        checks.append(("shell", [["shellcheck", *changed_sh]]))
    changed_py = _changed_existing(repo_path, changed_files, ".py")
    if changed_py:
        checks.append(("python", [["python3", "-m", "py_compile", *changed_py]]))
    return checks


def run_build(
    repo_path: Path,
    sandbox: Sandbox | None = None,
    changed_files: list[str] | None = None,
) -> tuple[bool, str, bool]:
    """Static-check the change, independent of whether the task requires tests.

    Returns (ok, output_tail, ran). ``ran`` is False only when no static check
    applies (e.g. a plain-JS repo with no tsconfig and no changed .sh/.py).
    Checks are additive per language: go/cargo build for compiled code,
    `tsc --noEmit` when a tsconfig exists, shellcheck over changed .sh files,
    and a py_compile syntax floor over changed .py files. A failure here means
    the change is statically broken and must block the leaf — this is the gate
    that makes landing non-compiling code impossible regardless of
    requires_tests.

    Tools are not probed host-side: the sandbox is the environment of record,
    so a detected language whose tool is missing there fails the command
    ("command not found") and the gate fails closed naming the tool — never
    a silent pass (the Observinator escape).
    """
    checks = _static_checks(repo_path, changed_files)
    if not checks:
        return True, "no static checks apply to this change", False

    if sandbox is None:
        from forge.task_worker.sandbox import make_sandbox

        sandbox = make_sandbox(repo_path)

    outputs: list[str] = []
    for label, cmds in checks:
        ok, output = _run_cmds(sandbox, cmds)
        outputs.append(f"## static-check: {label}\n{output}")
        if not ok:
            return False, _tail("\n".join(outputs)), True
    return True, _tail("\n".join(outputs)), True
