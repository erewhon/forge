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
- **Baseline-relative, not per-file absolute.** Changed-FILES scoping still inherits
  same-file debt: a pre-existing unsafe-fix violation in cli.py failed every leaf that
  touched the file across four runs, and since retries rewrite the leaf's code (not the
  pre-existing function), none could converge (test-as-spec pilot, live). Violations
  that survive autofix are compared against the same files at the leaf's BASE revision
  (keyed rule + digit-normalized message + file, line-drift tolerant); only NEW ones
  fail the leaf. Pre-existing ones are disclosed in the pass message, never blocking.

Scope: Python/ruff only, and only when the repo shows ruff intent (``ruff`` appears in
pyproject.toml — config section or dependency). Repo-scoped linters (``pnpm lint``,
``cargo clippy``) can't be cheaply confined to changed files and would hit the same
pre-existing-debt wall, so non-Python changes pass vacuously for now.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from forge.task_worker.sandbox import Sandbox

_LINT_TIMEOUT = 120
# 4000, not 1000: ruff renders each violation as a multi-line code frame, and the
# retry prompt is built from this tail — a budget that fits only the format-check
# block leaves the retry blind to the violations it must fix (observed: a leaf
# looped rewriting the same unsafe-fix idiom because the evidence never showed it).
_OUTPUT_TAIL = 4000


def _tail(text: str, n: int = _OUTPUT_TAIL) -> str:
    if len(text) <= n:
        return text
    cut = text[-n:]
    nl = cut.find("\n")
    if 0 <= nl < len(cut) - 1:
        return cut[nl + 1 :]
    return cut


_UV_NOISE_PREFIXES = ("Downloading ", "Downloaded ", "Installed ", "Resolved ", "Prepared ")


def _strip_uv_noise(text: str) -> str:
    """Drop uv/uvx progress lines so the evidence tail keeps ruff's violations, not
    package-manager chatter (belt to the ``uvx -q`` suspenders — ``uv run`` emits the
    same noise when the sandbox venv is cold)."""
    kept = [ln for ln in text.splitlines() if not ln.strip().startswith(_UV_NOISE_PREFIXES)]
    return "\n".join(kept)


# One ruff violation line: "path.py:12:5: E501 Line too long (105 > 100)".
_VIOLATION_RE = re.compile(
    r"^(?P<path>[^\s:][^:]*\.pyi?):(?P<line>\d+):(?P<col>\d+):\s+(?P<code>[A-Z]+\d+)\s+(?P<msg>.*)$"
)


def _violation_keys(output: str, strip_prefix: str = "") -> set[tuple[str, str, str]]:
    """Line-drift-tolerant violation identities: (file, rule, digit-normalized message).

    Digits in the message are normalized away so an E501 whose measured length shifted
    with unrelated edits still matches its baseline twin."""
    keys: set[tuple[str, str, str]] = set()
    for raw in output.splitlines():
        m = _VIOLATION_RE.match(raw.strip())
        if not m:
            continue
        path = m.group("path")
        if strip_prefix and path.startswith(strip_prefix):
            path = path[len(strip_prefix) :].lstrip("/")
        msg = re.sub(r"\d+", "#", m.group("msg")).strip()
        keys.add((path, m.group("code"), msg))
    return keys


def _baseline_keys(
    repo_path: Path,
    py_files: list[str],
    ruff_base: list[str],
    sandbox: Sandbox,
) -> set[tuple[str, str, str]] | None:
    """Violations already present in the touched files at the leaf's base revision.

    Materializes the base version of each touched file under the self-ignored
    ``.task_worker/`` dir (the sandbox sees the repo mount, so ruff must run on paths
    inside it; config discovery walks up and finds the repo's pyproject). Returns None
    when the baseline cannot be established — the caller fails closed on the full set
    rather than guessing."""
    from forge.task_worker.vcs import VCSError, get_base_file_content

    mirror_rel = f".task_worker/lint_base_{uuid.uuid4().hex[:8]}"
    mirror = repo_path / mirror_rel
    materialized: list[str] = []
    try:
        for f in py_files:
            try:
                content = get_base_file_content(repo_path, f)
            except VCSError:
                return None
            if content is None:
                continue  # new file at base: everything in it is genuinely new
            target = mirror / f
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            materialized.append(f"{mirror_rel}/{f}")
        if not materialized:
            return set()
        rc, out = _run(sandbox, [*ruff_base, "check", "--output-format", "concise", *materialized])
        # rc 0 = clean base, rc 1 = violations found; anything else is a ruff/config
        # error and the baseline is not trustworthy.
        if rc not in (0, 1):
            return None
        return _violation_keys(out, strip_prefix=mirror_rel)
    finally:
        shutil.rmtree(mirror, ignore_errors=True)


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
        # -q: uvx's download/install progress goes to stderr and, in an ephemeral
        # sandbox with a cold cache, is long enough to push ruff's actual violation
        # list out of the tailed evidence — the retry prompt then shows noise instead
        # of what to fix, and the leaf loops (test-as-spec live finding, 2026-07-30).
        return ["uvx", "-q", "ruff"]
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

    def check() -> tuple[int, int, str]:
        # concise output: one parseable line per violation — both the baseline
        # comparison and the evidence tail need identities, not code frames.
        rc_check, out_check = _run(
            sandbox, [*base, "check", "--output-format", "concise", *py_files]
        )
        rc_fmt, out_fmt = _run(sandbox, [*base, "format", "--check", *py_files])
        return rc_check, rc_fmt, f"{out_check}\n{out_fmt}"

    rc_check, rc_fmt, out = check()
    if rc_check == 0 and rc_fmt == 0:
        return True, "lint clean", False

    # Autofix on the changed files only, then re-judge: only violations that
    # survive the fix can fail the leaf.
    _run(sandbox, [*base, "check", "--fix", *py_files])
    _run(sandbox, [*base, "format", *py_files])
    rc_check, rc_fmt, out = check()
    if rc_check == 0 and rc_fmt == 0:
        return True, "lint clean after autofix", True
    if rc_fmt != 0 or rc_check not in (0, 1):
        # Format still red after formatting, or ruff itself errored (rc 2): not the
        # pre-existing-debt shape — fail on the full evidence.
        return False, _tail(_strip_uv_noise(out)), True

    # Violations survived autofix: judge them relative to the leaf's BASE revision.
    # Only violations NEW in this change fail the leaf; inherited same-file debt is
    # disclosed but never blocking (and never silently hidden — the count is stated).
    final_keys = _violation_keys(out)
    if not final_keys:
        return False, _tail(_strip_uv_noise(out)), True  # red but unparseable: fail closed
    base_keys = _baseline_keys(repo_path, py_files, base, sandbox)
    if base_keys is None:
        return False, _tail(_strip_uv_noise(out)), True  # no trustworthy baseline: fail closed
    new_keys = final_keys - base_keys
    ignored = len(final_keys & base_keys)
    if not new_keys:
        return (
            True,
            (
                f"lint clean relative to base ({ignored} pre-existing violation(s) in "
                f"touched files ignored)"
            ),
            True,
        )
    new_lines = [
        line
        for line in out.splitlines()
        if _violation_keys(line) and _violation_keys(line) <= new_keys
    ]
    evidence = "\n".join(new_lines) or out
    if ignored:
        evidence += (
            f"\n({ignored} pre-existing violation(s) in touched files ignored — "
            f"only the violations above are yours)"
        )
    return False, _tail(_strip_uv_noise(evidence)), True
