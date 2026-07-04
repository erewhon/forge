"""Dispatch-time epic context — the "sibling contracts" preamble (e2e dry-run Q1).

A leaf's spec is written before its dependencies' code exists, so specs encode the
architect's *guess* at sibling interfaces; the dry-run's drift case was a module
exposing ``VALID_UNITS`` while its sibling's spec assumed a factor table. The fix is
ground truth at dispatch time: for each direct dependency that has LANDED (journal
``leaf_dispatch`` records carry the commit), extract the public interface of the files
that commit touched — as they exist on disk *now*, which is what the new leaf actually
integrates against — and prepend a compact preamble to the spec the worker receives.

Contracts, not style: top-level function/class signatures and public module constants
(the dry-run's drift WAS a constant), rendered from the stdlib ``ast``. Non-Python
files are listed by path only. Everything is capped with an explicit truncation
marker — never silent (the inventory convention).

Dependency titles come from ``TaskInfo.deps`` (the Forge row), so hand-filed epics and
replan fix-ups get context too — no ``tree.json`` required. Building context must
NEVER break dispatch: callers wrap this in a degrade path (dispatch plain on error).
"""

from __future__ import annotations

import ast
import json
import subprocess
from pathlib import Path

from agents.coding_pipeline.models import LeafRow
from agents.task_worker.models import TaskInfo
from agents.task_worker.vcs import detect_vcs

_MAX_CHARS = 4000
_TIMEOUT = 30


# --- journal: which deps landed, and where ------------------------------------------


def landed_commits(run_dir: Path) -> dict[str, str]:
    """leaf title -> commit id for every leaf the journal records as landed.

    The LAST ``done`` record per title wins (a leaf re-run after a revert lands a
    fresh commit).
    """
    journal = run_dir / "journal.jsonl"
    if not journal.is_file():
        return {}
    landed: dict[str, str] = {}
    for line in journal.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("event") == "leaf_dispatch" and rec.get("status") == "done":
            commit = str(rec.get("commit_id", "") or "")
            if commit:
                landed[str(rec.get("leaf", ""))] = commit
    return landed


# --- VCS: what files a landed commit touched -----------------------------------------


def commit_files(repo: Path, commit: str) -> list[str]:
    """Paths the commit touched (added/modified), repo-relative. Deleted files are
    excluded — there is no interface to integrate against."""
    vcs = detect_vcs(repo)
    if vcs == "jj":
        res = subprocess.run(
            ["jj", "diff", "--no-pager", "-r", commit, "--summary"],
            capture_output=True,
            text=True,
            timeout=_TIMEOUT,
            cwd=repo,
        )
    elif vcs == "git":
        res = subprocess.run(
            ["git", "show", "--name-status", "--format=", commit],
            capture_output=True,
            text=True,
            timeout=_TIMEOUT,
            cwd=repo,
        )
    else:
        return []
    if res.returncode != 0:
        return []
    files: list[str] = []
    for line in res.stdout.splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            continue
        status, path = parts
        if status.upper().startswith(("A", "M")):
            files.append(path.strip())
    return files


# --- ast: the public surface of a Python file ----------------------------------------


def _render_args(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    sig = f"({ast.unparse(node.args)})"
    if node.returns is not None:
        sig += f" -> {ast.unparse(node.returns)}"
    return sig


def public_interface(source: str) -> list[str]:
    """Public top-level surface of a Python module: function/class signatures,
    public method signatures one level deep, and public constants (assignments).
    Returns [] for unparseable source — a broken file is not a contract."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    lines: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            if not node.name.startswith("_"):
                lines.append(f"def {node.name}{_render_args(node)}")
        elif isinstance(node, ast.ClassDef):
            if node.name.startswith("_"):
                continue
            bases = f"({', '.join(ast.unparse(b) for b in node.bases)})" if node.bases else ""
            lines.append(f"class {node.name}{bases}")
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef | ast.AsyncFunctionDef):
                    if not sub.name.startswith("_") or sub.name == "__init__":
                        lines.append(f"    def {sub.name}{_render_args(sub)}")
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and not target.id.startswith("_"):
                    lines.append(target.id)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and not node.target.id.startswith("_"):
                annotation = ast.unparse(node.annotation)
                lines.append(f"{node.target.id}: {annotation}")
    return lines


def _is_test_path(path: str) -> bool:
    parts = Path(path).parts
    return any(p in ("tests", "test") for p in parts) or Path(path).name.startswith("test_")


def _dep_block(repo: Path, title: str, commit: str) -> str:
    """One dependency's landed interface: files it touched, public surface for .py.

    Test files are path-only — test functions are not contracts, and their
    signatures would crowd out the surface that matters."""
    lines = [f'- "{title}" (commit {commit}):']
    for path in commit_files(repo, commit):
        full = repo / path
        if path.endswith(".py") and not _is_test_path(path) and full.is_file():
            surface = public_interface(full.read_text(encoding="utf-8", errors="replace"))
            if surface:
                lines.append(f"  - {path} — public surface:")
                lines.extend(f"      {s}" for s in surface)
                continue
        lines.append(f"  - {path}")
    return "\n".join(lines)


# --- the preamble --------------------------------------------------------------------


def build_leaf_context(
    task: TaskInfo,
    *,
    run_dir: Path,
    repo: Path,
    epic_goal: str,
    siblings: list[LeafRow],
    max_chars: int = _MAX_CHARS,
) -> str:
    """The epic-context preamble for one leaf, or "" when there is nothing to say.

    Landed direct dependencies get their real interfaces; every other epic leaf is
    titles-only (scope fencing). Whole dep blocks are dropped from the end to fit
    ``max_chars``, with an explicit marker.
    """
    landed = landed_commits(run_dir)
    dep_blocks = [_dep_block(repo, dep, landed[dep]) for dep in task.deps if dep in landed]

    other_titles = [
        f"- {row.task} [{row.status}]"
        for row in siblings
        if row.task != task.task and row.task not in task.deps
    ]

    if not dep_blocks and not other_titles:
        return ""

    parts = [
        "## Epic context (generated at dispatch — contracts, not style)",
        f"Epic: {epic_goal}".strip(),
    ]
    truncated = 0
    if dep_blocks:
        parts.append("\n### Landed interfaces you must integrate against")
        budget = max_chars - sum(len(p) + 1 for p in parts) - 400  # room for siblings
        kept: list[str] = []
        used = 0
        for block in dep_blocks:
            if used + len(block) > budget and kept:
                truncated += 1
                continue
            kept.append(block)
            used += len(block) + 1
        parts.extend(kept)
        if truncated:
            parts.append(f"(truncated: {truncated} more dependency interface(s) omitted)")
    if other_titles:
        parts.append("\n### Other leaves in this epic (do NOT implement these)")
        parts.extend(other_titles[:20])
        if len(other_titles) > 20:
            parts.append(f"(and {len(other_titles) - 20} more)")
    return "\n".join(parts)
