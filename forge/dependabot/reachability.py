"""Import-graph reachability signal for advisory prioritization.

A demote-only heuristic: is the vulnerable/bumped package actually imported by
this repo's code? Unreachable-vulnerability advisories get deprioritized; the
auto-merge gate is NEVER touched by this signal.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

# Directories to skip when walking the tree for .py files.
_SKIP_DIRS = {".venv", "venv", ".env", "node_modules", "__pycache__", ".git"}

# Patterns to match for top-level imports: ``import x`` and ``from x import y``.
_IMPORT_TOP = re.compile(r"^(?P<mod>[a-zA-Z0-9][a-zA-Z0-9._]*)")


def _skip_dir(name: str) -> bool:
    """Return True if *name* is a directory we should skip."""
    return name in _SKIP_DIRS


def imported_names(repo_root: Path) -> set[str]:
    """Walk ``*.py`` files under *repo_root*, collecting top-level imported module names.

    Skips hidden directories and ``.venv``. Returns the set of module name strings
    found at the top level of ``import`` / ``from ... import`` statements.
    """
    names: set[str] = set()
    py_files = sorted(repo_root.rglob("*.py"))
    for fpath in py_files:
        # Skip files inside skip directories
        try:
            if any(_skip_dir(part) for part in fpath.parts):
                continue
        except ValueError:
            continue
        try:
            source = fpath.read_text("utf-8", errors="replace")
        except OSError:
            continue
        try:
            tree = ast.parse(source, filename=str(fpath))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    # ``import x.y.z`` -> record "x" (top-level package)
                    top = alias.name.split(".")[0]
                    names.add(top)
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                # Relative imports (level > 0) are local — skip them.
                if node.module:
                    top = node.module.split(".")[0]
                    names.add(top)
    return names


def import_candidates(dist_name: str) -> set[str]:
    """Best-effort mapping from distribution name to plausible import names.

    Uses ``importlib.metadata.packages_distributions()`` reverse lookup when the
    dist is importable in the current env, plus dash/underscore normalization as
    fallback.
    """
    import importlib.metadata

    candidates: set[str] = set()

    # Try importlib.metadata reverse lookup first.
    try:
        dists = importlib.metadata.packages_distributions()
        for import_name, dist_names in dists.items():
            for dn in dist_names:
                if dn.lower().replace("-", "_") == dist_name.lower().replace("-", "_"):
                    candidates.add(import_name)
    except Exception:
        pass

    # Fallback: dash/underscore normalization.
    normalized = dist_name.lower().replace("-", "_")
    candidates.add(normalized)
    candidates.add(dist_name.lower())
    # Also add the dash version.
    candidates.add(dist_name.lower().replace("_", "-"))

    return candidates


def is_imported(repo_root: Path, dist_name: str) -> bool | None:
    """Is *dist_name* imported anywhere in the repo's Python files?

    Returns ``True`` when a candidate import name matches the collected imports,
    ``False`` when the walk succeeded and no candidate matches, and ``None`` when
    the candidate set is empty or the walk fails (never raises).
    """
    try:
        candidates = import_candidates(dist_name)
        if not candidates:
            return None
        found = imported_names(repo_root)
        if not found:
            # Walk found no .py files at all — can't determine
            return None
        return bool(candidates & found)
    except Exception:
        return None
