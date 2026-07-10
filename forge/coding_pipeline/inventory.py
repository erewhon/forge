"""A0 — repo and Forge inventory collection for the architect.

Pure collection, no LLM calls. The framing stage can only push back on a goal ("parity would
duplicate the sync layer — reframe") if it knows what already exists, so this gathers: an
ignore-aware depth-capped repo tree, heads of the key context files, module/test layout, the
detected toolchain, the Forge tasks already filed for the project, and a cheap lexical overlap
scan between the goal's terms and repo paths. Everything is size-capped with drops *counted*
(``Inventory.truncated``), never silent.
"""

from __future__ import annotations

import re
from pathlib import Path

from forge.coding_pipeline.config import settings
from forge.coding_pipeline.models import ExistingTask, FileHead, GoalSpec, Inventory
from forge.shared.automerge import is_test_path, slugify

# Never descend into these (plus simple dir entries from the repo's top-level .gitignore).
DEFAULT_IGNORES = frozenset(
    {
        ".git",
        ".jj",
        "__pycache__",
        ".venv",
        "venv",
        "node_modules",
        "target",
        "dist",
        "build",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        ".cache",
        "logs",
    }
)

# Files whose opening lines carry outsized context for the architect.
KEY_FILE_NAMES = (
    "CLAUDE.md",
    "README.md",
    "README.rst",
    "README",
    "pyproject.toml",
    "package.json",
    "Cargo.toml",
    "go.mod",
    "Justfile",
    "Makefile",
)

# Manifest/marker -> toolchain label.
TOOLCHAIN_MARKERS = (
    ("uv.lock", "uv/python"),
    ("pyproject.toml", "python (pyproject)"),
    ("pnpm-lock.yaml", "pnpm/node"),
    ("package.json", "node (package.json)"),
    ("Cargo.toml", "cargo/rust"),
    ("go.mod", "go"),
    ("Justfile", "just"),
)

TREE_MAX_ENTRIES = 400
KEY_FILE_HEAD_CHARS = 2_000
TEST_LAYOUT_MAX = 40
OVERLAPS_MAX = 40
EXISTING_TASKS_MAX = 100

_WORD_RE = re.compile(r"[a-z0-9_]{4,}")
_STOPWORDS = frozenset(
    {
        "this",
        "that",
        "with",
        "from",
        "into",
        "over",
        "should",
        "must",
        "have",
        "build",
        "make",
        "support",
        "existing",
        "using",
        "them",
        "then",
        "when",
        "where",
        "which",
        "their",
        "there",
    }
)


def run_dir_for(goal: GoalSpec) -> Path:
    """The epic run dir: ``runs_dir/<epic_slug>`` (slug derived from the goal when unset)."""
    return settings.runs_dir / (goal.epic_slug or slugify(goal.goal))


def _ignores_for(repo: Path) -> frozenset[str]:
    """DEFAULT_IGNORES plus plain directory entries from the repo's top-level .gitignore
    (simple names only — wildcard patterns are skipped, this is a heuristic, not git)."""
    extra: set[str] = set()
    gitignore = repo / ".gitignore"
    if gitignore.is_file():
        for line in gitignore.read_text().splitlines():
            entry = line.strip().rstrip("/")
            if entry and not entry.startswith("#") and not re.search(r"[*?\[\]!]", entry):
                extra.add(entry)
    return DEFAULT_IGNORES | extra


def _walk_tree(repo: Path, ignores: frozenset[str], depth: int) -> tuple[list[str], int]:
    """Indented tree lines (dirs end with '/'), depth-capped; returns (lines, dropped_count)."""
    lines: list[str] = []
    dropped = 0

    def _visit(directory: Path, level: int) -> None:
        nonlocal dropped
        try:
            entries = sorted(directory.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except OSError:
            return
        for entry in entries:
            if entry.name in ignores or entry.name.startswith("."):
                continue
            if len(lines) >= TREE_MAX_ENTRIES:
                dropped += 1
                continue
            indent = "  " * level
            lines.append(f"{indent}{entry.name}{'/' if entry.is_dir() else ''}")
            if entry.is_dir() and level + 1 < depth:
                _visit(entry, level + 1)

    _visit(repo, 0)
    return lines, dropped


def _key_files(repo: Path) -> list[FileHead]:
    heads: list[FileHead] = []
    for name in KEY_FILE_NAMES:
        path = repo / name
        if path.is_file():
            try:
                text = path.read_text(errors="replace")
            except OSError:
                continue
            heads.append(FileHead(path=name, head=text[:KEY_FILE_HEAD_CHARS]))
    return heads


def _modules(repo: Path, ignores: frozenset[str]) -> list[str]:
    return sorted(
        p.name
        for p in repo.iterdir()
        if p.is_dir() and p.name not in ignores and not p.name.startswith(".")
    )


def _test_layout(repo: Path, ignores: frozenset[str]) -> tuple[list[str], int]:
    """Directories (repo-relative) that contain at least one test file."""
    found: set[str] = set()
    for path in repo.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(repo)
        if any(part in ignores or part.startswith(".") for part in rel.parts[:-1]):
            continue
        if is_test_path(rel):
            found.add(str(rel.parent))
    ordered = sorted(found)
    dropped = max(0, len(ordered) - TEST_LAYOUT_MAX)
    return ordered[:TEST_LAYOUT_MAX], dropped


def _toolchain(repo: Path) -> list[str]:
    return [label for marker, label in TOOLCHAIN_MARKERS if (repo / marker).exists()]


def goal_terms(goal: GoalSpec) -> set[str]:
    """Lexical terms from the goal text for the overlap scan."""
    text = " ".join([goal.goal, goal.context, *goal.value_hints]).lower()
    return {w for w in _WORD_RE.findall(text) if w not in _STOPWORDS}


def _overlaps(repo: Path, ignores: frozenset[str], terms: set[str]) -> tuple[list[str], int]:
    """Repo paths whose name contains a goal term — the cheap scan that catches 'this already
    exists' before decomposition (the Nous-parity lesson)."""
    if not terms:
        return [], 0
    hits: list[str] = []
    for path in repo.rglob("*"):
        rel = path.relative_to(repo)
        if any(part in ignores or part.startswith(".") for part in rel.parts):
            continue
        stem = path.stem.lower()
        if any(term in stem for term in terms):
            hits.append(str(rel))
    hits.sort()
    dropped = max(0, len(hits) - OVERLAPS_MAX)
    return hits[:OVERLAPS_MAX], dropped


def fetch_project_tasks(project: str) -> list[ExistingTask]:
    """Compact rows of every Forge task (any status) filed for *project*.

    Reads through the task worker's daemon-backed Nous path; callers that already have rows
    (or tests) pass ``existing_tasks`` to :func:`collect_inventory` instead.
    """
    from nous_mcp.workflow import _query_tasks

    from forge.task_worker.nous_client import _read_db_content

    rows = _query_tasks(_read_db_content(), project=project, include_done=True, limit=None)
    return [
        ExistingTask(
            task=str(r.get("task", "")),
            status=str(r.get("status", "")),
            feature=str(r.get("feature") or ""),
            external_ref=str(r.get("external_ref") or ""),
        )
        for r in rows
    ]


def collect_inventory(
    goal: GoalSpec,
    repo: Path,
    *,
    existing_tasks: list[ExistingTask] | None = None,
) -> Inventory:
    """Collect the architect's context bundle for *repo*. Pure IO on the filesystem; the Forge
    read happens only when ``existing_tasks`` is not supplied."""
    ignores = _ignores_for(repo)
    truncated = 0

    tree_lines, dropped = _walk_tree(repo, ignores, settings.inventory_tree_depth)
    truncated += dropped

    tests, dropped = _test_layout(repo, ignores)
    truncated += dropped

    overlaps, dropped = _overlaps(repo, ignores, goal_terms(goal))
    truncated += dropped

    tasks = existing_tasks if existing_tasks is not None else fetch_project_tasks(goal.project)
    if len(tasks) > EXISTING_TASKS_MAX:
        truncated += len(tasks) - EXISTING_TASKS_MAX
        tasks = tasks[:EXISTING_TASKS_MAX]

    return Inventory(
        project=goal.project,
        repo=str(repo),
        tree="\n".join(tree_lines),
        key_files=_key_files(repo),
        modules=_modules(repo, ignores),
        test_layout=tests,
        toolchain=_toolchain(repo),
        existing_tasks=tasks,
        overlaps=overlaps,
        truncated=truncated,
    )


def render_inventory(inv: Inventory) -> str:
    """Markdown for the run dir + the architect prompt, trimmed (tree first) to the config cap."""
    parts: list[str] = [
        f"# Inventory — {inv.project}",
        f"\nRepo: `{inv.repo}`",
        f"Toolchain: {', '.join(inv.toolchain) or 'none detected'}",
        f"Modules: {', '.join(inv.modules) or '—'}",
    ]
    if inv.overlaps:
        parts.append("\n## Goal-term overlaps (existing code related to the ask)\n")
        parts.extend(f"- `{p}`" for p in inv.overlaps)
    if inv.test_layout:
        parts.append("\n## Test layout\n")
        parts.extend(f"- `{d}`" for d in inv.test_layout)
    if inv.existing_tasks:
        parts.append("\n## Forge tasks already filed for this project\n")
        parts.extend(
            f"- {t.task} — {t.status}"
            + (f" (feature: {t.feature})" if t.feature else "")
            + (f" [{t.external_ref}]" if t.external_ref else "")
            for t in inv.existing_tasks
        )
    for kf in inv.key_files:
        parts.append(f"\n## {kf.path} (head)\n\n```\n{kf.head}\n```")
    parts.append(f"\n## Repo tree (depth ≤ {settings.inventory_tree_depth})\n\n```")
    parts.append(inv.tree)
    parts.append("```")
    if inv.truncated:
        parts.append(f"\n*{inv.truncated} item(s) dropped by inventory caps.*")

    doc = "\n".join(parts)
    if len(doc) > settings.inventory_max_chars:
        overflow = len(doc) - settings.inventory_max_chars
        doc = doc[: settings.inventory_max_chars]
        doc += f"\n```\n\n*Inventory trimmed to fit the cap ({overflow} chars dropped).*"
    return doc


def write_inventory(inv: Inventory, run_dir: Path) -> Path:
    """Persist ``inventory.md`` + ``inventory.json`` into the epic run dir; returns the md path."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "inventory.json").write_text(inv.model_dump_json(indent=2))
    md_path = run_dir / "inventory.md"
    md_path.write_text(render_inventory(inv))
    return md_path
