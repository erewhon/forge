"""The no-LLM structural pass: tree-sitter walk of a mirror → ``RepoIndex`` + ``map.md``.

Symbols are deliberately shallow — top-level definitions plus one level of members inside
class-like containers. This is a map for orienting an agent (or a person), not an IDE index;
the summarizer stage reads actual file contents for anything deeper.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import re
import subprocess
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from forge.cartographer.models import (
    DEFAULT_IGNORES,
    FileEntry,
    MapConfig,
    ModuleNode,
    RepoIndex,
    TargetConfig,
    TargetInfo,
)

_EXT_LANGUAGE: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".rb": "ruby",
}

_ENTRY_STEMS = {"main", "index", "app", "cli", "__main__"}


@dataclass(frozen=True)
class _LangSpec:
    defs: dict[str, str]  # node type → symbol kind label
    imports: frozenset[str] = frozenset()
    containers: frozenset[str] = frozenset()  # def types whose bodies get one more level


_CLASS_KINDS = {
    "class_declaration": "class",
    "interface_declaration": "interface",
    "enum_declaration": "enum",
}

_LANG_SPECS: dict[str, _LangSpec] = {
    "python": _LangSpec(
        defs={"function_definition": "def", "class_definition": "class"},
        imports=frozenset({"import_statement", "import_from_statement"}),
        containers=frozenset({"class_definition"}),
    ),
    "javascript": _LangSpec(
        defs={
            "function_declaration": "function",
            "generator_function_declaration": "function",
            "class_declaration": "class",
            "method_definition": "method",  # only reachable inside a container walk
        },
        imports=frozenset({"import_statement"}),
        containers=frozenset({"class_declaration"}),
    ),
    "typescript": _LangSpec(
        defs={
            "function_declaration": "function",
            "class_declaration": "class",
            "abstract_class_declaration": "class",
            "interface_declaration": "interface",
            "enum_declaration": "enum",
            "type_alias_declaration": "type",
            "method_definition": "method",  # only reachable inside a container walk
        },
        imports=frozenset({"import_statement"}),
        containers=frozenset({"class_declaration", "abstract_class_declaration"}),
    ),
    "java": _LangSpec(
        defs={**_CLASS_KINDS, "record_declaration": "record", "method_declaration": "method"},
        imports=frozenset({"import_declaration"}),
        containers=frozenset({"class_declaration", "interface_declaration", "record_declaration"}),
    ),
    "go": _LangSpec(
        defs={"function_declaration": "func", "method_declaration": "func", "type_spec": "type"},
        imports=frozenset({"import_declaration"}),
    ),
    "rust": _LangSpec(
        defs={
            "function_item": "fn",
            "struct_item": "struct",
            "enum_item": "enum",
            "trait_item": "trait",
        },
        imports=frozenset({"use_declaration"}),
    ),
    "kotlin": _LangSpec(
        defs={
            "class_declaration": "class",
            "function_declaration": "fun",
            "object_declaration": "object",
        },
        imports=frozenset({"import_header"}),
        containers=frozenset({"class_declaration", "object_declaration"}),
    ),
    "ruby": _LangSpec(
        defs={"class": "class", "module": "module", "method": "def"},
        containers=frozenset({"class", "module"}),
    ),
}
_LANG_SPECS["tsx"] = _LANG_SPECS["typescript"]

_parsers: dict[str, object] = {}


def _parser(language: str):
    """Cached tree-sitter parser; a missing optional dependency yields a clear install hint."""
    if language not in _parsers:
        try:
            from tree_sitter_language_pack import get_parser
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise RuntimeError(
                "tree-sitter is not installed — install the map extra: `uv sync --extra map` "
                "(or `pip install 'forge[map]'`)."
            ) from exc
        _parsers[language] = get_parser(language)
    return _parsers[language]


def _node_name(node) -> str | None:
    named = node.child_by_field_name("name")
    if named is not None:
        return named.text.decode("utf-8", errors="replace")
    for child in node.named_children:
        if child.type in {"identifier", "type_identifier", "constant", "simple_identifier"}:
            return child.text.decode("utf-8", errors="replace")
    return None


def _unwrap(node):
    """See through wrapper nodes (python decorators, JS/TS export statements)."""
    if node.type == "decorated_definition":
        return node.child_by_field_name("definition") or node
    if node.type == "export_statement":
        return node.child_by_field_name("declaration") or node
    return node


def _one_line(text: bytes, limit: int = 160) -> str:
    line = " ".join(text.decode("utf-8", errors="replace").split())
    return line[:limit]


def _collect(
    node, spec: _LangSpec, symbols: list[str], imports: list[str], prefix: str = ""
) -> None:
    node = _unwrap(node)
    if node.type in spec.imports:
        imports.append(_one_line(node.text))
        return
    if node.type == "type_declaration":  # go: names live on the inner type_spec nodes
        for inner in node.named_children:
            _collect(inner, spec, symbols, imports, prefix)
        return
    kind = spec.defs.get(node.type)
    if kind is None:
        return
    name = _node_name(node)
    if name is None:
        return
    symbols.append(f"{kind} {prefix}{name}")
    if node.type in spec.containers and not prefix:  # one level of members, no deeper
        body = node.child_by_field_name("body")
        for member in body.named_children if body is not None else []:
            _collect(member, spec, symbols, imports, prefix=f"{name}.")


def parse_file(path: Path, language: str) -> tuple[list[str], list[str]]:
    """(symbols, imports) for one file. Unknown languages return empty results."""
    spec = _LANG_SPECS.get(language)
    if spec is None:
        return [], []
    tree = _parser(language).parse(path.read_bytes())
    symbols: list[str] = []
    imports: list[str] = []
    for child in tree.root_node.named_children:
        _collect(child, spec, symbols, imports)
    return symbols, imports


# --- module detection ---------------------------------------------------------------------------


def _package_json_name(path: Path) -> str | None:
    try:
        return json.loads(path.read_text()).get("name")
    except (OSError, ValueError):
        return None


def _pyproject_name(path: Path) -> str | None:
    try:
        return tomllib.loads(path.read_text()).get("project", {}).get("name")
    except (OSError, ValueError):
        return None


def _gradle_includes(text: str) -> list[str]:
    dirs: list[str] = []
    pattern = r"include\s*[(\s]\s*(['\"][^'\"]+['\"](?:\s*,\s*['\"][^'\"]+['\"])*)"
    for match in re.finditer(pattern, text):
        for entry in re.findall(r"['\"]([^'\"]+)['\"]", match.group(1)):
            dirs.append(entry.lstrip(":").replace(":", "/"))
    return dirs


def detect_modules(mirror: Path, target_name: str, ignored: set[Path]) -> list[ModuleNode]:
    """Workspace/module boundaries as ``ModuleNode``s (files are attached later)."""
    found: dict[str, str] = {}  # rel dir → module name

    for marker in sorted(mirror.rglob("*")):
        if not marker.is_file() or any(parent in ignored for parent in marker.parents):
            continue
        rel_dir = marker.parent.relative_to(mirror).as_posix()
        if marker.name == "package.json":
            found.setdefault(rel_dir, _package_json_name(marker) or marker.parent.name)
        elif marker.name == "pyproject.toml":
            found.setdefault(rel_dir, _pyproject_name(marker) or marker.parent.name)
        elif marker.name == "go.mod":
            first = marker.read_text().splitlines()[0] if marker.exists() else ""
            name = first.removeprefix("module").strip().rsplit("/", 1)[-1] or marker.parent.name
            found.setdefault(rel_dir, name)
        elif marker.name in ("settings.gradle", "settings.gradle.kts"):
            for sub in _gradle_includes(marker.read_text()):
                found.setdefault(
                    (marker.parent.relative_to(mirror) / sub).as_posix().lstrip("./"), sub
                )
        elif marker.name == "pom.xml":
            for sub in re.findall(r"<module>([^<]+)</module>", marker.read_text()):
                found.setdefault(
                    (marker.parent.relative_to(mirror) / sub.strip()).as_posix().lstrip("./"),
                    sub.strip(),
                )
            found.setdefault(rel_dir, marker.parent.name if rel_dir != "." else target_name)

    if "." not in found:
        found["."] = target_name
    return [ModuleNode(name=name, path=rel) for rel, name in sorted(found.items())]


def _module_for(rel_path: str, modules: list[ModuleNode]) -> str:
    best = ""
    best_len = -1
    for module in modules:
        prefix = "" if module.path == "." else module.path + "/"
        if rel_path.startswith(prefix) and len(prefix) > best_len:
            best, best_len = module.name, len(prefix)
    return best


# --- blob hashes --------------------------------------------------------------------------------


def git_blob_shas(mirror: Path) -> dict[str, str]:
    """Path → blob sha from the mirror's git index. Not a git repo → empty (callers fall back)."""
    proc = subprocess.run(
        ["git", "-C", str(mirror), "ls-files", "-s", "-z"], capture_output=True, text=True
    )
    if proc.returncode != 0:
        return {}
    shas: dict[str, str] = {}
    for record in proc.stdout.split("\0"):
        if not record:
            continue
        meta, _, path = record.partition("\t")
        parts = meta.split()
        if len(parts) >= 2:
            shas[path] = parts[1]
    return shas


# --- the pass -----------------------------------------------------------------------------------


@dataclass
class StructuralResult:
    index: RepoIndex
    parsed: int = 0
    skipped: list[str] = field(default_factory=list)


def _ignored_dirs(mirror: Path, target: TargetConfig) -> set[Path]:
    globs = (*DEFAULT_IGNORES, *target.ignore_globs)
    out: set[Path] = set()
    for path in mirror.rglob("*"):
        if path.is_dir() and any(fnmatch.fnmatch(path.name, g) for g in globs):
            out.add(path)
    return out


def build_index(config: MapConfig, target: TargetConfig, info: TargetInfo) -> StructuralResult:
    mirror = config.mirror_dir(target)
    if not mirror.is_dir():
        raise FileNotFoundError(
            f"no mirror for target '{target.name}' — run `forge map sync` first"
        )

    ignored = _ignored_dirs(mirror, target)
    modules = detect_modules(mirror, target.name, ignored)
    shas = git_blob_shas(mirror)
    result = StructuralResult(index=RepoIndex(target=info, modules=modules))

    for path in sorted(mirror.rglob("*")):
        if not path.is_file() or any(parent in ignored for parent in path.parents):
            continue
        language = target.language_hints.get(path.suffix) or _EXT_LANGUAGE.get(path.suffix)
        if language is None:
            continue
        rel = path.relative_to(mirror).as_posix()
        blob_sha = shas.get(rel) or hashlib.sha256(path.read_bytes()).hexdigest()
        entry = FileEntry(
            path=rel,
            module=_module_for(rel, modules),
            blob_sha=blob_sha,
            language=language,
            entry_point=path.stem in _ENTRY_STEMS,
        )
        if path.stat().st_size > config.max_file_bytes:
            entry.skipped = f"over max_file_bytes ({config.max_file_bytes})"
            result.skipped.append(rel)
        else:
            entry.symbols, entry.imports = parse_file(path, language)
            if language == "python" and not entry.entry_point:
                raw = path.read_bytes()
                entry.entry_point = b"__name__" in raw and b'"__main__"' in raw
            result.parsed += 1
        result.index.files.append(entry)

    for module in modules:
        module.files = [f.path for f in result.index.files if f.module == module.name]
    return result


def render_map(result: StructuralResult) -> str:
    """The human-readable ``map.md`` for one target."""
    index = result.index
    t = index.target
    lines = [f"# Repo map — {t.name}", ""]
    origin = f"{t.host}:{t.path}" if t.host else t.path
    lines.append(f"- **Checkout**: `{origin}`")
    lines.append(f"- **Synced**: {t.synced_at.isoformat() if t.synced_at else 'never'}"
                 + (" ⚠ STALE — last sync failed or never ran" if t.stale else ""))
    lines.append(
        f"- **Files mapped**: {result.parsed} parsed, {len(result.skipped)} skipped (size)"
    )
    lines.append("")
    for module in index.modules:
        entries = [f for f in index.files if f.module == module.name]
        if not entries:
            continue
        lines.append(f"## {module.name} (`{module.path}`, {len(entries)} files)")
        lines.append("")
        for entry in entries:
            mark = " *(entry point)*" if entry.entry_point else ""
            if entry.skipped:
                lines.append(f"- `{entry.path}`{mark} — skipped: {entry.skipped}")
                continue
            lines.append(f"- `{entry.path}`{mark}")
            if entry.symbols:
                lines.append(f"  - symbols: {'; '.join(entry.symbols)}")
            cross = [i for i in entry.imports if _is_cross_module(i, module, index.modules)]
            if cross:
                lines.append(f"  - cross-module imports: {'; '.join(cross)}")
        lines.append("")
    return "\n".join(lines)


def _is_cross_module(import_line: str, module: ModuleNode, modules: list[ModuleNode]) -> bool:
    return any(
        other.name != module.name and other.name and other.name in import_line
        for other in modules
    )


def write_outputs(config: MapConfig, target: TargetConfig, result: StructuralResult) -> Path:
    out = config.out_dir(target)
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.json").write_text(result.index.model_dump_json(indent=2) + "\n")
    (out / "map.md").write_text(render_map(result))
    return out
