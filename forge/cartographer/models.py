"""Config schema and the ``index.json`` contract for the cartographer.

``index.json`` is the interface between the structural pass and the later summarize/render
stages: ``files[].blob_sha`` is the summary-cache key, ``modules[]`` defines the rollup groups.
Change it deliberately — downstream stages validate against these models.
"""

from __future__ import annotations

import re
import tomllib
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from forge.shared.user_config import user_config_dir

#: Directory names never worth mapping, applied beneath every target on top of its own globs.
DEFAULT_IGNORES: tuple[str, ...] = (
    ".git",
    ".jj",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "dist",
    "build",
    "target",
    "vendor",
    ".idea",
    ".gradle",
)


class TargetConfig(BaseModel):
    """One checkout to map. ``host`` empty means a local path."""

    name: str
    path: str
    host: str | None = None
    ignore_globs: list[str] = Field(default_factory=list)
    # Extension → tree-sitter language overrides, e.g. {".inc": "php"}.
    language_hints: dict[str, str] = Field(default_factory=dict)

    @property
    def key(self) -> str:
        """Stable per-checkout directory slug: derived from host+path, never just the repo name."""
        raw = f"{self.host or 'local'}-{self.path}"
        return re.sub(r"[^A-Za-z0-9._-]+", "-", raw).strip("-").lower()


class MapConfig(BaseModel):
    targets: list[TargetConfig] = Field(default_factory=list)
    output_root: Path
    # Reserved for the summarizer stage; part of the schema so configs don't churn later.
    concurrency: int = 4
    max_file_bytes: int = 262_144

    def target_selection(self, names: list[str] | None) -> list[TargetConfig]:
        if not names:
            return list(self.targets)
        by_name = {t.name: t for t in self.targets}
        missing = [n for n in names if n not in by_name]
        if missing:
            raise KeyError(f"unknown target(s): {', '.join(missing)}")
        return [by_name[n] for n in names]

    def mirror_dir(self, target: TargetConfig) -> Path:
        return self.output_root / "mirror" / target.key

    def out_dir(self, target: TargetConfig) -> Path:
        return self.output_root / "out" / target.key


def default_config_path() -> Path:
    return user_config_dir() / "map.toml"


def load_config(path: Path | None = None) -> MapConfig:
    """Parse the map config (TOML). A missing file is a hard error — unlike forge's optional
    ``config.toml`` layer, the cartographer is useless without targets."""
    target = path or default_config_path()
    with target.open("rb") as fh:
        data = tomllib.load(fh)
    return MapConfig.model_validate(data)


# --- index.json ---------------------------------------------------------------------------------


class FileEntry(BaseModel):
    path: str  # relative to the mirror root, POSIX separators
    module: str
    blob_sha: str  # git blob hash (40 hex) or content sha256 (64 hex) for untracked files
    language: str
    symbols: list[str] = Field(default_factory=list)
    imports: list[str] = Field(default_factory=list)
    entry_point: bool = False
    skipped: str | None = None  # set (with a reason) when the file was not parsed


class ModuleNode(BaseModel):
    name: str
    path: str  # relative dir of the module root ("." for the repo root)
    files: list[str] = Field(default_factory=list)


class TargetInfo(BaseModel):
    name: str
    host: str | None = None
    path: str
    synced_at: datetime | None = None
    stale: bool = False


class RepoIndex(BaseModel):
    target: TargetInfo
    modules: list[ModuleNode] = Field(default_factory=list)
    files: list[FileEntry] = Field(default_factory=list)
