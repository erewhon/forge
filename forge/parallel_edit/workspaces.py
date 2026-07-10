"""Thin shim re-exporting forge.shared.workspaces with parallel_edit settings defaults.

Existing imports from this module must continue to work unchanged.
"""

from __future__ import annotations

from pathlib import Path

from forge.parallel_edit.config import settings
from forge.shared.workspaces import (
    DiffStat,
    JJError,
    _diff_exclude_fileset,
    _parse_diff_stat,
    collect_diff,
    create_workspace,
    ensure_git_marker,
    forget_workspace,
    resolve_base_rev,
)
from forge.shared.workspaces import (
    workspace_destination as _shared_workspace_destination,
)

__all__ = [
    "JJError",
    "DiffStat",
    "_diff_exclude_fileset",
    "_parse_diff_stat",
    "collect_diff",
    "create_workspace",
    "ensure_git_marker",
    "forget_workspace",
    "resolve_base_rev",
    "workspace_destination",
]


def workspace_destination(repo: Path, label: str) -> Path:
    """Fill base_dir/prefix from parallel_edit settings."""
    return _shared_workspace_destination(
        repo,
        label,
        base_dir=settings.workspace_base_dir,
        prefix=settings.workspace_name_prefix,
    )
