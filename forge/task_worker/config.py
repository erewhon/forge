from __future__ import annotations

from functools import cached_property
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class TaskWorkerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TASK_WORKER_")

    # Paths
    projects_dir: Path = Path.home() / "Projects" / "erewhon"
    nous_data_dir: Path = Path.home() / ".local" / "share" / "nous"

    # Nous targets
    notebook_name: str = "Forge"
    database_name: str = "Project Tasks"
    daemon_url: str = "http://127.0.0.1:7667"

    # Task execution
    task_timeout_seconds: int = 1800  # 30 min max per task
    default_max_files: int = 5  # used when task has no max_files set
    model_tier_default: str = "auto"  # router alias

    # Safety
    dry_run: bool = False  # when True, run OpenCode but don't commit or update Nous

    # Projects the worker is allowed to touch (allowlist).
    # Empty = no restriction (allows any project with a worker-ready task)
    allowed_projects: list[str] = []

    # Commit message prefix
    commit_prefix: str = "auto: "

    # --- Derived properties ---

    @cached_property
    def notebook_id(self) -> str:
        """Resolve notebook_name to its UUID via NousStorage."""
        from nous_mcp.storage import NousStorage

        storage = NousStorage(self.nous_data_dir)
        nb = storage.resolve_notebook(self.notebook_name)
        return nb["id"]

    @cached_property
    def database_id(self) -> str:
        """Resolve database_name to its database page id via NousStorage.

        Restricted to pages with pageType=='database' to disambiguate when a
        regular page shares the same title.
        """
        from nous_mcp.storage import NousStorage

        storage = NousStorage(self.nous_data_dir)
        name_lower = self.database_name.lower()
        all_dbs = storage.list_database_pages(self.notebook_id)

        # Dedupe by id — list_database_pages can yield duplicates.
        seen: dict[str, dict] = {}
        for p in all_dbs:
            seen.setdefault(p["id"], p)
        db_pages = list(seen.values())

        candidates = [
            p for p in db_pages if p.get("title", "").lower() == name_lower
        ]
        if not candidates:
            candidates = [
                p for p in db_pages
                if p.get("title", "").lower().startswith(name_lower)
            ]
        if len(candidates) == 1:
            return candidates[0]["id"]
        if not candidates:
            raise ValueError(
                f"No database page titled '{self.database_name}' in notebook "
                f"'{self.notebook_name}'"
            )
        titles = [c.get("title", "") for c in candidates]
        raise ValueError(
            f"Ambiguous database name '{self.database_name}': {titles}"
        )


settings = TaskWorkerSettings()
