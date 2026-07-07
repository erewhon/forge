from __future__ import annotations

from functools import cached_property
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class TaskWorkerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TASK_WORKER_")

    # Paths
    projects_dir: Path = Path.home() / "Projects" / "erewhon"

    # Nous targets
    notebook_name: str = "Forge"
    database_name: str = "Project Tasks"
    daemon_url: str = "http://127.0.0.1:7667"

    # Task execution
    task_timeout_seconds: int = 1800  # 30 min max per task
    default_max_files: int = 5  # used when task has no max_files set
    model_tier_default: str = "auto"  # router alias
    sandbox: str = "gaol-dx"  # default sandbox kind: "gaol-dx" or "gaol-run-once"

    # gaol run-once sandbox — ephemeral per-command containers for repos without a dx
    # container (the concurrent dispatcher's jj workspaces). Defaults mirror parallel_edit.
    gaol_binary: str = "gaol"
    runonce_runtime: str = "incus"
    runonce_image: str = "gaol-candidate-base"
    runonce_home: str = "/home/dev"
    runonce_memory: str | None = "4GiB"  # per-sandbox cap; None = uncapped
    runonce_cpus: int | None = 2
    # The sandbox NIC is NAT'd (incusbr0) and its DNS can't resolve LAN/mesh hostnames
    # (live smoke finding: opencode hung a full 30-minute timeout failing to reach the
    # router by name). Each name here is resolved ON THE HOST at run() time and injected
    # into the sandbox's /etc/hosts via run-once --add-host; unresolvable names are
    # skipped. "localhost" carries the LLM router.
    runonce_extra_hosts: list[str] = ["localhost"]
    # DHCP in a fresh container takes ~5-10s; without the readiness gate a
    # network-dependent command starts too early and hangs to its kill timeout.
    runonce_wait_network_secs: int = 30

    # Degenerate-session retry: a session that ends this fast with zero file changes is
    # an empty generation (router hiccup), not a real attempt — retry in-process up to
    # degenerate_retries times before recording the failure. Observed live (e2e dry-run):
    # a 2.8s zero-tool-call coder session burned a leaf's last attempt and escalated it.
    degenerate_session_seconds: float = 10.0
    degenerate_retries: int = 1

    # Safety
    dry_run: bool = False  # when True, run OpenCode but don't commit or update Nous

    # Projects the worker is allowed to touch (allowlist).
    # Empty = no restriction (allows any project with a worker-ready task)
    allowed_projects: list[str] = []

    # Commit message prefix
    commit_prefix: str = "auto: "

    # --- Derived properties ---

    def _storage(self):
        """Daemon-backed NousStorage (the constructor takes a client, not a data dir)."""
        from nous_mcp.daemon_client import NousDaemonClient
        from nous_mcp.storage import NousStorage

        return NousStorage(NousDaemonClient(base_url=self.daemon_url))

    @cached_property
    def notebook_id(self) -> str:
        """Resolve notebook_name to its UUID via NousStorage."""
        nb = self._storage().resolve_notebook(self.notebook_name)
        return nb["id"]

    @cached_property
    def database_id(self) -> str:
        """Resolve database_name to its database page id via NousStorage.

        Restricted to pages with pageType=='database' to disambiguate when a
        regular page shares the same title.
        """
        storage = self._storage()
        name_lower = self.database_name.lower()
        all_dbs = storage.list_database_pages(self.notebook_id)

        # Dedupe by id — list_database_pages can yield duplicates.
        seen: dict[str, dict] = {}
        for p in all_dbs:
            seen.setdefault(p["id"], p)
        db_pages = list(seen.values())

        candidates = [p for p in db_pages if p.get("title", "").lower() == name_lower]
        if not candidates:
            candidates = [p for p in db_pages if p.get("title", "").lower().startswith(name_lower)]
        if len(candidates) == 1:
            return candidates[0]["id"]
        if not candidates:
            raise ValueError(
                f"No database page titled '{self.database_name}' in notebook '{self.notebook_name}'"
            )
        titles = [c.get("title", "") for c in candidates]
        raise ValueError(f"Ambiguous database name '{self.database_name}': {titles}")


settings = TaskWorkerSettings()
