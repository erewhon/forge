"""The task-store port — the pipeline's single seam onto its task backend.

The coding pipeline and the task worker read, emit, and update tasks through one
``TaskStore`` Protocol, so the backend is swappable without touching any orchestration
logic. ``ForgeTaskStore`` (the default) delegates to the existing Nous-backed functions
unchanged; a ``GitHubTaskStore`` (issues) is the planned second adapter for running the
harness at work. ``get_task_store()`` selects one by config (env ``TASK_STORE_BACKEND``).

The surface is small and closed — seven operations cover every Forge touch the harness
makes:

    write:  emit, update_status
    read:   find_task, next_ready, get_spec, worker_gate, list_rows, in_progress_titles

Layering note: this module lives in ``forge.shared`` because both the pipeline and the
worker depend on it, but ``ForgeTaskStore`` is *definitionally* the Nous adapter, so it
does function-local imports of the concrete Forge implementation
(``task_worker.nous_client``, ``shared.forge_emit``, ``nous_mcp``, and the tested
``waves`` row normalizer). That lazy-import style is the house pattern those same modules
already use to keep heavy/cross deps off the import path — and here it also guarantees
there is no import cycle with the packages that consume the port. ``TaskInfo``, ``LeafRow``,
``EmitSpec``, and ``EmitSummary`` appear only in type hints (resolved lazily under
``from __future__ import annotations``).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic_settings import BaseSettings, SettingsConfigDict

if TYPE_CHECKING:
    from forge.coding_pipeline.models import LeafRow
    from forge.shared.forge_emit import EmitSpec, EmitSummary
    from forge.task_worker.models import TaskInfo


class TaskStoreSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TASK_STORE_")

    # "forge" (Nous, the default) or "github" (issues — the work-deployable adapter).
    backend: str = "forge"


settings = TaskStoreSettings()


@runtime_checkable
class TaskStore(Protocol):
    """One task backend. Adapters (Forge today, GitHub issues next) implement this."""

    # --- write ---------------------------------------------------------------

    def emit(
        self,
        specs: list[EmitSpec],
        *,
        project: str,
        status: str = "Spec Needed",
        execution_mode: str = "Manual",
        phase: str = "Polish",
        priority: int = 6,
        dry_run: bool = False,
        max_per_run: int | None = None,
        log: Callable[[str], None] | None = None,
    ) -> EmitSummary:
        """Create a batch of tasks, deduped idempotently by each spec's external_ref."""
        ...

    def update_status(
        self, task: str, status: str, notes: str = "", execution_mode: str | None = None
    ) -> None:
        """Set a task's status (and optionally its autonomy mode), appending notes."""
        ...

    # --- read: point ---------------------------------------------------------

    def find_task(self, name: str) -> TaskInfo | None:
        """Resolve a named task (any status) to a ``TaskInfo``, or None if absent."""
        ...

    def next_ready(self, projects: list[str]) -> TaskInfo | None:
        """The highest-priority worker-ready task within ``projects`` (empty = all)."""
        ...

    def get_spec(self, name: str) -> str:
        """The task's full spec as markdown (metadata header + page body)."""
        ...

    def worker_gate(self, name: str) -> str:
        """'' if the task may be worked (Ready AND Auto AND unblocked), else the reason."""
        ...

    # --- read: bulk ----------------------------------------------------------

    def list_rows(
        self, project: str, *, feature: str | None = None, include_done: bool = True
    ) -> list[LeafRow]:
        """All task rows for a project as ``LeafRow``s, blocked-state resolved."""
        ...

    def in_progress_titles(self, ref_prefix: str) -> list[str]:
        """Titles of In Progress tasks whose external_ref starts with ``ref_prefix``."""
        ...


class ForgeTaskStore:
    """The default backend: delegates to the Nous-backed implementation unchanged.

    Every method forwards to the existing free function (``task_worker.nous_client`` /
    ``shared.forge_emit`` / the ``waves`` normalizer), so behavior is identical to the
    pre-port pipeline and tests that patch those source functions keep working.
    """

    def emit(
        self,
        specs: list[EmitSpec],
        *,
        project: str,
        status: str = "Spec Needed",
        execution_mode: str = "Manual",
        phase: str = "Polish",
        priority: int = 6,
        dry_run: bool = False,
        max_per_run: int | None = None,
        log: Callable[[str], None] | None = None,
    ) -> EmitSummary:
        from forge.shared.forge_emit import emit_tasks

        return emit_tasks(
            specs,
            project=project,
            status=status,
            execution_mode=execution_mode,
            phase=phase,
            priority=priority,
            dry_run=dry_run,
            max_per_run=max_per_run,
            log=log,
        )

    def update_status(
        self, task: str, status: str, notes: str = "", execution_mode: str | None = None
    ) -> None:
        from forge.task_worker.nous_client import update_task_status

        update_task_status(task, status, notes=notes, execution_mode=execution_mode)

    def find_task(self, name: str) -> TaskInfo | None:
        from forge.task_worker.nous_client import find_task

        return find_task(name)

    def next_ready(self, projects: list[str]) -> TaskInfo | None:
        from forge.task_worker.nous_client import find_next_task

        return find_next_task(projects)

    def get_spec(self, name: str) -> str:
        from forge.task_worker.nous_client import get_task_spec

        return get_task_spec(name)

    def worker_gate(self, name: str) -> str:
        from forge.task_worker.nous_client import check_worker_gate

        return check_worker_gate(name)

    def list_rows(
        self, project: str, *, feature: str | None = None, include_done: bool = True
    ) -> list[LeafRow]:
        # Reuses the tested ``waves`` normalizer (null-as-manual, blocked resolution) so
        # there is exactly one row-building path across the pipeline.
        from nous_mcp.workflow import _query_tasks

        from forge.coding_pipeline.waves import _rows_from_raw
        from forge.task_worker.nous_client import _read_db_content

        db_content = _read_db_content()
        raw_rows = _query_tasks(
            db_content, project=project, feature=feature, include_done=include_done, limit=None
        )
        return _rows_from_raw(raw_rows, db_content)

    def in_progress_titles(self, ref_prefix: str) -> list[str]:
        from nous_mcp.workflow import _query_tasks

        from forge.task_worker.nous_client import _read_db_content

        rows = _query_tasks(_read_db_content(), status="In Progress", limit=None)
        return [
            str(r.get("task", ""))
            for r in rows
            if str(r.get("task", "")).strip()
            and str(r.get("external_ref", "") or "").startswith(ref_prefix)
        ]


def get_task_store() -> TaskStore:
    """The configured task backend. ``forge`` (default) today; ``github`` is the planned
    work-deployable adapter. Unknown values fail loudly rather than silently defaulting."""
    backend = settings.backend.strip().lower()
    if backend == "forge":
        return ForgeTaskStore()
    if backend == "github":
        from forge.shared.github_task_store import GitHubTaskStore

        return GitHubTaskStore()
    raise ValueError(f"unknown TASK_STORE_BACKEND {settings.backend!r} (known: 'forge', 'github')")
