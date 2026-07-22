"""Idempotent Forge-task emission for ensemble agents.

Turns an agent's confirmed findings into Project Tasks in a Forge project, reusing
``nous_mcp``'s own task-creation path — the same logic behind the ``create_task`` MCP
tool: a page in the project folder plus a Project Tasks DB row with the standard
properties. We capture that closure with a stand-in ``mcp`` rather than re-implementing
page+row creation over raw HTTP (the prior art is ``forge/task_worker/nous_client.py``,
which delegates writes to ``nous_mcp`` the same way).

Idempotency is by a stable ``external_ref`` the caller supplies: re-running an agent
over the same code does not duplicate tasks, because every existing task's external_ref
(including Done ones) is scanned first and matches are skipped.

This is shared plumbing. The refactor ensemble is the first consumer; testing and audit
adopt it later (suggested tests -> test tasks, confirmed bugs -> bug-fix tasks) by
building their own ``EmitSpec``s and calling ``emit_tasks``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class ForgeEmitSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FORGE_EMIT_")

    daemon_url: str = "http://127.0.0.1:7667"
    notebook: str = "Forge"
    database: str = "Project Tasks"
    max_per_run: int = 25  # safety cap on tasks created in a single emission


settings = ForgeEmitSettings()


# --- data shapes ------------------------------------------------------------


@dataclass(frozen=True)
class EmitSpec:
    """One task an agent wants emitted. Agents build these from confirmed findings.

    ``external_ref`` is the idempotency key and MUST be stable across runs — derive it
    from durable attributes (location, finding type), never from a run-assigned id.

    The gating fields (``status``/``execution_mode``/``phase``/``priority``) are per-spec
    overrides: when None the batch-level defaults passed to :func:`emit_tasks` apply, so
    ensemble consumers that gate a whole batch uniformly are unaffected. The guardrail
    fields (``max_files``/``requires_tests``/``model_tier``/``depends_on``) exist only
    per-spec — the coding pipeline's architect emits every leaf with its own autonomy
    tags and dependency list.
    """

    title: str
    content: str
    external_ref: str
    task_type: str = "chore"  # bug-fix | feature | refactor | docs | test | chore
    estimate: str | None = None  # xs | s | m | l | xl
    complexity: str | None = None  # routine | novel
    feature: str | None = None
    tags: str | None = None  # comma-separated extras; "task" + project auto-added
    status: str | None = None  # per-spec override of the batch status
    execution_mode: str | None = None  # per-spec override of the batch execution_mode
    phase: str | None = None  # per-spec override of the batch phase
    priority: int | None = None  # per-spec override of the batch priority
    max_files: int | None = None  # worker diff-sprawl guardrail
    requires_tests: bool | None = None  # worker must green tests before Done
    model_tier: str | None = None  # auto | auto-free | auto-full
    depends_on: str | None = None  # comma-separated task names (names must be comma-free)


@dataclass(frozen=True)
class EmitOutcome:
    external_ref: str
    title: str
    action: Literal["created", "skipped", "dry-run"]
    detail: str = ""  # page_id when created, reason when skipped


@dataclass
class EmitSummary:
    project: str
    created: list[EmitOutcome] = field(default_factory=list)
    skipped: list[EmitOutcome] = field(default_factory=list)
    planned: list[EmitOutcome] = field(default_factory=list)  # dry-run "would create"
    capped: int = 0  # how many were dropped by the per-run cap

    def line(self) -> str:
        return (
            f"emitted {len(self.created)} / planned {len(self.planned)} / "
            f"skipped(dedup) {len(self.skipped)} / capped {self.capped} "
            f"into project '{self.project}'"
        )


# --- nous_mcp task-creation reuse -------------------------------------------


class _CaptureMCP:
    """Stand-in for FastMCP whose ``@tool()`` decorator just records the function.

    Lets us grab ``nous_mcp``'s real ``create_task`` closure (wired to our storage /
    daemon) without registering a server or duplicating its body.
    """

    def __init__(self) -> None:
        self.tools: dict[str, Callable] = {}

    def tool(self, *_args, **_kwargs) -> Callable[[Callable], Callable]:
        def deco(fn: Callable) -> Callable:
            self.tools[fn.__name__] = fn
            return fn

        return deco


@dataclass(frozen=True)
class _NousCtx:
    create_task: Callable[..., str]
    create_project: Callable[..., str]
    storage: object
    notebook_id: str
    database_id: str


@lru_cache(maxsize=1)
def _ctx() -> _NousCtx:
    """Build (once) the nous_mcp create_task closure plus resolved notebook/database ids.

    Touches the daemon to resolve the database page, so tests patch ``_create_task`` and
    ``existing_external_refs`` rather than calling this.
    """
    from nous_mcp.daemon_client import NousDaemonClient
    from nous_mcp.storage import NousStorage
    from nous_mcp.workflow import register_workflow_tools

    daemon = NousDaemonClient(base_url=settings.daemon_url)
    storage = NousStorage(daemon)  # daemon-backed; reads/writes go through the daemon
    stub = _CaptureMCP()
    register_workflow_tools(stub, lambda: storage, lambda: daemon, lambda: True)

    nb = storage.resolve_notebook(settings.notebook)
    db_page = daemon.resolve_page(nb["id"], settings.database)
    return _NousCtx(
        create_task=stub.tools["create_task"],
        create_project=stub.tools["create_project"],
        storage=storage,
        notebook_id=nb["id"],
        database_id=db_page["id"],
    )


def _create_task(**kwargs) -> str:
    """Indirection over the captured nous_mcp create_task (so tests can patch it)."""
    return _ctx().create_task(**kwargs)


def ensure_project(project: str) -> None:
    """Ensure a Forge *project* exists (folder + task-DB select option) before emitting into it.
    Idempotent — a no-op when the project is already there. ``create_task`` requires the project to
    pre-exist, so callers emitting into a fresh project (e.g. radar trials) call this first."""
    _ctx().create_project(name=project, notebook=settings.notebook, database=settings.database)


def existing_external_refs() -> set[str]:
    """Every external_ref currently in the Project Tasks DB, including Done tasks.

    The dedup source: a finding whose stable external_ref is already here is not
    re-emitted, even if the prior task was completed and closed.
    """
    from nous_mcp.workflow import _query_tasks

    ctx = _ctx()
    db_content = ctx.storage.read_database_content(ctx.notebook_id, ctx.database_id)
    if not db_content:
        return set()
    tasks = _query_tasks(db_content, include_done=True)
    return {ref.strip() for t in tasks if (ref := str(t.get("external_ref") or "")).strip()}


# --- emission ---------------------------------------------------------------


def emit_task(
    *,
    project: str,
    title: str,
    content: str,
    external_ref: str,
    task_type: str = "chore",
    status: str = "Spec Needed",
    execution_mode: str = "Manual",
    phase: str = "Polish",
    priority: int = 6,
    estimate: str | None = None,
    complexity: str | None = None,
    feature: str | None = None,
    tags: str | None = None,
    max_files: int | None = None,
    requires_tests: bool | None = None,
    model_tier: str | None = None,
    depends_on: str | None = None,
    dry_run: bool = False,
    existing_refs: set[str] | None = None,
) -> EmitOutcome:
    """Create one Forge task, idempotently by ``external_ref``.

    Skips (no write) when ``external_ref`` already exists. Defaults gate the task away
    from the autonomous worker: ``status='Spec Needed'`` + ``execution_mode='Manual'``
    means a human reviews the AI proposal and flips it to Ready (and optionally
    Auto-OK) before the worker can implement it.

    Pass a prefetched ``existing_refs`` set for batch emission; it is mutated to include
    newly created refs so repeated calls within one run also dedup.
    """
    ref = external_ref.strip()
    refs = existing_refs if existing_refs is not None else existing_external_refs()
    if ref in refs:
        return EmitOutcome(ref, title, "skipped", "external_ref exists")
    if dry_run:
        if existing_refs is not None:
            existing_refs.add(ref)
        return EmitOutcome(ref, title, "dry-run", "would create")

    raw = _create_task(
        project=project,
        title=title,
        content=content,
        status=status,
        phase=phase,
        priority=priority,
        feature=feature,
        external_ref=ref,
        tags=tags,
        execution_mode=execution_mode,
        estimate=estimate,
        complexity=complexity,
        task_type=task_type,
        max_files=max_files,
        requires_tests=requires_tests,
        model_tier=model_tier,
        depends_on=depends_on,
    )
    if existing_refs is not None:
        existing_refs.add(ref)
    try:
        data = json.loads(raw)
        detail = str(data.get("page_id") or "")
    except (json.JSONDecodeError, TypeError):
        detail = ""
    return EmitOutcome(ref, title, "created", detail)


def emit_tasks(
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
    """Emit a batch of specs into one Forge project, deduped and capped.

    Reads the existing external_refs once, then for each spec: skips dedup hits, drops
    overflow past the per-run cap (counted, not silently truncated), else creates (or,
    in dry-run, plans) the task. The cap applies only to would-create tasks — dedup
    skips are free. A spec's own gating fields, when set, override the batch-level
    ``status``/``execution_mode``/``phase``/``priority`` for that spec only.
    """
    refs = existing_external_refs()
    cap = max_per_run if max_per_run is not None else settings.max_per_run
    summary = EmitSummary(project=project)
    creations = 0
    for spec in specs:
        ref = spec.external_ref.strip()
        if ref in refs:
            summary.skipped.append(EmitOutcome(ref, spec.title, "skipped", "external_ref exists"))
            if log:
                log(f"skip (exists): {spec.title}")
            continue
        if creations >= cap:
            summary.capped += 1
            if log:
                log(f"capped (>{cap}): {spec.title}")
            continue
        outcome = emit_task(
            project=project,
            title=spec.title,
            content=spec.content,
            external_ref=ref,
            task_type=spec.task_type,
            status=spec.status if spec.status is not None else status,
            execution_mode=(
                spec.execution_mode if spec.execution_mode is not None else execution_mode
            ),
            phase=spec.phase if spec.phase is not None else phase,
            priority=spec.priority if spec.priority is not None else priority,
            estimate=spec.estimate,
            complexity=spec.complexity,
            feature=spec.feature,
            tags=spec.tags,
            max_files=spec.max_files,
            requires_tests=spec.requires_tests,
            model_tier=spec.model_tier,
            depends_on=spec.depends_on,
            dry_run=dry_run,
            existing_refs=refs,
        )
        creations += 1
        if outcome.action == "created":
            summary.created.append(outcome)
            if log:
                log(f"created: {spec.title} -> {outcome.detail}")
        else:  # dry-run
            summary.planned.append(outcome)
            if log:
                log(f"would create: {spec.title}")
    return summary
