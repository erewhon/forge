"""Interactions with the Nous notebook: find tasks, read specs, update status.

Uses:
- nous_mcp.storage.NousStorage (daemon-backed) for reads (databases, pages).
- nous_mcp.daemon_client.NousDaemonClient for HTTP writes (tag/row updates,
  page appends). The daemon exposes the same endpoints we'd otherwise have
  to re-implement, so we delegate to it.
- nous_mcp.workflow._query_tasks / _find_task_row for the shared filter logic.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from nous_mcp.daemon_client import NousDaemonClient
from nous_mcp.markdown import export_page_to_markdown
from nous_mcp.storage import NousStorage
from nous_mcp.workflow import (
    ALL_STATUS_TAGS,
    STATUS_TAG_MAP,
    _find_task_row,
    _is_task_blocked,
    _query_tasks,
)

from agents.task_worker.config import settings
from agents.task_worker.models import TaskInfo


def _get_storage() -> NousStorage:
    # NousStorage is daemon-backed now (the constructor takes a client, not a data dir).
    return NousStorage(_get_daemon())


def _get_daemon() -> NousDaemonClient:
    return NousDaemonClient(base_url=settings.daemon_url)


def _read_db_content() -> dict:
    """Read the Project Tasks database content dict from disk."""
    storage = _get_storage()
    content = storage.read_database_content(settings.notebook_id, settings.database_id)
    if content is None:
        raise RuntimeError(
            f"Database '{settings.database_name}' not found in notebook '{settings.notebook_name}'"
        )
    return content


def _requires_tests_flag(raw: str) -> bool:
    """Coerce the Requires Tests cell label to a bool.

    null-as-true: if the label is missing or empty, treat as True (safer).
    """
    if not raw:
        return True
    return raw.strip().lower() in {"yes", "true", "y", "1"}


def _execution_mode_rank(mode: str) -> int:
    """Sort key: lower is higher priority. Auto-Preferred before Auto-OK."""
    m = (mode or "").lower()
    if m == "auto-preferred":
        return 0
    if m == "auto-ok":
        return 1
    return 2


def _task_from_raw(raw: dict) -> TaskInfo:
    """Convert a _query_tasks row dict into a TaskInfo."""
    return TaskInfo(
        id=str(raw.get("id", "")),
        task=raw["task"],
        project=raw["project"],
        status=raw["status"],
        priority=int(raw.get("priority", 99)),
        execution_mode=raw.get("execution_mode", "Manual"),
        model_tier=raw.get("model_tier") or "auto",
        estimate=raw.get("estimate") or "",
        complexity=raw.get("complexity") or "",
        task_type=raw.get("task_type") or "",
        max_files=raw.get("max_files"),
        requires_tests=_requires_tests_flag(raw.get("requires_tests", "")),
        deps=list(raw.get("deps", [])),
    )


def _normalize_task_name(task_name: str) -> str:
    name = task_name.strip()
    if name.lower().startswith("task: "):
        name = name[6:].strip()
    return name


def _find_raw_task(db_content: dict, task_name: str) -> dict | None:
    """Find a task row dict (any status, including Done) by case-insensitive title match."""
    name = _normalize_task_name(task_name).lower()
    for raw in _query_tasks(db_content, include_done=True, limit=None):
        if str(raw.get("task", "")).strip().lower() == name:
            return raw
    return None


def find_task(task_name: str) -> TaskInfo | None:
    """Resolve a named task (any status) to a TaskInfo, or None if it doesn't exist.

    For callers that dispatch a *specific* task (the coding pipeline's dispatcher) rather than
    picking the next worker-ready one. Resolution alone grants nothing: ``run_one`` re-checks
    the worker gate itself before touching anything.
    """
    raw = _find_raw_task(_read_db_content(), task_name)
    return _task_from_raw(raw) if raw is not None else None


def _gate_reason(status: str, execution_mode: str, blocking: list[str]) -> str:
    """Pure worker-gate decision: '' when Ready AND Auto AND unblocked, else the refusal reason.

    Null execution_mode is Manual (null-as-manual, same as the query filters).
    """
    if status.strip().lower() != "ready":
        return f"status is '{status}', not Ready"
    mode = (execution_mode or "Manual").strip()
    if mode.lower() not in {"auto-ok", "auto-preferred"}:
        return f"execution_mode is '{mode}', not Auto-OK/Auto-Preferred"
    if blocking:
        return f"blocked by unmet dependencies: {', '.join(blocking)}"
    return ""


def check_worker_gate(task_name: str) -> str:
    """Fresh-read worker-gate check for one named task. '' = allowed; else the refusal reason.

    Reads the database again rather than trusting the caller's TaskInfo, so a task retagged
    or blocked after selection is refused.
    """
    db_content = _read_db_content()
    raw = _find_raw_task(db_content, task_name)
    if raw is None:
        return f"task '{_normalize_task_name(task_name)}' not found in the tasks database"
    _, blocking = _is_task_blocked(raw, db_content)
    return _gate_reason(str(raw.get("status", "")), str(raw.get("execution_mode", "")), blocking)


def find_next_task(allowed_projects: list[str]) -> TaskInfo | None:
    """Find the highest-priority worker-ready task.

    Applies the shared ``_query_tasks(worker_ready=True)`` filter — which already
    enforces status=Ready, execution_mode IN (Auto-OK, Auto-Preferred), blocked=False.
    Then narrows to ``allowed_projects`` if non-empty, and sorts by:
      1. Execution mode: Auto-Preferred first
      2. Priority (lower integer = higher)
      3. Task name (stable tiebreaker)
    """
    db_content = _read_db_content()

    # worker_ready=True applies the implicit filters. _query_tasks does not
    # run the blocked post-filter itself (that happens upstream in the MCP
    # tool), so we replicate it here.
    raw_tasks = _query_tasks(db_content, worker_ready=True, limit=None)
    unblocked: list[dict] = []
    for t in raw_tasks:
        is_blocked, _ = _is_task_blocked(t, db_content)
        if not is_blocked:
            unblocked.append(t)

    if allowed_projects:
        allowed_lower = {p.lower() for p in allowed_projects}
        unblocked = [t for t in unblocked if t["project"].lower() in allowed_lower]

    if not unblocked:
        return None

    unblocked.sort(
        key=lambda t: (
            _execution_mode_rank(t.get("execution_mode", "")),
            int(t.get("priority", 99)),
            t.get("task", ""),
        )
    )

    return _task_from_raw(unblocked[0])


def get_task_spec(task_name: str) -> str:
    """Build the full task spec as markdown.

    Replicates the logic of ``nous_mcp.workflow.get_task_spec`` without having
    to call the MCP tool closure.
    """
    from nous_mcp.workflow import (
        _get_row_status,
        _parse_depends_on,
        _resolve_dep_row,
    )

    storage = _get_storage()
    daemon = _get_daemon()
    notebook_id = settings.notebook_id

    # Normalize task name
    name = task_name.strip()
    if name.lower().startswith("task: "):
        name = name[6:].strip()

    # Resolve page — try "Task: X" first, then bare name
    page_title = f"Task: {name}"
    try:
        page = daemon.resolve_page(notebook_id, page_title)
    except Exception:
        try:
            page = daemon.resolve_page(notebook_id, name)
        except Exception as e:
            raise ValueError(f"Task page not found: tried '{page_title}' and '{name}'") from e

    page_content = export_page_to_markdown(page)

    # Pull database row for metadata
    db_content = storage.read_database_content(notebook_id, settings.database_id)
    row = None
    prop_map: dict = {}
    if db_content:
        row, _, prop_map = _find_task_row(db_content, name)

    def _label(prop_name: str) -> str:
        prop = prop_map.get(prop_name)
        if not prop or not row:
            return ""
        cell = row.get("cells", {}).get(prop["id"], "")
        if not cell:
            return ""
        options = {o["id"]: o["label"] for o in prop.get("options", [])}
        return options.get(cell, str(cell))

    project_name = "Unknown"
    status = "Unknown"
    priority = "—"
    phase = "—"
    external_ref = "None"
    max_files_str = ""

    if row and db_content:
        status = _get_row_status(row, db_content)
        cells = row.get("cells", {})

        if "project" in prop_map:
            p = prop_map["project"]
            cell = cells.get(p["id"], "")
            opts = {o["id"]: o["label"] for o in p.get("options", [])}
            project_name = opts.get(cell, str(cell) if cell else "Unknown")

        if "priority" in prop_map:
            cell = cells.get(prop_map["priority"]["id"], "")
            priority = str(cell) if cell else "—"

        if "phase" in prop_map:
            p = prop_map["phase"]
            cell = cells.get(p["id"], "")
            opts = {o["id"]: o["label"] for o in p.get("options", [])}
            phase = opts.get(cell, str(cell) if cell else "—")

        if "external ref" in prop_map:
            cell = cells.get(prop_map["external ref"]["id"], "")
            if cell:
                external_ref = str(cell)

        if "max files" in prop_map:
            cell = cells.get(prop_map["max files"]["id"], "")
            if cell not in ("", None):
                max_files_str = str(cell)

    exec_mode = _label("execution mode")
    model_tier_val = _label("model tier")
    estimate_val = _label("estimate")
    complexity_val = _label("complexity")
    task_type_val = _label("task type")
    requires_tests_val = _label("requires tests")

    # Dependency status
    dep_section = "None"
    blocking: list[str] = []
    if row and db_content:
        depends_on_cell = ""
        if "depends on" in prop_map:
            p = prop_map["depends on"]
            depends_on_cell = row.get("cells", {}).get(p["id"], "")
        parsed = _parse_depends_on(depends_on_cell)
        if parsed:
            dep_parts: list[str] = []
            for uid, dep_name in parsed:
                dep_row = _resolve_dep_row(db_content, uid, dep_name)
                if dep_row:
                    dep_status = _get_row_status(dep_row, db_content)
                    satisfied = dep_status.lower() == "done"
                    marker = "done" if satisfied else f"**{dep_status}**"
                    dep_parts.append(f"- {dep_name}: {marker}")
                    if not satisfied:
                        blocking.append(dep_name)
                else:
                    dep_parts.append(f"- {dep_name}: **Not Found**")
                    blocking.append(dep_name)
            dep_section = "\n".join(dep_parts)

    # Guardrails
    guardrails: list[str] = []
    if status.lower() == "done":
        guardrails.append("> **Note:** This task is already marked Done.")
    elif status.lower() == "in progress":
        guardrails.append(
            "> **Warning:** This task is already In Progress — another agent may be working on it."
        )
    if blocking:
        guardrails.append(f"> **Blocked:** Dependencies not yet Done: {', '.join(blocking)}")

    parts: list[str] = []
    if guardrails:
        parts.append("\n".join(guardrails))
        parts.append("")

    parts.append("## Task Metadata")
    parts.append(f"- **Project:** {project_name}")
    parts.append(f"- **Status:** {status}")
    parts.append(f"- **Priority:** {priority}")
    parts.append(f"- **Phase:** {phase}")
    parts.append(f"- **External Ref:** {external_ref}")
    parts.append(
        f"- **Execution Mode:** {exec_mode}"
        if exec_mode
        else "- **Execution Mode:** Manual (default)"
    )
    if model_tier_val:
        parts.append(f"- **Model Tier:** {model_tier_val}")
    if estimate_val:
        parts.append(f"- **Estimate:** {estimate_val}")
    if complexity_val:
        parts.append(f"- **Complexity:** {complexity_val}")
    if task_type_val:
        parts.append(f"- **Task Type:** {task_type_val}")
    if max_files_str:
        parts.append(f"- **Max Files:** {max_files_str}")
    if requires_tests_val:
        parts.append(f"- **Requires Tests:** {requires_tests_val}")
    parts.append("- **Dependencies:**")
    if dep_section == "None":
        parts.append("  None")
    else:
        for line in dep_section.split("\n"):
            parts.append(f"  {line}")

    parts.append("")
    parts.append("---")
    parts.append("")
    parts.append(page_content)

    return "\n".join(parts)


def _resolve_task_page(daemon: NousDaemonClient, task_name: str) -> dict:
    """Resolve a task page by "Task: X" title or bare name."""
    page_title = f"Task: {task_name}"
    try:
        return daemon.resolve_page(settings.notebook_id, page_title)
    except Exception:
        return daemon.resolve_page(settings.notebook_id, task_name)


def update_task_status(
    task: str, status: str, notes: str = "", execution_mode: str | None = None
) -> None:
    """Update a task's status in both the page tags and the database row.

    Optionally appends an Implementation Notes block to the task page.
    When setting to Done, auto-sets the Completed cell to today.
    ``execution_mode`` additionally updates the autonomy gate cell — the
    pipeline's escalation path flips it to Manual so a human re-arming the
    task doesn't silently re-enter the auto pool.
    """
    from nous_mcp.markdown import markdown_to_blocks

    daemon = _get_daemon()
    notebook_id = settings.notebook_id

    name = task.strip()
    if name.lower().startswith("task: "):
        name = name[6:].strip()

    # 1. Update page tags
    page = _resolve_task_page(daemon, name)
    current_tags: list[str] = page.get("tags", []) or []
    new_tags = [t for t in current_tags if t.lower() not in ALL_STATUS_TAGS]
    new_status_tag = STATUS_TAG_MAP.get(status.lower(), status.lower().replace(" ", "-"))
    new_tags.append(new_status_tag)
    daemon.update_page(notebook_id, page["id"], tags=new_tags)

    # 2. Update database row
    db_content = _read_db_content()
    row, _, _ = _find_task_row(db_content, name)
    if row is not None:
        update_cells: dict[str, Any] = {"Status": status}
        if status.lower() == "done":
            update_cells["Completed"] = date.today().isoformat()
        if execution_mode is not None:
            update_cells["Execution Mode"] = execution_mode
        daemon.update_database_rows(
            notebook_id,
            settings.database_id,
            [{"row": row["id"], "cells": update_cells}],
        )

    # 3. Append implementation notes
    if notes:
        today_iso = date.today().isoformat()
        notes_md = f"\n\n## Implementation Notes\n\n### {today_iso} — Status: {status}\n\n{notes}\n"
        blocks = markdown_to_blocks(notes_md)
        daemon.append_to_page(notebook_id, page["id"], blocks=blocks)
