"""GitBugTaskStore — the ``TaskStore`` adapter backed by git-bug (issues as git objects).

The third backend behind the port (``TASK_STORE_BACKEND=git-bug``). git-bug stores each
bug as a chain of git objects under ``refs/bugs/*`` in the repo itself, so tasks travel
with the repo: they ride the same HTTP-git channel used to move code onto a locked-down
work box, need no tracker API or server, work offline, and their operation history doubles
as an audit log. This is also the native task surface for Soft Serve sprinkles, whose
issues are git-bug bugs — tasks the pipeline files here render in that forge's UI.

The encoding is ``task_conventions`` verbatim (proved by GitHubTaskStore):
- **task = bug.** The bug title is the logical key; the spec markdown is the bug's
  description (first comment) below a ``pipeline-meta`` block.
- **status = label + open/closed.** Open bug + ``status:*`` label; **Done is a closed bug**.
- **deps by title** in the meta block, resolved against bug titles.

Pinned CLI surface (git-bug **v0.10.1**, verified live):
- list:   ``git-bug bug --format json`` → id/title/status/labels (NO bodies)
- show:   ``git-bug bug show <id> --format json`` → adds ``comments[]`` with ``message``
- create: ``git-bug bug new --non-interactive -t TITLE -m BODY`` (labels added after)
- labels: ``git-bug bug label new|rm <id> LABEL`` (free-form — no pre-registration, so
  there is no ``ensure_labels`` bootstrap here at all)
- status: ``git-bug bug status close|open <id>``
- notes:  ``git-bug bug comment new <id> -m MSG --non-interactive``
- meta edit: ``git-bug bug comment edit --non-interactive -m BODY <first-comment-id>``

Because list output carries no bodies, meta-dependent operations do one ``show`` per bug
(N+1). Fine at per-repo scale; revisit only if a work repo grows thousands of open bugs.

Setup: the repo needs a git-bug identity once (``git-bug user new``). Without one, write
commands fail and the CLI's stderr says exactly that — surfaced verbatim in our error.
Bug refs sync separately from code: ``git bug push`` / ``git bug pull``.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Protocol

from pydantic_settings import BaseSettings, SettingsConfigDict

from forge.coding_pipeline.models import LeafRow
from forge.queue.models import QueueRow
from forge.shared.forge_emit import EmitOutcome, EmitSpec, EmitSummary
from forge.shared.task_conventions import (
    AUTO_MODES,
    LABEL_TO_STATUS,
    STATUS_TO_LABEL,
    bool_null_true,
    format_meta_block,
    mode_rank,
    normalize_title,
    parse_int,
    parse_int_or_none,
    parse_meta_block,
    set_meta_field,
    split_deps,
    strip_meta_block,
)
from forge.task_worker.models import TaskInfo


class GitBugTaskStoreSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GIT_BUG_TASK_STORE_")

    repo_path: str = "."  # the repo whose refs/bugs/* hold the tasks
    project: str = ""  # Forge-style project name; defaults to the repo directory name
    max_per_run: int = 25  # safety cap on bugs created in one emission


settings = GitBugTaskStoreSettings()


# --- the git-bug client seam --------------------------------------------------


@dataclass
class _Bug:
    id: str  # full hash (stable key for show/label/status ops)
    title: str
    state: str  # "open" | "closed"
    labels: list[str]
    description: str = ""  # first-comment message; only populated by show_bug
    first_comment_id: str = ""  # only populated by show_bug; needed for meta edits
    comments: list[str] = field(default_factory=list)  # messages, description first


class GitBugReader(Protocol):
    """The read-only git-bug surface (least agency — mirror of ``GhReader``)."""

    def list_bugs(self) -> list[_Bug]: ...
    def show_bug(self, bug_id: str) -> _Bug: ...


class GitBugWriter(Protocol):
    """The write-capable git-bug surface — only the emit / status-update path holds one."""

    def create_bug(self, *, title: str, body: str) -> str: ...
    def add_label(self, bug_id: str, label: str) -> None: ...
    def remove_label(self, bug_id: str, label: str) -> None: ...
    def close_bug(self, bug_id: str) -> None: ...
    def open_bug(self, bug_id: str) -> None: ...
    def add_comment(self, bug_id: str, body: str) -> None: ...
    def edit_comment(self, comment_id: str, body: str) -> None: ...


class GitBugClient(GitBugReader, GitBugWriter, Protocol):
    """The full surface: both narrow interfaces (prefer handing out one or the other)."""


class SubprocessGitBugClient:
    """A ``GitBugClient`` that shells out to the ``git-bug`` CLI in one repo."""

    def __init__(self, repo_path: str | Path) -> None:
        self.repo_path = Path(repo_path)

    def _run(self, args: list[str]) -> str:
        proc = subprocess.run(
            ["git-bug", *args],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"git-bug {' '.join(args)} failed: {proc.stderr.strip()}")
        return proc.stdout

    def list_bugs(self) -> list[_Bug]:
        # No status filter: the bare listing returns open AND closed bugs (verified
        # v0.10.1); `--status all` is not a valid query value.
        raw = json.loads(self._run(["bug", "--format", "json"]) or "[]")
        return [
            _Bug(
                id=str(d["id"]),
                title=str(d["title"]),
                state=str(d["status"]).lower(),
                labels=[str(lbl) for lbl in d.get("labels") or []],
            )
            for d in raw
        ]

    def show_bug(self, bug_id: str) -> _Bug:
        d = json.loads(self._run(["bug", "show", bug_id, "--format", "json"]))
        comments = d.get("comments") or []
        return _Bug(
            id=str(d["id"]),
            title=str(d["title"]),
            state=str(d["status"]).lower(),
            labels=[str(lbl) for lbl in d.get("labels") or []],
            description=str(comments[0].get("message") or "") if comments else "",
            first_comment_id=str(comments[0].get("id") or "") if comments else "",
            comments=[str(c.get("message") or "") for c in comments],
        )

    def create_bug(self, *, title: str, body: str) -> str:
        out = self._run(["bug", "new", "--non-interactive", "-t", title, "-m", body])
        # "<shortid> created" — the short id is a valid selector for every other command.
        return out.strip().split()[0]

    def add_label(self, bug_id: str, label: str) -> None:
        self._run(["bug", "label", "new", bug_id, label])

    def remove_label(self, bug_id: str, label: str) -> None:
        self._run(["bug", "label", "rm", bug_id, label])

    def close_bug(self, bug_id: str) -> None:
        self._run(["bug", "status", "close", bug_id])

    def open_bug(self, bug_id: str) -> None:
        self._run(["bug", "status", "open", bug_id])

    def add_comment(self, bug_id: str, body: str) -> None:
        self._run(["bug", "comment", "new", bug_id, "-m", body, "--non-interactive"])

    def edit_comment(self, comment_id: str, body: str) -> None:
        self._run(["bug", "comment", "edit", "--non-interactive", "-m", body, comment_id])


class _ReadOnlyGitBugWriter:
    """Writer seat of a read-only store: writes fail fast with the reason (least agency
    is only real if the missing capability is visibly missing — same as the gh store)."""

    def __getattr__(self, name: str):
        raise PermissionError(
            f"this GitBugTaskStore is read-only (constructed with a GitBugReader and no "
            f"GitBugWriter) — write operation {name!r} is not available"
        )


# --- the store ----------------------------------------------------------------


class GitBugTaskStore:
    """``TaskStore`` over git-bug bugs in one repo. See the module docstring for the mapping.

    Read paths go through ``self._reader`` (a ``GitBugReader``); only emit and
    status updates touch ``self._writer``. Construct with ``gb=`` for both capabilities,
    or ``reader=`` alone for a read-only store whose writes fail fast."""

    def __init__(
        self,
        gb: GitBugClient | None = None,
        *,
        reader: GitBugReader | None = None,
        writer: GitBugWriter | None = None,
        project: str | None = None,
    ) -> None:
        repo_path = settings.repo_path.strip() or "."
        if gb is None and reader is None and writer is None:
            gb = SubprocessGitBugClient(repo_path)
        if gb is not None and (reader is not None or writer is not None):
            raise ValueError("pass either gb= (both capabilities) or reader=/writer=, not both")
        resolved_reader = reader if reader is not None else gb
        if resolved_reader is None:
            raise ValueError("a GitBugReader is required: every store operation reads bugs")
        self._reader: GitBugReader = resolved_reader
        self._writer: GitBugWriter = (
            writer if writer is not None else (gb or _ReadOnlyGitBugWriter())
        )
        self.project = project or settings.project.strip() or Path(repo_path).resolve().name

    # --- read helpers ----------------------------------------------------

    def _status_of(self, bug: _Bug) -> str:
        if bug.state == "closed":
            return "Done"
        for label in bug.labels:
            if label in LABEL_TO_STATUS:
                return LABEL_TO_STATUS[label]
        return "Spec Needed"  # open with no status label: not worker-ready (fail-safe)

    def _full_bugs(self) -> list[_Bug]:
        """Every bug with its description loaded (one show per bug — see module note)."""
        return [self._reader.show_bug(b.id) for b in self._reader.list_bugs()]

    def _leaf_rows(self, bugs: list[_Bug]) -> list[LeafRow]:
        done_titles = {b.title for b in bugs if self._status_of(b) == "Done"}
        rows: list[LeafRow] = []
        for bug in bugs:
            meta = parse_meta_block(bug.description)
            deps = split_deps(meta.get("depends_on", ""))
            blocking = [d for d in deps if d not in done_titles]
            rows.append(
                LeafRow(
                    task=bug.title,
                    status=self._status_of(bug),
                    execution_mode=meta.get("execution_mode") or "Manual",
                    priority=parse_int(meta.get("priority"), 99),
                    blocked=bool(blocking),
                    blocked_by=blocking,
                    external_ref=meta.get("external_ref", ""),
                )
            )
        return rows

    def _task_info(self, bug: _Bug) -> TaskInfo:
        meta = parse_meta_block(bug.description)
        return TaskInfo(
            id=bug.id,
            task=bug.title,
            project=self.project,
            status=self._status_of(bug),
            priority=parse_int(meta.get("priority"), 99),
            execution_mode=meta.get("execution_mode") or "Manual",
            model_tier=meta.get("model_tier") or "auto",
            estimate=meta.get("estimate") or "",
            complexity=meta.get("complexity") or "",
            task_type=meta.get("task_type") or "",
            max_files=parse_int_or_none(meta.get("max_files")),
            requires_tests=bool_null_true(meta.get("requires_tests")),
            deps=split_deps(meta.get("depends_on", "")),
            external_ref=meta.get("external_ref", ""),
        )

    def _find_bug(self, name: str) -> _Bug | None:
        title = normalize_title(name).lower()
        for bug in self._reader.list_bugs():
            if bug.title.strip().lower() == title:
                return self._reader.show_bug(bug.id)
        return None

    # --- write -----------------------------------------------------------

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
        log=None,
    ) -> EmitSummary:
        existing = {
            ref
            for bug in self._full_bugs()
            if (ref := parse_meta_block(bug.description).get("external_ref", "").strip())
        }
        cap = max_per_run if max_per_run is not None else settings.max_per_run
        summary = EmitSummary(project=project)
        creations = 0
        for spec in specs:
            ref = spec.external_ref.strip()
            if ref in existing:
                summary.skipped.append(
                    EmitOutcome(ref, spec.title, "skipped", "external_ref exists")
                )
                if log:
                    log(f"skip (exists): {spec.title}")
                continue
            if creations >= cap:
                summary.capped += 1
                if log:
                    log(f"capped (>{cap}): {spec.title}")
                continue
            eff_status = spec.status if spec.status is not None else status
            eff_mode = spec.execution_mode if spec.execution_mode is not None else execution_mode
            eff_phase = spec.phase if spec.phase is not None else phase
            eff_priority = spec.priority if spec.priority is not None else priority
            existing.add(ref)
            creations += 1
            if dry_run:
                summary.planned.append(EmitOutcome(ref, spec.title, "dry-run", "would create"))
                if log:
                    log(f"would create: {spec.title}")
                continue
            body = self._bug_body(spec, mode=eff_mode, phase=eff_phase, priority=eff_priority)
            bug_id = self._writer.create_bug(title=spec.title, body=body)
            label = STATUS_TO_LABEL.get(eff_status.strip().lower())
            if label:
                # `bug new` has no --label flag; the label lands as a second operation.
                self._writer.add_label(bug_id, label)
            summary.created.append(EmitOutcome(ref, spec.title, "created", bug_id))
            if log:
                log(f"created: {spec.title} -> {bug_id}")
        return summary

    def _bug_body(self, spec: EmitSpec, *, mode: str, phase: str, priority: int) -> str:
        fields: dict[str, str] = {"external_ref": spec.external_ref}
        if spec.feature:
            fields["feature"] = spec.feature
        fields["task_type"] = spec.task_type
        fields["execution_mode"] = mode
        if spec.model_tier:
            fields["model_tier"] = spec.model_tier
        fields["priority"] = str(priority)
        if spec.max_files is not None:
            fields["max_files"] = str(spec.max_files)
        if spec.requires_tests is not None:
            fields["requires_tests"] = "true" if spec.requires_tests else "false"
        if spec.estimate:
            fields["estimate"] = spec.estimate
        if spec.complexity:
            fields["complexity"] = spec.complexity
        fields["phase"] = phase
        if spec.depends_on:
            fields["depends_on"] = spec.depends_on
        return f"{format_meta_block(fields)}\n\n{spec.content}"

    def update_status(
        self, task: str, status: str, notes: str = "", execution_mode: str | None = None
    ) -> None:
        bug = self._find_bug(task)
        if bug is None:
            raise ValueError(f"bug not found for task {task!r}")
        target = status.strip()
        status_labels = [lbl for lbl in bug.labels if lbl.startswith("status:")]
        if target.lower() == "done":
            for lbl in status_labels:
                self._writer.remove_label(bug.id, lbl)
            self._writer.close_bug(bug.id)
        else:
            if bug.state == "closed":
                self._writer.open_bug(bug.id)
            new_label = STATUS_TO_LABEL.get(target.lower())
            for lbl in status_labels:
                if lbl != new_label:
                    self._writer.remove_label(bug.id, lbl)
            if new_label and new_label not in bug.labels:
                self._writer.add_label(bug.id, new_label)
        if execution_mode is not None:
            if not bug.first_comment_id:
                raise ValueError(f"bug {bug.id} has no description comment to carry meta")
            self._writer.edit_comment(
                bug.first_comment_id,
                set_meta_field(bug.description, "execution_mode", execution_mode),
            )
        if notes:
            self._writer.add_comment(
                bug.id, f"**{date.today().isoformat()} — Status: {status}**\n\n{notes}"
            )

    # --- read ------------------------------------------------------------

    def find_task(self, name: str) -> TaskInfo | None:
        bug = self._find_bug(name)
        return self._task_info(bug) if bug is not None else None

    def next_ready(self, projects: list[str]) -> TaskInfo | None:
        if projects and self.project.lower() not in {p.lower() for p in projects}:
            return None
        bugs = self._full_bugs()
        done_titles = {b.title for b in bugs if self._status_of(b) == "Done"}
        candidates: list[_Bug] = []
        for bug in bugs:
            meta = parse_meta_block(bug.description)
            mode = (meta.get("execution_mode") or "Manual").strip().lower()
            if self._status_of(bug) != "Ready" or mode not in AUTO_MODES:
                continue
            if any(d not in done_titles for d in split_deps(meta.get("depends_on", ""))):
                continue
            candidates.append(bug)
        candidates.sort(
            key=lambda b: (
                mode_rank(parse_meta_block(b.description).get("execution_mode") or "Manual"),
                parse_int(parse_meta_block(b.description).get("priority"), 99),
                b.title,
            )
        )
        return self._task_info(candidates[0]) if candidates else None

    def worker_gate(self, name: str) -> str:
        from forge.task_worker.nous_client import _gate_reason

        title = normalize_title(name)
        rows = {r.task: r for r in self._leaf_rows(self._full_bugs())}
        row = rows.get(title)
        if row is None:
            return f"task {title!r} not found in the issue tracker"
        return _gate_reason(row.status, row.execution_mode, row.blocked_by)

    def get_spec(self, name: str) -> str:
        bug = self._find_bug(name)
        if bug is None:
            raise ValueError(f"bug not found for task {name!r}")
        rows = {r.task: r for r in self._leaf_rows(self._full_bugs())}
        row = rows.get(bug.title)
        meta = parse_meta_block(bug.description)
        status = self._status_of(bug)

        parts: list[str] = []
        if status.lower() == "done":
            parts.append("> **Note:** This task is already marked Done.\n")
        elif status.lower() == "in progress":
            parts.append(
                "> **Warning:** This task is already In Progress — another agent may be on it.\n"
            )
        if row and row.blocked:
            parts.append(f"> **Blocked:** Dependencies not yet Done: {', '.join(row.blocked_by)}\n")

        parts.append("## Task Metadata")
        parts.append(f"- **Project:** {self.project}")
        parts.append(f"- **Status:** {status}")
        parts.append(f"- **Priority:** {meta.get('priority', '—')}")
        parts.append(f"- **Execution Mode:** {meta.get('execution_mode', 'Manual')}")
        for label, key in (
            ("Model Tier", "model_tier"),
            ("Estimate", "estimate"),
            ("Complexity", "complexity"),
            ("Task Type", "task_type"),
            ("Max Files", "max_files"),
            ("Requires Tests", "requires_tests"),
        ):
            if meta.get(key):
                parts.append(f"- **{label}:** {meta[key]}")
        deps = split_deps(meta.get("depends_on", ""))
        parts.append("- **Dependencies:**")
        if not deps:
            parts.append("  None")
        else:
            done_titles = {r for r, lr in rows.items() if lr.status == "Done"}
            for dep in deps:
                marker = "done" if dep in done_titles else "**not done**"
                parts.append(f"  - {dep}: {marker}")
        parts.append("\n---\n")
        parts.append(strip_meta_block(bug.description))
        return "\n".join(parts)

    def list_rows(
        self, project: str, *, feature: str | None = None, include_done: bool = True
    ) -> list[LeafRow]:
        bugs = self._full_bugs()
        rows = self._leaf_rows(bugs)
        if feature is not None:
            by_title = {b.title: parse_meta_block(b.description).get("feature", "") for b in bugs}
            rows = [r for r in rows if by_title.get(r.task, "") == feature]
        if not include_done:
            rows = [r for r in rows if r.status != "Done"]
        return rows

    def in_progress_titles(self, ref_prefix: str) -> list[str]:
        titles: list[str] = []
        for bug in self._reader.list_bugs():
            if bug.state != "open" or STATUS_TO_LABEL["in progress"] not in bug.labels:
                continue
            full = self._reader.show_bug(bug.id)
            ref = parse_meta_block(full.description).get("external_ref", "")
            if bug.title.strip() and ref.startswith(ref_prefix):
                titles.append(bug.title)
        return titles

    def queue(self, *, project: str | None = None) -> list[QueueRow]:
        # Single-project store: one repo == one project, so any other name is empty.
        if project is not None and project.lower() != self.project.lower():
            return []
        bugs = self._full_bugs()
        metas = {b.title: parse_meta_block(b.description) for b in bugs}
        return [
            QueueRow(
                project=self.project,
                task=row.task,
                status=row.status,
                execution_mode=row.execution_mode,
                priority=row.priority,
                blocked=row.blocked,
                blocked_by=row.blocked_by,
                feature=metas.get(row.task, {}).get("feature", ""),
                model_tier=metas.get(row.task, {}).get("model_tier", ""),
            )
            for row in self._leaf_rows(bugs)
            if row.status != "Done" and row.task.strip()
        ]
