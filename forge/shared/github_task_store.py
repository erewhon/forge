"""GitHubTaskStore — the ``TaskStore`` adapter backed by GitHub issues.

The second backend behind the port (``TASK_STORE_BACKEND=github``), for running the coding
harness against a work GitHub instead of Forge/Nous. Nothing above the port changes: the
architect, orchestrator, and worker keep keying tasks by title and deps by name — which is
exact parity with the Forge backend, so this adapter is a translation layer, not a redesign.

The encoding (decided with the user):
- **task = issue.** The issue title is the logical key (exact-title resolution); the spec
  markdown is the issue body *below* a metadata block.
- **status = label + open/closed.** ``Spec Needed`` / ``Ready`` / ``In Progress`` are an open
  issue with a ``status:*`` label; **Done is a closed issue**.
- **everything else = a ``pipeline-meta`` block** in the body (execution_mode, model_tier,
  priority, max_files, requires_tests, task_type, complexity, estimate, feature, external_ref,
  and ``depends_on`` as comma-separated names). No GitHub Projects setup required.
- **deps by name** (portable body-convention), resolved by title lookup exactly as Forge
  resolves ``Depends On`` names against the tasks DB — batch-emitted issues have stable names
  before they have numbers, so name-based deps need no emit ordering.

Transport is the ``gh`` CLI (ambient auth, works in a locked-down env), wrapped behind the
``GhClient`` seam so the store is unit-testable against an in-memory fake.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import date
from typing import Protocol

from pydantic_settings import BaseSettings, SettingsConfigDict

from forge.coding_pipeline.models import LeafRow
from forge.queue.models import QueueRow
from forge.shared.forge_emit import EmitOutcome, EmitSpec, EmitSummary

# The portable encoding (meta block, status labels, deps-by-title) lives in
# task_conventions so GitBugTaskStore shares it verbatim. Names are re-exported
# here because this module defined them first.
from forge.shared.task_conventions import (
    AUTO_MODES as _AUTO_MODES,
)
from forge.shared.task_conventions import (
    LABEL_TO_STATUS as _LABEL_TO_STATUS,
)
from forge.shared.task_conventions import (
    STATUS_TO_LABEL as _STATUS_TO_LABEL,
)
from forge.shared.task_conventions import (
    bool_null_true as _bool_null_true,
)
from forge.shared.task_conventions import (
    format_meta_block as format_meta_block,
)
from forge.shared.task_conventions import (
    mode_rank as _mode_rank,
)
from forge.shared.task_conventions import (
    normalize_title as _normalize_title,
)
from forge.shared.task_conventions import (
    parse_int as _int,
)
from forge.shared.task_conventions import (
    parse_int_or_none as _int_or_none,
)
from forge.shared.task_conventions import (
    parse_meta_block as parse_meta_block,
)
from forge.shared.task_conventions import (
    set_meta_field as set_meta_field,
)
from forge.shared.task_conventions import (
    split_deps as _split_deps,
)
from forge.shared.task_conventions import (
    strip_meta_block as strip_meta_block,
)
from forge.task_worker.models import TaskInfo

# color (hex, no '#') + description for each status label; created idempotently by
# ``ensure_labels`` because ``gh issue create --label`` fails on a label the repo lacks.
_STATUS_LABEL_STYLE = {
    "status:spec-needed": ("ededed", "Pipeline: spec needed (a human must ready it)"),
    "status:ready": ("0e8a16", "Pipeline: ready for the worker"),
    "status:in-progress": ("fbca04", "Pipeline: in progress"),
    "status:done": ("1d76db", "Pipeline: done"),
}

# Ordered keys for a stable, diff-friendly meta block.
_META_KEYS = (
    "external_ref",
    "feature",
    "task_type",
    "execution_mode",
    "model_tier",
    "priority",
    "max_files",
    "requires_tests",
    "estimate",
    "complexity",
    "phase",
    "depends_on",
)
_META_START = "<!-- pipeline-meta"
_META_END = "-->"


class GitHubTaskStoreSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GITHUB_TASK_STORE_")

    repo: str = ""  # "owner/repo" — required
    project: str = ""  # Forge-style project name this repo maps to; defaults to the repo name
    max_per_run: int = 25  # safety cap on issues created in one emission
    list_limit: int = 1000  # --limit for `gh issue list`


settings = GitHubTaskStoreSettings()


# --- the gh client seam -----------------------------------------------------


@dataclass
class _Issue:
    number: int
    title: str
    body: str
    state: str  # "open" | "closed"
    labels: list[str]


class GhReader(Protocol):
    """The read-only GitHub surface, bound to one repo — everything a component that only
    queries issues needs. Kept disjoint from ``GhWriter`` (least agency, Zero Trust Part IV
    Phase 5): a read-path consumer handed a ``GhReader`` structurally cannot close, relabel,
    or comment on issues, because the capability is not in its hands."""

    def list_issues(self, *, state: str, label: str | None = None) -> list[_Issue]: ...


class GhWriter(Protocol):
    """The write-capable GitHub surface — only the emit / status-update path holds one.
    Deliberately does NOT extend ``GhReader``: a writer is never silently handed read surface
    it didn't ask for, and vice versa."""

    def create_issue(self, *, title: str, body: str, labels: list[str]) -> int: ...
    def edit_labels(self, number: int, *, add: list[str], remove: list[str]) -> None: ...
    def edit_body(self, number: int, body: str) -> None: ...
    def close_issue(self, number: int) -> None: ...
    def reopen_issue(self, number: int) -> None: ...
    def comment(self, number: int, body: str) -> None: ...
    def ensure_label(self, name: str, *, color: str, description: str) -> None: ...


class GhClient(GhReader, GhWriter, Protocol):
    """The full GitHub surface: both narrow interfaces. The real impl shells out to ``gh``;
    tests supply an in-memory fake. Prefer typing collaborators as ``GhReader`` or
    ``GhWriter`` — this combined Protocol exists for the store itself and for callers that
    genuinely hold both capabilities."""


class SubprocessGhClient:
    """A ``GhClient`` that shells out to the ``gh`` CLI against one ``owner/repo``."""

    def __init__(self, repo: str, *, list_limit: int = 1000) -> None:
        self.repo = repo
        self.list_limit = list_limit

    def _run(self, args: list[str]) -> str:
        proc = subprocess.run(
            ["gh", *args, "--repo", self.repo],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"gh {' '.join(args)} failed: {proc.stderr.strip()}")
        return proc.stdout

    def list_issues(self, *, state: str, label: str | None = None) -> list[_Issue]:
        args = [
            "issue",
            "list",
            "--state",
            state,
            "--json",
            "number,title,body,state,labels",
            "--limit",
            str(self.list_limit),
        ]
        if label:
            args += ["--label", label]
        raw = json.loads(self._run(args) or "[]")
        return [
            _Issue(
                number=int(d["number"]),
                title=str(d["title"]),
                body=str(d.get("body") or ""),
                state=str(d["state"]).lower(),
                labels=[lbl["name"] for lbl in d.get("labels", [])],
            )
            for d in raw
        ]

    def create_issue(self, *, title: str, body: str, labels: list[str]) -> int:
        args = ["issue", "create", "--title", title, "--body", body]
        for label in labels:
            args += ["--label", label]
        url = self._run(args).strip().splitlines()[-1]
        return int(url.rsplit("/", 1)[-1])

    def edit_labels(self, number: int, *, add: list[str], remove: list[str]) -> None:
        if not add and not remove:
            return
        args = ["issue", "edit", str(number)]
        for label in add:
            args += ["--add-label", label]
        for label in remove:
            args += ["--remove-label", label]
        self._run(args)

    def edit_body(self, number: int, body: str) -> None:
        self._run(["issue", "edit", str(number), "--body", body])

    def close_issue(self, number: int) -> None:
        self._run(["issue", "close", str(number)])

    def reopen_issue(self, number: int) -> None:
        self._run(["issue", "reopen", str(number)])

    def comment(self, number: int, body: str) -> None:
        self._run(["issue", "comment", str(number), "--body", body])

    def ensure_label(self, name: str, *, color: str, description: str) -> None:
        # --force makes it idempotent: create if absent, update in place if present.
        self._run(
            ["label", "create", name, "--color", color, "--description", description, "--force"]
        )


# --- the store --------------------------------------------------------------


class _ReadOnlyWriter:
    """The writer seat of a store constructed without write capability: every write fails
    fast with the reason, instead of an AttributeError deep in a `gh` call. Least agency is
    only real if the missing capability is *visibly* missing."""

    def __getattr__(self, name: str):
        raise PermissionError(
            f"this GitHubTaskStore is read-only (constructed with a GhReader and no "
            f"GhWriter) — write operation {name!r} is not available"
        )


class GitHubTaskStore:
    """``TaskStore`` over GitHub issues in one repo. See the module docstring for the mapping.

    The store holds both capability seams but wires each method to the narrowest one:
    read paths go through ``self._reader`` (a ``GhReader``), the emit / status-update path
    through ``self._writer`` (a ``GhWriter``). Construct with ``gh=`` for both capabilities
    (the default ``SubprocessGhClient`` satisfies both), or with ``reader=`` alone for a
    read-only store whose write methods fail fast with ``PermissionError``."""

    def __init__(
        self,
        gh: GhClient | None = None,
        *,
        reader: GhReader | None = None,
        writer: GhWriter | None = None,
        project: str | None = None,
    ) -> None:
        repo = settings.repo.strip()
        if gh is None and reader is None and writer is None:
            if not repo:
                raise ValueError(
                    "GITHUB_TASK_STORE_REPO must be set (owner/repo) for the github backend"
                )
            gh = SubprocessGhClient(repo, list_limit=settings.list_limit)
        if gh is not None and (reader is not None or writer is not None):
            raise ValueError("pass either gh= (both capabilities) or reader=/writer=, not both")
        resolved_reader = reader if reader is not None else gh
        if resolved_reader is None:
            raise ValueError("a GhReader is required: every store operation reads issues")
        self._reader: GhReader = resolved_reader
        self._writer: GhWriter = writer if writer is not None else (gh or _ReadOnlyWriter())
        self.project = project or settings.project.strip() or (repo.split("/")[-1] if repo else "")

    # --- read helpers ----------------------------------------------------

    def _status_of(self, issue: _Issue) -> str:
        if issue.state == "closed":
            return "Done"
        for label in issue.labels:
            if label in _LABEL_TO_STATUS:
                return _LABEL_TO_STATUS[label]
        return "Spec Needed"  # open with no status label: not worker-ready (fail-safe)

    def _leaf_rows(self, issues: list[_Issue]) -> list[LeafRow]:
        done_titles = {i.title for i in issues if self._status_of(i) == "Done"}
        rows: list[LeafRow] = []
        for issue in issues:
            meta = parse_meta_block(issue.body)
            deps = _split_deps(meta.get("depends_on", ""))
            blocking = [d for d in deps if d not in done_titles]
            rows.append(
                LeafRow(
                    task=issue.title,
                    status=self._status_of(issue),
                    execution_mode=meta.get("execution_mode") or "Manual",
                    priority=_int(meta.get("priority"), 99),
                    blocked=bool(blocking),
                    blocked_by=blocking,
                    external_ref=meta.get("external_ref", ""),
                )
            )
        return rows

    def _task_info(self, issue: _Issue) -> TaskInfo:
        meta = parse_meta_block(issue.body)
        return TaskInfo(
            id=str(issue.number),
            task=issue.title,
            project=self.project,
            status=self._status_of(issue),
            priority=_int(meta.get("priority"), 99),
            execution_mode=meta.get("execution_mode") or "Manual",
            model_tier=meta.get("model_tier") or "auto",
            estimate=meta.get("estimate") or "",
            complexity=meta.get("complexity") or "",
            task_type=meta.get("task_type") or "",
            max_files=_int_or_none(meta.get("max_files")),
            requires_tests=_bool_null_true(meta.get("requires_tests")),
            deps=_split_deps(meta.get("depends_on", "")),
        )

    def _find_issue(self, name: str) -> _Issue | None:
        title = _normalize_title(name).lower()
        for issue in self._reader.list_issues(state="all"):
            if issue.title.strip().lower() == title:
                return issue
        return None

    # --- write -----------------------------------------------------------

    def ensure_labels(self) -> None:
        """Idempotently create the ``status:*`` labels the backend relies on.

        A live ``gh issue create --label status:ready`` fails if the label doesn't exist in
        the repo, so ``emit`` calls this once before creating. Safe to call standalone as a
        one-time bootstrap step for a fresh work repo.
        """
        for name, (color, description) in _STATUS_LABEL_STYLE.items():
            self._writer.ensure_label(name, color=color, description=description)

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
            for issue in self._reader.list_issues(state="all")
            if (ref := parse_meta_block(issue.body).get("external_ref", "").strip())
        }
        cap = max_per_run if max_per_run is not None else settings.max_per_run
        summary = EmitSummary(project=project)
        if not dry_run and specs:
            self.ensure_labels()  # the status:* labels must exist before create --label
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
            body = self._issue_body(spec, mode=eff_mode, phase=eff_phase, priority=eff_priority)
            label = _STATUS_TO_LABEL.get(eff_status.strip().lower())
            number = self._writer.create_issue(
                title=spec.title, body=body, labels=[label] if label else []
            )
            summary.created.append(EmitOutcome(ref, spec.title, "created", str(number)))
            if log:
                log(f"created: {spec.title} -> #{number}")
        return summary

    def _issue_body(self, spec: EmitSpec, *, mode: str, phase: str, priority: int) -> str:
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
        issue = self._find_issue(task)
        if issue is None:
            raise ValueError(f"issue not found for task {task!r}")
        target = status.strip()
        status_labels = [lbl for lbl in issue.labels if lbl.startswith("status:")]
        if target.lower() == "done":
            self._writer.edit_labels(issue.number, add=[], remove=status_labels)
            self._writer.close_issue(issue.number)
        else:
            if issue.state == "closed":
                self._writer.reopen_issue(issue.number)
            new_label = _STATUS_TO_LABEL.get(target.lower())
            self._writer.edit_labels(
                issue.number,
                add=[new_label] if new_label else [],
                remove=[lbl for lbl in status_labels if lbl != new_label],
            )
        if execution_mode is not None:
            self._writer.edit_body(
                issue.number, set_meta_field(issue.body, "execution_mode", execution_mode)
            )
        if notes:
            self._writer.comment(
                issue.number, f"**{date.today().isoformat()} — Status: {status}**\n\n{notes}"
            )

    # --- read ------------------------------------------------------------

    def find_task(self, name: str) -> TaskInfo | None:
        issue = self._find_issue(name)
        return self._task_info(issue) if issue is not None else None

    def next_ready(self, projects: list[str]) -> TaskInfo | None:
        if projects and self.project.lower() not in {p.lower() for p in projects}:
            return None
        issues = self._reader.list_issues(state="all")
        done_titles = {i.title for i in issues if self._status_of(i) == "Done"}
        candidates: list[_Issue] = []
        for issue in issues:
            meta = parse_meta_block(issue.body)
            mode = (meta.get("execution_mode") or "Manual").strip().lower()
            if self._status_of(issue) != "Ready" or mode not in _AUTO_MODES:
                continue
            if any(d not in done_titles for d in _split_deps(meta.get("depends_on", ""))):
                continue
            candidates.append(issue)
        candidates.sort(
            key=lambda i: (
                _mode_rank(parse_meta_block(i.body).get("execution_mode") or "Manual"),
                _int(parse_meta_block(i.body).get("priority"), 99),
                i.title,
            )
        )
        return self._task_info(candidates[0]) if candidates else None

    def worker_gate(self, name: str) -> str:
        from forge.task_worker.nous_client import _gate_reason

        title = _normalize_title(name)
        rows = {r.task: r for r in self._leaf_rows(self._reader.list_issues(state="all"))}
        row = rows.get(title)
        if row is None:
            return f"task {title!r} not found in the issue tracker"
        return _gate_reason(row.status, row.execution_mode, row.blocked_by)

    def get_spec(self, name: str) -> str:
        issue = self._find_issue(name)
        if issue is None:
            raise ValueError(f"issue not found for task {name!r}")
        rows = {r.task: r for r in self._leaf_rows(self._reader.list_issues(state="all"))}
        row = rows.get(issue.title)
        meta = parse_meta_block(issue.body)
        status = self._status_of(issue)

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
        deps = _split_deps(meta.get("depends_on", ""))
        parts.append("- **Dependencies:**")
        if not deps:
            parts.append("  None")
        else:
            done_titles = {r for r, lr in rows.items() if lr.status == "Done"}
            for dep in deps:
                marker = "done" if dep in done_titles else "**not done**"
                parts.append(f"  - {dep}: {marker}")
        parts.append("\n---\n")
        parts.append(strip_meta_block(issue.body))
        return "\n".join(parts)

    def list_rows(
        self, project: str, *, feature: str | None = None, include_done: bool = True
    ) -> list[LeafRow]:
        issues = self._reader.list_issues(state="all")
        rows = self._leaf_rows(issues)
        if feature is not None:
            by_title = {i.title: parse_meta_block(i.body).get("feature", "") for i in issues}
            rows = [r for r in rows if by_title.get(r.task, "") == feature]
        if not include_done:
            rows = [r for r in rows if r.status != "Done"]
        return rows

    def in_progress_titles(self, ref_prefix: str) -> list[str]:
        issues = self._reader.list_issues(state="open", label=_STATUS_TO_LABEL["in progress"])
        return [
            issue.title
            for issue in issues
            if issue.title.strip()
            and parse_meta_block(issue.body).get("external_ref", "").startswith(ref_prefix)
        ]

    def queue(self, *, project: str | None = None) -> list[QueueRow]:
        # Single-project store: one repo == one project, so any other name is empty.
        if project is not None and project.lower() != self.project.lower():
            return []
        issues = self._reader.list_issues(state="all")
        metas = {i.title: parse_meta_block(i.body) for i in issues}
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
            for row in self._leaf_rows(issues)
            if row.status != "Done" and row.task.strip()
        ]
