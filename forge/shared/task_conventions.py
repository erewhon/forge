"""The portable task-encoding conventions shared by the per-repo TaskStore backends.

GitHubTaskStore proved these conventions; GitBugTaskStore reuses them verbatim so a task
body is readable by either backend (and by the Soft Serve sprinkles UI, which renders
git-bug issues):

- **status** — a ``status:*`` label for Spec Needed / Ready / In Progress; **Done is the
  tracker's closed state**, with the label as optional bookkeeping.
- **metadata** — a ``pipeline-meta`` HTML-comment block at the top of the body
  (execution_mode, model_tier, priority, max_files, requires_tests, external_ref, ...,
  and ``depends_on`` as comma-separated task titles).
- **dependencies by title** — resolved against the same tracker's titles, exactly like
  Forge resolves ``Depends On`` names; batch-emitted tasks have stable names before they
  have numbers, so name-based deps need no emit ordering.
"""

from __future__ import annotations

AUTO_MODES = {"auto-ok", "auto-preferred"}

# status <-> label. Done is represented by the tracker's closed state (label optional).
STATUS_TO_LABEL = {
    "spec needed": "status:spec-needed",
    "ready": "status:ready",
    "in progress": "status:in-progress",
    "done": "status:done",
}
LABEL_TO_STATUS = {
    "status:spec-needed": "Spec Needed",
    "status:ready": "Ready",
    "status:in-progress": "In Progress",
    "status:done": "Done",
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


def format_meta_block(fields: dict[str, str]) -> str:
    """Render an ordered ``pipeline-meta`` HTML-comment block from string fields."""
    lines = [_META_START]
    for key in _META_KEYS:
        if key in fields and fields[key] != "":
            lines.append(f"{key}: {fields[key]}")
    lines.append(_META_END)
    return "\n".join(lines)


def parse_meta_block(body: str) -> dict[str, str]:
    """Parse the ``pipeline-meta`` block out of a task body into a str->str dict."""
    start = body.find(_META_START)
    if start == -1:
        return {}
    end = body.find(_META_END, start + len(_META_START))
    if end == -1:
        return {}
    inner = body[start + len(_META_START) : end]
    meta: dict[str, str] = {}
    for line in inner.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()
    return meta


def strip_meta_block(body: str) -> str:
    """The task body with its ``pipeline-meta`` block removed — the spec itself."""
    start = body.find(_META_START)
    if start == -1:
        return body.strip()
    end = body.find(_META_END, start + len(_META_START))
    if end == -1:
        return body.strip()
    remainder = body[:start] + body[end + len(_META_END) :]
    return remainder.strip()


def set_meta_field(body: str, key: str, value: str) -> str:
    """Return *body* with one meta field updated (block + spec content preserved)."""
    meta = parse_meta_block(body)
    meta[key] = value
    spec = strip_meta_block(body)
    return f"{format_meta_block(meta)}\n\n{spec}" if spec else format_meta_block(meta)


def split_deps(raw: str) -> list[str]:
    return [d.strip() for d in raw.split(",") if d.strip()]


def parse_int(raw: str | None, default: int) -> int:
    try:
        return int(str(raw))
    except (TypeError, ValueError):
        return default


def parse_int_or_none(raw: str | None) -> int | None:
    try:
        return int(str(raw))
    except (TypeError, ValueError):
        return None


def bool_null_true(raw: str | None) -> bool:
    """null-as-true (matches Forge): missing requires_tests means tests ARE required."""
    if raw is None:
        return True
    return str(raw).strip().lower() in {"true", "yes", "y", "1"}


def mode_rank(mode: str) -> int:
    return 0 if mode.strip().lower() == "auto-preferred" else 1


def normalize_title(name: str) -> str:
    name = name.strip()
    if name.lower().startswith("task: "):
        name = name[6:].strip()
    return name
