"""The session **baton** — a durable, structured handoff artifact that makes session state portable
across any boundary: a Claude outage, an overnight gap, or an A/B run of the same work under two
models. It is the context-compaction summary made durable and machine-readable, written to
``.forge/baton.md`` in the target repo.

This is the *real* Switcheroo primitive; outage-failover to a local OpenCode fleet is only one of
three consumers:

1. **Outage failover** — drain worker-shaped Forge leaves via a local fleet while Claude is down.
2. **Overnight resumption** — a fresh session picks a long task back up cleanly from the baton.
3. **A/B testing** — resume the *same* baton under model A vs B and compare tokens / cache / output;
   the baton is what makes the two runs start from identical state.

Two constraints, borrowed from the sibling rules-layer file (:mod:`forge.shared.lessons`):

- **Markdown-first, human-editable.** The body is authoritative and hand-editable under plain
  ``## Section`` headings — a person or an agent can open the baton and read/patch it. Only the
  machine anchor (VCS id, timestamps) lives in YAML frontmatter.
- **Never silently lose a decision.** ``goal`` / ``plan`` / ``next_action`` / ``working_set`` are
  *current state* and are meant to change every write. ``decisions`` are *rationale*, and losing one
  loses the "why" a later reader needs — so :func:`write_baton` **accretes** decisions (union with
  what is already on disk) and only prunes when the caller says ``allow_prune=True``.

The baton is also **version-lined to the working copy**: each write records the current jj change id
(or git commit) so "where we were" (the baton) and "what changed" (the switch-back ``jj diff``) line
up against the same base.
"""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from forge.task_worker.vcs import detect_vcs

#: The durable baton file, relative to the target repo. Shares the machine-managed ``.forge/``
#: directory with the lessons file, kept apart from the human-authored AGENTS.md.
BATON_PATH = ".forge/baton.md"

_VCS_TIMEOUT = 15

#: Body section headings, in render order, tagged by how the section is parsed/rendered:
#: ``"para"`` (a free-text paragraph) or ``"bullets"`` (a ``- `` list). The heading text is what
#: appears in the file; the field name is the :class:`Baton` attribute it round-trips to.
_SECTIONS: list[tuple[str, str, str]] = [
    ("Goal", "goal", "para"),
    ("Next action", "next_action", "para"),
    ("Plan", "plan", "bullets"),
    ("Working set", "working_set", "bullets"),
    ("Decisions", "decisions", "bullets"),
    ("Notes", "notes", "para"),
]

_INTRO = (
    "> Durable session baton — where we are, what's next, and why. Resume from this plus the VCS\n"
    "> diff since `change_id`. Honor the decisions; they each encode a choice already made.\n"
)


class Baton(BaseModel):
    """A durable session-continuation summary. Small by design — a summary, not a transcript."""

    goal: str = ""  #: The current top-level goal being pursued.
    next_action: str = ""  #: The single concrete next step — the first thing a resumer should do.
    plan: list[str] = Field(default_factory=list)  #: Remaining steps, ordered.
    working_set: list[str] = Field(default_factory=list)  #: Files/paths in play right now.
    decisions: list[str] = Field(default_factory=list)  #: Key decisions & rationale (accretes).
    notes: str = ""  #: Freeform extra context that doesn't fit the structured fields.

    # --- machine anchor (frontmatter) ---
    vcs: str | None = None  #: "jj" | "git" | None, as detected at write time.
    branch: str | None = None  #: Bookmark/branch at write time, best-effort.
    change_id: str | None = (
        None  #: jj change id (or git commit) of the working copy — the diff base.
    )
    created_at: str | None = None  #: ISO-8601 UTC of first write; preserved across rewrites.
    updated_at: str | None = None  #: ISO-8601 UTC of the most recent write.


def baton_path(repo: Path) -> Path:
    return repo / BATON_PATH


# ---------------------------------------------------------------------------
# VCS anchor — version-line the baton to the working copy
# ---------------------------------------------------------------------------


def _run_vcs(args: list[str], repo: Path) -> str | None:
    """Best-effort VCS probe: the trimmed stdout, or ``None`` on any failure. A baton must still
    write when VCS introspection fails, so callers never see an exception from here."""
    try:
        result = subprocess.run(
            args, cwd=repo, capture_output=True, text=True, timeout=_VCS_TIMEOUT
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    out = result.stdout.strip()
    return out or None


def capture_vcs_anchor(repo: Path) -> dict[str, str | None]:
    """The current working-copy anchor — ``{vcs, branch, change_id}`` — so a later switch-back can
    diff against exactly the state the baton describes. Fully best-effort: every field falls back to
    ``None`` rather than raising, so a baton is always writable."""
    vcs = detect_vcs(repo)
    if vcs == "jj":
        change_id = _run_vcs(
            ["jj", "--no-pager", "log", "-r", "@", "--no-graph", "-T", "change_id.short()"], repo
        )
        # jj has no single "current branch"; the nearest bookmark(s) on @ are the useful analog.
        branch = _run_vcs(
            ["jj", "--no-pager", "log", "-r", "@", "--no-graph", "-T", "bookmarks"], repo
        )
        return {"vcs": "jj", "branch": branch, "change_id": change_id}
    if vcs == "git":
        branch = _run_vcs(["git", "rev-parse", "--abbrev-ref", "HEAD"], repo)
        change_id = _run_vcs(["git", "rev-parse", "--short", "HEAD"], repo)
        return {"vcs": "git", "branch": branch, "change_id": change_id}
    return {"vcs": None, "branch": None, "change_id": None}


# ---------------------------------------------------------------------------
# Parse / render
# ---------------------------------------------------------------------------


def _split_frontmatter(text: str) -> tuple[dict, str]:
    """Split a leading ``---`` YAML frontmatter block from the markdown body. Missing or malformed
    frontmatter yields ``({}, text)`` — parsing a baton never fails on a hand-edited file."""
    if text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        if end != -1:
            try:
                data = yaml.safe_load(text[4:end]) or {}
            except yaml.YAMLError:
                data = {}
            if isinstance(data, dict):
                return data, text[end + 5 :]
    return {}, text


def _parse_body(body: str) -> dict[str, object]:
    """Pull the known ``## Section`` blocks out of a baton body. Forgiving by design: unknown
    headings are ignored, sections may appear in any order or be absent, and bullet vs paragraph is
    decided per the section's declared kind."""
    kind = {field: k for _, field, k in _SECTIONS}
    heading_to_field = {h.lower(): field for h, field, _ in _SECTIONS}

    current: str | None = None
    buckets: dict[str, list[str]] = {field: [] for _, field, _ in _SECTIONS}
    for raw in body.splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if stripped.startswith("## "):
            current = heading_to_field.get(stripped[3:].strip().lower())
            continue
        if current is None or stripped.startswith(">"):
            continue
        buckets[current].append(line)

    out: dict[str, object] = {}
    for field, lines in buckets.items():
        if kind[field] == "bullets":
            out[field] = [ln.strip()[2:].strip() for ln in lines if ln.lstrip().startswith("- ")]
        else:
            out[field] = "\n".join(ln for ln in lines).strip()
    return out


def parse_baton(text: str) -> Baton:
    """Parse a baton file's full text back into a :class:`Baton`."""
    fm, body = _split_frontmatter(text)
    fields = _parse_body(body)
    for key in ("vcs", "branch", "change_id", "created_at", "updated_at"):
        if fm.get(key) is not None:
            fields[key] = fm[key]
    return Baton.model_validate(fields)


def read_baton(repo: Path) -> Baton | None:
    """The baton for *repo*, or ``None`` when there is none."""
    path = baton_path(repo)
    if not path.is_file():
        return None
    return parse_baton(path.read_text())


def render_baton(baton: Baton) -> str:
    """The full ``.forge/baton.md`` text: YAML machine anchor, then human-editable sections. Empty
    sections are omitted so the file stays a lean summary."""
    anchor = {
        "updated_at": baton.updated_at,
        "created_at": baton.created_at,
        "vcs": baton.vcs,
        "branch": baton.branch,
        "change_id": baton.change_id,
    }
    anchor = {k: v for k, v in anchor.items() if v is not None}
    front = yaml.safe_dump(anchor, sort_keys=False, default_flow_style=False).strip()

    parts = [f"---\n{front}\n---\n", "# Session baton\n\n", _INTRO]
    for heading, field, section_kind in _SECTIONS:
        value = getattr(baton, field)
        if not value:
            continue
        parts.append(f"\n## {heading}\n")
        if section_kind == "bullets":
            parts.append("\n".join(f"- {item}" for item in value) + "\n")
        else:
            parts.append(f"{value}\n")
    return "".join(parts)


def render_baton_preamble(baton: Baton | None) -> str:
    """Wrap a baton as a prompt preamble that seeds a resuming consumer (a local fleet worker, a
    fresh session, an A/B run). ``None`` or an empty baton yields ``""``. Mirrors
    :func:`forge.shared.lessons.render_lessons_preamble` so the two blocks compose in one prompt."""
    if baton is None or not (baton.goal or baton.next_action or baton.plan):
        return ""
    lines = [
        "## Session baton — resume from here (READ FIRST)",
        "",
        "You are continuing an in-progress session. This is where it stood; pick it up from the "
        "next action. Honor the decisions — each one is a choice already made.",
        "",
    ]
    if baton.goal:
        lines += [f"**Goal:** {baton.goal}", ""]
    if baton.next_action:
        lines += [f"**Next action:** {baton.next_action}", ""]
    if baton.plan:
        lines += ["**Remaining plan:**", *[f"- {step}" for step in baton.plan], ""]
    if baton.working_set:
        lines += ["**Working set:**", *[f"- {f}" for f in baton.working_set], ""]
    if baton.decisions:
        lines += ["**Decisions (honor these):**", *[f"- {d}" for d in baton.decisions], ""]
    if baton.notes:
        lines += [f"**Notes:** {baton.notes}", ""]
    if baton.change_id:
        lines += [
            f"_Working copy anchored at {baton.vcs or 'vcs'} `{baton.change_id}`; changes since "
            "then are what the previous session left in flight._",
            "",
        ]
    return "\n".join(lines) + "\n---\n\n"


# ---------------------------------------------------------------------------
# Write — with drift discipline
# ---------------------------------------------------------------------------


def _merge_decisions(existing: list[str], incoming: list[str]) -> list[str]:
    """Union of the two lists, existing first, order-preserving, deduped — the accretion that makes
    decisions never silently drop."""
    merged = list(existing)
    for d in incoming:
        if d not in merged:
            merged.append(d)
    return merged


def write_baton(repo: Path, baton: Baton, *, allow_prune: bool = False) -> Baton:
    """Persist *baton* to ``.forge/baton.md`` and return the effective baton actually written.

    Applies the primitive's two guarantees:

    - **Decisions accrete.** Any decision already on disk is preserved unless ``allow_prune=True``
      (the explicit "yes, I really mean to drop rationale" escape hatch). Every other field is taken
      from *baton* as current state.
    - **Version-lined.** The VCS anchor (``vcs`` / ``branch`` / ``change_id``) is captured from the
      working copy when the caller left it unset, and ``created_at`` is preserved across rewrites
      while ``updated_at`` is stamped now.
    """
    existing = read_baton(repo)

    if not allow_prune and existing is not None:
        baton = baton.model_copy(
            update={"decisions": _merge_decisions(existing.decisions, baton.decisions)}
        )

    if baton.change_id is None and baton.vcs is None:
        baton = baton.model_copy(update=capture_vcs_anchor(repo))

    now = datetime.now(UTC).isoformat(timespec="seconds")
    created = baton.created_at or (existing.created_at if existing else None) or now
    baton = baton.model_copy(update={"created_at": created, "updated_at": now})

    path = baton_path(repo)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_baton(baton))
    return baton


def record_decision(repo: Path, decision: str) -> bool:
    """Append a single *decision* to the repo's baton (deduped), preserving all other state. Returns
    ``True`` when it was newly recorded. A thin convenience over read → append → write for the
    common "note this choice" case; mirrors :func:`forge.shared.lessons.append_lesson`."""
    decision = decision.strip()
    baton = read_baton(repo) or Baton()
    if decision in baton.decisions:
        return False
    baton = baton.model_copy(update={"decisions": [*baton.decisions, decision]})
    write_baton(repo, baton)
    return True
