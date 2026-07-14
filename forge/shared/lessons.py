"""The rules layer of the two-layer loop spine: a durable, per-repo lessons file the worker reads
at the start of EVERY run, plus the hill-climbing hook that proposes a new lesson when the loop
catches the same class of mistake twice.

The progress layer (wave journal + task status + the provenance refs) says WHERE the loop is; this
rules layer says WHAT IT LEARNED — so a repeated mistake becomes a durable instruction instead of a
forever-repeated failure. Two hard constraints shape the design:

- **Short by design.** Every byte rides in every worker prompt, so appends are budgeted with
  oldest-first eviction (`MAX_LESSONS`).
- **Never silent.** Appending to the *active* file is a human decision (the memory-drift caution):
  the loop only PROPOSES — it writes candidate lessons to a visible, per-run artifact for review.
"""

from __future__ import annotations

from pathlib import Path

#: The durable lessons file, relative to the target repo. A dedicated machine-managed file (not the
#: human-authored AGENTS.md) so append/eviction and the drift caution stay clean.
REPO_LESSONS_PATH = ".forge/lessons.md"
#: Proposed-but-unapproved lessons, per run — the visible artifact a human promotes from.
PROPOSED_LESSONS_NAME = "lessons.proposed.md"
#: Every lesson is paid for on every worker run; keep the rules layer lean.
MAX_LESSONS = 30

_LESSONS_HEADER = (
    "# Repository lessons\n\nDurable lessons read by the autonomous worker before every run.\n"
)


def lessons_path(repo: Path) -> Path:
    return repo / REPO_LESSONS_PATH


def read_lessons(repo: Path) -> str:
    """The raw lessons file content for *repo*, or ``""`` when there is none."""
    path = lessons_path(repo)
    return path.read_text() if path.is_file() else ""


def lesson_bullets(text: str) -> list[str]:
    """The actual lesson lines (``- ...`` bullets), in order — headers/blanks/prose ignored."""
    return [line.strip() for line in text.splitlines() if line.lstrip().startswith("- ")]


def render_lessons_preamble(lessons_text: str) -> str:
    """Wrap the repo's lessons as a prompt preamble the worker reads before the task, or ``""``
    when there are none. Re-rendered from the bullets so the injected block is clean regardless of
    how the file itself is formatted."""
    bullets = lesson_bullets(lessons_text)
    if not bullets:
        return ""
    return (
        "## Repository lessons — learned from earlier runs (READ FIRST)\n\n"
        "Durable lessons from past automation on THIS repository. Honor them; each one encodes a "
        "mistake made before:\n\n" + "\n".join(bullets) + "\n\n---\n\n"
    )


def _as_bullet(lesson: str) -> str:
    lesson = lesson.strip()
    return lesson if lesson.startswith("- ") else f"- {lesson}"


def append_lesson(repo: Path, lesson: str, *, max_lessons: int = MAX_LESSONS) -> bool:
    """The APPROVED append: add a one-line *lesson* to the repo's durable lessons file, deduped,
    with oldest-first eviction past *max_lessons*. Returns ``True`` when a new lesson was written,
    ``False`` when it was already present. This is the human-invoked promotion step — the loop
    proposes, a human calls this."""
    bullet = _as_bullet(lesson)
    existing = lesson_bullets(read_lessons(repo))
    if bullet in existing:
        return False
    kept = (existing + [bullet])[-max_lessons:]  # oldest fall off when over budget
    path = lessons_path(repo)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_LESSONS_HEADER + "\n" + "\n".join(kept) + "\n")
    return True


def draft_lesson(reason: str, *, count: int) -> str:
    """Deterministically draft a one-line lesson from a recurring failure reason — cheap and
    LLM-free (a lesson proposal must not itself burn the token budget). A human refines the wording
    on approval; the point here is to surface the recurring class, not to write prose."""
    first_line = next((ln.strip() for ln in reason.splitlines() if ln.strip()), "(no detail)")
    return (
        f"- Recurring failure (seen {count}× this epic): {first_line[:180]} "
        "— fix the root cause; a blind retry will not clear it."
    )


def proposed_path(run_dir: Path) -> Path:
    return run_dir / PROPOSED_LESSONS_NAME


def proposed_lessons(run_dir: Path) -> list[str]:
    path = proposed_path(run_dir)
    return lesson_bullets(path.read_text()) if path.is_file() else []


def propose_lesson(run_dir: Path, lesson: str) -> bool:
    """Record a candidate lesson in the run's visible proposals artifact (deduped). Returns
    ``True`` when newly proposed. This never touches the active lessons file — a human promotes a
    proposal with :func:`append_lesson`."""
    bullet = _as_bullet(lesson)
    if bullet in proposed_lessons(run_dir):
        return False
    path = proposed_path(run_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        ""
        if path.is_file()
        else "# Proposed lessons (unreviewed)\n\nPromote into .forge/lessons.md to adopt.\n\n"
    )
    with path.open("a") as fh:
        fh.write(header + bullet + "\n")
    return True
