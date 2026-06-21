"""Skeleton book-config scaffolding for `meta book init`.

Writes a ready-to-edit ``book.yaml`` that already validates against ``BookConfig`` (so
``meta book ./book.yaml --dry-run`` works immediately) and nudges toward strong research
questions — including a counter-narrative one per chapter, which the verifier weights 1.5×.
"""

from __future__ import annotations

from pathlib import Path

DEFAULT_FILENAME = "book.yaml"

# A valid BookConfig (title + description + chapters[number/title/description/research_questions]),
# pre-filled with placeholder prose the user replaces. Kept valid so a fresh init can be dry-run
# straight away.
BOOK_SKELETON = """\
# Book research config for `meta book`. Replace the placeholders, then run:
#   meta book ./book.yaml --max-sprints 10
#
# Output defaults to an absolute path under the meta repo; to keep it beside this file:
#   BOOK_RESEARCHER_PROJECT_DIR="$PWD/research" meta book ./book.yaml --max-sprints 10
#
# See RESEARCH-WORKFLOWS.md for how to write strong research questions: specific,
# source-demanding, one claim each, with a counter-narrative question per chapter.

title: "Untitled Book"
description: "One or two sentences: what the book is about and its central thesis."

chapters:
  - number: 1
    title: "Introduction"
    description: "Overview and thesis statement."
    research_questions:
      - "What is the central argument, stated precisely?"
      - "What existing work covers this topic, and where does it disagree?"

  - number: 2
    title: "Chapter title"
    description: "What this chapter establishes."
    research_questions:
      - "A specific, source-demanding question (ask for dates / primary sources)?"
      - "What is the strongest opposing view or disconfirming evidence?"
"""


def resolve_target(path: str | None) -> Path:
    """Resolve the init target. A directory (or trailing-slash path) gets ``book.yaml`` appended."""
    if not path:
        return Path(DEFAULT_FILENAME).resolve()
    p = Path(path).expanduser()
    if p.is_dir():
        p = p / DEFAULT_FILENAME
    return p.resolve()


def write_skeleton(path: str | None = None, *, force: bool = False) -> Path:
    """Write the skeleton config to ``path`` (default ``./book.yaml``).

    Raises ``FileExistsError`` if the target exists and ``force`` is False — init never
    clobbers an existing config silently.
    """
    target = resolve_target(path)
    if target.exists() and not force:
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(BOOK_SKELETON)
    return target
