"""Shared plumbing for the outline lifecycle commands (lint --critic, revise, decompose).

One place builds the failover pool from ``settings.outline_models`` so every outline call routes
the same way: local first, hosted fallback, Anthropic only when the harness backend is switched
to it. Also the YAML round-trip helpers — the outline is a human's file with comments, so edits
go through ruamel's round-trip loader rather than a load/dump that would strip them.
"""

from __future__ import annotations

import difflib
import io
import re
from pathlib import Path

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq

from forge.book_researcher.config import settings
from forge.book_researcher.models import BookConfig
from forge.shared.ensemble import ApiExecutor, Pool


def outline_pool(models: list[str] | None = None) -> Pool:
    """The outline role's failover pool: ``models`` (default ``settings.outline_models``) in
    preference order on the router, or the Anthropic model when that backend is selected."""
    if settings.llm_backend == "anthropic":
        return Pool(
            role="outline",
            executors=[
                ApiExecutor(
                    label=f"anthropic:{settings.anthropic_model}",
                    kind="anthropic",
                    model=settings.anthropic_model,
                )
            ],
        )
    aliases = models or settings.outline_models
    return Pool(
        role="outline",
        executors=[
            ApiExecutor(
                label=f"router:{alias}",
                kind="openai",
                model=alias,
                base_url=settings.openai_base_url,
                api_key=settings.openai_api_key,
            )
            for alias in aliases
        ],
    )


# --- YAML round-trip ------------------------------------------------------------------------------


def _yaml() -> YAML:
    y = YAML()
    y.preserve_quotes = True
    y.width = 100
    y.indent(mapping=2, sequence=4, offset=2)
    return y


def load_yaml_doc(text: str) -> CommentedMap:
    doc = _yaml().load(text)
    if not isinstance(doc, CommentedMap):
        raise ValueError("outline YAML is not a mapping at the top level")
    return doc


def dump_yaml_doc(doc: CommentedMap) -> str:
    buf = io.StringIO()
    _yaml().dump(doc, buf)
    return buf.getvalue()


def _literal(text: str):
    """Multi-line prose as a block scalar, single-line as a plain string."""
    from ruamel.yaml.scalarstring import LiteralScalarString

    return LiteralScalarString(text) if "\n" in text else text


def book_to_doc(book: BookConfig) -> CommentedMap:
    """A fresh CommentedMap for a BookConfig, fields in the order a human would write them and
    empty optionals omitted, so a generated outline reads like a hand-written one."""
    doc = CommentedMap()
    doc["title"] = book.title
    doc["description"] = _literal(book.description)
    if book.guidance:
        doc["guidance"] = CommentedSeq(book.guidance)
    if book.sources.reachable or book.sources.blocked or book.sources.notes:
        src = CommentedMap()
        if book.sources.reachable:
            src["reachable"] = CommentedSeq(book.sources.reachable)
        if book.sources.blocked:
            src["blocked"] = CommentedSeq(book.sources.blocked)
        if book.sources.notes:
            src["notes"] = CommentedSeq(book.sources.notes)
        doc["sources"] = src
    chapters = CommentedSeq()
    for ch in book.chapters:
        m = CommentedMap()
        m["number"] = ch.number
        m["title"] = ch.title
        m["description"] = _literal(ch.description)
        if ch.sources:
            m["sources"] = CommentedSeq(ch.sources)
        if ch.guidance:
            m["guidance"] = CommentedSeq(ch.guidance)
        m["research_questions"] = CommentedSeq(ch.research_questions)
        chapters.append(m)
    doc["chapters"] = chapters
    return doc


def render_outline_yaml(book: BookConfig, header: str = "") -> str:
    """Serialise a BookConfig as outline YAML with an optional leading comment block."""
    body = dump_yaml_doc(book_to_doc(book))
    if not header:
        return body
    commented = "\n".join(f"# {line}".rstrip() for line in header.strip().splitlines())
    return f"{commented}\n\n{body}"


def write_outline(book: BookConfig, path: Path, *, header: str = "", force: bool = False) -> Path:
    """Write the outline; refuses to clobber an existing file unless ``force``."""
    if path.exists() and not force:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_outline_yaml(book, header))
    return path


# --- fuzzy question matching -------------------------------------------------------------------

MATCH_THRESHOLD = 0.75


def norm_question(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower()).rstrip("?.! ")


def match_question(needle: str, haystack: list[str]) -> tuple[int, float] | None:
    """Index + similarity of the best match for ``needle`` in ``haystack`` (normalised), or None
    below :data:`MATCH_THRESHOLD`. Substring containment counts as a match either way — a model
    that echoes a truncated question is still pointing at the right one."""
    n = norm_question(needle)
    if not n:
        return None
    best: tuple[int, float] | None = None
    for i, candidate in enumerate(haystack):
        c = norm_question(candidate)
        if not c:
            continue
        if n == c:
            return i, 1.0
        if (len(n) >= 25 and n in c) or (len(c) >= 25 and c in n):
            ratio = 0.99
        else:
            ratio = difflib.SequenceMatcher(None, n, c).ratio()
        if best is None or ratio > best[1]:
            best = (i, ratio)
    if best is None or best[1] < MATCH_THRESHOLD:
        return None
    return best
