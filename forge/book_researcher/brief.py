"""Prompt briefs: what the planner and researcher are told about the book beyond the question.

Before this module the researcher's whole brief was the question string plus a chapter number —
no title, no chapter description, none of the discipline rules the human wrote in YAML comments
(which ``yaml.safe_load`` discards). That is why every question had to name its subject, its
repository, and its blocked hosts inline. The brief carries that context as data instead, so
questions can be short and the rules apply to every sprint uniformly.
"""

from __future__ import annotations

from forge.book_researcher.models import BookConfig, ChapterOutline, Reachability, SourcePolicy


def effective_policy(book: BookConfig, reachability: Reachability | None) -> SourcePolicy:
    """The YAML's source policy merged with the probe's findings.

    The human's ``blocked`` list always wins (a host they blocked stays blocked even if the root
    URL answers 200 — they may know the documents 403). Probe results fill in the rest.
    """
    reachable = list(book.sources.reachable)
    blocked = list(book.sources.blocked)
    if reachability is not None:
        for host, verdict in sorted(reachability.hosts.items()):
            if verdict.state == "blocked" and host not in blocked:
                blocked.append(host)
            elif verdict.state == "reachable" and host not in reachable and host not in blocked:
                reachable.append(host)
    # A probe-reachable host the human later blocked must not stay in reachable.
    reachable = [h for h in reachable if h not in blocked]
    return SourcePolicy(reachable=reachable, blocked=blocked, notes=list(book.sources.notes))


def _bullets(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def render_source_policy(policy: SourcePolicy) -> str:
    parts: list[str] = []
    if policy.reachable:
        parts.append(
            "Prefer these repositories (verified reachable):\n" + _bullets(policy.reachable)
        )
    if policy.blocked:
        parts.append(
            "These hosts are NOT reachable from here — do not cite them, do not ask for documents "
            "that live only there, and say so if the record is only there:\n"
            + _bullets(policy.blocked)
        )
    if policy.notes:
        parts.append("Source notes:\n" + _bullets(policy.notes))
    return "\n\n".join(parts)


def render_book_brief(book: BookConfig, policy: SourcePolicy) -> str:
    """Book-level brief for the planner: thesis, book-wide rules, source policy."""
    parts = [f"Book: {book.title}", f"Description: {book.description.strip()}"]
    if book.guidance:
        parts.append(
            "Book-wide research rules (apply to every question):\n" + _bullets(book.guidance)
        )
    sources = render_source_policy(policy)
    if sources:
        parts.append(sources)
    return "\n\n".join(parts)


def render_chapter_brief(book: BookConfig, chapter: ChapterOutline, policy: SourcePolicy) -> str:
    """Chapter-level brief for the researcher: book + chapter context, rules, and where the
    primary record for this chapter lives."""
    parts = [
        f"Book: {book.title} — {book.description.strip()}",
        f"Chapter {chapter.number}: {chapter.title}\n{chapter.description.strip()}",
    ]
    rules = list(book.guidance) + list(chapter.guidance)
    if rules:
        parts.append("Research rules for this chapter (mandatory):\n" + _bullets(rules))
    if chapter.sources:
        parts.append("Primary-source repositories for this chapter:\n" + _bullets(chapter.sources))
    sources = render_source_policy(policy)
    if sources:
        parts.append(sources)
    return "\n\n".join(parts)


def render_chapter_line(chapter: ChapterOutline) -> str:
    """One chapter as the planner sees it in the outline listing."""
    line = f"  Chapter {chapter.number}: {chapter.title} - {chapter.description.strip()}"
    if chapter.sources:
        line += f"\n    Primary sources: {', '.join(chapter.sources)}"
    if chapter.guidance:
        line += f"\n    Chapter rules: {' | '.join(chapter.guidance)}"
    return line
