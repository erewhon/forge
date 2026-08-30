"""`forge book revise` — turn verifier reviews into proposed outline edits.

The panel's feedback is outline-shaped ("the question references a July 2023 judgment — there was
no July 2023 judgment; a writer would inherit the confusion"), but until now it only reached the
*next sprint's planner*, and the per-run "chapter exhausted" state was forgotten at exit. Course-
correcting an outline meant a human reading every review JSON. This command reads them instead.

Three stages, each usable alone:

1. :func:`gather_evidence` — per chapter: attempts, passes, scores, what is covered (and at what
   confidence), failed questions, and the most recent reviews' follow-ups. No model.
   ``--report`` stops here and prints it.
2. :func:`propose_revision` — one structured call on the outline pool: evidence → a list of
   ``OutlineEdit``. Each edit is narrow (replace/add/remove one question, add one rule, block one
   host) with the sprint ids it rests on, so a local model does fine and a human can audit it.
3. :func:`apply_edits` — apply the resolvable edits to the outline YAML via a comment-preserving
   round-trip. Never touches ``book.yaml`` itself unless ``--apply``; the default output is
   ``<name>.proposed.yaml`` + ``<name>.revision.md`` + a diff to read.
"""

from __future__ import annotations

import difflib
import json
from dataclasses import dataclass, field
from pathlib import Path

from ruamel.yaml.comments import CommentedMap, CommentedSeq

from forge.book_researcher.brief import effective_policy, render_book_brief
from forge.book_researcher.models import (
    BookConfig,
    OutlineEdit,
    Reachability,
    RevisionProposal,
    SourcePolicy,
    SprintContract,
    SprintFindings,
    VerificationResult,
)
from forge.book_researcher.outline import (
    dump_yaml_doc,
    insert_key,
    load_yaml_doc,
    match_question,
    quoted,
)
from forge.shared.datectx import today_line
from forge.shared.ensemble import Pool
from forge.shared.llm import RESEARCH_FAILED_PREFIX
from forge.shared.panel import structured

# --- evidence -------------------------------------------------------------------------------------


@dataclass
class ReviewSummary:
    sprint_id: str
    overall: int
    passed: bool
    weakest: str  # dimension name
    questions: list[str]
    follow_ups: list[str]


@dataclass
class ChapterEvidence:
    chapter: int
    attempts: int = 0
    passes: int = 0
    best_score: int | None = None
    covered: list[tuple[str, str]] = field(default_factory=list)  # (question, confidence)
    failed_questions: list[str] = field(default_factory=list)
    reviews: list[ReviewSummary] = field(default_factory=list)  # most recent first, capped


@dataclass
class RevisionEvidence:
    chapters: dict[int, ChapterEvidence]
    reachability: Reachability | None
    total_sprints: int


def _weakest_dimension(r: VerificationResult) -> str:
    s = r.scores
    dims = {
        "source_diversity": s.source_diversity,
        "claim_verification": s.claim_verification,
        "counter_narrative": s.counter_narrative,
        "depth": s.depth,
        "actionability": s.actionability,
    }
    return min(dims, key=lambda k: dims[k])


def gather_evidence(
    book: BookConfig, project_dir: Path, *, max_reviews: int = 4, max_follow_ups: int = 6
) -> RevisionEvidence:
    """Read sprints/ and knowledge/ into per-chapter evidence. Tolerates missing/partial files."""
    from forge.book_researcher.probe import load_reachability

    sprints_dir = project_dir / "sprints"
    knowledge_dir = project_dir / "knowledge"
    chapters = {ch.number: ChapterEvidence(chapter=ch.number) for ch in book.chapters}

    contracts: dict[str, SprintContract] = {}
    if sprints_dir.is_dir():
        for path in sorted(sprints_dir.glob("sprint-*.json")):
            if path.name.endswith("-review.json"):
                continue
            try:
                c = SprintContract.model_validate_json(path.read_text())
            except Exception:
                continue
            contracts[c.sprint_id] = c

    reviews: list[tuple[SprintContract, VerificationResult]] = []
    if sprints_dir.is_dir():
        for path in sorted(sprints_dir.glob("sprint-*-review.json")):
            try:
                r = VerificationResult.model_validate_json(path.read_text())
            except Exception:
                continue
            c = contracts.get(r.sprint_id)
            if c is None or c.chapter not in chapters:
                continue
            reviews.append((c, r))

    for c, r in reviews:
        ev = chapters[c.chapter]
        ev.attempts += 1
        if r.passed:
            ev.passes += 1
        if ev.best_score is None or r.scores.overall > ev.best_score:
            ev.best_score = r.scores.overall
    # Most recent reviews first, capped per chapter — the newest feedback describes the outline
    # as it stands now.
    for c, r in reversed(reviews):
        ev = chapters[c.chapter]
        if len(ev.reviews) >= max_reviews:
            continue
        ev.reviews.append(
            ReviewSummary(
                sprint_id=r.sprint_id,
                overall=r.scores.overall,
                passed=r.passed,
                weakest=_weakest_dimension(r),
                questions=list(c.questions),
                follow_ups=[q for q in r.follow_up_questions if q.strip()][:max_follow_ups],
            )
        )

    if knowledge_dir.is_dir():
        for ch_dir in sorted(knowledge_dir.iterdir()):
            if not ch_dir.name.startswith("chapter-"):
                continue
            try:
                number = int(ch_dir.name.split("-", 1)[1])
            except ValueError:
                continue
            ev = chapters.get(number)
            if ev is None:
                continue
            for path in sorted(ch_dir.glob("sprint-*.json")):
                try:
                    sf = SprintFindings.model_validate_json(path.read_text())
                except Exception:
                    continue
                for f in sf.findings:
                    if f.answer.startswith(RESEARCH_FAILED_PREFIX):
                        ev.failed_questions.append(f.question)
                    else:
                        ev.covered.append((f.question, f.confidence))

    return RevisionEvidence(
        chapters=chapters,
        reachability=load_reachability(project_dir),
        total_sprints=len(contracts),
    )


# Caps keep the digest inside a local model's context on a real project (ten chapters, a dozen
# sprints, an imported recon run with twenty findings in one chapter). Questions are cut, not
# dropped: a truncated question still says what was asked.
MAX_COVERED_PER_CHAPTER = 12
MAX_QUESTION_CHARS = 200


def _cut(text: str, limit: int = MAX_QUESTION_CHARS) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def render_evidence(book: BookConfig, ev: RevisionEvidence) -> str:
    """The evidence digest — what the model sees, and what ``--report`` prints."""
    lines = [f"# Revision evidence: {book.title}", "", f"Sprints run: {ev.total_sprints}", ""]
    if ev.reachability is not None:
        blocked = sorted(h for h, v in ev.reachability.hosts.items() if v.state == "blocked")
        reachable = sorted(h for h, v in ev.reachability.hosts.items() if v.state == "reachable")
        lines.append(f"Probed hosts — reachable: {', '.join(reachable) or 'none'}")
        lines.append(f"Probed hosts — blocked: {', '.join(blocked) or 'none'}")
        lines.append("")
    for ch in book.chapters:
        e = ev.chapters[ch.number]
        lines.append(f"## Chapter {ch.number}: {ch.title}")
        if e.attempts == 0:
            lines.append("No sprints yet.")
        else:
            lines.append(
                f"Sprints: {e.attempts}, passed: {e.passes}, best score: {e.best_score}/10"
            )
        lines.append("Outline questions:")
        lines.extend(f"  {i}. {q}" for i, q in enumerate(ch.research_questions, 1))
        if e.covered:
            shown = e.covered[-MAX_COVERED_PER_CHAPTER:]
            hidden = len(e.covered) - len(shown)
            lines.append(
                "Covered (question → confidence)"
                + (f", most recent {len(shown)} of {len(e.covered)}" if hidden else "")
                + ":"
            )
            lines.extend(f"  - [{conf}] {_cut(q)}" for q, conf in shown)
        if e.failed_questions:
            lines.append("Research FAILED (empty/tool failure) for:")
            lines.extend(f"  - {_cut(q)}" for q in e.failed_questions)
        for r in e.reviews:
            verdict = "PASS" if r.passed else "FAIL"
            lines.append(
                f"Review sprint {r.sprint_id}: {verdict} {r.overall}/10, weakest on {r.weakest}"
            )
            lines.append("  asked: " + " | ".join(_cut(q, 120) for q in r.questions))
            for fu in r.follow_ups:
                lines.append(f"  - {_cut(fu, 400)}")
        lines.append("")
    return "\n".join(lines)


# --- proposal -------------------------------------------------------------------------------------

REVISE_SYSTEM = """\
You are the OUTLINE EDITOR for a non-fiction book research harness. The outline (book.yaml) lists \
chapters and research questions; each sprint, a planner picks 2-4 questions, a web-searching \
researcher answers each one in its own call, and an adversarial panel grades the answers 1-10 on \
source_diversity, claim_verification, counter_narrative (1.5x), depth (1.5x), and actionability.

You are given the outline, the book-wide rules, the source policy (which hosts are reachable), \
and the EVIDENCE from the sprints run so far: scores, what is covered and at what confidence, \
research failures, and the panel's follow-up challenges. Propose the smallest set of edits to the \
OUTLINE that would raise the next sprints' scores. Each edit must be one of:
- replace_question: a question that keeps failing for a reason in its wording — it names a \
non-existent event, bundles several claims, relies on a blocked host, or reads as a topic. \
"old" must quote the outline question VERBATIM; "new" is the rewrite.
- remove_question: covered at high confidence in a passed sprint, or unanswerable in principle.
- add_question: a gap the reviews keep raising that no outline question targets — including a \
"verify or retract" question when reviews found a contradiction or a suspect figure.
- add_guidance / add_chapter_guidance: a rule that would have prevented a recurring failure \
("never supply a docket number a retrieved source did not show verbatim"). Prefer a rule over \
rewriting many questions the same way.
- add_chapter_source: a reachable repository the chapter should draw on.
- block_host: a host the evidence shows the researcher cannot fetch (403, tool failures).
- note: an observation for the human with no edit (e.g. "chapter 3 is done; stop sprinting it").

Rules: every edit carries a one-sentence reason and the sprint ids it rests on. Do not invent \
facts about the subject — reason from the evidence given. Do not rewrite questions that are \
passing. Fewer, sharper edits beat many; ten is plenty. Questions must name their subject \
explicitly, ask one thing, and demand a source. Never point a question at a host listed as \
unreachable. Do not write today's date or "as of <month>" into a rule or question — the \
harness tells the researcher the date on every call, and a hard-coded one goes stale.

Return ONLY JSON:
{"summary": "<2-3 sentences>", "edits": [{"op": "<op>", "chapter": <n or null>, "old": \
"<verbatim or null>", "new": "<text or null>", "reason": "<one sentence>", "evidence": ["<sprint \
id>", ...]}]}"""


def _render_outline(book: BookConfig) -> str:
    lines = []
    for ch in book.chapters:
        lines.append(f"Chapter {ch.number}: {ch.title}")
        lines.append(f"  {ch.description.strip()}")
        if ch.sources:
            lines.append(f"  sources: {', '.join(ch.sources)}")
        if ch.guidance:
            lines.append("  guidance: " + " | ".join(ch.guidance))
        for q in ch.research_questions:
            lines.append(f"  - {q}")
    return "\n".join(lines)


def propose_revision(
    book: BookConfig,
    ev: RevisionEvidence,
    *,
    pool: Pool,
    max_tokens: int = 16384,
    timeout: float = 300.0,
) -> RevisionProposal:
    """One structured call: outline + evidence → RevisionProposal. Raises if the pool is
    exhausted without a schema-valid payload (the raw output is included for diagnosis)."""
    policy = effective_policy(book, ev.reachability)
    user = (
        render_book_brief(book, policy)
        + "\n\n# Outline\n\n"
        + _render_outline(book)
        + "\n\n"
        + render_evidence(book, ev)
        + "\nPropose the outline edits."
    )
    result = structured(
        pool=pool,
        schema=RevisionProposal,
        system=f"{today_line()}\n\n{REVISE_SYSTEM}",
        user=user,
        max_tokens=max_tokens,
        timeout=timeout,
        predicate=lambda p: len(p.edits) > 0,
    )
    if result.value is None:
        raw = (result.raw or "").strip()
        tail = f"\nLast raw output:\n{raw[:1500]}" if raw else ""
        raise RuntimeError(
            "revise: no model in the outline pool produced a schema-valid proposal "
            f"({result.error}). Check the router aliases in BOOK_RESEARCHER_OUTLINE_MODELS.{tail}"
        )
    return result.value


# --- resolution + application -------------------------------------------------------------------


@dataclass
class ResolvedEdit:
    edit: OutlineEdit
    applicable: bool
    note: str = ""  # why not, or what it matched
    matched: str | None = None  # the outline question `old` resolved to


ADD_DUPLICATE_THRESHOLD = 0.6


def _near_duplicate(new: str, existing: list[str]) -> bool:
    """An added question that paraphrases one already in the chapter. The bar is lower than
    for replace-matching: a model asked for gaps tends to restate a question it saw, with a
    source demand bolted on, and that is a duplicate a human would reject."""
    import difflib

    from forge.book_researcher.outline import norm_question

    n = norm_question(new)
    for q in existing:
        c = norm_question(q)
        if difflib.SequenceMatcher(None, n, c).ratio() >= ADD_DUPLICATE_THRESHOLD:
            return True
        # Same opening clause (first eight words) is the usual paraphrase signature.
        if " ".join(n.split()[:8]) == " ".join(c.split()[:8]):
            return True
    return False


def resolve_edits(book: BookConfig, edits: list[OutlineEdit]) -> list[ResolvedEdit]:
    """Decide which edits can be applied mechanically. A replace/remove whose ``old`` does not
    resolve to an outline question (fuzzily) is kept in the report but not applied."""
    out: list[ResolvedEdit] = []
    for e in edits:
        # Models say `add_guidance` with a chapter number when they mean a chapter rule (Gemma
        # did, on the first real run); honour the chapter rather than promote the rule book-wide.
        if e.op == "add_guidance" and e.chapter is not None:
            e = e.model_copy(update={"op": "add_chapter_guidance"})
        ch = book.chapter(e.chapter) if e.chapter is not None else None
        if e.op == "note":
            out.append(ResolvedEdit(e, False, "note — no edit"))
            continue
        if e.op in ("add_guidance", "block_host"):
            ok = bool(e.new and e.new.strip())
            out.append(ResolvedEdit(e, ok, "" if ok else "missing text"))
            continue
        if ch is None:
            out.append(ResolvedEdit(e, False, f"chapter {e.chapter} is not in the outline"))
            continue
        if e.op in ("add_question", "add_chapter_guidance", "add_chapter_source"):
            if not (e.new and e.new.strip()):
                out.append(ResolvedEdit(e, False, "missing text"))
            elif e.op == "add_question" and _near_duplicate(e.new, ch.research_questions):
                out.append(ResolvedEdit(e, False, "already in the outline (near-duplicate)"))
            else:
                out.append(ResolvedEdit(e, True))
            continue
        # replace_question / remove_question
        m = match_question(e.old or "", ch.research_questions)
        if m is None:
            out.append(ResolvedEdit(e, False, "`old` does not match any question in the chapter"))
            continue
        idx, ratio = m
        if e.op == "replace_question" and not (e.new and e.new.strip()):
            out.append(ResolvedEdit(e, False, "replace without a replacement"))
            continue
        out.append(
            ResolvedEdit(e, True, f"matched ({ratio:.2f})", matched=ch.research_questions[idx])
        )
    return out


def _chapter_map(doc: CommentedMap, number: int) -> CommentedMap | None:
    for ch in doc.get("chapters") or []:
        if isinstance(ch, CommentedMap) and ch.get("number") == number:
            return ch
    return None


def _append(
    container: CommentedMap,
    key: str,
    value: str,
    *,
    after: tuple[str, ...] = (),
    before: str = "",
) -> None:
    seq = container.get(key)
    if seq is None:
        seq = CommentedSeq()
        insert_key(container, key, seq, after=after, before=before)
    if value not in seq:
        seq.append(quoted(value))


def apply_edits(yaml_text: str, resolved: list[ResolvedEdit]) -> str:
    """Apply the applicable edits to the outline YAML, preserving comments and layout."""
    doc = load_yaml_doc(yaml_text)
    for r in resolved:
        if not r.applicable:
            continue
        e = r.edit
        if e.op == "add_guidance":
            _append(
                doc,
                "guidance",
                e.new.strip(),  # type: ignore[union-attr]
                after=("description", "title"),
                before="chapters",
            )
            continue
        if e.op == "block_host":
            sources = doc.get("sources")
            if sources is None:
                sources = CommentedMap()
                insert_key(
                    doc,
                    "sources",
                    sources,
                    after=("guidance", "description", "title"),
                    before="chapters",
                )
            _append(sources, "blocked", e.new.strip())  # type: ignore[union-attr]
            continue
        ch = _chapter_map(doc, e.chapter)  # type: ignore[arg-type]
        if ch is None:
            continue
        if e.op == "add_question":
            _append(ch, "research_questions", e.new.strip())  # type: ignore[union-attr]
        elif e.op == "add_chapter_guidance":
            _append(
                ch,
                "guidance",
                e.new.strip(),  # type: ignore[union-attr]
                after=("sources", "description"),
                before="research_questions",
            )
        elif e.op == "add_chapter_source":
            _append(
                ch,
                "sources",
                e.new.strip(),  # type: ignore[union-attr]
                after=("description",),
                before="guidance",
            )
        elif e.op in ("replace_question", "remove_question"):
            qs = ch.get("research_questions")
            if qs is None:
                continue
            m = match_question(r.matched or e.old or "", list(qs))
            if m is None:
                continue
            idx = m[0]
            if e.op == "replace_question":
                qs[idx] = e.new.strip()  # type: ignore[union-attr]
            else:
                del qs[idx]
    return dump_yaml_doc(doc)


def render_revision_report(
    book: BookConfig, proposal: RevisionProposal, resolved: list[ResolvedEdit], diff: str
) -> str:
    applied = [r for r in resolved if r.applicable]
    skipped = [r for r in resolved if not r.applicable]
    lines = [f"# Outline revision: {book.title}", "", proposal.summary, ""]
    lines.append(f"## Proposed edits ({len(applied)} applicable, {len(skipped)} not applied)")
    lines.append("")
    for r in resolved:
        e = r.edit
        where = f"chapter {e.chapter}" if e.chapter is not None else "book"
        flag = "APPLY" if r.applicable else "SKIP "
        lines.append(f"- **{flag} {e.op}** ({where}) — {e.reason}")
        if e.evidence:
            lines.append(f"  - evidence: sprints {', '.join(e.evidence)}")
        if e.old:
            lines.append(f"  - old: {e.old}")
        if r.matched and r.matched != e.old:
            lines.append(f"  - matched outline question: {r.matched}")
        if e.new:
            lines.append(f"  - new: {e.new}")
        if r.note and not r.applicable:
            lines.append(f"  - not applied: {r.note}")
    lines.append("")
    if diff.strip():
        lines += ["## Diff", "", "```diff", diff.rstrip(), "```", ""]
    return "\n".join(lines)


def unified_diff(before: str, after: str, name: str) -> str:
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=name,
            tofile=f"{name} (proposed)",
        )
    )


def proposal_paths(config_path: Path) -> tuple[Path, Path]:
    stem = config_path.with_suffix("")
    return Path(f"{stem}.proposed.yaml"), Path(f"{stem}.revision.md")


def save_proposal_json(project_dir: Path, proposal: RevisionProposal) -> Path:
    """Keep the raw proposal beside the sprints so a rejected one can still be audited later."""
    path = project_dir / "revisions"
    path.mkdir(parents=True, exist_ok=True)
    n = len(list(path.glob("revision-*.json"))) + 1
    out = path / f"revision-{n:03d}.json"
    out.write_text(json.dumps(proposal.model_dump(), indent=2))
    return out


def policy_for(book: BookConfig, ev: RevisionEvidence) -> SourcePolicy:
    return effective_policy(book, ev.reachability)
