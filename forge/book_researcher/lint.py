"""`forge book lint` — catch what the verifier panel will dock, before spending a run on it.

Two layers:

- **Deterministic rules** (no model, runs in milliseconds): the failure modes the panel's reviews
  keep naming — a chapter with no counter-narrative question (weighted 1.5x), compound questions
  the panel can't grade cleanly, questions that lean on the chapter title for their subject (one
  such question came back answered about an unrelated assault case), blocked hosts named as
  sources, skeleton placeholders, duplicates, numbering.
- **Critic** (optional, one structured call per chapter on a local model): reads each question
  against the verifier's own five-dimension rubric and predicts where it will lose points, with
  a rewrite. Cheap enough to run on every edit; still opt-in because it spends tokens.

Errors block `revise --apply` and `decompose`; warnings are advice. Heuristics are tuned for a
low false-positive rate on a real outline (the corruption book) rather than for recall.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from forge.book_researcher.models import (
    BookConfig,
    ChapterCritique,
    ChapterOutline,
    LintFinding,
    SourcePolicy,
)
from forge.shared.ensemble import Pool
from forge.shared.panel import structured

MAX_QUESTIONS_PER_CHAPTER = 5
MIN_QUESTIONS_PER_CHAPTER = 2
MIN_QUESTION_WORDS = 8

# Exact (normalised) matches on the skeleton's placeholder strings — not substrings, since a real
# description can legitimately contain "what this chapter establishes".
_PLACEHOLDERS = frozenset(
    {
        "untitled book",
        "one or two sentences: what the book is about and its central thesis",
        "chapter title",
        "what this chapter establishes",
        "a specific, source-demanding question (ask for dates / primary sources)",
        "sample research project",
    }
)


def _is_placeholder(text: str) -> bool:
    return _norm(text) in _PLACEHOLDERS


_COUNTER_RE = re.compile(
    r"strongest (?:\w+ ){0,2}(?:argument|case|defen[cs]e|objection|counter)"
    r"|\boppos(?:ing|ite|ition)\b|\bdisconfirm|\bcounter[- ]?(?:narrative|argument|case|evidence)"
    r"|\brebut|\bcritics?\b|\bcriticisms?\b|\bskeptic|\balternative explanation|\bdevil's advocate"
    r"|\blegitimate (?:\w+ )?(?:rationale|justification|reason)|\bdefen[cs]e of\b"
    r"|\bargue[sd]? (?:that|against)|\bweakness|\bmethodological|\bretracted|\bcorrected"
    r"|\bcase for the\b|\bbest case for\b|\bin defen[cs]e\b|\bwhat (?:do|does|would) .{0,40}"
    r"(?:defenders|proponents|supporters|advocates) (?:say|argue|contend|claim)"
    r"|\bthe argument that\b|\bbase rate\b|\brather than\b|\bcompare[sd]? (?:with|to)\b"
    r"|\babsent\b|\bnormally\b|\bconsistent with how\b|\bordinary\b|\blawful\b"
    r"|\bhow do .{0,60}(?:assess|view|read|judge|rate)\b",
    re.IGNORECASE,
)
_SOURCE_DEMAND_RE = re.compile(
    r"primary[- ]source|\bcite|\bciting|\bper the\b|\baccording to\b|\bfiling|\bdocket"
    r"|\bwith dates\b|\bsource|\brecord\b|\brecords\b|\breport|\bvia\b|\bdisclos|\bopinion\b"
    r"|\bdecision text|\btranscript|\bstatute|\bregister|\bnamed\b|\bname the\b|\bdata\b"
    r"|\bwho (?:advances|makes|says|argues)|\bquot(?:e|ing)\b|\bdocument",
    re.IGNORECASE,
)
_VAGUE_START_RE = re.compile(
    r"^(?:tell me about|describe|discuss|explain|give an overview|overview of|summari[sz]e"
    r"|what about|what is the history of|talk about)\b",
    re.IGNORECASE,
)
_VAGUE_WORDS_RE = re.compile(
    r"\beverything\b|\ball of the\b|\betc\.?(?:\s|$)|\bin general\b|\bbroadly\b|\bsome of the\b",
    re.IGNORECASE,
)
_WH = r"(?:what|which|who|whom|how|when|where|why)"
# A wh-word opening a clause: at the start, or after a comma/semicolon/dash + and/or. Two of
# these ("what did X hold, and on what evidence?") is one question with its evidence demanded —
# the panel grades that fine. Three is a list of questions wearing one question mark.
_WH_CLAUSE_RE = re.compile(
    rf"(?:^|[,;—–-]\s*(?:and|or)\s+|[,;—–-]\s+)(?:on\s+|to\s+|in\s+|per\s+|by\s+)?{_WH}\b",
    re.IGNORECASE,
)
# Each question is researched in its OWN call, with the chapter brief but without the sibling
# questions. "The case" resolves from the brief; "these removals" / "the above" / "elsewhere in
# this book" only resolve from questions the researcher will never see.
_ANAPHORA_RE = re.compile(
    r"\b(?:these|those)\s+\w+|\bthis kind\b|\bof this kind\b|\bthe above\b|\bthe same\b"
    r"|\belsewhere in (?:this|the) book\b|\bthe (?:previous|preceding|earlier|prior) "
    r"(?:question|chapter|section)s?\b|\bas (?:above|noted|discussed)\b|\bsuch (?:trading|conduct"
    r"|claims?|payments?|awards?|cases?)\b",
    re.IGNORECASE,
)
_INSTRUCTION_RE = re.compile(
    r"\bdo not\b|\bdon't\b|\bnever\b|\bonly what\b|\breport only\b|\bunless it appears\b"
    r"|\bsay exactly\b|\bis (?:unreachable|blocked)\b|\bare (?:unreachable|blocked)\b"
    r"|\bcannot be reached\b",
    re.IGNORECASE,
)
_NEGATED_HOST_RE = re.compile(
    r"(?:unreachable|blocked|not reachable|cannot|can't|do not|don't|never|instead of|rather "
    r"than|avoid)",
    re.IGNORECASE,
)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower().rstrip("?.! "))


def _words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9][\w'’.-]*", text)


def _has_explicit_subject(question: str) -> bool:
    """A proper noun (capitalised, not sentence-initial), a digit, a quoted name, or a URL/host."""
    if re.search(r"\d", question) or re.search(r"[\"“'‘][^\"”'’]{3,}[\"”'’]", question):
        return True
    if re.search(r"\b[a-z0-9-]+\.(?:gov|org|com|net|edu|io)\b", question, re.IGNORECASE):
        return True
    words = _words(question)
    for i, w in enumerate(words):
        if i == 0:
            continue
        if w[0].isupper() and w not in {"I"}:
            prev = words[i - 1]
            # A capital after "?" / ":" is sentence-initial too.
            if prev.endswith((".", "?", ":")):
                continue
            return True
    return False


def _mentions_host(text: str, host: str) -> bool:
    return re.search(rf"(?<![\w.-]){re.escape(host)}(?![\w-])", text, re.IGNORECASE) is not None


def _host_negated(text: str, host: str) -> bool:
    """True when the host is mentioned as a warning ("X is unreachable"), not as a source."""
    for m in re.finditer(rf"(?<![\w.-]){re.escape(host)}(?![\w-])", text, re.IGNORECASE):
        window = text[max(0, m.start() - 80) : m.end() + 40]
        if _NEGATED_HOST_RE.search(window):
            return True
    return False


# --- rules ----------------------------------------------------------------------------------------


def _book_rules(book: BookConfig) -> Iterable[LintFinding]:
    for text, where in ((book.title, "title"), (book.description, "description")):
        if _is_placeholder(text):
            yield LintFinding(
                rule="placeholder",
                severity="error",
                message=f"book {where} still has skeleton placeholder text: {text.strip()[:60]!r}",
            )
    if len(book.description.split()) < 8:
        yield LintFinding(
            rule="thin-description",
            severity="warn",
            message="book description is under 8 words — the planner steers by it; state "
            "the thesis",
        )

    seen_numbers: dict[int, int] = {}
    for ch in book.chapters:
        seen_numbers[ch.number] = seen_numbers.get(ch.number, 0) + 1
    for number, count in seen_numbers.items():
        if count > 1:
            yield LintFinding(
                rule="chapter-numbering",
                severity="error",
                chapter=number,
                message=f"chapter number {number} is used {count} times — knowledge/ dirs collide",
            )
    numbers = [ch.number for ch in book.chapters]
    if (
        numbers
        and numbers != list(range(1, len(numbers) + 1))
        and len(seen_numbers) == len(numbers)
    ):
        yield LintFinding(
            rule="chapter-numbering",
            severity="warn",
            message=f"chapter numbers are not 1..{len(numbers)} in order: {numbers}",
        )

    seen_q: dict[str, int] = {}
    for ch in book.chapters:
        for q in ch.research_questions:
            key = _norm(q)
            if key in seen_q:
                yield LintFinding(
                    rule="duplicate-question",
                    severity="error",
                    chapter=ch.number,
                    question=q,
                    message=f"duplicate of a question in chapter {seen_q[key]}",
                )
            else:
                seen_q[key] = ch.number


def _chapter_rules(ch: ChapterOutline, policy: SourcePolicy) -> Iterable[LintFinding]:
    n = ch.number
    for text, where in ((ch.title, "title"), (ch.description, "description")):
        if _is_placeholder(text):
            yield LintFinding(
                rule="placeholder",
                severity="error",
                chapter=n,
                message=f"chapter {where} still has skeleton placeholder text: "
                f"{text.strip()[:60]!r}",
            )
    if len(ch.description.split()) < 5:
        yield LintFinding(
            rule="thin-description",
            severity="error" if not ch.description.strip() else "warn",
            chapter=n,
            message="chapter description is missing or under 5 words — the researcher now reads "
            "it as the chapter brief; say what the chapter must establish",
        )

    qs = ch.research_questions
    if not qs:
        yield LintFinding(
            rule="no-questions",
            severity="error",
            chapter=n,
            message="chapter has no research questions — the planner has nothing to sprint on",
        )
        return
    if len(qs) > MAX_QUESTIONS_PER_CHAPTER:
        yield LintFinding(
            rule="too-many-questions",
            severity="warn",
            chapter=n,
            message=f"{len(qs)} questions (max {MAX_QUESTIONS_PER_CHAPTER} recommended) — a sprint "
            "takes 2-4, so extras wait several sprints; split the chapter or trim",
        )
    elif len(qs) < MIN_QUESTIONS_PER_CHAPTER:
        yield LintFinding(
            rule="too-few-questions",
            severity="warn",
            chapter=n,
            message=f"only {len(qs)} question(s) — one sprint will exhaust the chapter",
        )

    if not any(_COUNTER_RE.search(q) for q in qs):
        yield LintFinding(
            rule="no-counter-narrative",
            severity="warn",
            chapter=n,
            message="no counter-narrative question (strongest opposing view / disconfirming "
            "evidence) — the panel weights counter_narrative 1.5x and has a lens hunting for it",
        )
    if not ch.sources and not any(_SOURCE_DEMAND_RE.search(q) for q in qs):
        yield LintFinding(
            rule="no-source-demand",
            severity="warn",
            chapter=n,
            message="no chapter sources and no question demands a source ('per the filing', "
            "'with dates and primary sources') — the researcher will answer from memory",
        )

    for q in qs:
        yield from _question_rules(n, q, policy)


def _question_rules(n: int, q: str, policy: SourcePolicy) -> Iterable[LintFinding]:
    words = _words(q)
    if len(words) < MIN_QUESTION_WORDS or _VAGUE_START_RE.search(q) or _VAGUE_WORDS_RE.search(q):
        yield LintFinding(
            rule="vague-question",
            severity="warn",
            chapter=n,
            question=q,
            message="reads as a topic, not a question — name the specific claim, entity, period, "
            "or figure to establish (vague questions score low on depth and claim_verification)",
        )
    if q.count("?") > 1 or len(_WH_CLAUSE_RE.findall(q)) >= 3:
        yield LintFinding(
            rule="compound-question",
            severity="warn",
            chapter=n,
            question=q,
            message="several questions in one — split them so the panel can grade each claim "
            "cleanly and a partial answer is not scored as a miss",
        )
    if (m := _ANAPHORA_RE.search(q)) and not _has_explicit_subject(q):
        yield LintFinding(
            rule="no-explicit-subject",
            severity="warn",
            chapter=n,
            question=q,
            message=f"refers to {m.group(0)!r} — each question is researched in its own call "
            "without the sibling questions, so name what it refers to (it was once answered "
            "about an unrelated case for exactly this reason)",
        )
    for host in policy.blocked:
        if _mentions_host(q, host) and not _host_negated(q, host):
            yield LintFinding(
                rule="blocked-host-named",
                severity="warn",
                chapter=n,
                question=q,
                message=f"names {host}, which the proxy cannot reach — point at a reachable "
                "repository or the question is unanswerable as posed",
            )
    if len(words) > 40 and _INSTRUCTION_RE.search(q):
        yield LintFinding(
            rule="instruction-in-question",
            severity="warn",
            chapter=n,
            question=q,
            message="long question carrying instructions ('do NOT', 'report ONLY', 'X is "
            "unreachable') — move the rule to chapter `guidance:` / `sources:` so it applies to "
            "every sprint and the question can be short",
        )


def lint_book(book: BookConfig, policy: SourcePolicy | None = None) -> list[LintFinding]:
    """All deterministic findings, errors first, then in outline order."""
    policy = policy or book.sources
    findings = list(_book_rules(book))
    for ch in book.chapters:
        findings.extend(_chapter_rules(ch, policy))
    findings.sort(key=lambda f: (f.severity != "error", f.chapter or 0))
    return findings


def has_errors(findings: list[LintFinding]) -> bool:
    return any(f.severity == "error" for f in findings)


# --- critic (optional, model-backed) --------------------------------------------------------------

CRITIC_SYSTEM = """\
You are a research-question CRITIC for a non-fiction book. Each question you are given will be \
handed, alone, to a web-searching research model, and its answer will then be graded by an \
adversarial panel on five dimensions (7+ passes):
- source_diversity: multiple independent source types; primary sources over advocacy summaries
- claim_verification: every claim backed by a retrieved source, hedged when uncertain
- counter_narrative (weighted 1.5x): opposing views and disconfirming evidence engaged
- depth (weighted 1.5x): specific dates, figures, named people, mechanisms — not generalities
- actionability: a writer can draft a section from the answer

For EACH question predict the dimension it will lose the most points on and why, given only the \
question's wording and the chapter brief. Good questions name their subject explicitly, ask for \
one claim, demand a source ("per the filing", "with dates"), and point at a repository the \
researcher can actually reach. Offer a rewrite only when it would raise the predicted score; \
otherwise set rewrite to null. Do not invent facts about the subject — judge the QUESTION.

Return ONLY JSON:
{"chapter": <n>, "critiques": [{"question": "<verbatim>", "weakest_dimension": "<one of the \
five>", "predicted_score": <1-10>, "problem": "<one sentence>", "rewrite": "<text or null>"}]}"""


def critique_chapter(
    book: BookConfig,
    chapter: ChapterOutline,
    policy: SourcePolicy,
    *,
    pool: Pool,
    max_tokens: int = 8192,
    timeout: float = 300.0,
) -> ChapterCritique | None:
    """One structured call: the chapter's questions → per-question weakest dimension + rewrite."""
    from forge.book_researcher.brief import render_chapter_brief

    user = (
        render_chapter_brief(book, chapter, policy)
        + "\n\nQuestions to critique (echo each verbatim):\n"
        + "\n".join(f"{i}. {q}" for i, q in enumerate(chapter.research_questions, 1))
    )
    expected = {_norm(q) for q in chapter.research_questions}

    def _covers(c: ChapterCritique) -> bool:
        return c.chapter == chapter.number and bool(
            {_norm(x.question) for x in c.critiques} & expected
        )

    result = structured(
        pool=pool,
        schema=ChapterCritique,
        system=CRITIC_SYSTEM,
        user=user,
        max_tokens=max_tokens,
        timeout=timeout,
        predicate=_covers,
    )
    if result.value is None:
        print(f"  critic produced nothing usable for chapter {chapter.number}: {result.error}")
    return result.value


def critiques_to_findings(
    critiques: list[ChapterCritique], *, threshold: int = 7
) -> list[LintFinding]:
    """Critic verdicts below ``threshold`` become warn-level findings under the `critic` rule."""
    out: list[LintFinding] = []
    for c in critiques:
        for q in c.critiques:
            if q.predicted_score >= threshold:
                continue
            msg = f"predicted {q.predicted_score}/10, weakest on {q.weakest_dimension}: {q.problem}"
            if q.rewrite:
                msg += f"\n      rewrite: {q.rewrite}"
            out.append(
                LintFinding(
                    rule="critic",
                    severity="warn",
                    chapter=c.chapter,
                    question=q.question,
                    message=msg,
                )
            )
    return out


# --- report ---------------------------------------------------------------------------------------


def render_lint_report(book: BookConfig, findings: list[LintFinding]) -> str:
    errors = sum(1 for f in findings if f.severity == "error")
    warns = len(findings) - errors
    lines = [
        f"Lint: {book.title} — {len(book.chapters)} chapters, {errors} errors, {warns} warnings"
    ]
    if not findings:
        lines.append("  clean")
        return "\n".join(lines)

    by_chapter: dict[int | None, list[LintFinding]] = {}
    for f in findings:
        by_chapter.setdefault(f.chapter, []).append(f)
    for chapter in sorted(by_chapter, key=lambda c: (c is not None, c or 0)):
        header = "book" if chapter is None else f"chapter {chapter}"
        ch = book.chapter(chapter) if chapter is not None else None
        if ch is not None:
            header += f": {ch.title}"
        lines.append(f"\n{header}")
        for f in by_chapter[chapter]:
            tag = "ERROR" if f.severity == "error" else "warn "
            lines.append(f"  {tag} [{f.rule}] {f.message}")
            if f.question:
                q = f.question if len(f.question) <= 110 else f.question[:107] + "..."
                lines.append(f"        Q: {q}")
    return "\n".join(lines)
