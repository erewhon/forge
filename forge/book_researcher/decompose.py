"""`forge book decompose` — derive an outline from a reconnaissance run, with a human gate.

The outline that worked was not brainstormed from an idea; it was derived from evidence — a
broad `forge research` recon run (its synthesis, its findings, and above all its verifier reviews,
which said *why* the broad framing kept failing). This command reproduces that derivation and
splits it at the one point that needs taste:

1. **Framing** (one structured call, or a human-written YAML): thesis, 2-3 candidate slicing
   principles with tradeoffs (by mechanism / by chronology / by actor), a recommendation, a
   chapter sketch with primary sources per chapter, and book-wide guidance. Persisted as
   ``outline-framing.json`` + ``.md`` in the project dir. ``approved`` is forced False on model
   output; only ``--approve`` (a human action) or ``--framing <file>`` (a human-authored framing)
   opens the gate. Mirrors the coding pipeline's A1→A2 gate.
2. **Decomposition** (one structured call with ``BookConfig`` as the schema and the linter as the
   predicate): approved framing + recon digest + reachability → ``book.yaml``. Output that fails
   the lint's error rules or the shape rules is retried and failed over inside the pool, so what
   lands on disk already passes ``forge book lint``.

Requires a recon slug rather than a bare idea on purpose: decomposing from nothing is how the
first attempt went wrong.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import yaml

from forge.book_researcher.brief import effective_policy
from forge.book_researcher.lint import has_errors, lint_book
from forge.book_researcher.models import (
    BookConfig,
    OutlineFraming,
    Reachability,
    SourcePolicy,
)
from forge.book_researcher.outline import write_outline
from forge.general_researcher.config import GeneralResearcherSettings
from forge.general_researcher.models import SprintFindings as ReconSprint
from forge.general_researcher.models import VerificationResult as ReconReview
from forge.shared.datectx import today_line
from forge.shared.ensemble import Pool
from forge.shared.llm import RESEARCH_FAILED_PREFIX
from forge.shared.panel import structured
from forge.shared.source_check import extract_url

MIN_CHAPTERS = 4
MAX_CHAPTERS = 16
MIN_QUESTIONS = 2
MAX_QUESTIONS = 5


class DecomposeError(RuntimeError):
    """A stage could not produce a usable artifact; ``raw`` carries the model's last output."""

    def __init__(self, *args: object, raw: str = "") -> None:
        super().__init__(*args)
        self.raw = raw


class FramingExistsError(DecomposeError):
    pass


class FramingNotApprovedError(DecomposeError):
    pass


# --- recon digest ---------------------------------------------------------------------------------


@dataclass
class ReconDigest:
    slug: str
    question: str
    context: str
    sub_questions: list[str]
    synthesis: str
    findings: list[tuple[str, str, list[str]]]  # (question, confidence, hosts)
    follow_ups: list[str]
    best_score: int | None
    sprint_count: int


def load_recon(slug: str, root: Path | None = None) -> ReconDigest:
    """Read a `forge research` topic dir. Raises FileNotFoundError with the available slugs."""
    root = root or GeneralResearcherSettings().project_dir
    topic_dir = root / slug
    if not topic_dir.is_dir():
        available = sorted(p.name for p in root.iterdir() if p.is_dir()) if root.is_dir() else []
        hint = f" Available: {', '.join(available)}." if available else ""
        raise FileNotFoundError(f"no research topic at {topic_dir}.{hint}")

    question, context, subs = slug, "", []
    topic_file = topic_dir / "topic.yaml"
    if topic_file.is_file():
        data = yaml.safe_load(topic_file.read_text()) or {}
        question = data.get("question", slug)
        context = data.get("context", "") or ""
        subs = list(data.get("sub_questions", []) or [])

    synthesis = ""
    if (topic_dir / "synthesis.md").is_file():
        synthesis = (topic_dir / "synthesis.md").read_text()

    findings: list[tuple[str, str, list[str]]] = []
    for path in sorted((topic_dir / "findings").glob("sprint-*.json")):
        try:
            sf = ReconSprint.model_validate_json(path.read_text())
        except Exception:
            continue
        for f in sf.findings:
            if f.answer.startswith(RESEARCH_FAILED_PREFIX):
                continue
            hosts = sorted({h for s in f.sources if (u := extract_url(s)) and (h := _host(u))})
            findings.append((f.question, f.confidence, hosts))

    follow_ups: list[str] = []
    best: int | None = None
    sprint_count = 0
    for path in sorted((topic_dir / "sprints").glob("sprint-*-review.json")):
        try:
            r = ReconReview.model_validate_json(path.read_text())
        except Exception:
            continue
        sprint_count += 1
        best = r.scores.overall if best is None else max(best, r.scores.overall)
        follow_ups.extend(q for q in r.follow_up_questions if q.strip())

    return ReconDigest(
        slug=slug,
        question=question,
        context=context,
        sub_questions=subs,
        synthesis=synthesis,
        findings=findings,
        follow_ups=follow_ups,
        best_score=best,
        sprint_count=sprint_count,
    )


def _host(url: str) -> str | None:
    from urllib.parse import urlparse

    return urlparse(url).hostname


def render_recon(
    d: ReconDigest, *, max_synthesis_chars: int = 12000, max_follow_ups: int = 24
) -> str:
    lines = [
        f"# Reconnaissance run: {d.slug}",
        f"Question: {d.question}",
    ]
    if d.context:
        lines.append(f"Context: {d.context}")
    if d.sub_questions:
        lines.append("Sub-questions: " + " | ".join(d.sub_questions))
    lines.append(
        f"Sprints: {d.sprint_count}, best verification score: "
        f"{d.best_score if d.best_score is not None else 'n/a'}/10"
    )
    if d.synthesis:
        syn = d.synthesis
        if len(syn) > max_synthesis_chars:
            syn = syn[:max_synthesis_chars] + "\n\n[... synthesis truncated ...]"
        lines += ["", "## Synthesis", "", syn]
    if d.findings:
        lines += ["", "## Findings (question → confidence → source hosts)"]
        for q, conf, hosts in d.findings:
            lines.append(f"- [{conf}] {q}" + (f" ← {', '.join(hosts)}" if hosts else ""))
    if d.follow_ups:
        lines += ["", "## Verifier challenges (why the broad framing kept failing)"]
        seen: set[str] = set()
        for fu in d.follow_ups:
            key = fu.strip().lower()
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"- {fu.strip()}")
            if len(seen) >= max_follow_ups:
                break
    return "\n".join(lines)


# --- framing --------------------------------------------------------------------------------------

FRAMING_SYSTEM = """\
You are framing a NON-FICTION BOOK's research outline from a reconnaissance run. This is the \
editorial step, not the outline itself: decide how the book should be SLICED into chapters so \
that every chapter's research questions are satisfiable by a web-searching researcher and \
gradable by an adversarial panel (source_diversity, claim_verification, counter_narrative, depth, \
actionability).

Mandates:
- Read the verifier challenges hard: they say why the broad framing kept failing (single-sourced, \
advocacy summaries instead of primary records, contradictions, fabricated identifiers). The slice \
you recommend must make those failures avoidable — e.g. slice by MECHANISM when each mechanism \
has its own evidentiary standard and its own primary-source repository, not by date.
- Propose 2-3 genuinely different slicing principles with tradeoffs, then recommend one.
- Sketch the chapters under the recommended slice: a title, one line on what it establishes, and \
the primary-source repositories (hosts) where its record actually lives. Include a framing/\
standards chapter first when the subject needs a consistent evidentiary standard, and a \
"case for the other side" chapter argued at full strength.
- Write 4-8 book-wide guidance rules the researcher must follow on every question — the \
discipline the challenges show was missing (adjudicated vs alleged, verify-or-retract, never \
invent identifiers, name the subject explicitly).
- Do not invent facts beyond the recon; you are organising evidence, not adding it.

Return ONLY JSON:
{"thesis": str, "audience": str, "slicing_options": [{"name": str, "principle": str, \
"tradeoffs": str}], "recommended": str, "chapter_sketch": [{"title": str, "one_line": str, \
"primary_sources": [str]}], "guidance": [str]}
Do not include an "approved" field — approval is a human decision."""


def propose_framing(
    recon: ReconDigest,
    *,
    pool: Pool,
    reachability: Reachability | None = None,
    max_tokens: int = 16384,
    timeout: float = 300.0,
) -> OutlineFraming:
    user = render_recon(recon)
    if reachability is not None:
        blocked = sorted(h for h, v in reachability.hosts.items() if v.state == "blocked")
        reachable = sorted(h for h, v in reachability.hosts.items() if v.state == "reachable")
        user += (
            "\n\n## Source reachability (probed through the researcher's egress)\n"
            f"reachable: {', '.join(reachable) or 'none probed'}\n"
            f"blocked: {', '.join(blocked) or 'none'}\n"
        )
    user += "\n\nFrame the book."
    result = structured(
        pool=pool,
        schema=OutlineFraming,
        system=f"{today_line()}\n\n{FRAMING_SYSTEM}",
        user=user,
        max_tokens=max_tokens,
        timeout=timeout,
        predicate=lambda f: MIN_CHAPTERS <= len(f.chapter_sketch) <= MAX_CHAPTERS,
    )
    if result.value is None:
        raise DecomposeError(f"framing produced no usable proposal: {result.error}", raw=result.raw)
    framing = result.value
    framing.approved = False  # only approve_framing / a human-written framing may flip this
    return framing


def render_framing(f: OutlineFraming) -> str:
    lines = ["# Outline framing", "", f"**Thesis:** {f.thesis}", ""]
    if f.audience:
        lines += [f"**Audience:** {f.audience}", ""]
    if f.slicing_options:
        lines.append("## Slicing options")
        for o in f.slicing_options:
            lines += [f"- **{o.name}** — {o.principle}", f"  - tradeoffs: {o.tradeoffs}"]
        lines.append("")
    lines += [f"**Recommended:** {f.recommended}", "", "## Chapter sketch"]
    for i, ch in enumerate(f.chapter_sketch, 1):
        src = f" (sources: {', '.join(ch.primary_sources)})" if ch.primary_sources else ""
        lines.append(f"{i}. **{ch.title}** — {ch.one_line}{src}")
    if f.guidance:
        lines += ["", "## Book-wide guidance"]
        lines.extend(f"- {g}" for g in f.guidance)
    lines += [
        "",
        "---",
        "APPROVED"
        if f.approved
        else "NOT APPROVED — edit outline-framing.json as you like, then "
        "`forge book decompose <slug> --approve`",
    ]
    return "\n".join(lines) + "\n"


def persist_framing(framing: OutlineFraming, path: Path, *, force: bool = False) -> Path:
    if path.exists() and not force:
        raise FramingExistsError(
            f"{path} already exists — edit it, `--approve` it, or re-propose with --force"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(framing.model_dump_json(indent=2))
    path.with_suffix(".md").write_text(render_framing(framing))
    return path


def load_framing(path: Path) -> OutlineFraming | None:
    if not path.is_file():
        return None
    return OutlineFraming.model_validate(json.loads(path.read_text()))


def load_human_framing(path: Path) -> OutlineFraming:
    """A framing the human wrote by hand (YAML or JSON) — approved by construction."""
    text = path.read_text()
    data = yaml.safe_load(text) if path.suffix in (".yaml", ".yml") else json.loads(text)
    framing = OutlineFraming.model_validate(data)
    framing.approved = True
    return framing


def approve_framing(path: Path) -> OutlineFraming:
    framing = load_framing(path)
    if framing is None:
        raise DecomposeError(f"no framing at {path} — run `forge book decompose <slug>` first")
    framing.approved = True
    path.write_text(framing.model_dump_json(indent=2))
    path.with_suffix(".md").write_text(render_framing(framing))
    return framing


# --- decomposition --------------------------------------------------------------------------------

DECOMPOSE_SYSTEM = """\
You are decomposing an APPROVED framing into a research outline for a non-fiction book. The \
outline drives an automated loop: each sprint a planner picks 2-4 of the questions, a \
web-searching researcher answers each one IN ITS OWN CALL (it sees the question plus the chapter \
brief, not the other questions), and an adversarial panel grades the answers on \
source_diversity, claim_verification, counter_narrative (1.5x), depth (1.5x), actionability.

Write the outline so those graders are satisfiable:
- Follow the approved chapter sketch; one chapter per sketch entry, numbered 1..N in order.
- Each chapter: a description of 2-4 sentences saying what it must establish and what standard \
of evidence applies; `sources` listing the primary-source repositories (hosts) for that chapter, \
using ONLY hosts that are reachable per the source policy; %d-%d `research_questions`.
- Each question: ONE claim, names its subject explicitly (a person, case, filing, entity, period \
— never "the case" or "these payments" alone), demands a source ("per the filing", "with dates \
and primary sources", "who advances it"), and is answerable from a reachable repository.
- Every chapter has exactly one counter-narrative question: the strongest opposing view, the \
legitimate rationale, disconfirming evidence, or who argues the other side.
- Where the recon's verifier challenges found a contradiction or a suspect figure, add a \
"verify or retract" question that asks for the record verbatim.
- Put discipline in `guidance` (book-wide rules, from the framing) and per-chapter `guidance`, \
NOT inside the questions; keep questions under ~40 words.
- Do not invent facts beyond the recon and framing.

Return ONLY JSON matching:
{"title": str, "description": str, "guidance": [str], "sources": {"reachable": [str], \
"blocked": [str], "notes": [str]}, "chapters": [{"number": int, "title": str, "description": \
str, "sources": [str], "guidance": [str], "research_questions": [str]}]}"""


def _decompose_user(
    framing: OutlineFraming, recon: ReconDigest, policy: SourcePolicy, chapter_count: int | None
) -> str:
    from forge.book_researcher.brief import render_source_policy

    parts = [
        render_framing(framing),
        "\n## Source policy\n",
        render_source_policy(policy) or "(none)",
    ]
    if chapter_count:
        parts.append(f"\nTarget chapter count: {chapter_count}.")
    parts += ["\n", render_recon(recon), "\n\nWrite the outline."]
    return "\n".join(parts)


def _shape_ok(book: BookConfig, policy: SourcePolicy) -> bool:
    if not (MIN_CHAPTERS <= len(book.chapters) <= MAX_CHAPTERS):
        return False
    for ch in book.chapters:
        if not (MIN_QUESTIONS <= len(ch.research_questions) <= MAX_QUESTIONS):
            return False
    findings = lint_book(book, policy)
    if has_errors(findings):
        return False
    # A chapter without a counter-narrative question is the panel's favourite dock; refuse it.
    return not any(f.rule == "no-counter-narrative" for f in findings)


def decompose(
    framing: OutlineFraming,
    recon: ReconDigest,
    *,
    pool: Pool,
    reachability: Reachability | None = None,
    chapter_count: int | None = None,
    max_tokens: int = 16384,
    timeout: float = 300.0,
) -> BookConfig:
    """Approved framing + recon → a BookConfig that already passes the linter's error rules."""
    if not framing.approved:
        raise FramingNotApprovedError(
            "framing has not been approved by a human — read outline-framing.md, then "
            "`forge book decompose <slug> --approve` before decomposition"
        )
    # Seed the policy from the probe so the model only names hosts that answer.
    seed = BookConfig(title="", description="", chapters=[])
    policy = effective_policy(seed, reachability)
    system = DECOMPOSE_SYSTEM % (MIN_QUESTIONS, MAX_QUESTIONS)
    result = structured(
        pool=pool,
        schema=BookConfig,
        system=f"{today_line()}\n\n{system}",
        user=_decompose_user(framing, recon, policy, chapter_count),
        max_tokens=max_tokens,
        timeout=timeout,
        predicate=lambda b: _shape_ok(b, policy),
    )
    if result.value is None:
        raise DecomposeError(
            f"decomposition produced no outline that passes lint: {result.error}",
            raw=result.raw,
        )
    book = result.value
    # The human's framing guidance is authoritative; the model may add, never drop.
    for rule in framing.guidance:
        if rule not in book.guidance:
            book.guidance.insert(0, rule)
    book.guidance = list(dict.fromkeys(book.guidance))
    # Probe results are data the YAML should carry too, so the outline is self-describing.
    merged = effective_policy(book, reachability)
    book.sources = merged
    return book


def provenance_header(recon: ReconDigest, framing: OutlineFraming) -> str:
    return (
        f"Research outline derived by `forge book decompose {recon.slug}` on "
        f"{date.today():%Y-%m-%d} "
        f"from the recon run ({recon.sprint_count} sprints, best score "
        f"{recon.best_score if recon.best_score is not None else 'n/a'}/10).\n"
        f"Slice: {framing.recommended}\n"
        "Edit freely — `forge book lint` checks it, `forge book revise` proposes edits from "
        "reviews."
    )


def write_decomposed(
    book: BookConfig, path: Path, recon: ReconDigest, framing: OutlineFraming, *, force: bool
) -> Path:
    return write_outline(book, path, header=provenance_header(recon, framing), force=force)
