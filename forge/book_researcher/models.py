from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# --- the outline (book.yaml) ---------------------------------------------------------------------


class SourcePolicy(BaseModel):
    """Where the primary record lives and what the tool proxy can actually reach.

    The researcher only ever saw the question string, so this knowledge used to be crammed into
    every question ("... via CourtListener; nycourts.gov is unreachable"). Now it is data: the
    harness injects it into the planner and researcher prompts, and `forge book probe` maintains
    the machine-owned half of it (``reachability.json``) without touching the human-authored YAML.
    """

    reachable: list[str] = []  # hosts / repositories the researcher should prefer
    blocked: list[str] = []  # hosts the proxy egress cannot fetch — never cite, never ask for
    notes: list[str] = []  # free-form ("EDGAR full-text search works; browse-edgar does not")


class ChapterOutline(BaseModel):
    number: int
    title: str
    description: str
    research_questions: list[str]
    sources: list[str] = []  # primary-source repositories for this chapter
    guidance: list[str] = []  # chapter-specific rules ("never ask for a docket number")


class BookConfig(BaseModel):
    title: str
    description: str
    chapters: list[ChapterOutline]
    guidance: list[str] = []  # book-wide rules, read by planner + researcher on every call
    sources: SourcePolicy = Field(default_factory=SourcePolicy)

    def chapter(self, number: int) -> ChapterOutline | None:
        for ch in self.chapters:
            if ch.number == number:
                return ch
        return None


# --- the sprint loop ------------------------------------------------------------------------------


class SprintContract(BaseModel):
    sprint_id: str
    chapter: int
    questions: list[str]
    success_criteria: list[str]
    priority: Literal["high", "medium", "low"]


class ResearchFinding(BaseModel):
    question: str
    answer: str
    sources: list[str]
    confidence: Literal["high", "medium", "low"]


class SprintFindings(BaseModel):
    sprint_id: str
    chapter: int
    findings: list[ResearchFinding]
    raw_search_notes: str = ""


class VerificationScores(BaseModel):
    source_diversity: int
    claim_verification: int
    counter_narrative: int
    depth: int
    actionability: int
    overall: int


class VerificationResult(BaseModel):
    sprint_id: str
    scores: VerificationScores
    passed: bool
    feedback: str
    follow_up_questions: list[str]


# --- reachability (`forge book probe`) -----------------------------------------------------------

HostState = Literal["reachable", "blocked", "unknown"]


class HostVerdict(BaseModel):
    host: str
    url: str
    status: int | None = None
    state: HostState
    note: str = ""


class Reachability(BaseModel):
    """Machine-owned record of what the proxy egress could fetch, written beside the sprints.

    Kept out of book.yaml on purpose: the YAML is the human's, this is the probe's. The two are
    merged at load time into the effective ``SourcePolicy`` the prompts see.
    """

    probed_at: str
    proxy: str | None = None
    hosts: dict[str, HostVerdict] = {}


# --- outline lint ---------------------------------------------------------------------------------

Severity = Literal["error", "warn"]


class LintFinding(BaseModel):
    rule: str
    severity: Severity
    message: str
    chapter: int | None = None
    question: str | None = None


class QuestionCritique(BaseModel):
    """A local model's read of one question against the verifier's own rubric."""

    question: str
    weakest_dimension: Literal[
        "source_diversity", "claim_verification", "counter_narrative", "depth", "actionability"
    ]
    predicted_score: int = Field(ge=1, le=10)
    problem: str
    rewrite: str | None = None


class ChapterCritique(BaseModel):
    chapter: int
    critiques: list[QuestionCritique]


# --- outline revision (`forge book revise`) ------------------------------------------------------

EditOp = Literal[
    "replace_question",
    "add_question",
    "remove_question",
    "add_guidance",  # book-wide rule
    "add_chapter_guidance",
    "add_chapter_source",
    "block_host",
    "note",  # an observation with no edit — surfaced to the human, never applied
]


class OutlineEdit(BaseModel):
    op: EditOp
    chapter: int | None = None
    old: str | None = None  # exact text of the question being replaced/removed
    new: str | None = None
    reason: str
    evidence: list[str] = []  # sprint ids the reason rests on


class RevisionProposal(BaseModel):
    summary: str
    edits: list[OutlineEdit]


# --- decomposition (`forge book decompose`) ------------------------------------------------------


class SlicingOption(BaseModel):
    name: str
    principle: str  # "by mechanism: each has its own evidentiary standard and primary record"
    tradeoffs: str


class ChapterSketch(BaseModel):
    title: str
    one_line: str
    primary_sources: list[str] = []


class OutlineFraming(BaseModel):
    """The one editorial judgment in the pipeline, held for a human.

    Everything downstream (questions, sources, guidance) is evidence-in/structured-out, but HOW a
    book is sliced — by mechanism, by chronology, by actor — is taste, and a wrong slice makes
    every question harder to satisfy. Decomposition hard-refuses until ``approved`` is flipped by
    a human action or the framing was human-written to begin with.
    """

    thesis: str
    audience: str = ""
    slicing_options: list[SlicingOption] = []
    recommended: str
    chapter_sketch: list[ChapterSketch]
    guidance: list[str] = []
    approved: bool = False
