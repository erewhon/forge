"""Data shapes for the coding pipeline (design: ``coding-pipeline-design.md``).

The architect turns a ``GoalSpec`` into a ``FramingProposal`` (A1 — gated on human approval),
then a tree of ``LeafSpec``s (A2). The orchestrator dispatches leaves and journals each wave as a
``WaveRecord``: the ``WaveReport`` of what happened plus the ``ReplanAction``s taken (A4). Leaf
titles are validated comma-free here because Forge's ``Depends On`` cell format splits entries on
commas — catching it at model construction beats catching it at emission.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, Field, field_validator

_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")


# --- goal input -------------------------------------------------------------


class GoalSpec(BaseModel):
    """The "build X" input to ``meta build plan`` — authored by a human, never a model."""

    goal: str
    project: str  # Forge project name (must already exist)
    repo: Path | None = None  # target repo; None = the project's conventional checkout
    context: str = ""  # constraints, links, prior art the architect should read
    value_hints: list[str] = []  # what should ship user-visible value first
    epic_slug: str | None = None  # override the architect's derived slug

    @classmethod
    def load(cls, path: Path) -> GoalSpec:
        """Load from ``.yaml``/``.yml`` (the whole document) or ``.md`` (YAML frontmatter holds
        the fields; the markdown body is appended to ``context``)."""
        text = path.read_text()
        if path.suffix in {".yaml", ".yml"}:
            return cls.model_validate(yaml.safe_load(text))
        if path.suffix in {".md", ".markdown"}:
            data, body = _split_frontmatter(text)
            if body:
                data["context"] = f"{data.get('context', '')}\n\n{body}".strip()
            return cls.model_validate(data)
        raise ValueError(f"unsupported goal spec format '{path.suffix}' (use .yaml/.yml or .md)")


def _split_frontmatter(text: str) -> tuple[dict, str]:
    """Split a markdown document into (frontmatter dict, body). Frontmatter is required — a goal
    spec without ``goal``/``project`` fields isn't loadable."""
    if not text.startswith("---"):
        raise ValueError("markdown goal spec needs YAML frontmatter with 'goal' and 'project'")
    parts = text.split("\n---", 1)
    if len(parts) != 2:
        raise ValueError("markdown goal spec frontmatter is not closed with '---'")
    data = yaml.safe_load(parts[0].removeprefix("---")) or {}
    if not isinstance(data, dict):
        raise ValueError("markdown goal spec frontmatter must be a YAML mapping")
    return data, parts[1].lstrip("\n")


# --- A0: inventory ----------------------------------------------------------


class FileHead(BaseModel):
    """The opening lines of a key file (CLAUDE.md, README, manifests) for architect context."""

    path: str
    head: str


class ExistingTask(BaseModel):
    """Compact row of a Forge task already filed for the project — decomposition dedup context."""

    task: str
    status: str
    feature: str = ""
    external_ref: str = ""


class Inventory(BaseModel):
    """What A0 hands the framing stage: enough repo + Forge reality to push back on the goal."""

    project: str
    repo: str
    tree: str  # indented directory tree, depth-capped, ignore-aware
    key_files: list[FileHead] = []
    modules: list[str] = []  # top-level non-ignored directories
    test_layout: list[str] = []  # directories containing test files
    toolchain: list[str] = []  # detected: uv/python, pnpm/node, cargo/rust, ...
    existing_tasks: list[ExistingTask] = []
    overlaps: list[str] = []  # repo paths lexically related to the goal's terms
    truncated: int = 0  # items dropped by size caps — counted, never silent


# --- A1: framing ------------------------------------------------------------


class FramingOption(BaseModel):
    name: str
    summary: str
    tradeoffs: str = ""


class FramingProposal(BaseModel):
    """The architect's first output — a scoping position, not a tree. Push-back is first-class:
    ``restated_goal`` may disagree with ``goal_as_stated`` (``rescoped=True``). Decomposition
    hard-refuses until a human flips ``approved`` — no code path may set it."""

    goal_as_stated: str
    restated_goal: str
    rescoped: bool = False
    inventory_summary: str = ""
    gap_analysis: str = ""
    options: list[FramingOption] = []
    recommendation: str
    value_ordering: list[str] = []  # which slices ship user-visible value first
    risks: list[str] = []
    epic_slug: str
    branch: str | None = None  # suggested; orchestrator defaults to {branch_prefix}/{epic_slug}
    approved: bool = False

    @field_validator("epic_slug")
    @classmethod
    def _slug_safe(cls, v: str) -> str:
        if not _SLUG_RE.match(v):
            raise ValueError(f"epic_slug {v!r} must be a lowercase [a-z0-9-] slug")
        return v


# --- A2: decomposition ------------------------------------------------------


class BoundednessCheck(BaseModel):
    """The five worker-shaped criteria. A leaf failing any is split further or terminally tagged
    novel/Manual — 'a human does this one' is a valid stop, not an error."""

    single_concern: bool  # no unresolved design choice inside
    bounded_diff: bool  # expected diff fits the leaf's max_files
    small_estimate: bool  # estimate <= m
    testable_acceptance: bool  # acceptance criteria checkable by the project suite
    files_named: bool  # the spec names its target files/modules
    notes: str = ""

    @property
    def worker_shaped(self) -> bool:
        return (
            self.single_concern
            and self.bounded_diff
            and self.small_estimate
            and self.testable_acceptance
            and self.files_named
        )


class LeafSpec(BaseModel):
    """One tree leaf with its full autonomy tag set. ``content`` must be a complete worker spec
    (what/why, acceptance criteria, files hint, test expectations) — the worker sees nothing else.
    Defaults are the conservative-tagging position: Manual until the architect argues otherwise."""

    title: str
    content: str
    feature: str
    depends_on: list[str] = []  # leaf titles within the same tree
    priority: int = 5
    phase: str = "Feature"
    status: Literal["Ready", "Spec Needed"] = "Ready"
    execution_mode: Literal["Manual", "Auto-OK", "Auto-Preferred"] = "Manual"
    complexity: Literal["routine", "novel"] | None = None
    estimate: Literal["xs", "s", "m", "l", "xl"] | None = None
    task_type: Literal["bug-fix", "feature", "refactor", "docs", "test", "chore"] = "feature"
    requires_tests: bool = True
    max_files: int | None = None
    # "coder" is the conservative-tagging floor for autonomous leaves — the router's
    # bare "auto" often answers text-only in opencode sessions (e2e dry-run finding).
    model_tier: Literal["auto", "auto-free", "auto-full", "coder"] | None = None
    boundedness: BoundednessCheck | None = None

    @field_validator("title")
    @classmethod
    def _title_comma_free(cls, v: str) -> str:
        if "," in v:
            raise ValueError(
                f"leaf title {v!r} contains a comma — Forge Depends On cells split on commas"
            )
        return v

    @field_validator("depends_on")
    @classmethod
    def _deps_comma_free(cls, v: list[str]) -> list[str]:
        for dep in v:
            if "," in dep:
                raise ValueError(f"dependency title {dep!r} contains a comma")
        return v


class TaskTree(BaseModel):
    """The decomposition output: the flat leaf list whose deps encode the tree.

    Persisted as ``tree.json`` in the run dir; A3 emission maps it to Forge."""

    leaves: list[LeafSpec] = []


# --- wave planning ------------------------------------------------------------


class LeafRow(BaseModel):
    """A normalized Forge task row as the wave planner sees it (null-as-manual applied,
    blocked state resolved)."""

    task: str
    status: str
    execution_mode: str = "Manual"
    priority: int = 99
    blocked: bool = False
    blocked_by: list[str] = []
    external_ref: str = ""  # lets consumers spot fix-up leaves (pipeline:<epic>:fix:*)


class BlockedLeaf(BaseModel):
    task: str
    blocked_by: list[str] = []


class WavePlan(BaseModel):
    """What the orchestrator may dispatch next, plus the counts it needs to tell
    "dry" (epic exhausted → run the epic gate) from "waiting on humans".

    Scope is the epic (external_ref-prefix membership); ``feature`` is the optional
    narrowing filter that produced this plan, "" when the whole epic was planned."""

    epic_slug: str = ""
    feature: str = ""
    project: str
    dispatch: list[str] = []  # leaf titles in dispatch order, capped at wave_size
    ready_manual: int = 0  # Ready but human-owned (Manual, unblocked)
    spec_needed: int = 0
    in_progress: int = 0
    done: int = 0
    blocked: list[BlockedLeaf] = []

    @property
    def dry(self) -> bool:
        """Nothing dispatchable and nothing that could ever become dispatchable —
        the epic's tree is exhausted; the orchestrator proceeds to the epic gate."""
        return (
            not self.dispatch
            and not self.blocked
            and self.ready_manual == 0
            and self.spec_needed == 0
            and self.in_progress == 0
        )

    @property
    def waiting_on_human(self) -> bool:
        """Nothing dispatchable now, but human-owned or blocked leaves remain —
        the orchestrator reports and exits cleanly rather than spinning."""
        return not self.dispatch and not self.dry


# --- wave execution ---------------------------------------------------------


class LeafOutcome(BaseModel):
    """The pipeline's journal record of one dispatched leaf. The dispatcher maps whatever
    ``task_worker.run_one`` returns into this shape — task_worker is a lower layer and never
    imports this package."""

    leaf: str  # leaf title
    status: Literal["done", "failed", "skipped"]
    reason: str = ""
    commit_id: str | None = None
    changed_files: list[str] = []
    duration_s: float = 0.0


class SuiteResult(BaseModel):
    """Whole-project test run on the wave's accumulated state — the hard wave gate."""

    passed: bool
    output_tail: str = ""


class ReviewFinding(BaseModel):
    """One confirmed finding from the wave's advisory review. ``slug`` must be stable across
    replans — it keys the fix-up leaf's external_ref (``pipeline:{epic}:fix:{slug}``)."""

    slug: str
    summary: str
    severity: Literal["critical", "high", "medium", "low"] = "medium"
    file: str | None = None
    confirmed: bool = False


class WaveReport(BaseModel):
    """What a wave actually did — the replan stage's entire input."""

    wave: int
    outcomes: list[LeafOutcome] = []
    suite: SuiteResult | None = None
    findings: list[ReviewFinding] = []
    # The review funnel: raw candidates collected, whether consolidation ran clean
    # (False = fail-open passthrough), and candidates dropped as already covered by
    # open fix-up leaves.
    raw_findings: int = 0
    consolidation_ok: bool = True
    dropped_covered: list[str] = []
    diff_stat: str = ""

    @property
    def landed(self) -> list[LeafOutcome]:
        return [o for o in self.outcomes if o.status == "done"]

    @property
    def failed(self) -> list[LeafOutcome]:
        return [o for o in self.outcomes if o.status == "failed"]

    @property
    def suite_green(self) -> bool:
        return self.suite is not None and self.suite.passed


# --- A4: replan actions -----------------------------------------------------
# A discriminated union so journal round-trips revive the right class. Escalation at the attempt
# cap is deterministic pre-rule code in the orchestrator; the LLM proposes the rest.


class FixupAction(BaseModel):
    """A confirmed wave finding becomes a new leaf (ref ``pipeline:{epic}:fix:{finding_slug}``)."""

    kind: Literal["fixup"] = "fixup"
    finding_slug: str
    leaf: LeafSpec


class RespecAction(BaseModel):
    """A failed leaf (attempts below the cap) gets a rewritten/split spec and returns to Ready."""

    kind: Literal["respec"] = "respec"
    leaf_title: str
    revised: LeafSpec
    rationale: str = ""


class SplitSubtreeAction(BaseModel):
    """Repeated failures across one feature: park its remaining leaves as Spec Needed and re-run
    decomposition on the subtree with the wave history as context."""

    kind: Literal["split_subtree"] = "split_subtree"
    feature: str
    rationale: str = ""


class EscalateAction(BaseModel):
    """Leaf at the attempt cap → Spec Needed + Manual with diagnostics; a human takes it."""

    kind: Literal["escalate"] = "escalate"
    leaf_title: str
    diagnostics: str = ""


class IntegrationFixAction(BaseModel):
    """Wave suite red though leaves individually passed: a strong-tier leaf for the interaction."""

    kind: Literal["integration_fix"] = "integration_fix"
    leaf: LeafSpec
    rationale: str = ""


class HaltAction(BaseModel):
    """The framing was invalidated by what landed — stop the run and report to the human."""

    kind: Literal["halt"] = "halt"
    reason: str


ReplanAction = Annotated[
    FixupAction
    | RespecAction
    | SplitSubtreeAction
    | EscalateAction
    | IntegrationFixAction
    | HaltAction,
    Field(discriminator="kind"),
]


class WaveRecord(BaseModel):
    """One wave's full journal entry (``pipeline-runs/<epic>/wave-NNN.json``). The journal stamps
    ``timestamp`` at write time."""

    wave: int
    timestamp: str | None = None
    dispatched: list[str] = []  # leaf titles, in dispatch order
    report: WaveReport
    actions: list[ReplanAction] = []
