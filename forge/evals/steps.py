"""Step adapters — render production system/user messages for every graded prompt surface.

One adapter per graded prompt surface, keyed by step.  Each adapter's ``build`` method reads
the ``GoldCase`` inputs (already resolved text from fixture files) and produces a ``PromptSpec``
containing the exact system prompt, user message, and Pydantic schema the production code uses.

Importing private helpers from agent modules is the POINT — it guarantees the eval renders
what production renders.  Agent modules must not import ``forge.evals``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field

import yaml
from pydantic import BaseModel

from forge.evals.fixtures import read_input
from forge.evals.models import GoldCase, StepName

# ---------------------------------------------------------------------------
# Wire models for steps that need them
# ---------------------------------------------------------------------------


class ConfirmVerdict(BaseModel):
    """Wire shape for the review-confirm verdict — matches the prompt's declared JSON."""

    real: bool
    reason: str


class TestGapItem(BaseModel):
    target: str
    gap_type: str
    why_it_matters: str
    suggested_test: str
    severity: str


class GapsEnvelope(BaseModel):
    """Wire shape for the testgap-find envelope — matches ``_GAP_SHAPE``."""

    gaps: list[TestGapItem] = field(default_factory=list)


class SkepticVerdict(BaseModel):
    """Wire shape for the testgap-skeptic verdict."""

    real: bool
    confidence: str
    severity: str
    reasoning: str


# ---------------------------------------------------------------------------
# Core types
# ---------------------------------------------------------------------------


@dataclass
class PromptSpec:
    system: str
    user: str
    schema: type[BaseModel]


@dataclass
class StepAdapter:
    step: str
    build: Callable[[GoldCase], PromptSpec]


# ---------------------------------------------------------------------------
# Helper: parse JSON input files from a GoldCase
# ---------------------------------------------------------------------------


def _read_json(case: GoldCase, name: str) -> dict:
    """Read a JSON input file and return the parsed dict."""
    text = read_input(case, name)
    return json.loads(text)


def _read_yaml(case: GoldCase, name: str) -> dict:
    """Read a YAML input file and return the parsed dict."""
    text = read_input(case, name)
    return yaml.safe_load(text) or {}


# ---------------------------------------------------------------------------
# replan adapter
# ---------------------------------------------------------------------------


def _build_replan(case: GoldCase) -> PromptSpec:
    from forge.coding_pipeline.architect import (
        REPLAN_SYSTEM,
        ReplanEnvelope,
        _replan_user,
    )
    from forge.coding_pipeline.models import (
        FramingProposal,
        LeafSpec,
        WaveReport,
    )

    framing = FramingProposal.model_validate(_read_json(case, "framing.json"))
    tree = [LeafSpec.model_validate(d) for d in _read_json(case, "tree.json")]
    report = WaveReport.model_validate(_read_json(case, "report.json"))
    attempts = _read_json(case, "attempts.json")

    from forge.coding_pipeline.architect import deterministic_escalations

    escalated = deterministic_escalations(report, attempts)
    user = _replan_user(framing, tree, report, attempts, escalated)

    return PromptSpec(system=REPLAN_SYSTEM, user=user, schema=ReplanEnvelope)


# ---------------------------------------------------------------------------
# decompose adapter
# ---------------------------------------------------------------------------


def _build_decompose(case: GoldCase) -> PromptSpec:
    from forge.coding_pipeline.architect import (
        DECOMPOSE_SYSTEM,
        _decompose_user,
    )
    from forge.coding_pipeline.models import FramingProposal, Inventory, TaskTree

    framing = FramingProposal.model_validate(_read_json(case, "framing.json"))
    inventory = Inventory.model_validate(_read_json(case, "inventory.json"))
    user = _decompose_user(framing, inventory)

    return PromptSpec(system=DECOMPOSE_SYSTEM, user=user, schema=TaskTree)


# ---------------------------------------------------------------------------
# boundedness adapter
# ---------------------------------------------------------------------------


def _build_boundedness(case: GoldCase) -> PromptSpec:
    from forge.coding_pipeline.architect import (
        BOUNDEDNESS_SYSTEM,
        LeafBoundedness,
        _leaf_summary,
    )
    from forge.coding_pipeline.models import LeafSpec

    leaf = LeafSpec.model_validate(_read_json(case, "leaf.json"))
    user = _leaf_summary(leaf)

    return PromptSpec(system=BOUNDEDNESS_SYSTEM, user=user, schema=LeafBoundedness)


# ---------------------------------------------------------------------------
# review-findings adapter
# ---------------------------------------------------------------------------


def _build_review_findings(case: GoldCase) -> PromptSpec:
    from forge.coding_pipeline.verify import (
        FINDINGS_SYSTEM,
        FindingsEnvelope,
    )

    diff = read_input(case, "diff.patch")
    user = f"Wave diff:\n\n{diff}"

    return PromptSpec(system=FINDINGS_SYSTEM, user=user, schema=FindingsEnvelope)


# ---------------------------------------------------------------------------
# review-confirm adapter
# ---------------------------------------------------------------------------


def build_confirm_user(diff: str, summary: str, file: str | None, severity: str) -> str:
    """Mirror the ``make_user`` lambda in ``verify.confirm_findings`` exactly.

    Shared by the review-confirm adapter and the runner's review pipeline flow
    so the eval's confirm calls render byte-identically to production's."""
    return (
        f"Finding: {summary}\nFile: {file or 'unspecified'} "
        f"(severity claimed: {severity})\n\nWave diff:\n\n{diff}"
    )


def _build_review_confirm(case: GoldCase) -> PromptSpec:
    from forge.coding_pipeline.verify import (
        CONFIRM_SYSTEM,
    )

    diff = read_input(case, "diff.patch")
    candidate = _read_json(case, "candidate.json")

    user = build_confirm_user(
        diff,
        candidate.get("summary", ""),
        candidate.get("file"),
        candidate.get("severity", "medium"),
    )

    return PromptSpec(system=CONFIRM_SYSTEM, user=user, schema=ConfirmVerdict)


# ---------------------------------------------------------------------------
# testgap-find adapter
# ---------------------------------------------------------------------------


def _build_testgap_find(case: GoldCase) -> PromptSpec:
    from forge.testing_ensemble.prompts import FINDER_ANGLES, finder_system

    context = read_input(case, "context.md")
    # The angle rides in the expected block (case config the loader already
    # carries) — a self-referencing "case.yaml" input entry is not a contract
    # the loader supports.
    angle_key = (case.expected or {}).get("angle", "coverage")

    # Find the matching angle directive from FINDER_ANGLES
    angle_directive = ""
    focus = angle_key
    for name, directive in FINDER_ANGLES:
        if name == angle_key:
            angle_directive = directive
            break

    if not angle_directive:
        angle_directive = FINDER_ANGLES[0][1]
        focus = FINDER_ANGLES[0][0]

    system = finder_system(focus, angle_directive)

    return PromptSpec(system=system, user=context, schema=GapsEnvelope)


# ---------------------------------------------------------------------------
# testgap-skeptic adapter
# ---------------------------------------------------------------------------


def _build_testgap_skeptic(case: GoldCase) -> PromptSpec:
    from forge.testing_ensemble.prompts import SKEPTIC_BASE, verify_user

    context = read_input(case, "context.md")
    gap = _read_json(case, "gap.json")

    # Convert dict to model for verify_user which expects a model
    from forge.testing_ensemble.models import TestGap

    gap_model = TestGap.model_validate(gap)
    user = verify_user(context, gap_model.model_dump_json())

    return PromptSpec(system=SKEPTIC_BASE, user=user, schema=SkepticVerdict)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


ADAPTERS: dict[str, StepAdapter] = {
    "replan": StepAdapter(step="replan", build=_build_replan),
    "decompose": StepAdapter(step="decompose", build=_build_decompose),
    "boundedness": StepAdapter(step="boundedness", build=_build_boundedness),
    "review-findings": StepAdapter(step="review-findings", build=_build_review_findings),
    "review-confirm": StepAdapter(step="review-confirm", build=_build_review_confirm),
    "testgap-find": StepAdapter(step="testgap-find", build=_build_testgap_find),
    "testgap-skeptic": StepAdapter(step="testgap-skeptic", build=_build_testgap_skeptic),
}


def get_adapter(step: StepName) -> StepAdapter:
    """Return the adapter for *step*. Raises KeyError when *step* is unknown."""
    adapter = ADAPTERS.get(step)
    if adapter is None:
        raise KeyError(f"unknown step: {step}")
    return adapter
