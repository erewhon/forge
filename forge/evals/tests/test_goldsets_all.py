"""Regression guard for the decompose / boundedness / review / testgap gold sets.

Same contract as ``test_goldsets_replan.py``: the checked-in gold sets must
stay loadable, renderable through the production adapters, and satisfiable —
every case's ``golden.json`` (the frozen reference answer and, for non-holdout
cases, the few-shot exemplar source) must pass its grader. review-findings
goldens are ideal FINDER envelopes; the guard wraps them into the pipeline
artifact a perfect confirm stage would produce.
"""

from __future__ import annotations

import json

import pytest

from forge.evals.config import settings
from forge.evals.fixtures import load_goldsets
from forge.evals.models import GoldCase
from forge.evals.runner import GRADERS
from forge.evals.steps import ADAPTERS

STEPS = (
    "decompose",
    "boundedness",
    "review-findings",
    "review-confirm",
    "testgap-find",
    "testgap-skeptic",
)

# Authored sizes — shrinkage means someone deleted gold.
MIN_CASES = {
    "decompose": 2,
    "boundedness": 9,
    "review-findings": 4,
    "review-confirm": 8,
    "testgap-find": 4,
    "testgap-skeptic": 6,
}


def _cases(step: str) -> list[GoldCase]:
    return load_goldsets(settings.goldsets_dir, step=step)


def _all_cases() -> list[GoldCase]:
    return [c for step in STEPS for c in _cases(step)]


def _golden_raw(case: GoldCase) -> str:
    golden = (case.case_dir / "golden.json").read_text()
    if case.step != "review-findings":
        return golden
    # Wrap the ideal finder envelope into the pipeline artifact of a perfect
    # confirm stage (everything real survives; nothing else was emitted).
    data = json.loads(golden)
    candidates = [
        {
            "slug": f"g{i}",
            "summary": f["summary"],
            "file": f.get("file"),
            "severity": f.get("severity", "medium"),
        }
        for i, f in enumerate(data.get("findings", []))
    ]
    return json.dumps(
        {
            "finder_valid": True,
            "finder_raw": golden,
            "candidates": candidates,
            "confirmed": candidates,
            "votes": [],
            "confirm_errors": 0,
        }
    )


@pytest.mark.parametrize("step", STEPS)
def test_goldset_size_and_holdout(step: str):
    cases = _cases(step)
    assert len(cases) >= MIN_CASES[step], f"{step} gold set shrank below its authored size"
    holdouts = sum(1 for c in cases if c.holdout)
    assert holdouts / len(cases) >= 1 / 3, f"{step} holdout fraction fell below 1/3"


@pytest.mark.parametrize("case", _all_cases(), ids=lambda c: f"{c.step}/{c.case_id}")
def test_case_renders_through_production_adapter(case: GoldCase):
    spec = ADAPTERS[case.step].build(case)
    assert spec.system.strip()
    assert spec.user.strip()


@pytest.mark.parametrize("case", _all_cases(), ids=lambda c: f"{c.step}/{c.case_id}")
def test_golden_answer_passes_grader(case: GoldCase):
    result = GRADERS[case.step](case, _golden_raw(case))
    failed = [f"{c.name}: {c.detail}" for c in result.checks if not c.passed]
    assert result.passed, f"golden answer no longer passes: {failed}"


def test_real_anchor_fixtures_present():
    """The f559315d bug batch and the captured wave decoys are the anchor
    fixtures — losing them loses the epic's best real-world evidence."""
    ids = {c.case_id for c in _cases("review-findings")}
    assert "real-evals-plumbing-batch" in ids
    assert "clean-diff-temperature" in ids
    confirm_ids = {c.case_id for c in _cases("review-confirm")}
    assert {"real-typo-goldets", "real-loader-collapse", "real-baseline-shared-file"} <= confirm_ids
    assert {"real-decoy-infinite-recursion", "real-decoy-missing-aggregation"} <= confirm_ids
