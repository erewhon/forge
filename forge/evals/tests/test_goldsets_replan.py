"""Regression guard for the CHECKED-IN replan gold set.

The gold set is a durable asset (Fable-authored, distill-evals epic): these
tests keep it loadable and satisfiable forever — every case must render
through the real replan adapter, and every case's ``golden.json`` (the frozen
reference answer, also the few-shot exemplar source for non-holdout cases)
must pass the grader. A schema or prompt-builder change that breaks either
fails HERE, not silently at the next scorecard run.
"""

from __future__ import annotations

import json

import pytest

from forge.evals.config import settings
from forge.evals.fixtures import load_goldsets, read_input
from forge.evals.graders.replan import grade
from forge.evals.models import GoldCase
from forge.evals.steps import ADAPTERS

# The sketch's decision families + the enum-stress regression set.
EXPECTED_FAMILIES = (
    "under-cap-respec",
    "confirmed-fixup",
    "integration-red",
    "mixed",
    "split-subtree",
    "halt",
    "clean-noop",
    "respect-escalation",
)
ENUM_STRESS_IDS = {
    "enum-task-type",
    "enum-phase",
    "enum-estimate",
    "enum-complexity",
    "enum-model-tier",
    "enum-status",
}


def _cases() -> list[GoldCase]:
    return load_goldsets(settings.goldsets_dir, step="replan")


def test_goldset_loads_with_taxonomy_coverage():
    cases = _cases()
    assert len(cases) >= 20, "replan gold set shrank below its authored size"

    ids = {c.case_id for c in cases}
    assert ENUM_STRESS_IDS <= ids, f"enum-stress cases missing: {ENUM_STRESS_IDS - ids}"
    for family in EXPECTED_FAMILIES:
        assert any(i.startswith(family) or family in i for i in ids), (
            f"no case for family '{family}'"
        )

    holdouts = sum(1 for c in cases if c.holdout)
    assert holdouts / len(cases) >= 1 / 3, "holdout fraction fell below 1/3"


@pytest.mark.parametrize("case", _cases(), ids=lambda c: c.case_id)
def test_case_renders_through_production_adapter(case: GoldCase):
    spec = ADAPTERS["replan"].build(case)
    assert "REPLAN" in spec.system
    assert "## Wave" in spec.user  # _replan_user's report section rendered


@pytest.mark.parametrize("case", _cases(), ids=lambda c: c.case_id)
def test_golden_answer_passes_grader(case: GoldCase):
    golden = (case.case_dir / "golden.json").read_text()
    result = grade(case, golden)
    failed = [f"{c.name}: {c.detail}" for c in result.checks if not c.passed]
    assert result.passed, f"golden answer no longer passes: {failed}"


@pytest.mark.parametrize("case", _cases(), ids=lambda c: c.case_id)
def test_forces_model_path_flag_is_honest(case: GoldCase):
    """The declared forces_model_path must match what production's short-circuit
    would compute from the inputs (clean-noop cases are the deliberate False)."""
    rep = json.loads(read_input(case, "report.json"))
    attempts = json.loads(read_input(case, "attempts.json"))
    confirmed = any(f.get("confirmed") for f in rep.get("findings", []))
    failed_under_cap = [
        o for o in rep["outcomes"] if o["status"] == "failed" and attempts.get(o["leaf"], 0) < 2
    ]
    landed = [o for o in rep["outcomes"] if o["status"] == "done"]
    integration_red = bool(landed) and not rep["suite"]["passed"]
    computed = bool(confirmed or failed_under_cap or integration_red)
    assert case.expected.get("forces_model_path") == computed
