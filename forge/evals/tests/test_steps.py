"""Tests for step adapters that render production system/user messages.

Each adapter test verifies:
- ``system`` is identity-equal (``is``) to the production constant
- ``user`` is byte-equal to what the production builder produces
- The ``schema`` validates against the production-declared JSON shape
- Unknown step raises ``KeyError``
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from pydantic import BaseModel

from agents.evals.models import GoldCase
from agents.evals.steps import (
    ADAPTERS,
    ConfirmVerdict,
    GapsEnvelope,
    PromptSpec,
    SkepticVerdict,
    StepAdapter,
    get_adapter,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_json(path: Path, name: str, data: dict | list) -> None:
    (path / name).write_text(json.dumps(data, indent=2))


def _write_text(path: Path, name: str, text: str) -> None:
    (path / name).write_text(text)


def _make_gold_case(
    step: str,
    case_dir: Path,
    inputs: dict[str, str] | None = None,
    expected: dict | None = None,
) -> GoldCase:
    return GoldCase(
        step=step,
        case_id=f"test-{step}",
        case_dir=case_dir,
        schema_version=1,
        inputs=inputs or {},
        expected=expected or {},
    )


# ---------------------------------------------------------------------------
# Unknown step
# ---------------------------------------------------------------------------


def test_unknown_step_raises_keyerror():
    with pytest.raises(KeyError, match="unknown step"):
        get_adapter("nonexistent")  # type: ignore[arg-type]


def test_adapter_registry_has_all_steps():
    from agents.evals.models import StepName

    known_steps: set[StepName] = {
        "replan",
        "decompose",
        "boundedness",
        "review-findings",
        "review-confirm",
        "testgap-find",
        "testgap-skeptic",
    }
    assert set(ADAPTERS.keys()) == known_steps


# ---------------------------------------------------------------------------
# replan adapter
# ---------------------------------------------------------------------------


def _make_replan_fixtures(tmp_path: Path) -> GoldCase:
    _write_json(
        tmp_path,
        "framing.json",
        {
            "goal_as_stated": "Build a feature",
            "restated_goal": "Build the feature better",
            "rescoped": False,
            "inventory_summary": "lots of stuff",
            "gap_analysis": "nothing here",
            "recommendation": "Go for it",
            "value_ordering": ["core"],
            "risks": ["maybe"],
            "epic_slug": "my-feature",
            "branch": "feat/my-feature",
            "approved": True,
        },
    )
    _write_json(
        tmp_path,
        "tree.json",
        [
            {
                "title": "leaf one",
                "content": "x" * 250,
                "feature": "Core",
                "depends_on": [],
                "priority": 1,
                "phase": "Feature",
                "status": "Ready",
                "execution_mode": "Auto-OK",
                "complexity": "routine",
                "estimate": "s",
                "task_type": "feature",
                "requires_tests": True,
                "max_files": 3,
                "model_tier": "auto-full",
            }
        ],
    )
    _write_json(
        tmp_path,
        "report.json",
        {
            "wave": 1,
            "outcomes": [],
            "findings": [],
            "raw_findings": 0,
            "consolidation_ok": True,
            "dropped_covered": [],
            "diff_stat": "0 files",
        },
    )
    _write_json(tmp_path, "attempts.json", {"leaf one": 1})

    case = _make_gold_case(
        "replan",
        tmp_path,
        {
            "framing.json": "framing.json",
            "tree.json": "tree.json",
            "report.json": "report.json",
            "attempts.json": "attempts.json",
        },
    )
    return case


def test_replan_system_is_identity_to_constant():
    from agents.coding_pipeline.architect import REPLAN_SYSTEM

    case = _make_replan_fixtures(Path(tempfile.mkdtemp()))
    adapter = ADAPTERS["replan"]
    spec = adapter.build(case)
    assert spec.system is REPLAN_SYSTEM


def test_replan_user_matches_production():
    case = _make_replan_fixtures(Path(tempfile.mkdtemp()))
    adapter = ADAPTERS["replan"]
    spec = adapter.build(case)

    # Call the production builder directly and compare
    from agents.coding_pipeline.architect import (
        _replan_user,
        deterministic_escalations,
    )
    from agents.coding_pipeline.models import FramingProposal, LeafSpec, WaveReport

    framing = FramingProposal.model_validate(
        json.loads((case.case_dir / "framing.json").read_text())
    )
    tree = [
        LeafSpec.model_validate(d) for d in json.loads((case.case_dir / "tree.json").read_text())
    ]
    report = WaveReport.model_validate(json.loads((case.case_dir / "report.json").read_text()))
    attempts = json.loads((case.case_dir / "attempts.json").read_text())

    escalated = deterministic_escalations(report, attempts)
    expected_user = _replan_user(framing, tree, report, attempts, escalated)

    assert spec.user == expected_user


def test_replan_schema_is_replan_envelope():
    from agents.coding_pipeline.architect import ReplanEnvelope

    case = _make_replan_fixtures(Path(tempfile.mkdtemp()))
    adapter = ADAPTERS["replan"]
    spec = adapter.build(case)
    assert spec.schema is ReplanEnvelope


# ---------------------------------------------------------------------------
# decompose adapter
# ---------------------------------------------------------------------------


def _make_decompose_fixtures(tmp_path: Path) -> GoldCase:
    _write_json(
        tmp_path,
        "framing.json",
        {
            "goal_as_stated": "Build X",
            "restated_goal": "Build X but smarter",
            "rescoped": False,
            "inventory_summary": "some stuff",
            "gap_analysis": "gap",
            "recommendation": "do it",
            "value_ordering": [],
            "risks": [],
            "epic_slug": "x",
            "approved": True,
        },
    )
    _write_json(
        tmp_path,
        "inventory.json",
        {
            "project": "meta",
            "repo": "/path/to/meta",
            "tree": "- agents/\n  - evals/",
            "key_files": [],
            "modules": ["agents"],
            "test_layout": ["agents/evals/tests"],
            "toolchain": ["uv"],
            "existing_tasks": [],
            "overlaps": [],
            "truncated": 0,
        },
    )

    case = _make_gold_case(
        "decompose",
        tmp_path,
        {
            "framing.json": "framing.json",
            "inventory.json": "inventory.json",
        },
    )
    return case


def test_decompose_system_is_identity_to_constant():
    from agents.coding_pipeline.architect import DECOMPOSE_SYSTEM

    case = _make_decompose_fixtures(Path(tempfile.mkdtemp()))
    adapter = ADAPTERS["decompose"]
    spec = adapter.build(case)
    assert spec.system is DECOMPOSE_SYSTEM


def test_decompose_user_matches_production():
    from agents.coding_pipeline.architect import _decompose_user
    from agents.coding_pipeline.models import FramingProposal, Inventory

    case = _make_decompose_fixtures(Path(tempfile.mkdtemp()))
    adapter = ADAPTERS["decompose"]
    spec = adapter.build(case)

    framing = FramingProposal.model_validate(
        json.loads((case.case_dir / "framing.json").read_text())
    )
    inventory = Inventory.model_validate(json.loads((case.case_dir / "inventory.json").read_text()))

    expected_user = _decompose_user(framing, inventory)
    assert spec.user == expected_user


def test_decompose_schema_is_task_tree():
    from agents.coding_pipeline.models import TaskTree

    case = _make_decompose_fixtures(Path(tempfile.mkdtemp()))
    adapter = ADAPTERS["decompose"]
    spec = adapter.build(case)
    assert spec.schema is TaskTree


# ---------------------------------------------------------------------------
# boundedness adapter
# ---------------------------------------------------------------------------


def _make_boundedness_fixtures(tmp_path: Path) -> GoldCase:
    _write_json(
        tmp_path,
        "leaf.json",
        {
            "title": "wire the flux capacitor",
            "content": "x" * 300,
            "feature": "Time Travel",
            "depends_on": [],
            "priority": 1,
            "phase": "Feature",
            "status": "Ready",
            "execution_mode": "Auto-OK",
            "complexity": "routine",
            "estimate": "s",
            "task_type": "feature",
            "requires_tests": True,
            "max_files": 3,
            "model_tier": "auto-full",
        },
    )

    case = _make_gold_case(
        "boundedness",
        tmp_path,
        {
            "leaf.json": "leaf.json",
        },
    )
    return case


def test_boundedness_system_is_identity_to_constant():
    from agents.coding_pipeline.architect import BOUNDEDNESS_SYSTEM

    case = _make_boundedness_fixtures(Path(tempfile.mkdtemp()))
    adapter = ADAPTERS["boundedness"]
    spec = adapter.build(case)
    assert spec.system is BOUNDEDNESS_SYSTEM


def test_boundedness_user_matches_production():
    from agents.coding_pipeline.architect import _leaf_summary
    from agents.coding_pipeline.models import LeafSpec

    case = _make_boundedness_fixtures(Path(tempfile.mkdtemp()))
    adapter = ADAPTERS["boundedness"]
    spec = adapter.build(case)

    leaf = LeafSpec.model_validate(json.loads((case.case_dir / "leaf.json").read_text()))
    expected_user = _leaf_summary(leaf)
    assert spec.user == expected_user


def test_boundedness_schema_is_leaf_boundedness():
    from agents.coding_pipeline.architect import LeafBoundedness

    case = _make_boundedness_fixtures(Path(tempfile.mkdtemp()))
    adapter = ADAPTERS["boundedness"]
    spec = adapter.build(case)
    assert spec.schema is LeafBoundedness


# ---------------------------------------------------------------------------
# review-findings adapter
# ---------------------------------------------------------------------------


def _make_review_findings_fixtures(tmp_path: Path) -> GoldCase:
    _write_text(tmp_path, "diff.patch", "diff --git a/foo.py b/foo.py\n+print('hello')")

    case = _make_gold_case(
        "review-findings",
        tmp_path,
        {
            "diff.patch": "diff.patch",
        },
    )
    return case


def test_review_findings_system_is_identity_to_constant():
    from agents.coding_pipeline.verify import FINDINGS_SYSTEM

    case = _make_review_findings_fixtures(Path(tempfile.mkdtemp()))
    adapter = ADAPTERS["review-findings"]
    spec = adapter.build(case)
    assert spec.system is FINDINGS_SYSTEM


def test_review_findings_user_shape():
    case = _make_review_findings_fixtures(Path(tempfile.mkdtemp()))
    adapter = ADAPTERS["review-findings"]
    spec = adapter.build(case)
    assert spec.user.startswith("Wave diff:\n\n")

    # Verify the user message contains the diff content
    diff_content = (case.case_dir / "diff.patch").read_text()
    assert diff_content in spec.user


def test_review_findings_schema_is_findings_envelope():
    from agents.coding_pipeline.verify import FindingsEnvelope

    case = _make_review_findings_fixtures(Path(tempfile.mkdtemp()))
    adapter = ADAPTERS["review-findings"]
    spec = adapter.build(case)
    assert spec.schema is FindingsEnvelope


# ---------------------------------------------------------------------------
# review-confirm adapter
# ---------------------------------------------------------------------------


def _make_review_confirm_fixtures(tmp_path: Path) -> GoldCase:
    _write_text(tmp_path, "diff.patch", "diff --git a/bar.py b/bar.py\n+bad code")
    _write_json(
        tmp_path,
        "candidate.json",
        {
            "summary": "unused variable",
            "file": "bar.py",
            "severity": "low",
        },
    )

    case = _make_gold_case(
        "review-confirm",
        tmp_path,
        {
            "diff.patch": "diff.patch",
            "candidate.json": "candidate.json",
        },
    )
    return case


def test_review_confirm_system_is_identity_to_constant():
    from agents.coding_pipeline.verify import CONFIRM_SYSTEM

    case = _make_review_confirm_fixtures(Path(tempfile.mkdtemp()))
    adapter = ADAPTERS["review-confirm"]
    spec = adapter.build(case)
    assert spec.system is CONFIRM_SYSTEM


def test_review_confirm_user_shape():
    case = _make_review_confirm_fixtures(Path(tempfile.mkdtemp()))
    adapter = ADAPTERS["review-confirm"]
    spec = adapter.build(case)

    # Should follow the make_user lambda shape from verify.confirm_findings
    assert "Finding: unused variable" in spec.user
    assert "File: bar.py" in spec.user
    assert "(severity claimed: low)" in spec.user
    assert "Wave diff:\n\n" in spec.user


def test_review_confirm_schema_is_confirm_verdict():
    case = _make_review_confirm_fixtures(Path(tempfile.mkdtemp()))
    adapter = ADAPTERS["review-confirm"]
    spec = adapter.build(case)
    assert spec.schema is ConfirmVerdict


# ---------------------------------------------------------------------------
# testgap-find adapter
# ---------------------------------------------------------------------------


def _make_testgap_find_fixtures(tmp_path: Path) -> GoldCase:
    _write_text(
        tmp_path,
        "context.md",
        "## SOURCE\n\ndef foo(): pass\n\n## EXISTING TESTS\n\ndef test_foo(): pass",
    )

    # case.yaml is also read as YAML for the "angle" key
    with open(tmp_path / "case.yaml", "w") as f:
        f.write(
            "step: testgap-find\n"
            "case_id: test-testgap-find\n"
            "schema_version: 1\n"
            "inputs:\n  context.md: context.md\n"
            "angle: error-paths\n"
        )

    case = _make_gold_case(
        "testgap-find",
        tmp_path,
        {
            "context.md": "context.md",
            "case.yaml": "case.yaml",
        },
    )
    return case


def test_testgap_find_system_matches_finder_system():
    from agents.testing_ensemble.prompts import FINDER_ANGLES, finder_system

    case = _make_testgap_find_fixtures(Path(tempfile.mkdtemp()))
    adapter = ADAPTERS["testgap-find"]
    spec = adapter.build(case)

    # The system should be built by finder_system with the correct angle
    expected_system = finder_system("error-paths", FINDER_ANGLES[1][1])
    assert spec.system == expected_system


def test_testgap_find_user_is_context():
    case = _make_testgap_find_fixtures(Path(tempfile.mkdtemp()))
    adapter = ADAPTERS["testgap-find"]
    spec = adapter.build(case)
    context = (case.case_dir / "context.md").read_text()
    assert spec.user == context


def test_testgap_find_schema_is_gaps_envelope():
    case = _make_testgap_find_fixtures(Path(tempfile.mkdtemp()))
    adapter = ADAPTERS["testgap-find"]
    spec = adapter.build(case)
    assert spec.schema is GapsEnvelope


# ---------------------------------------------------------------------------
# testgap-skeptic adapter
# ---------------------------------------------------------------------------


def _make_testgap_skeptic_fixtures(tmp_path: Path) -> GoldCase:
    _write_text(tmp_path, "context.md", "## SOURCE\n\ndef bar(): pass\n\n## EXISTING TESTS\n\n")
    _write_json(
        tmp_path,
        "gap.json",
        {
            "target": "bar::no error handling",
            "gap_type": "error-path",
            "why_it_matters": "exceptions will crash",
            "suggested_test": "test bar raises",
            "severity": "high",
        },
    )

    case = _make_gold_case(
        "testgap-skeptic",
        tmp_path,
        {
            "context.md": "context.md",
            "gap.json": "gap.json",
        },
    )
    return case


def test_testgap_skeptic_system_is_identity_to_skeptic_base():
    from agents.testing_ensemble.prompts import SKEPTIC_BASE

    case = _make_testgap_skeptic_fixtures(Path(tempfile.mkdtemp()))
    adapter = ADAPTERS["testgap-skeptic"]
    spec = adapter.build(case)
    assert spec.system is SKEPTIC_BASE


def test_testgap_skeptic_user_shape():
    from agents.testing_ensemble.models import TestGap
    from agents.testing_ensemble.prompts import verify_user

    case = _make_testgap_skeptic_fixtures(Path(tempfile.mkdtemp()))
    adapter = ADAPTERS["testgap-skeptic"]
    spec = adapter.build(case)

    context = (case.case_dir / "context.md").read_text()
    gap_data = json.loads((case.case_dir / "gap.json").read_text())
    expected_user = verify_user(context, TestGap.model_validate(gap_data).model_dump_json())
    assert spec.user == expected_user


def test_testgap_skeptic_schema_is_skeptic_verdict():
    case = _make_testgap_skeptic_fixtures(Path(tempfile.mkdtemp()))
    adapter = ADAPTERS["testgap-skeptic"]
    spec = adapter.build(case)
    assert spec.schema is SkepticVerdict


# ---------------------------------------------------------------------------
# Wire model validation tests
# ---------------------------------------------------------------------------


def test_confirm_verdict_validates_correct_shape():
    data = {"real": True, "reason": "this is clearly a bug"}
    v = ConfirmVerdict.model_validate(data)
    assert v.real is True
    assert v.reason == "this is clearly a bug"


def test_confirm_verdict_rejects_missing_fields():
    with pytest.raises(Exception):  # ValidationError
        ConfirmVerdict.model_validate({"real": True})  # missing 'reason'


def test_gaps_envelope_validates_empty_gaps():
    v = GapsEnvelope.model_validate({"gaps": []})
    assert v.gaps == []


def test_gaps_envelope_validates_with_gaps():
    v = GapsEnvelope.model_validate(
        {
            "gaps": [
                {
                    "target": "foo::bar",
                    "gap_type": "coverage",
                    "why_it_matters": "lost coverage",
                    "suggested_test": "test bar",
                    "severity": "high",
                }
            ]
        }
    )
    assert len(v.gaps) == 1
    assert v.gaps[0].target == "foo::bar"


def test_skeptic_verdict_validates_correct_shape():
    data = {
        "real": False,
        "confidence": "high",
        "severity": "medium",
        "reasoning": "already covered",
    }
    v = SkepticVerdict.model_validate(data)
    assert v.real is False
    assert v.confidence == "high"
    assert v.severity == "medium"
    assert v.reasoning == "already covered"


# ---------------------------------------------------------------------------
# Adapter protocol tests
# ---------------------------------------------------------------------------


def test_step_adapter_has_required_fields():
    spec = PromptSpec(system="s", user="u", schema=BaseModel)
    adapter = StepAdapter(step="replan", build=lambda case: spec)
    assert adapter.step == "replan"
    assert callable(adapter.build)


def test_prompt_spec_has_required_fields():
    spec = PromptSpec(system="sys", user="usr", schema=BaseModel)
    assert spec.system == "sys"
    assert spec.user == "usr"
    assert spec.schema is BaseModel
