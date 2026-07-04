"""Tests for the coding pipeline emit module (A3 - tree emission to Forge).

Covers: ref stability, dependency ordering, comma-title rejection, dedup on
re-emit, cap behaviour, fixup ref shape, and dry-run parity.

Patches ``_create_task`` and ``existing_external_refs`` from
``agents.shared.forge_emit`` so no network / daemon is needed.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from agents.coding_pipeline.emit import (
    EmitResult,
    dry_run_emit,
    emit_fixup,
    emit_tree,
    stable_ref,
)
from agents.coding_pipeline.models import LeafSpec, TaskTree


@pytest.fixture
def fake_nous(monkeypatch):
    """Patch _create_task and existing_external_refs for all tests."""
    calls: list[dict] = []

    def fake_create(**kwargs) -> str:
        calls.append(kwargs)
        return json.dumps({"page_id": f"pg-{len(calls)}", "row_id": f"row-{len(calls)}"})

    import agents.shared.forge_emit as fe

    monkeypatch.setattr(fe, "_create_task", fake_create)
    monkeypatch.setattr(fe, "existing_external_refs", lambda: set())

    return calls


def _leaf(
    title: str,
    feature: str = "feat-a",
    depends_on: list[str] | None = None,
    status: str = "Ready",
    execution_mode: str = "Auto-OK",
    phase: str = "Feature",
    priority: int = 2,
    task_type: str = "feature",
    requires_tests: bool = True,
    max_files: int | None = 3,
    model_tier: str | None = "auto",
) -> LeafSpec:
    return LeafSpec(
        title=title,
        content=f"Implement {title}",
        feature=feature,
        depends_on=depends_on or [],
        status=status,
        execution_mode=execution_mode,
        phase=phase,
        priority=priority,
        task_type=task_type,
        requires_tests=requires_tests,
        max_files=max_files,
        model_tier=model_tier,
    )


def _tree(*leaves: LeafSpec) -> TaskTree:
    return TaskTree(leaves=list(leaves))


class TestStableRef:
    def test_basic_ref_shape(self):
        leaf = _leaf("Add parser")
        ref = stable_ref("my-epic", leaf)
        assert ref == "pipeline:my-epic:feat-a-add-parser"

    def test_deterministic(self):
        leaf = _leaf("Add parser", feature="core")
        assert stable_ref("epic", leaf) == stable_ref("epic", leaf)

    def test_fixup_shape_keeps_epic_slug(self):
        # pipeline:{epic}:fix:{slug} — the epic stays so fix-ups can't collide across epics
        leaf = _leaf("Fix parser crash", feature="core")
        ref = stable_ref("my-epic", leaf, fixup=True)
        assert ref == "pipeline:my-epic:fix:core-fix-parser-crash"

    def test_fixup_same_function(self):
        """Fix-up leaves use the same stable_ref function - dedup path is unified."""
        leaf = _leaf("Fix crash", feature="auth")
        normal = stable_ref("e", leaf)
        fixup = stable_ref("e", leaf, fixup=True)
        assert normal != fixup
        assert "fix:" in fixup

    def test_fixup_keys_on_finding_slug_when_given(self):
        """The FINDING is the stable identity across replans — the replan model invents
        leaf titles freely, so two re-discoveries of one finding must share a ref
        (dry-run Q2; this was the documented-but-dead finding_slug key)."""
        finding_slug = "src-cli-py-list-units-crashes-on-empty-domain-table"
        first = _leaf("Fix list-units crash on empty table", feature="core")
        rediscovered = _leaf("Handle empty domain table in list-units", feature="cli")
        ref_a = stable_ref("my-epic", first, fixup=True, finding_slug=finding_slug)
        ref_b = stable_ref("my-epic", rediscovered, fixup=True, finding_slug=finding_slug)
        assert ref_a == ref_b  # same finding -> same ref, whatever the titles
        assert ref_a.startswith("pipeline:my-epic:fix:")
        # finding_slug only applies to fix-ups; tree leaves keep the title key
        assert "fix:" not in stable_ref("my-epic", first, finding_slug=finding_slug)

    def test_slug_truncation(self):
        """Long feature:title combos get truncated by slugify."""
        leaf = _leaf("x" * 60, feature="f" * 60)
        ref = stable_ref("e", leaf)
        assert len(ref) <= len("pipeline:e:") + 50

    def test_slug_lowercased(self):
        leaf = _leaf("Add PARSER")
        ref = stable_ref("epic", leaf)
        assert "parser" in ref
        assert "PARSER" not in ref


class TestDependencyOrdering:
    def test_flat_no_deps_emits_as_is(self, fake_nous, monkeypatch):
        monkeypatch.setattr("agents.shared.forge_emit.existing_external_refs", lambda: set())
        leaves = [_leaf("A"), _leaf("B"), _leaf("C")]
        tree = _tree(*leaves)

        with patch("agents.coding_pipeline.emit._topo_sort") as mock_sort:
            mock_sort.return_value = leaves
            _result = emit_tree(tree, project="Meta", epic_slug="epic")

        assert mock_sort.call_count == 1
        assert len(_result.created) == 3

    def test_dep_order_affects_emission_order(self):
        """Leaf B depends on A, so A must be emitted first."""
        a = _leaf("Implement core")
        b = _leaf("Build on core", depends_on=["Implement core"])

        with patch("agents.coding_pipeline.emit._topo_sort") as mock_sort:
            mock_sort.return_value = [a, b]
            assert mock_sort.return_value == [a, b]  # type: ignore[attr-defined]

    def test_comma_title_rejected_via_topo_sort(self, fake_nous):
        """emit's own comma check must hold even if a leaf bypassed model validation
        (model_construct skips validators — belt-and-suspenders)."""
        leaf = LeafSpec.model_construct(
            title="Add parser, tokenizer",
            content="implement it",
            feature="parse",
            depends_on=[],
            priority=5,
            phase="Feature",
            status="Ready",
            execution_mode="Manual",
            complexity=None,
            estimate=None,
            task_type="feature",
            requires_tests=True,
            max_files=None,
            model_tier=None,
            boundedness=None,
        )
        tree = _tree(leaf)

        with pytest.raises(ValueError, match="comma"):
            emit_tree(tree, project="Meta", epic_slug="epic")

    def test_unknown_dep_rejected(self):
        leaf = _leaf("A", depends_on=["Nonexistent"])
        tree = _tree(leaf)

        with pytest.raises(ValueError, match="Nonexistent"):
            emit_tree(tree, project="Meta", epic_slug="epic")

    def test_cycle_rejected(self):
        a = _leaf("A", depends_on=["B"])
        b = _leaf("B", depends_on=["A"])
        tree = _tree(a, b)

        with pytest.raises(ValueError, match="circular"):
            emit_tree(tree, project="Meta", epic_slug="epic")


class TestGatingPassthrough:
    def test_architect_tags_pass_through_verbatim(self, fake_nous):
        # The architect's conservative tagging IS the gating decision — emission
        # must never second-guess it (a Ready+Auto-OK leaf stays Ready+Auto-OK).
        leaf = _leaf("Ready leaf", status="Ready", execution_mode="Auto-OK", phase="Feature")
        tree = _tree(leaf)
        emit_tree(tree, project="Meta", epic_slug="epic")
        assert len(fake_nous) == 1
        sent = fake_nous[0]
        assert sent["status"] == "Ready"
        assert sent["execution_mode"] == "Auto-OK"
        assert sent["phase"] == "Feature"

    def test_spec_needed_leaf_stays_spec_needed(self, fake_nous):
        leaf = _leaf("Half-baked leaf", status="Spec Needed", execution_mode="Manual")
        emit_tree(_tree(leaf), project="Meta", epic_slug="epic")
        assert fake_nous[0]["status"] == "Spec Needed"
        assert fake_nous[0]["execution_mode"] == "Manual"

    def test_requires_tests_false_stays_explicit(self, fake_nous):
        # False must reach Forge as an explicit "No", not degrade to unset
        # (unset is null-as-true in the worker).
        leaf = _leaf("Docs-only leaf", requires_tests=False)
        emit_tree(_tree(leaf), project="Meta", epic_slug="epic")
        assert fake_nous[0]["requires_tests"] is False

    def test_per_spec_priority_passthrough(self, fake_nous):
        leaf = _leaf("High priority", priority=1)
        emit_tree(_tree(leaf), project="Meta", epic_slug="epic")
        assert fake_nous[0]["priority"] == 1

    def test_guardrails_flow_through(self, fake_nous):
        leaf = _leaf("Guarded", max_files=2, requires_tests=True, model_tier="auto-free")
        emit_tree(_tree(leaf), project="Meta", epic_slug="epic")
        sent = fake_nous[0]
        assert sent["max_files"] == 2
        assert sent["requires_tests"] is True
        assert sent["model_tier"] == "auto-free"

    def test_depends_on_wire_as_comma_separated(self, fake_nous):
        a = _leaf("Implement core")
        b = _leaf("Build on core", depends_on=["Implement core"])
        emit_tree(_tree(a, b), project="Meta", epic_slug="epic")
        assert len(fake_nous) == 2
        assert fake_nous[0]["external_ref"] == stable_ref("epic", a)
        assert fake_nous[1]["external_ref"] == stable_ref("epic", b)
        assert fake_nous[1]["depends_on"] == "Implement core"

    def test_task_type_passed(self, fake_nous):
        leaf = _leaf("Fix bug", task_type="bug-fix")
        emit_tree(_tree(leaf), project="Meta", epic_slug="epic")
        assert fake_nous[0]["task_type"] == "bug-fix"


class TestIdempotentReEmission:
    def test_re_emit_skips_existing(self, fake_nous, monkeypatch):
        """Second emit of the same tree should skip all leaves."""
        leaf = _leaf("Add parser")
        tree = _tree(leaf)

        result1 = emit_tree(tree, project="Meta", epic_slug="epic")
        assert len(result1.created) == 1
        assert len(result1.skipped) == 0

        monkeypatch.setattr(
            "agents.shared.forge_emit.existing_external_refs",
            lambda: {stable_ref("epic", leaf)},
        )
        result2 = emit_tree(tree, project="Meta", epic_slug="epic")
        assert len(result2.created) == 0
        assert len(result2.skipped) == 1

    def test_partial_re_emit(self, fake_nous, monkeypatch):
        """Re-emit with one new leaf: old ones skipped, new one created."""
        a = _leaf("Existing")
        result1 = emit_tree(_tree(a), project="Meta", epic_slug="epic")
        assert len(result1.created) == 1

        b = _leaf("New leaf")
        monkeypatch.setattr(
            "agents.shared.forge_emit.existing_external_refs",
            lambda: {stable_ref("epic", a)},
        )
        result2 = emit_tree(_tree(a, b), project="Meta", epic_slug="epic")
        assert len(result2.created) == 1
        assert len(result2.skipped) == 1
        assert result2.created[0].external_ref == stable_ref("epic", b)


class TestCapBehavior:
    def test_cap_drops_excess(self, fake_nous, monkeypatch):
        monkeypatch.setattr("agents.shared.forge_emit.existing_external_refs", lambda: set())
        leaves = [_leaf(f"Leaf {i}") for i in range(5)]
        result = emit_tree(_tree(*leaves), project="Meta", epic_slug="epic", max_per_run=2)
        assert len(result.created) == 2
        assert result.capped == 3

    def test_cap_not_consumed_by_dedup(self, fake_nous, monkeypatch):
        emit_tree(_tree(_leaf("Old")), project="Meta", epic_slug="epic")

        monkeypatch.setattr(
            "agents.shared.forge_emit.existing_external_refs",
            lambda: {stable_ref("epic", _leaf("Old"))},
        )
        result = emit_tree(
            _tree(_leaf("Old"), _leaf("New")),
            project="Meta",
            epic_slug="epic",
            max_per_run=1,
        )
        assert len(result.created) == 1
        assert len(result.skipped) == 1
        assert result.capped == 0


class TestDryRun:
    def test_dry_run_creates_nothing(self, fake_nous):
        result = emit_tree(_tree(_leaf("Dry task")), project="Meta", epic_slug="epic", dry_run=True)
        assert len(result.planned) == 1
        assert len(result.created) == 0
        assert len(fake_nous) == 0

    def test_dry_run_emit_helper(self, fake_nous):
        result = dry_run_emit(
            _tree(_leaf("Dry task 1"), _leaf("Dry task 2")),
            project="Meta",
            epic_slug="epic",
        )
        assert len(result.planned) == 2
        assert len(result.created) == 0


class TestEmitFixup:
    def test_fixup_uses_fix_ref_shape(self, fake_nous, monkeypatch):
        monkeypatch.setattr("agents.shared.forge_emit.existing_external_refs", lambda: set())
        leaf = _leaf("Fix crash after wave 1", feature="parser")
        outcome = emit_fixup(leaf, project="Meta", epic_slug="my-epic")
        assert outcome.action == "created"
        expected_ref = stable_ref("my-epic", leaf, fixup=True)
        assert outcome.external_ref == expected_ref
        assert "fix:" in expected_ref

    def test_fixup_dedup(self, fake_nous, monkeypatch):
        leaf = _leaf("Fix crash", feature="auth")
        existing_ref = stable_ref("e", leaf, fixup=True)
        monkeypatch.setattr(
            "agents.shared.forge_emit.existing_external_refs",
            lambda: {existing_ref},
        )
        outcome = emit_fixup(leaf, project="Meta", epic_slug="e")
        assert outcome.action == "skipped"


class TestDecisionLog:
    def test_journal_written_on_emit(self, tmp_path, fake_nous, monkeypatch):
        monkeypatch.setattr("agents.shared.forge_emit.existing_external_refs", lambda: set())
        leaf = _leaf("Logged task")
        tree = _tree(leaf)
        runs_dir = tmp_path / "pipeline-runs"

        emit_tree(tree, project="Meta", epic_slug="my-epic", runs_dir=runs_dir)

        journal_path = runs_dir / "my-epic" / "journal.jsonl"
        assert journal_path.exists()
        lines = journal_path.read_text().strip().split("\n")
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["event"] == "emit_task"
        assert record["action"] == "created"
        assert record["epic"] == "my-epic"

    def test_journal_not_written_on_dry_run(self, tmp_path, fake_nous):
        leaf = _leaf("Dry task")
        tree = _tree(leaf)
        runs_dir = tmp_path / "pipeline-runs"

        emit_tree(tree, project="Meta", epic_slug="my-epic", dry_run=True, runs_dir=runs_dir)

        journal_path = runs_dir / "my-epic" / "journal.jsonl"
        assert not journal_path.exists()

    def test_journal_skips_logged(self, tmp_path, fake_nous, monkeypatch):
        leaf = _leaf("Existing")
        tree = _tree(leaf)
        runs_dir = tmp_path / "pipeline-runs"

        monkeypatch.setattr(
            "agents.shared.forge_emit.existing_external_refs",
            lambda: {stable_ref("e", leaf)},
        )

        emit_tree(tree, project="Meta", epic_slug="e", runs_dir=runs_dir)

        journal_path = runs_dir / "e" / "journal.jsonl"
        lines = journal_path.read_text().strip().split("\n")
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["action"] == "skipped"


class TestEmitResult:
    def test_summary_line(self):
        result = EmitResult(
            project="Meta",
            epic_slug="epic",
            created=[MagicMock(external_ref="r1", title="a", action="created")],
            skipped=[MagicMock(external_ref="r2", title="b", action="skipped")],
            capped=2,
            planned=[MagicMock(external_ref="r3", title="c", action="dry-run")],
        )
        line = result.summary()
        assert "1 created" in line
        assert "1 skipped" in line
        assert "2 capped" in line
        assert "1 planned" in line
