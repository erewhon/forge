"""Reachability signal tests — AST collector, name normalization, emit demotion, policy gate."""

from __future__ import annotations

import textwrap
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from forge.dependabot.emit import _demote, emit_advisory
from forge.dependabot.models import BumpCandidate, EvidenceBundle
from forge.dependabot.policy import auto_eligible
from forge.dependabot.reachability import import_candidates, imported_names, is_imported

# --- imported_names (AST collector) ---------------------------------------------------------


def test_imported_names_finds_import_and_from_import():
    with TemporaryDirectory() as td:
        root = Path(td)
        (root / "main.py").write_text(
            textwrap.dedent("""\
                import os
                from pathlib import Path
                import json as j
                from collections import defaultdict

                print("hello")
            """),
            encoding="utf-8",
        )
        found = imported_names(root)
        assert "os" in found
        assert "pathlib" in found
        assert "json" in found
        assert "collections" in found


def test_imported_names_skips_venv_and_hidden_dirs():
    with TemporaryDirectory() as td:
        root = Path(td)
        # A file inside .venv should be skipped
        venv_dir = root / ".venv" / "lib"
        venv_dir.mkdir(parents=True)
        (venv_dir / "something.py").write_text("import requests\n", encoding="utf-8")
        # A file outside .venv should be collected
        (root / "app.py").write_text("import flask\n", encoding="utf-8")
        found = imported_names(root)
        assert "flask" in found
        assert "requests" not in found


def test_imported_names_skips_node_modules():
    with TemporaryDirectory() as td:
        root = Path(td)
        nm = root / "node_modules" / "pkg"
        nm.mkdir(parents=True)
        (nm / "index.py").write_text("import something\n", encoding="utf-8")
        (root / "main.py").write_text("import click\n", encoding="utf-8")
        found = imported_names(root)
        assert "click" in found
        assert "something" not in found


def test_imported_names_skips_undecodable_files():
    with TemporaryDirectory() as td:
        root = Path(td)
        # A file with binary content that can't be parsed as Python
        (root / "bad.py").write_bytes(b"\x00\x01\x02\xff\xfe")
        (root / "good.py").write_text("import requests\n", encoding="utf-8")
        found = imported_names(root)
        assert "requests" in found


def test_imported_names_returns_empty_when_no_py_files():
    with TemporaryDirectory() as td:
        root = Path(td)
        (root / "readme.txt").write_text("hello", encoding="utf-8")
        found = imported_names(root)
        assert found == set()


def test_imported_names_collects_top_level_only():
    """import x.y.z should record only 'x' (top-level package)."""
    with TemporaryDirectory() as td:
        root = Path(td)
        (root / "main.py").write_text(
            "import foo.bar.baz\nfrom a.b import c\n",
            encoding="utf-8",
        )
        found = imported_names(root)
        assert "foo" in found
        assert "a" in found
        assert "foo.bar.baz" not in found
        assert "a.b" not in found


def test_imported_names_skips_relative_imports():
    """Relative imports (from . import x) are local and should not be collected."""
    with TemporaryDirectory() as td:
        root = Path(td)
        (root / "main.py").write_text(
            textwrap.dedent("""\
                from . import local_mod
                from ..parent import something
                import os
            """),
            encoding="utf-8",
        )
        found = imported_names(root)
        assert "os" in found
        # Relative imports have node.level > 0 and node.module is None for "from . import x"
        # so they should not be collected at all
        assert "local_mod" not in found


# --- import_candidates (name normalization) -------------------------------------------------


def test_import_candidates_normalizes_dashes():
    candidates = import_candidates("my-package")
    assert "my_package" in candidates
    assert "my-package" in candidates


def test_import_candidates_normalizes_underscores():
    candidates = import_candidates("my_package")
    assert "my_package" in candidates
    assert "my-package" in candidates


def test_import_candidates_uses_importlib_metadata_when_available():
    """For a known stdlib or installed package, importlib.metadata reverse lookup adds the
    import name."""
    # 'os' is a stdlib module and should resolve via importlib.metadata
    candidates = import_candidates("os")
    assert "os" in candidates


def test_import_candidates_always_includes_normalized_forms():
    """Even when importlib.metadata fails, dash/underscore normalization is the fallback."""
    for name in ("requests", "REQUESTS", "Requests"):
        candidates = import_candidates(name)
        lower = name.lower()
        assert lower.replace("-", "_") in candidates
        assert lower.replace("_", "-") in candidates


# --- is_imported (full path) ----------------------------------------------------------------


def test_is_imported_returns_true_when_package_imported():
    with TemporaryDirectory() as td:
        root = Path(td)
        (root / "main.py").write_text("import requests\n", encoding="utf-8")
        result = is_imported(root, "requests")
        assert result is True


def test_is_imported_returns_false_when_not_imported():
    with TemporaryDirectory() as td:
        root = Path(td)
        (root / "main.py").write_text("import os\n", encoding="utf-8")
        result = is_imported(root, "nonexistent-package")
        assert result is False


def test_is_imported_returns_none_on_no_py_files():
    with TemporaryDirectory() as td:
        root = Path(td)
        (root / "readme.txt").write_text("hello", encoding="utf-8")
        result = is_imported(root, "requests")
        assert result is None


def test_is_imported_never_raises():
    with TemporaryDirectory() as td:
        root = Path(td)
        # Even with weird inputs, should not raise
        assert is_imported(root, "") is None
        assert is_imported(root, "123weird") is not True  # might be None or False


# --- _demote (priority demotion) ------------------------------------------------------------


def test_demote_false_raises_priority_by_two():
    pri, note = _demote(6, False)
    assert pri == 8
    assert note == (
        "⚠ vulnerable package not imported by this repo's code "
        "(import-graph heuristic) — deprioritized"
    )


def test_demote_true_unchanged():
    pri, note = _demote(6, True)
    assert pri == 6
    assert note is None


def test_demote_none_unchanged():
    pri, note = _demote(6, None)
    assert pri == 6
    assert note is None


def test_demote_clamps_to_max():
    pri, _ = _demote(9, False)
    assert pri == 9


def test_demote_clamps_to_min():
    pri, _ = _demote(1, False)
    assert pri == 3


def test_demote_near_max_clamps():
    pri, _ = _demote(8, False)
    assert pri == 9


def test_demote_already_at_max_stays():
    pri, note = _demote(9, False)
    assert pri == 9
    assert note is None  # no change, no note


# --- emit_advisory reachability demotion ----------------------------------------------------


def test_emit_advisory_includes_demotion_when_reachable_false():
    candidate = BumpCandidate(name="fake-pkg", current="1.0", latest="2.0", delta="minor")
    evidence = EvidenceBundle(
        candidate=candidate,
        complete=True,
        target_yanked=False,
        reachable=False,
        lockfile_changes=["fake-pkg 1.0->2.0"],
    )
    with patch("forge.dependabot.emit.get_task_store") as mock_store:
        emit_advisory(
            candidate,
            evidence,
            "some reason",
            project="Meta",
            branch="deps/fake-pkg-2-0",
            base_priority=6,
        )
        (specs,), kwargs = mock_store.return_value.emit.call_args
        assert kwargs["project"] == "Meta"
        assert kwargs["status"] == "Ready"
        spec = specs[0]
        assert spec.priority == 8
        assert "not imported by this repo's code" in spec.content


def test_emit_advisory_no_demotion_when_reachable_true():
    candidate = BumpCandidate(name="fake-pkg", current="1.0", latest="2.0", delta="minor")
    evidence = EvidenceBundle(
        candidate=candidate,
        complete=True,
        target_yanked=False,
        reachable=True,
        lockfile_changes=["fake-pkg 1.0->2.0"],
    )
    with patch("forge.dependabot.emit.get_task_store") as mock_store:
        emit_advisory(
            candidate,
            evidence,
            "some reason",
            project="Meta",
            branch="deps/fake-pkg-2-0",
            base_priority=6,
        )
        (specs,), _ = mock_store.return_value.emit.call_args
        assert specs[0].priority == 6


def test_emit_advisory_no_demotion_when_reachable_none():
    candidate = BumpCandidate(name="fake-pkg", current="1.0", latest="2.0", delta="minor")
    evidence = EvidenceBundle(
        candidate=candidate,
        complete=True,
        target_yanked=False,
        reachable=None,
        lockfile_changes=["fake-pkg 1.0->2.0"],
    )
    with patch("forge.dependabot.emit.get_task_store") as mock_store:
        emit_advisory(
            candidate,
            evidence,
            "some reason",
            project="Meta",
            branch="deps/fake-pkg-2-0",
            base_priority=6,
        )
        (specs,), _ = mock_store.return_value.emit.call_args
        assert specs[0].priority == 6


def test_emit_advisory_no_demotion_when_evidence_is_none():
    candidate = BumpCandidate(name="fake-pkg", current="1.0", latest="2.0", delta="minor")
    with patch("forge.dependabot.emit.get_task_store") as mock_store:
        emit_advisory(
            candidate,
            None,
            "some reason",
            project="Meta",
            branch="deps/fake-pkg-2-0",
            base_priority=6,
        )
        (specs,), _ = mock_store.return_value.emit.call_args
        assert specs[0].priority == 6


# --- policy: reachable is NOT in auto_eligible ----------------------------------------------


def test_auto_eligible_ignores_reachable_true():
    """reachable=True should not affect eligibility."""
    candidate = BumpCandidate(name="idna", current="3.11", latest="3.15", delta="minor")
    evidence = EvidenceBundle(
        candidate=candidate,
        target_yanked=False,
        complete=True,
        target_attested=True,
        reachable=True,
        lockfile_changes=["idna 3.11->3.15"],
    )
    eligible, _ = auto_eligible(evidence)
    assert eligible


def test_auto_eligible_ignores_reachable_false():
    """reachable=False should NOT make the bump ineligible — demote only."""
    candidate = BumpCandidate(name="idna", current="3.11", latest="3.15", delta="minor")
    evidence = EvidenceBundle(
        candidate=candidate,
        target_yanked=False,
        complete=True,
        target_attested=True,
        reachable=False,
        lockfile_changes=["idna 3.11->3.15"],
    )
    eligible, _ = auto_eligible(evidence)
    assert eligible


def test_auto_eligible_ignores_reachable_none():
    """reachable=None should NOT make the bump ineligible."""
    candidate = BumpCandidate(name="idna", current="3.11", latest="3.15", delta="minor")
    evidence = EvidenceBundle(
        candidate=candidate,
        target_yanked=False,
        complete=True,
        target_attested=True,
        reachable=None,
        lockfile_changes=["idna 3.11->3.15"],
    )
    eligible, _ = auto_eligible(evidence)
    assert eligible


def test_auto_eligible_identical_for_reachable_true_false_none():
    """Truth table: reachable=True/False/None all produce the same eligibility verdict."""
    candidate = BumpCandidate(name="idna", current="3.11", latest="3.15", delta="minor")
    results = []
    for val in (True, False, None):
        evidence = EvidenceBundle(
            candidate=candidate,
            target_yanked=False,
            complete=True,
            target_attested=True,
            reachable=val,
            lockfile_changes=["idna 3.11->3.15"],
        )
        eligible, reason = auto_eligible(evidence)
        results.append((eligible, reason))
    # All three should be identical
    assert results[0] == results[1] == results[2]
