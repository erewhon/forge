"""Advisory Forge emission — spec shape and the project=None no-op."""

from __future__ import annotations

from unittest.mock import patch

from forge.dependabot.emit import _title, emit_advisory, external_ref
from forge.dependabot.models import BumpCandidate, EvidenceBundle


def _candidate() -> BumpCandidate:
    return BumpCandidate(name="idna", current="3.11", latest="3.15", delta="minor")


def test_external_ref_is_stable_per_package_and_target():
    assert external_ref(_candidate()) == "deps:idna:3.15"


def test_title_is_comma_free():
    assert "," not in _title(_candidate())


def test_project_none_is_a_noop():
    with patch("forge.dependabot.emit.get_task_store") as mock_store:
        assert emit_advisory(_candidate(), None, "reason", project=None, branch=None) is None
        mock_store.return_value.emit.assert_not_called()


def test_spec_carries_reason_branch_and_evidence():
    evidence = EvidenceBundle(
        candidate=_candidate(), lockfile_changes=["idna 3.11->3.15"], complete=True
    )
    with patch("forge.dependabot.emit.get_task_store") as mock_store:
        emit_advisory(
            _candidate(),
            evidence,
            "version delta is major",
            project="Meta",
            branch="deps/idna-3-15",
            log=lambda m: None,
        )
    (specs,), kwargs = mock_store.return_value.emit.call_args
    assert kwargs["project"] == "Meta"
    assert kwargs["status"] == "Ready"
    spec = specs[0]
    assert spec.external_ref == "deps:idna:3.15"
    assert "version delta is major" in spec.content
    assert "deps/idna-3-15" in spec.content
    assert "evidence complete: yes" in spec.content
