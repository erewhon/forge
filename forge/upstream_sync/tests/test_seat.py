"""Collision seat validation: citations enforced mechanically, failures never crash."""

from __future__ import annotations

import json

import pytest

from forge.upstream_sync import seat as st
from forge.upstream_sync.config import settings
from forge.upstream_sync.models import LayerManifest


def _layer() -> LayerManifest:
    return LayerManifest(added=["SPRINKLES.md"], modified=["README.md"])


def _verdict(monkeypatch, raw: str):
    monkeypatch.setattr(st, "complete", lambda *a, **kw: raw)
    return st.collision_verdict(
        layer=_layer(),
        upstream_files=["pkg/web/routes.go"],
        upstream_log="abc123 upstream change",
        upstream_stat=" pkg/web/routes.go | 10 +-",
        overlap=[],
        overlap_diff="",
    )


def test_cited_finding_survives(monkeypatch):
    raw = json.dumps(
        {
            "collision": True,
            "findings": [{"file": "pkg/web/routes.go", "reason": "layer registers routes here"}],
            "notes": "",
        }
    )
    v = _verdict(monkeypatch, raw)
    assert v.collision is True
    assert v.findings[0].file == "pkg/web/routes.go"


def test_uncited_finding_is_demoted_and_collision_downgraded(monkeypatch):
    raw = json.dumps(
        {
            "collision": True,
            "findings": [{"file": "made/up/file.go", "reason": "vibes"}],
            "notes": "",
        }
    )
    v = _verdict(monkeypatch, raw)
    assert v.collision is False  # true without a citable finding is worry, not evidence
    assert v.findings == []
    assert "demoted" in v.notes and "made/up/file.go" in v.notes


def test_layer_files_are_citable_too(monkeypatch):
    raw = json.dumps(
        {
            "collision": True,
            "findings": [{"file": "SPRINKLES.md", "reason": "add/add divergence"}],
        }
    )
    v = _verdict(monkeypatch, raw)
    assert v.collision is True


def test_unparseable_output_is_unknown(monkeypatch):
    v = _verdict(monkeypatch, "I feel like there might be a problem somewhere")
    assert v.collision is None
    assert "unparseable" in v.notes


def test_seat_exception_is_unknown_not_a_crash(monkeypatch):
    def boom(*a, **kw):
        raise RuntimeError("router down")

    monkeypatch.setattr(st, "complete", boom)
    v = st.collision_verdict(
        layer=_layer(),
        upstream_files=[],
        upstream_log="",
        upstream_stat="",
        overlap=[],
        overlap_diff="",
    )
    assert v.collision is None
    assert "router down" in v.notes


def test_disabled_seat_is_unknown(monkeypatch):
    monkeypatch.setattr(settings, "seat_enabled", False)
    v = st.collision_verdict(
        layer=_layer(),
        upstream_files=[],
        upstream_log="",
        upstream_stat="",
        overlap=[],
        overlap_diff="",
    )
    assert v.collision is None
    assert "disabled" in v.notes


@pytest.mark.parametrize("payload", [{}, {"findings": []}])
def test_missing_collision_key_is_unknown(monkeypatch, payload):
    v = _verdict(monkeypatch, json.dumps(payload))
    assert v.collision is None
