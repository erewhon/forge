"""Tests for the redundancy report module — prompt builder, parser, renderer."""

from __future__ import annotations

from unittest.mock import MagicMock

from forge.dependabot.models import BumpCandidate, RedundancyCluster, RedundancyReport
from forge.dependabot.redundancy import build_redundancy_prompt, call_model, render_report


def _deps() -> list[BumpCandidate]:
    return [
        BumpCandidate(name="httpx", current="0.27.0", latest="0.28.1", delta="minor"),
        BumpCandidate(name="requests", current="2.31.0", latest="2.32.3", delta="minor"),
        BumpCandidate(name="pydantic", current="2.6.0", latest="2.9.2", delta="minor"),
        BumpCandidate(name="click", current="8.1.7", latest="8.1.8", delta="patch"),
    ]


# --- prompt builder ---


def test_prompt_includes_dep_names():
    deps = _deps()
    system, user = build_redundancy_prompt(deps)
    for dep in deps:
        assert dep.name in user
        assert dep.current in user


def test_prompt_requests_json_clusters_response():
    system, user = build_redundancy_prompt(_deps())
    assert "clusters" in user
    assert "purpose" in user
    assert "migration_note" in user


# --- parser round-trip (canned model reply) ---


def test_call_model_parse_valid_cluster(monkeypatch):
    canned_response = {
        "clusters": [
            {
                "purpose": "HTTP client",
                "packages": ["httpx", "requests"],
                "keep": "httpx",
                "migration_note": "Migrate to httpx for async support",
            }
        ]
    }
    mock_return = "```json\n" + str(canned_response).replace("'", '"') + "\n```"
    mock_complete = MagicMock(return_value=mock_return)
    monkeypatch.setattr("forge.dependabot.redundancy.complete", mock_complete)

    report = call_model(_deps())
    assert len(report.clusters) == 1
    cluster = report.clusters[0]
    assert cluster.purpose == "HTTP client"
    assert cluster.packages == ["httpx", "requests"]
    assert cluster.keep == "httpx"
    assert "Migrate" in cluster.migration_note


def test_call_model_non_json_returns_empty(monkeypatch):
    mock_complete = MagicMock(return_value="just some plain text from the model")
    monkeypatch.setattr("forge.dependabot.redundancy.complete", mock_complete)

    report = call_model(_deps())
    assert report.clusters == []


def test_call_model_no_clusters_key_returns_empty(monkeypatch):
    mock_complete = MagicMock(return_value='{"data": "no clusters key here"}')
    monkeypatch.setattr("forge.dependabot.redundancy.complete", mock_complete)

    report = call_model(_deps())
    assert report.clusters == []


def test_call_model_empty_clusters_key_returns_empty(monkeypatch):
    mock_complete = MagicMock(return_value='{"clusters": []}')
    monkeypatch.setattr("forge.dependabot.redundancy.complete", mock_complete)

    report = call_model(_deps())
    assert report.clusters == []


# --- renderer ---


def test_render_report_with_clusters():
    report = RedundancyReport(
        clusters=[
            RedundancyCluster(
                purpose="HTTP client",
                packages=["httpx", "requests"],
                keep="httpx",
                migration_note="Migrate to httpx.",
            )
        ]
    )
    rendered = render_report(report, _deps())
    assert "# Dependency Redundancy Report" in rendered
    assert "**Dependencies scanned:** 4" in rendered
    assert "## Cluster 1: HTTP client" in rendered
    assert "httpx" in rendered
    assert "requests" in rendered
    assert "Migrate to httpx." in rendered


def test_render_report_empty_clusters():
    report = RedundancyReport(clusters=[])
    rendered = render_report(report, _deps())
    assert "# Dependency Redundancy Report" in rendered
    assert "**Dependencies scanned:** 4" in rendered
    assert "no overlapping-purpose" in rendered or "Each library" in rendered


def test_render_report_multiple_clusters():
    report = RedundancyReport(
        clusters=[
            RedundancyCluster(
                purpose="HTTP client",
                packages=["httpx", "requests"],
                keep="httpx",
                migration_note="Use httpx.",
            ),
            RedundancyCluster(
                purpose="JSON validation",
                packages=["pydantic", "jsonschema"],
                keep="pydantic",
                migration_note="Stick with pydantic.",
            ),
        ]
    )
    rendered = render_report(report, _deps())
    assert "## Cluster 1: HTTP client" in rendered
    assert "## Cluster 2: JSON validation" in rendered
    assert "**Clusters found:** 2" in rendered
