"""Evidence assembly tests — fetchers injected, NO network."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from agents.dependabot.models import AuditFinding, BumpCandidate
from agents.dependabot.supply_chain import (
    changelog_url,
    collect_evidence,
    package_age_days,
    split_findings,
    typosquat_suspect,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _candidate(**over) -> BumpCandidate:
    base = dict(name="idna", current="3.11", latest="3.15", delta="minor")
    base.update(over)
    return BumpCandidate(**base)


def _meta() -> dict:
    return json.loads((FIXTURES / "pypi_meta.json").read_text())


# --- split_findings -------------------------------------------------------------------------


def test_finding_fixed_by_bump_lands_in_current():
    f = AuditFinding(package="idna", vuln_id="PYSEC-1", fix_versions=["3.15"])
    current, target = split_findings([f], _candidate())
    assert current == [f] and target == []


def test_finding_not_fixed_lands_in_target():
    f = AuditFinding(package="idna", vuln_id="PYSEC-2", fix_versions=["9.9.9"])
    current, target = split_findings([f], _candidate())
    assert current == [] and target == [f]


def test_unparseable_fix_version_is_conservative():
    f = AuditFinding(package="idna", vuln_id="PYSEC-3", fix_versions=["not-a-version"])
    current, target = split_findings([f], _candidate())
    assert current == [] and target == [f]


def test_no_fix_versions_is_conservative():
    f = AuditFinding(package="idna", vuln_id="PYSEC-4")
    _, target = split_findings([f], _candidate())
    assert target == [f]


def test_other_packages_findings_ignored():
    f = AuditFinding(package="requests", vuln_id="PYSEC-5", fix_versions=["1.0"])
    current, target = split_findings([f], _candidate())
    assert current == [] and target == []


def test_mixed_width_fix_version_compares():
    # target 3.15 vs fix "3.14.1" — padding must not break the compare
    f = AuditFinding(package="idna", vuln_id="PYSEC-6", fix_versions=["3.14.1"])
    current, _ = split_findings([f], _candidate())
    assert current == [f]


# --- signal helpers -------------------------------------------------------------------------


def test_package_age_days_from_earliest_upload():
    meta = {"urls": [{"upload_time_iso_8601": "2026-07-01T00:00:00Z"}]}
    now = datetime(2026, 7, 6, tzinfo=UTC)
    assert package_age_days(meta, now=now) == 5


def test_package_age_days_none_without_uploads():
    assert package_age_days({"urls": []}) is None


def test_changelog_url_prefers_changelog_keys():
    urls = {"Homepage": "https://h", "Changelog": "https://c"}
    assert changelog_url(urls) == "https://c"


def test_changelog_url_falls_back_to_repo():
    assert changelog_url({"Repository": "https://r"}) == "https://r"
    assert changelog_url({}) is None
    assert changelog_url(None) is None


def test_typosquat_one_edit_away_is_suspect():
    assert typosquat_suspect("reqeusts") == "requests"  # transposition
    assert typosquat_suspect("request") == "requests"  # deletion


def test_typosquat_exact_member_is_not_suspect():
    assert typosquat_suspect("requests") is None


def test_typosquat_distant_name_is_clean():
    assert typosquat_suspect("meta-agents-dependabot") is None


# --- collect_evidence -----------------------------------------------------------------------


def test_happy_path_is_complete():
    meta = _meta()
    bundle = collect_evidence(
        _candidate(),
        [AuditFinding(package="idna", vuln_id="PYSEC-2026-215", fix_versions=["3.15"])],
        ["idna 3.11->3.15"],
        fetch_version=lambda n, v: meta["version"],
        fetch_project=lambda n: meta["project"],
        now=datetime(2026, 7, 6, tzinfo=UTC),
    )
    assert bundle.complete
    assert bundle.target_yanked is False
    assert bundle.package_age_days is not None
    assert bundle.changelog_url == "https://github.com/kjd/idna/blob/master/HISTORY.md"
    assert [f.vuln_id for f in bundle.findings_current] == ["PYSEC-2026-215"]
    assert bundle.findings_target == []
    assert bundle.lockfile_changes == ["idna 3.11->3.15"]


def test_failed_version_fetch_is_incomplete_but_assembled():
    meta = _meta()
    bundle = collect_evidence(
        _candidate(),
        [],
        ["idna 3.11->3.15"],
        fetch_version=lambda n, v: None,
        fetch_project=lambda n: meta["project"],
    )
    assert not bundle.complete
    assert bundle.target_yanked is None
    assert bundle.package_age_days is None
    # package-level project_urls still give the changelog fallback
    assert bundle.changelog_url is not None


def test_failed_project_fetch_is_incomplete():
    meta = _meta()
    bundle = collect_evidence(
        _candidate(),
        [],
        ["idna 3.11->3.15"],
        fetch_version=lambda n, v: meta["version"],
        fetch_project=lambda n: None,
    )
    assert not bundle.complete


def test_empty_lock_delta_is_incomplete():
    meta = _meta()
    bundle = collect_evidence(
        _candidate(),
        [],
        [],
        fetch_version=lambda n, v: meta["version"],
        fetch_project=lambda n: meta["project"],
    )
    assert not bundle.complete


def test_yanked_target_is_flagged():
    meta = _meta()
    yanked = json.loads(json.dumps(meta["version"]))
    yanked["info"]["yanked"] = True
    bundle = collect_evidence(
        _candidate(),
        [],
        ["idna 3.11->3.15"],
        fetch_version=lambda n, v: yanked,
        fetch_project=lambda n: meta["project"],
    )
    assert bundle.target_yanked is True


def test_empty_findings_is_not_incompleteness():
    meta = _meta()
    bundle = collect_evidence(
        _candidate(),
        [],
        ["idna 3.11->3.15"],
        fetch_version=lambda n, v: meta["version"],
        fetch_project=lambda n: meta["project"],
    )
    assert bundle.complete
    assert bundle.findings_current == [] and bundle.findings_target == []
