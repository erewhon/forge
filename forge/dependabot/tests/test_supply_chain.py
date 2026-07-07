"""Evidence assembly tests — fetchers injected, NO network."""

from __future__ import annotations

import io
import json
import tarfile
from datetime import UTC, datetime
from pathlib import Path

from agents.dependabot.models import AuditFinding, BumpCandidate
from agents.dependabot.supply_chain import (
    changelog_url,
    collect_evidence,
    install_script_change,
    maintainer_change,
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


# --- v2 provenance signals ------------------------------------------------------------------


def test_maintainer_same_identity_is_false():
    # Real shape (idna capture 2026-07-06): name embedded in author_email, no maintainer.
    info = {"author": None, "author_email": "Kim Davies <kim+pypi@gumleaf.org>"}
    assert maintainer_change(info, dict(info)) is False


def test_maintainer_email_change_is_true():
    cur = {"author_email": "Kim Davies <kim+pypi@gumleaf.org>"}
    tgt = {"author_email": "Kim Davies <totally-new-owner@evil.example>"}
    assert maintainer_change(cur, tgt) is True


def test_maintainer_multi_email_set_compare():
    # Real shape (mcp capture): author name + comma-separated maintainer_email addresses.
    cur = {"author": "Anthropic, PBC.", "maintainer_email": "David <d@a.com>, Justin <j@a.com>"}
    tgt = {"author": "Anthropic, PBC.", "maintainer_email": "Justin <j@a.com>, David <d@a.com>"}
    assert maintainer_change(cur, tgt) is False  # same SET, different order


def test_maintainer_name_fallback_when_no_emails():
    cur = {"author": "Alice"}
    assert maintainer_change(cur, {"author": "Alice"}) is False
    assert maintainer_change(cur, {"author": "Mallory"}) is True


def test_maintainer_undeterminable_is_none():
    assert maintainer_change(None, {"author": "Alice"}) is None
    assert maintainer_change({}, {"author": "Alice"}) is None  # no identity on one side
    assert maintainer_change({"author": "Alice"}, {}) is None


def _sdist_bytes(members: dict[str, str]) -> bytes:
    """A tiny in-memory .tar.gz sdist with the given root-relative members."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, content in members.items():
            data = content.encode()
            info = tarfile.TarInfo(name=f"pkg-1.0/{name}")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _meta_with_sdist(url: str) -> dict:
    return {"urls": [{"packagetype": "sdist", "url": url}]}


_PYPROJECT = '[build-system]\nbuild-backend = "hatchling.build"\n'


def test_install_scripts_identical_sdists_is_false():
    blobs = {
        "cur": _sdist_bytes({"pyproject.toml": _PYPROJECT}),
        "tgt": _sdist_bytes({"pyproject.toml": _PYPROJECT}),
    }
    result = install_script_change(
        _meta_with_sdist("cur"), _meta_with_sdist("tgt"), fetch_bytes=lambda u: blobs[u]
    )
    assert result is False


def test_install_scripts_new_setup_py_is_true():
    blobs = {
        "cur": _sdist_bytes({"pyproject.toml": _PYPROJECT}),
        "tgt": _sdist_bytes({"pyproject.toml": _PYPROJECT, "setup.py": "import os"}),
    }
    result = install_script_change(
        _meta_with_sdist("cur"), _meta_with_sdist("tgt"), fetch_bytes=lambda u: blobs[u]
    )
    assert result is True


def test_install_scripts_backend_change_is_true():
    changed = _PYPROJECT.replace("hatchling.build", "evil_backend.hooks")
    blobs = {
        "cur": _sdist_bytes({"pyproject.toml": _PYPROJECT}),
        "tgt": _sdist_bytes({"pyproject.toml": changed}),
    }
    result = install_script_change(
        _meta_with_sdist("cur"), _meta_with_sdist("tgt"), fetch_bytes=lambda u: blobs[u]
    )
    assert result is True


def test_install_scripts_no_sdists_anywhere_is_false():
    # Pure-wheel releases have no install-time script surface.
    assert install_script_change({"urls": []}, {"urls": []}, fetch_bytes=lambda u: None) is False


def test_install_scripts_one_sided_sdist_is_none():
    blobs = {"tgt": _sdist_bytes({"pyproject.toml": _PYPROJECT})}
    result = install_script_change(
        {"urls": []}, _meta_with_sdist("tgt"), fetch_bytes=lambda u: blobs[u]
    )
    assert result is None


def test_install_scripts_fetch_failure_is_none():
    result = install_script_change(
        _meta_with_sdist("cur"), _meta_with_sdist("tgt"), fetch_bytes=lambda u: None
    )
    assert result is None


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


def test_v2_signal_unavailability_does_not_mark_incomplete():
    """Contract: maintainer/install-script signals are best-effort — a current-version fetch
    failure yields None for both, while `complete` (target fetches + delta) stays True."""
    meta = _meta()

    def fetch_version(name, version):
        return None if version == "3.11" else meta["version"]  # current fetch fails

    bundle = collect_evidence(
        _candidate(),
        [],
        ["idna 3.11->3.15"],
        fetch_version=fetch_version,
        fetch_project=lambda n: meta["project"],
        fetch_bytes=lambda u: None,
    )
    assert bundle.complete  # target-side evidence is all present
    assert bundle.maintainer_changed is None
    assert bundle.new_install_scripts is None


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
