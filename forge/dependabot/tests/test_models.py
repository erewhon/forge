from __future__ import annotations

import pytest

from forge.dependabot.config import DependabotSettings, settings
from forge.dependabot.models import (
    AuditFinding,
    BumpCandidate,
    BumpResult,
    EvidenceBundle,
)


class TestSettings:
    def test_default_branch_prefix(self):
        assert settings.branch_prefix == "deps"

    def test_default_signoff_max_tokens(self):
        assert settings.signoff_max_tokens == 4096

    def test_default_signoff_timeout(self):
        assert settings.signoff_timeout == 180.0

    def test_default_scan_timeout(self):
        assert settings.scan_timeout == 120

    def test_default_audit_timeout(self):
        assert settings.audit_timeout == 300

    def test_default_max_candidates(self):
        assert settings.max_candidates == 20

    def test_env_override_branch_prefix(self, monkeypatch):
        monkeypatch.setenv("DEPENDABOT_BRANCH_PREFIX", "dependabot/")
        s = DependabotSettings()
        assert s.branch_prefix == "dependabot/"

    def test_env_override_max_candidates(self, monkeypatch):
        monkeypatch.setenv("DEPENDABOT_MAX_CANDIDATES", "5")
        s = DependabotSettings()
        assert s.max_candidates == 5


class TestBumpCandidate:
    def test_defaults(self):
        c = BumpCandidate(name="requests", current="2.28.0", latest="2.31.0", delta="patch")
        assert c.direct is True

    def test_no_defaults(self):
        c = BumpCandidate(name="flask", current="2.0", latest="3.0", delta="major", direct=False)
        assert c.direct is False


class TestAuditFinding:
    def test_defaults(self):
        f = AuditFinding(package="pkg", vuln_id="CVE-2024-0001")
        assert f.fix_versions == []
        assert f.description == ""
        assert f.aliases == []

    def test_with_values(self):
        f = AuditFinding(
            package="pkg",
            vuln_id="CVE-2024-0001",
            fix_versions=["1.1.0", "2.0.0"],
            description="Remote code execution",
            aliases=["GHSA-xxxx"],
        )
        assert f.fix_versions == ["1.1.0", "2.0.0"]
        assert f.description == "Remote code execution"
        assert f.aliases == ["GHSA-xxxx"]


class TestEvidenceBundle:
    def test_defaults(self):
        candidate = BumpCandidate(name="pkg", current="1.0", latest="1.1", delta="patch")
        e = EvidenceBundle(candidate=candidate)
        assert e.findings_current == []
        assert e.findings_target == []
        assert e.target_yanked is None
        assert e.package_age_days is None
        assert e.changelog_url is None
        assert e.typosquat_suspect is None
        assert e.lockfile_changes == []
        assert e.complete is False

    def test_with_values(self):
        candidate = BumpCandidate(name="pkg", current="1.0", latest="2.0", delta="minor")
        finding = AuditFinding(package="pkg", vuln_id="CVE-2024-9999")
        e = EvidenceBundle(
            candidate=candidate,
            findings_current=[finding],
            findings_target=[],
            target_yanked=False,
            package_age_days=5,
            changelog_url="https://example.com/CHANGES",
            typosquat_suspect="pkq",
            lockfile_changes=["pkg==2.0.0"],
            complete=True,
        )
        assert len(e.findings_current) == 1
        assert e.complete is True


class TestBumpResult:
    def test_defaults(self):
        r = BumpResult(status="planned")
        assert r.reason == ""
        assert r.candidate is None
        assert r.branch is None
        assert r.change_id is None
        assert r.merged_to_main is False
        assert r.evidence is None
        assert r.tests_passed is None

    def test_with_values(self):
        candidate = BumpCandidate(name="pkg", current="1.0", latest="1.1", delta="patch")
        r = BumpResult(
            status="branched",
            reason="auto-merged",
            candidate=candidate,
            branch="deps/pkg-1.1",
            merged_to_main=True,
            tests_passed=True,
        )
        assert r.status == "branched"
        assert r.merged_to_main is True
        assert r.tests_passed is True


class TestDeltaClassValidation:
    def test_valid_deltas(self):
        for valid in ("patch", "minor", "major", "unknown"):
            candidate = BumpCandidate(name="pkg", current="1.0", latest="1.1", delta=valid)
            assert candidate.delta == valid

    def test_junk_delta_is_rejected(self):
        with pytest.raises(ValueError):
            BumpCandidate(name="pkg", current="1.0", latest="1.1", delta="junk")  # type: ignore[arg-type]
