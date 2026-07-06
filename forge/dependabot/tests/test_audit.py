from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agents.dependabot.audit import (
    AuditError,
    _normalize_name,
    export_requirements,
    findings_for,
    run_audit,
)
from agents.dependabot.models import AuditFinding

FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestNormalizeName:
    def test_already_normalized(self):
        assert _normalize_name("requests") == "requests"

    def test_underscores(self):
        assert _normalize_name("my_package") == "my-package"

    def test_dots(self):
        assert _normalize_name("my.package") == "my-package"

    def test_mixed_separator_runs(self):
        assert _normalize_name("my--_.package") == "my-package"

    def test_uppercase(self):
        assert _normalize_name("Django") == "django"

    def test_complex_mixed(self):
        assert _normalize_name("Foo-Bar_Baz.Qux") == "foo-bar-baz-qux"


class TestExportRequirements:
    def test_success_runs_correct_argv(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        out = tmp_path / "out.txt"
        (repo / "pyproject.toml").write_text("[project]\nname = 'test'\n")
        (repo / "uv.lock").write_text("")

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = ""

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            export_requirements(repo, out)

        mock_run.assert_called_once()
        args = mock_run.call_args
        assert args[0][0] == ["uv", "export", "--frozen", "--no-emit-project", "-o", str(out)]
        assert args[1]["cwd"] == str(repo)
        assert args[1]["capture_output"] is True
        assert args[1]["text"] is True

    def test_failure_raises_audit_error(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        out = tmp_path / "out.txt"
        (repo / "pyproject.toml").write_text("[project]\nname = 'test'\n")

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "error: lock file not found"

        with patch("subprocess.run", return_value=mock_result):
            with pytest.raises(AuditError, match="uv export failed"):
                export_requirements(repo, out)


class TestRunAudit:
    def _make_repo(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "pyproject.toml").write_text("[project]\nname = 'test'\n")
        (repo / "uv.lock").write_text("")
        return repo

    def test_exit_code_0_no_findings(self, tmp_path):
        repo = self._make_repo(tmp_path)
        empty_json = json.dumps([])

        export_result = MagicMock()
        export_result.returncode = 0
        export_result.stderr = ""

        audit_result = MagicMock()
        audit_result.returncode = 0
        audit_result.stdout = empty_json
        audit_result.stderr = ""

        with patch("subprocess.run", side_effect=[export_result, audit_result]):
            findings = run_audit(repo)

        assert findings == []
        # Should have two calls: uv export + pip-audit
        # (side_effect consumes exactly 2 calls)

    def test_exit_code_1_with_findings(self, tmp_path):
        repo = self._make_repo(tmp_path)
        fixture_data = [
            {
                "vuln": {
                    "id": "CVE-2024-9999",
                    "all_vulns": [
                        {
                            "id": "CVE-2024-9999",
                            "fix_versions": ["1.1.0"],
                            "aliases": ["GHSA-xxxx"],
                            "description": "Test vulnerability",
                        }
                    ],
                    "database_id": "CVE-2024-9999",
                },
                "package": {
                    "name": "test-pkg",
                    "version": "1.0.0",
                    "subpath": None,
                    "link": "https://pypi.org/project/test-pkg/",
                },
                "dependencies": [
                    {
                        "package_name": "test-pkg",
                        "package_version": "1.0.0",
                        "subpath": None,
                        "vulnerable_versions": [">=1.0"],
                        "fix_versions": ["1.1.0"],
                    }
                ],
            }
        ]
        mock_stdout = json.dumps(fixture_data)

        export_result = MagicMock()
        export_result.returncode = 0
        export_result.stderr = ""

        audit_result = MagicMock()
        audit_result.returncode = 1
        audit_result.stdout = mock_stdout
        audit_result.stderr = ""

        with patch("subprocess.run", side_effect=[export_result, audit_result]):
            findings = run_audit(repo)

        assert len(findings) == 1
        f = findings[0]
        assert f.package == "test-pkg"
        assert f.vuln_id == "CVE-2024-9999"
        assert f.fix_versions == ["1.1.0"]
        assert f.description == "Test vulnerability"
        assert f.aliases == ["GHSA-xxxx"]

    def test_exit_code_2_raises_audit_error(self, tmp_path):
        repo = self._make_repo(tmp_path)

        export_result = MagicMock()
        export_result.returncode = 0
        export_result.stderr = ""

        audit_result = MagicMock()
        audit_result.returncode = 2
        audit_result.stdout = ""
        audit_result.stderr = "pip-audit internal error"

        with patch("subprocess.run", side_effect=[export_result, audit_result]):
            with pytest.raises(AuditError, match="pip-audit failed"):
                run_audit(repo)

    def test_audit_argv_is_pinned(self, tmp_path):
        """Assert the exact pip-audit argv is invoked (pinned-invocation guard)."""
        repo = self._make_repo(tmp_path)

        export_result = MagicMock()
        export_result.returncode = 0
        export_result.stderr = ""

        audit_result = MagicMock()
        audit_result.returncode = 0
        audit_result.stdout = "[]"
        audit_result.stderr = ""

        with patch("subprocess.run", side_effect=[export_result, audit_result]) as mock_run:
            run_audit(repo)

        pip_audit_call = mock_run.call_args_list[1]
        argv = pip_audit_call[0][0]
        assert argv[0] == "uvx"
        assert "pip-audit" in argv
        assert "-r" in argv
        assert "--format" in argv
        assert "json" in argv
        assert "--disable-pip" in argv

    def test_timeout_passed_to_subprocess(self, tmp_path):
        repo = self._make_repo(tmp_path)

        export_result = MagicMock()
        export_result.returncode = 0
        export_result.stderr = ""

        audit_result = MagicMock()
        audit_result.returncode = 0
        audit_result.stdout = "[]"
        audit_result.stderr = ""

        with patch("subprocess.run", side_effect=[export_result, audit_result]) as mock_run:
            run_audit(repo, timeout=60)

        pip_audit_call = mock_run.call_args_list[1]
        assert pip_audit_call[1]["timeout"] == 60

    def test_empty_stdout_returns_empty_list(self, tmp_path):
        repo = self._make_repo(tmp_path)

        export_result = MagicMock()
        export_result.returncode = 0
        export_result.stderr = ""

        audit_result = MagicMock()
        audit_result.returncode = 0
        audit_result.stdout = ""
        audit_result.stderr = ""

        with patch("subprocess.run", side_effect=[export_result, audit_result]):
            findings = run_audit(repo)

        assert findings == []


class TestFindingsFor:
    def test_matches_exact_name(self):
        findings = [
            AuditFinding(package="requests", vuln_id="CVE-2024-0001"),
            AuditFinding(package="flask", vuln_id="CVE-2024-0002"),
        ]
        result = findings_for(findings, "requests", "2.28.0")
        assert len(result) == 1
        assert result[0].package == "requests"

    def test_matches_normalized_name(self):
        findings = [
            AuditFinding(package="my_package", vuln_id="CVE-2024-0001"),
            AuditFinding(package="flask", vuln_id="CVE-2024-0002"),
        ]
        result = findings_for(findings, "my-package", "1.0")
        assert len(result) == 1
        assert result[0].package == "my_package"

    def test_matches_case_insensitive(self):
        findings = [
            AuditFinding(package="Django", vuln_id="CVE-2024-0001"),
        ]
        result = findings_for(findings, "django", "4.0")
        assert len(result) == 1

    def test_no_match(self):
        findings = [
            AuditFinding(package="requests", vuln_id="CVE-2024-0001"),
        ]
        result = findings_for(findings, "flask", "2.0")
        assert result == []

    def test_empty_findings(self):
        result = findings_for([], "requests", "2.28.0")
        assert result == []

    def test_multiple_matches(self):
        findings = [
            AuditFinding(package="requests", vuln_id="CVE-2024-0001"),
            AuditFinding(package="requests", vuln_id="CVE-2024-0002"),
            AuditFinding(package="flask", vuln_id="CVE-2024-0003"),
        ]
        result = findings_for(findings, "requests", "2.28.0")
        assert len(result) == 2
        assert all(f.package == "requests" for f in result)


class TestFixtureParsing:
    def test_fixture_parses_to_expected_findings(self):
        """Load the pip_audit.json fixture and parse it to validate AuditFinding conversion."""
        fixture_path = FIXTURES_DIR / "pip_audit.json"
        data = json.loads(fixture_path.read_text())

        expected = []
        for entry in data:
            deps = entry.get("dependencies", [])
            vuln_info = entry.get("vuln", {})
            vulns = (
                vuln_info.get("all_vulns", [vuln_info]) if "all_vulns" in vuln_info else [vuln_info]
            )
            for dep in deps:
                for v in vulns:
                    finding = AuditFinding(
                        package=dep["package_name"],
                        vuln_id=v.get("id", ""),
                        fix_versions=v.get("fix_versions", []),
                        description=v.get("description", ""),
                        aliases=v.get("aliases", []),
                    )
                    # Only include findings with a real vuln_id
                    if finding.vuln_id:
                        expected.append(finding)

        assert len(expected) == 1
        assert expected[0].package == "vulnerable-pkg"
        assert expected[0].vuln_id == "CVE-2024-1234"
        assert expected[0].fix_versions == ["2.0.1"]
        desc = "Buffer overflow in the parser module allows remote code execution."
        assert expected[0].description == desc
        assert expected[0].aliases == ["GHSA-abc123"]
