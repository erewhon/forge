from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agents.dependabot.audit import (
    AuditError,
    _filter_auditable,
    _normalize_name,
    _parse_findings,
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


class TestFilterAuditable:
    def test_drops_path_deps_keeps_pinned_blocks(self):
        # Shape captured from this repo's real `uv export` output: a local path dep has no
        # `==` and no hash, and breaks pip-audit's hash-checking mode.
        text = (
            "# comment\n"
            "../nous/nous-py\n"
            "    # via meta-agents\n"
            "anyio==4.12.1 \\\n"
            "    --hash=sha256:aaaa \\\n"
            "    --hash=sha256:bbbb\n"
        )
        filtered = _filter_auditable(text)
        assert "../nous/nous-py" not in filtered
        assert "anyio==4.12.1" in filtered
        assert "--hash=sha256:aaaa" in filtered  # continuation follows its header's fate
        assert "# comment" in filtered

    def test_editable_lines_dropped(self):
        text = "-e ../somewhere/pkg\nrequests==2.31.0\n"
        filtered = _filter_auditable(text)
        assert "-e" not in filtered
        assert "requests==2.31.0" in filtered


class TestExportRequirements:
    def test_success_runs_correct_argv(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        out = tmp_path / "out.txt"

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

        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "error: lock file not found"

        with patch("subprocess.run", return_value=mock_result):
            with pytest.raises(AuditError, match="uv export failed"):
                export_requirements(repo, tmp_path / "out.txt")


def _fake_subprocess(audit_returncode: int, audit_stdout: str):
    """A subprocess.run stand-in: the uv export call writes a real file (run_audit reads it
    back to filter), the pip-audit call returns the canned result."""
    calls: list[list[str]] = []

    def fake_run(argv, **kwargs):
        calls.append(list(argv) if not isinstance(argv, str) else [argv])
        result = MagicMock()
        result.stderr = ""
        if argv[0] == "uv":  # the export
            Path(argv[argv.index("-o") + 1]).write_text("pkg==1.0.0\n")
            result.returncode = 0
        else:  # pip-audit
            result.returncode = audit_returncode
            result.stdout = audit_stdout
        return result

    return fake_run, calls


class TestRunAudit:
    def _repo(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        return repo

    def test_exit_code_0_no_findings(self, tmp_path):
        fake, _ = _fake_subprocess(0, json.dumps({"dependencies": [], "fixes": []}))
        with patch("subprocess.run", side_effect=fake):
            assert run_audit(self._repo(tmp_path)) == []

    def test_exit_code_1_with_findings(self, tmp_path):
        # Real pip-audit --format json schema (captured from a live run 2026-07-06).
        payload = {
            "dependencies": [
                {"name": "clean-pkg", "version": "3.0.0", "vulns": []},
                {
                    "name": "test-pkg",
                    "version": "1.0.0",
                    "vulns": [
                        {
                            "id": "PYSEC-2026-0001",
                            "fix_versions": ["1.1.0"],
                            "aliases": ["CVE-2026-9999", "GHSA-xxxx"],
                            "description": "Test vulnerability",
                        }
                    ],
                },
            ],
            "fixes": [],
        }
        fake, _ = _fake_subprocess(1, json.dumps(payload))
        with patch("subprocess.run", side_effect=fake):
            findings = run_audit(self._repo(tmp_path))

        assert len(findings) == 1
        f = findings[0]
        assert f.package == "test-pkg"
        assert f.vuln_id == "PYSEC-2026-0001"
        assert f.fix_versions == ["1.1.0"]
        assert f.description == "Test vulnerability"
        assert f.aliases == ["CVE-2026-9999", "GHSA-xxxx"]

    def test_exit_code_2_raises_audit_error(self, tmp_path):
        fake, _ = _fake_subprocess(2, "")
        with patch("subprocess.run", side_effect=fake):
            with pytest.raises(AuditError, match="pip-audit failed"):
                run_audit(self._repo(tmp_path))

    def test_audit_argv_is_pinned(self, tmp_path):
        """Assert the exact pip-audit argv is invoked (pinned-invocation guard)."""
        fake, calls = _fake_subprocess(0, json.dumps({"dependencies": []}))
        with patch("subprocess.run", side_effect=fake):
            run_audit(self._repo(tmp_path))

        argv = calls[1]
        assert argv[0] == "uvx"
        assert "pip-audit" in argv
        assert "-r" in argv
        assert "--format" in argv
        assert "json" in argv
        assert "--disable-pip" in argv

    def test_export_output_is_filtered_before_audit(self, tmp_path):
        """The path-dep filter runs between export and pip-audit."""
        seen: dict[str, str] = {}

        def fake_run(argv, **kwargs):
            result = MagicMock()
            result.stderr = ""
            if argv[0] == "uv":
                Path(argv[argv.index("-o") + 1]).write_text("../local/pkg\nrequests==2.31.0\n")
                result.returncode = 0
            else:
                req = Path(argv[argv.index("-r") + 1])
                seen["contents"] = req.read_text()
                result.returncode = 0
                result.stdout = json.dumps({"dependencies": []})
            return result

        with patch("subprocess.run", side_effect=fake_run):
            run_audit(tmp_path)

        assert "../local/pkg" not in seen["contents"]
        assert "requests==2.31.0" in seen["contents"]

    def test_timeout_passed_to_subprocess(self, tmp_path):
        captured: dict = {}

        def fake_run(argv, **kwargs):
            result = MagicMock()
            result.stderr = ""
            if argv[0] == "uv":
                Path(argv[argv.index("-o") + 1]).write_text("pkg==1.0.0\n")
                result.returncode = 0
            else:
                captured.update(kwargs)
                result.returncode = 0
                result.stdout = json.dumps({"dependencies": []})
            return result

        with patch("subprocess.run", side_effect=fake_run):
            run_audit(self._repo(tmp_path), timeout=60)

        assert captured["timeout"] == 60

    def test_empty_stdout_returns_empty_list(self, tmp_path):
        fake, _ = _fake_subprocess(0, "")
        with patch("subprocess.run", side_effect=fake):
            assert run_audit(self._repo(tmp_path)) == []


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
        findings = [AuditFinding(package="Django", vuln_id="CVE-2024-0001")]
        assert len(findings_for(findings, "django", "4.0")) == 1

    def test_no_match(self):
        findings = [AuditFinding(package="requests", vuln_id="CVE-2024-0001")]
        assert findings_for(findings, "flask", "2.0") == []

    def test_empty_findings(self):
        assert findings_for([], "requests", "2.28.0") == []

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
    def test_fixture_parses_via_the_real_parser(self):
        """The fixture is a trimmed REAL pip-audit output; parse it with the production
        parser (not a reimplementation) so schema drift fails here first."""
        findings = _parse_findings((FIXTURES_DIR / "pip_audit.json").read_text())
        assert len(findings) == 1
        f = findings[0]
        assert f.package == "idna"
        assert f.vuln_id == "PYSEC-2026-215"
        assert f.fix_versions == ["3.15"]
        assert "CVE-2026-45409" in f.aliases
