"""PnpmEcosystem adapter tests — fixtures live-captured from pnpm 11.8.0 fleet clones.

``pnpm_outdated.json`` (example-org/example-lib) and ``pnpm_audit.json`` (example-org/example-app,
trimmed) pin the shapes the module docstring of ``pnpm.py`` documents: no "current" field
in outdated output, exit 1 = results found, npm-audit-v1 advisories with range-typed
``patched_versions``.
"""

from __future__ import annotations

from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

import pytest

from forge.dependabot.audit import AuditError
from forge.dependabot.bump import BumpError
from forge.dependabot.ecosystems import pnpm as pnpm_mod
from forge.dependabot.ecosystems.pnpm import (
    PnpmEcosystem,
    _locked_versions,
    _parse_pnpm_lock_diff,
)
from forge.dependabot.models import AuditFinding, BumpCandidate
from forge.dependabot.policy import auto_eligible
from forge.dependabot.scan import ScanError

FIXTURES = Path(__file__).parent / "fixtures"
OUTDATED_FIXTURE = (FIXTURES / "pnpm_outdated.json").read_text()
AUDIT_FIXTURE = (FIXTURES / "pnpm_audit.json").read_text()

LOCKFILE_V9 = """\
lockfileVersion: '9.0'

importers:

  .:
    dependencies:
      react-router:
        specifier: ^7.10
        version: 7.10.0
      styled-thing:
        specifier: ^5.0
        version: 5.0.0(typescript@5.9.3)
    devDependencies:
      prettier:
        specifier: ^3.4
        version: 3.8.1
      typescript:
        specifier: ^5.7
        version: 5.9.3

packages:

  prettier@3.8.1:
    resolution: {integrity: sha512-xxx}
"""


def _repo(tmp_path: Path) -> Path:
    (tmp_path / "pnpm-lock.yaml").write_text(LOCKFILE_V9)
    return tmp_path


def _proc(returncode: int, stdout: str = "", stderr: str = "") -> CompletedProcess:
    return CompletedProcess(args=["pnpm"], returncode=returncode, stdout=stdout, stderr=stderr)


# ---------------------------------------------------------------------------
# _locked_versions
# ---------------------------------------------------------------------------


class TestLockedVersions:
    def test_v9_importers_with_peer_suffix_stripped(self, tmp_path):
        locked = _locked_versions(_repo(tmp_path))
        assert locked["prettier"] == "3.8.1"
        assert locked["typescript"] == "5.9.3"
        assert locked["styled-thing"] == "5.0.0"  # (typescript@5.9.3) stripped

    def test_old_top_level_shape_and_string_entries(self, tmp_path):
        (tmp_path / "pnpm-lock.yaml").write_text(
            "lockfileVersion: 5.4\ndependencies:\n  lodash: 4.17.21\n"
        )
        assert _locked_versions(tmp_path) == {"lodash": "4.17.21"}

    def test_missing_or_broken_lockfile_is_empty(self, tmp_path):
        assert _locked_versions(tmp_path) == {}
        (tmp_path / "pnpm-lock.yaml").write_text(": not: valid: yaml: [")
        assert _locked_versions(tmp_path) == {}


# ---------------------------------------------------------------------------
# scan_outdated
# ---------------------------------------------------------------------------


class TestScanOutdated:
    def test_fixture_with_locked_current_versions(self, tmp_path):
        repo = _repo(tmp_path)
        with patch("forge.dependabot.ecosystems.pnpm.subprocess.run") as m:
            m.return_value = _proc(1, OUTDATED_FIXTURE)  # exit 1 = results found
            candidates = PnpmEcosystem().scan_outdated(repo)

        by_name = {c.name: c for c in candidates}
        prettier = by_name["prettier"]
        assert prettier.current == "3.8.1"  # from the LOCKFILE, not the outdated JSON
        assert prettier.latest == "3.9.5"
        assert prettier.delta == "minor"
        assert by_name["typescript"].delta == "major"
        assert [c.delta for c in candidates] == ["minor", "major"]  # lowest-delta-first sort

    def test_entry_missing_from_lockfile_is_dropped(self, tmp_path):
        repo = _repo(tmp_path)
        outdated = '{"ghost-pkg": {"latest": "2.0.0", "wanted": "1.0.0"}}'
        with patch("forge.dependabot.ecosystems.pnpm.subprocess.run") as m:
            m.return_value = _proc(1, outdated)
            candidates = PnpmEcosystem().scan_outdated(repo)
        assert candidates == []  # unprovable current is not a candidate

    def test_nothing_outdated_is_empty(self, tmp_path):
        repo = _repo(tmp_path)
        with patch("forge.dependabot.ecosystems.pnpm.subprocess.run") as m:
            m.return_value = _proc(0, "{}")
            assert PnpmEcosystem().scan_outdated(repo) == []

    def test_unexpected_exit_raises_scan_error(self, tmp_path):
        with patch("forge.dependabot.ecosystems.pnpm.subprocess.run") as m:
            m.return_value = _proc(2, "", "ERR_PNPM_NO_LOCKFILE")
            with pytest.raises(ScanError, match="ERR_PNPM_NO_LOCKFILE"):
                PnpmEcosystem().scan_outdated(_repo(tmp_path))


# ---------------------------------------------------------------------------
# apply_bump — --ignore-scripts is the security floor
# ---------------------------------------------------------------------------


def _candidate(name="prettier", current="3.8.1", latest="3.9.5", delta="patch"):
    return BumpCandidate(name=name, current=current, latest=latest, delta=delta)


class TestApplyBump:
    def test_update_carries_ignore_scripts_always(self, monkeypatch, tmp_path):
        commands = []

        def fake_run(cmd, **kw):
            commands.append(cmd)
            return _proc(0)

        monkeypatch.setattr(pnpm_mod.subprocess, "run", fake_run)
        monkeypatch.setattr(
            pnpm_mod, "get_changed_files", lambda repo: ["package.json", "pnpm-lock.yaml"]
        )
        changed = PnpmEcosystem().apply_bump(tmp_path, _candidate())
        assert changed == ["package.json", "pnpm-lock.yaml"]
        # The security floor: lifecycle scripts of untrusted packages must never run.
        assert commands[0] == ["pnpm", "update", "prettier", "--ignore-scripts"]

    def test_failure_reverts_and_raises(self, monkeypatch, tmp_path):
        reverted = []
        monkeypatch.setattr(
            pnpm_mod.subprocess, "run", lambda cmd, **kw: _proc(1, "", "ERR_PNPM_FETCH")
        )
        monkeypatch.setattr(pnpm_mod, "revert_changes", lambda repo: reverted.append(repo))
        with pytest.raises(BumpError, match="ERR_PNPM_FETCH"):
            PnpmEcosystem().apply_bump(tmp_path, _candidate())
        assert reverted


# ---------------------------------------------------------------------------
# lockfile_delta — pnpm-lock.yaml package-header parsing
# ---------------------------------------------------------------------------


LOCK_DIFF = """\
diff --git a/pnpm-lock.yaml b/pnpm-lock.yaml
--- a/pnpm-lock.yaml
+++ b/pnpm-lock.yaml
@@ -10,7 +10,7 @@
-  prettier@3.8.1:
+  prettier@3.9.5:
-  '@types/node@20.1.2':
+  '@types/node@20.2.0':
-  styled-thing@5.0.0(typescript@5.9.3):
+  styled-thing@5.0.0(typescript@7.0.2):
diff --git a/other.yaml b/other.yaml
--- a/other.yaml
+++ b/other.yaml
@@ -1 +1 @@
-  red-herring@1.0.0:
+  red-herring@2.0.0:
"""


class TestLockDelta:
    def test_pairs_scoped_names_and_same_version_filtering(self):
        deltas = _parse_pnpm_lock_diff(LOCK_DIFF)
        assert "prettier 3.8.1->3.9.5" in deltas
        assert "@types/node 20.1.2->20.2.0" in deltas  # quoted scoped key
        # styled-thing changed only its peer suffix — same version, filtered.
        assert not any(d.startswith("styled-thing") for d in deltas)
        # Other files' hunks never leak in.
        assert not any("red-herring" in d for d in deltas)
        assert len(deltas) == 2

    def test_garbage_is_empty(self):
        assert _parse_pnpm_lock_diff("not a diff") == []


# ---------------------------------------------------------------------------
# run_audit — pnpm audit primary, osv-scanner fallback, fail-closed floor
# ---------------------------------------------------------------------------


class TestRunAudit:
    def test_advisories_parsed_with_ghsa_ids_and_range_lower_bounds(self, monkeypatch, tmp_path):
        seen = {}

        def fake_run(cmd, **kw):
            seen["cmd"] = cmd
            return _proc(1, AUDIT_FIXTURE)  # exit 1 = advisories found

        monkeypatch.setattr(pnpm_mod.subprocess, "run", fake_run)
        findings = PnpmEcosystem().run_audit(tmp_path)

        assert seen["cmd"] == ["pnpm", "audit", "--json"]
        by_id = {f.vuln_id: f for f in findings}
        assert set(by_id) == {"GHSA-2w69-qvjg-hvjx", "GHSA-8v8x-cx79-35w7"}
        first = by_id["GHSA-2w69-qvjg-hvjx"]
        assert first.package == "react-router"
        assert first.fix_versions == ["7.11.1"]  # the >= lower bound of the patched range
        assert "XSS" in first.description

    def test_unexpected_exit_raises(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            pnpm_mod.subprocess, "run", lambda cmd, **kw: _proc(2, "", "registry down")
        )
        with pytest.raises(AuditError, match="registry down"):
            PnpmEcosystem().run_audit(tmp_path)

    def test_missing_pnpm_falls_back_to_osv_scanner(self, monkeypatch, tmp_path):
        def fake_run(cmd, **kw):
            if cmd[0] == "pnpm":
                raise FileNotFoundError("pnpm")
            assert cmd == ["osv-scanner", "--format", "json", "--lockfile", "pnpm-lock.yaml"]
            return _proc(0, "")

        monkeypatch.setattr(pnpm_mod.subprocess, "run", fake_run)
        assert PnpmEcosystem().run_audit(tmp_path) == []

    def test_neither_scanner_fails_closed(self, monkeypatch, tmp_path):
        def fake_run(cmd, **kw):
            raise FileNotFoundError(cmd[0])

        monkeypatch.setattr(pnpm_mod.subprocess, "run", fake_run)
        with pytest.raises(AuditError, match="pnpm nor osv-scanner"):
            PnpmEcosystem().run_audit(tmp_path)


# ---------------------------------------------------------------------------
# collect_evidence + policy — the demote-not-block contract
# ---------------------------------------------------------------------------


class TestEvidence:
    def _findings(self) -> list[AuditFinding]:
        return [
            AuditFinding(package="react-router", vuln_id="GHSA-FIXED", fix_versions=["7.11.1"]),
            AuditFinding(package="react-router", vuln_id="GHSA-REMAINS", fix_versions=["7.12.0"]),
        ]

    def test_findings_split_current_vs_target(self, tmp_path):
        candidate = _candidate(name="react-router", current="7.10.0", latest="7.11.1")
        bundle = PnpmEcosystem().collect_evidence(
            candidate, self._findings(), ["react-router 7.10.0->7.11.1"], repo_root=tmp_path
        )
        assert [f.vuln_id for f in bundle.findings_current] == ["GHSA-FIXED"]
        assert [f.vuln_id for f in bundle.findings_target] == ["GHSA-REMAINS"]

    def test_policy_demotes_with_the_adapter_reason(self, tmp_path):
        bundle = PnpmEcosystem().collect_evidence(_candidate(), [], [], repo_root=tmp_path)
        eligible, reason = auto_eligible(bundle)
        assert eligible is False
        assert "advisory track" in reason and "pnpm" in reason
