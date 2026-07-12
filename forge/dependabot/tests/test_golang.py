"""GoEcosystem adapter tests — fixtures live-captured from go 1.26.4 / govulncheck v1.6.0.

``go_list_outdated.json`` and ``govulncheck_module.json`` are trimmed captures from the
soft-serve-with-sprinkles repo (2026-07-11); the shapes they pin are the ones the module
docstring of ``golang.py`` documents.
"""

from __future__ import annotations

from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

import pytest

from forge.dependabot.audit import AuditError
from forge.dependabot.bump import BumpError
from forge.dependabot.config import settings
from forge.dependabot.ecosystems import golang
from forge.dependabot.ecosystems.golang import GoEcosystem, _parse_gomod_diff
from forge.dependabot.models import AuditFinding, BumpCandidate
from forge.dependabot.policy import auto_eligible
from forge.dependabot.scan import ScanError

FIXTURES = Path(__file__).parent / "fixtures"
GO_LIST_FIXTURE = (FIXTURES / "go_list_outdated.json").read_text()
GOVULNCHECK_FIXTURE = (FIXTURES / "govulncheck_module.json").read_text()


def _proc(returncode: int, stdout: str = "", stderr: str = "") -> CompletedProcess:
    return CompletedProcess(args=["go"], returncode=returncode, stdout=stdout, stderr=stderr)


# ---------------------------------------------------------------------------
# scan_outdated
# ---------------------------------------------------------------------------


class TestScanOutdated:
    def test_fixture_yields_direct_updates_only(self):
        with patch("forge.dependabot.ecosystems.golang.subprocess.run") as m:
            m.return_value = _proc(0, GO_LIST_FIXTURE)
            candidates = GoEcosystem().scan_outdated(Path("/fake/repo"))

        names = [c.name for c in candidates]
        # Main module, the no-update module, and the Indirect module are all skipped.
        assert names == ["charm.land/bubbles/v2", "charm.land/bubbletea/v2"]
        assert "cloud.google.com/go/compute/metadata" not in names

    def test_versions_stripped_and_delta_classified(self):
        with patch("forge.dependabot.ecosystems.golang.subprocess.run") as m:
            m.return_value = _proc(0, GO_LIST_FIXTURE)
            candidates = GoEcosystem().scan_outdated(Path("/fake/repo"))

        bubbles = next(c for c in candidates if c.name == "charm.land/bubbles/v2")
        assert bubbles.current == "2.1.0"
        assert bubbles.latest == "2.1.1"
        assert bubbles.delta == "patch"

    def test_sorted_patch_first_and_capped(self, monkeypatch):
        stream = "\n".join(
            [
                '{"Path": "example.com/major", "Version": "v1.0.0",'
                ' "Update": {"Version": "v2.0.0"}}',
                '{"Path": "example.com/patch", "Version": "v1.0.0",'
                ' "Update": {"Version": "v1.0.1"}}',
                '{"Path": "example.com/minor", "Version": "v1.0.0",'
                ' "Update": {"Version": "v1.1.0"}}',
            ]
        )
        with patch("forge.dependabot.ecosystems.golang.subprocess.run") as m:
            m.return_value = _proc(0, stream)
            candidates = GoEcosystem().scan_outdated(Path("/fake/repo"))
        assert [c.delta for c in candidates] == ["patch", "minor", "major"]

        monkeypatch.setattr(settings, "max_candidates", 2)
        with patch("forge.dependabot.ecosystems.golang.subprocess.run") as m:
            m.return_value = _proc(0, stream)
            capped = GoEcosystem().scan_outdated(Path("/fake/repo"))
        assert len(capped) == 2

    def test_nonzero_exit_raises_scan_error(self):
        with patch("forge.dependabot.ecosystems.golang.subprocess.run") as m:
            m.return_value = _proc(1, "", "go: no modules")
            with pytest.raises(ScanError, match="no modules"):
                GoEcosystem().scan_outdated(Path("/fake/repo"))


# ---------------------------------------------------------------------------
# apply_bump
# ---------------------------------------------------------------------------


def _candidate(name="charm.land/bubbles/v2", current="2.1.0", latest="2.1.1", delta="patch"):
    return BumpCandidate(name=name, current=current, latest=latest, delta=delta)


class TestApplyBump:
    def test_runs_go_get_then_tidy_and_returns_changed(self, monkeypatch):
        commands: list[list[str]] = []

        def fake_run(cmd, **kw):
            commands.append(cmd)
            return _proc(0)

        monkeypatch.setattr(golang.subprocess, "run", fake_run)
        monkeypatch.setattr(golang, "get_changed_files", lambda repo: ["go.mod", "go.sum"])

        changed = GoEcosystem().apply_bump(Path("/fake/repo"), _candidate())
        assert changed == ["go.mod", "go.sum"]
        assert commands[0] == ["go", "get", "charm.land/bubbles/v2@v2.1.1"]
        assert commands[1] == ["go", "mod", "tidy"]

    def test_go_get_failure_reverts_and_raises(self, monkeypatch):
        reverted = []

        def fake_run(cmd, **kw):
            return _proc(1, "", "invalid version")

        monkeypatch.setattr(golang.subprocess, "run", fake_run)
        monkeypatch.setattr(golang, "revert_changes", lambda repo: reverted.append(repo))

        with pytest.raises(BumpError, match="go get .*invalid version"):
            GoEcosystem().apply_bump(Path("/fake/repo"), _candidate())
        assert reverted  # working copy cleaned before raising

    def test_tidy_failure_also_reverts(self, monkeypatch):
        reverted = []

        def fake_run(cmd, **kw):
            return _proc(0) if cmd[:2] == ["go", "get"] else _proc(1, "", "tidy exploded")

        monkeypatch.setattr(golang.subprocess, "run", fake_run)
        monkeypatch.setattr(golang, "revert_changes", lambda repo: reverted.append(repo))

        with pytest.raises(BumpError, match="go mod tidy .*tidy exploded"):
            GoEcosystem().apply_bump(Path("/fake/repo"), _candidate())
        assert reverted


# ---------------------------------------------------------------------------
# lockfile_delta — go.mod diff parsing
# ---------------------------------------------------------------------------


GOMOD_DIFF = """\
diff --git a/go.mod b/go.mod
--- a/go.mod
+++ b/go.mod
@@ -10,7 +10,7 @@
 module github.com/charmbracelet/soft-serve
-go 1.25.9
+go 1.25.10
 require (
-\tcharm.land/bubbles/v2 v2.1.0
+\tcharm.land/bubbles/v2 v2.1.1
-\tgolang.org/x/sys v0.43.0 // indirect
+\tgolang.org/x/sys v0.44.0 // indirect
-\tgithub.com/git-bug/git-bug v0.10.1 // indirect
+\tgithub.com/git-bug/git-bug v0.10.1
 )
-require golang.org/x/crypto v0.51.0
+require golang.org/x/crypto v0.52.0
diff --git a/other/file.txt b/other/file.txt
--- a/other/file.txt
+++ b/other/file.txt
@@ -1 +1 @@
-example.com/red-herring v1.0.0
+example.com/red-herring v2.0.0
"""


class TestGomodDelta:
    def test_pairs_extracted_from_gomod_hunks_only(self):
        deltas = _parse_gomod_diff(GOMOD_DIFF)
        assert "charm.land/bubbles/v2 2.1.0->2.1.1" in deltas
        assert "golang.org/x/sys 0.43.0->0.44.0" in deltas  # indirect churn is honest output
        assert "golang.org/x/crypto 0.51.0->0.52.0" in deltas  # single-line require form
        # The go directive (no v prefix) and other files' hunks never leak in, and a
        # same-version pair (annotation churn from go mod tidy) is not a delta.
        assert not any("red-herring" in d for d in deltas)
        assert not any("git-bug" in d for d in deltas)
        assert not any(d.startswith("go ") for d in deltas)
        assert len(deltas) == 3

    def test_unparseable_diff_is_empty_not_an_error(self):
        assert _parse_gomod_diff("complete nonsense") == []


# ---------------------------------------------------------------------------
# run_audit — govulncheck primary, osv-scanner fallback, fail-closed floor
# ---------------------------------------------------------------------------


OSV_SCANNER_JSON = """\
{"results": [{"source": {"path": "go.sum", "type": "lockfile"}, "packages": [
  {"package": {"name": "github.com/yuin/goldmark", "version": "1.7.8", "ecosystem": "Go"},
   "vulnerabilities": [
     {"id": "GO-2026-5320", "summary": "XSS in goldmark",
      "aliases": ["CVE-2026-5160"],
      "affected": [{"package": {"name": "github.com/yuin/goldmark", "ecosystem": "Go"},
                    "ranges": [{"type": "SEMVER",
                                "events": [{"introduced": "0"}, {"fixed": "1.7.17"}]}]}]}
   ]}
]}]}
"""


class TestRunAudit:
    def test_govulncheck_findings_join_catalog(self, tmp_path, monkeypatch):
        seen = {}

        def fake_run(cmd, **kw):
            seen["cmd"], seen["cwd"] = cmd, kw.get("cwd")
            return _proc(0, GOVULNCHECK_FIXTURE)

        monkeypatch.setattr(golang.subprocess, "run", fake_run)
        findings = GoEcosystem().run_audit(tmp_path)

        # Package scan over ./... from the repo root — module scan's results silently
        # depend on the launch directory (18 vs 27 findings on the live capture).
        assert seen["cmd"] == ["govulncheck", "-scan", "package", "-json", "./..."]
        assert seen["cwd"] == str(tmp_path)

        by_id = {f.vuln_id: f for f in findings}
        assert len(findings) == 3  # findings, NOT the 3-entry osv catalog misread as results
        goldmark = by_id["GO-2026-5320"]
        assert goldmark.package == "github.com/yuin/goldmark"
        assert goldmark.fix_versions == ["1.7.17"]  # v stripped
        assert "CVE-2026-5160" in goldmark.aliases
        assert "Cross-site Scripting" in goldmark.description
        # A finding with no fixed_version (unfixed vuln) still surfaces, with no fix versions.
        assert by_id["GO-2022-0470"].fix_versions == []

    def test_govulncheck_failure_is_an_audit_error(self, tmp_path, monkeypatch):
        # e.g. a repo with no .go files anywhere fails the package load — fail closed.
        monkeypatch.setattr(
            golang.subprocess, "run", lambda *a, **kw: _proc(1, "", "no Go files matched")
        )
        with pytest.raises(AuditError, match="no Go files matched"):
            GoEcosystem().run_audit(tmp_path)

    def test_missing_govulncheck_falls_back_to_osv_scanner(self, tmp_path, monkeypatch):
        repo = tmp_path

        def fake_run(cmd, **kw):
            if cmd[0] == "govulncheck":
                raise FileNotFoundError("govulncheck")
            assert cmd == ["osv-scanner", "--format", "json", "--lockfile", "go.sum"]
            return _proc(1, OSV_SCANNER_JSON)  # exit 1 = vulns found = successful scan

        monkeypatch.setattr(golang.subprocess, "run", fake_run)
        findings = GoEcosystem().run_audit(repo)
        assert len(findings) == 1
        assert findings[0].vuln_id == "GO-2026-5320"
        assert findings[0].fix_versions == ["1.7.17"]

    def test_neither_scanner_installed_fails_closed(self, tmp_path, monkeypatch):
        def fake_run(cmd, **kw):
            raise FileNotFoundError(cmd[0])

        monkeypatch.setattr(golang.subprocess, "run", fake_run)
        with pytest.raises(AuditError, match="govulncheck nor osv-scanner"):
            GoEcosystem().run_audit(tmp_path)


# ---------------------------------------------------------------------------
# collect_evidence + policy — the demote-not-block contract
# ---------------------------------------------------------------------------


class TestEvidence:
    def _findings(self) -> list[AuditFinding]:
        return [
            AuditFinding(
                package="golang.org/x/net",
                vuln_id="GO-FIXED-BY-BUMP",
                fix_versions=["0.53.0"],
            ),
            AuditFinding(
                package="golang.org/x/net",
                vuln_id="GO-STILL-PRESENT",
                fix_versions=["0.55.0"],
            ),
        ]

    def test_findings_split_current_vs_target(self, tmp_path):
        candidate = _candidate(name="golang.org/x/net", current="0.52.0", latest="0.53.0")
        bundle = GoEcosystem().collect_evidence(
            candidate, self._findings(), ["golang.org/x/net 0.52.0->0.53.0"], repo_root=tmp_path
        )
        assert [f.vuln_id for f in bundle.findings_current] == ["GO-FIXED-BY-BUMP"]
        assert [f.vuln_id for f in bundle.findings_target] == ["GO-STILL-PRESENT"]
        assert bundle.lockfile_changes == ["golang.org/x/net 0.52.0->0.53.0"]

    def test_evidence_is_incomplete_by_construction_with_reason(self, tmp_path):
        bundle = GoEcosystem().collect_evidence(_candidate(), [], [], repo_root=tmp_path)
        assert bundle.complete is False
        assert bundle.incomplete_reason is not None
        assert "PyPI-only" in bundle.incomplete_reason

    def test_policy_demotes_go_bump_with_the_adapter_reason(self, tmp_path):
        """The acceptance contract: a Go candidate with no provenance signals lands on the
        advisory track and the reason names the absent signals — never a crash, never a pass."""
        bundle = GoEcosystem().collect_evidence(_candidate(), [], [], repo_root=tmp_path)
        eligible, reason = auto_eligible(bundle)
        assert eligible is False
        assert "advisory track" in reason and "PyPI-only" in reason
