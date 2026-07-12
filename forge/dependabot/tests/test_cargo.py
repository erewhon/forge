"""CargoEcosystem adapter tests — fixtures live-captured 2026-07-12 (cargo 1.95.0 /
cargo-audit via brew / index.crates.io).

``crates_index_anyhow.ndjson`` is pure live capture (trimmed); index edge cases the live
crate lacks (a yanked HIGHEST version, a prerelease) are appended inline in the tests and
marked synthetic. ``cargo_audit.json`` is trimmed from a real protectinator audit.
"""

from __future__ import annotations

import json
from pathlib import Path
from subprocess import CompletedProcess

import pytest

from forge.dependabot.audit import AuditError
from forge.dependabot.bump import BumpError
from forge.dependabot.ecosystems import cargo as cargo_mod
from forge.dependabot.ecosystems.cargo import (
    CargoEcosystem,
    _direct_deps,
    _index_latest,
    _index_prefix,
    _locked_versions,
)
from forge.dependabot.models import AuditFinding, BumpCandidate
from forge.dependabot.policy import auto_eligible
from forge.dependabot.scan import ScanError  # noqa: F401 — parity import with sibling suites

FIXTURES = Path(__file__).parent / "fixtures"
INDEX_FIXTURE = (FIXTURES / "crates_index_anyhow.ndjson").read_text()
AUDIT_FIXTURE = (FIXTURES / "cargo_audit.json").read_text()

CARGO_TOML = """\
[package]
name = "app"
version = "0.1.0"

[dependencies]
anyhow = "1.0"
renamed-alias = { package = "real-crate", version = "0.5" }
local-thing = { path = "../local-thing" }

[dev-dependencies]
tempfile = "3"

[workspace]
members = ["crates/a"]

[workspace.dependencies]
serde = { version = "1.0", features = ["derive"] }
"""

CARGO_LOCK = """\
version = 4

[[package]]
name = "anyhow"
version = "1.0.102"
source = "registry+https://github.com/rust-lang/crates.io-index"

[[package]]
name = "serde"
version = "1.0.219"
source = "registry+https://github.com/rust-lang/crates.io-index"

[[package]]
name = "real-crate"
version = "0.5.1"
source = "registry+https://github.com/rust-lang/crates.io-index"

[[package]]
name = "tempfile"
version = "3.15.0"
source = "registry+https://github.com/rust-lang/crates.io-index"

[[package]]
name = "ambiguous-crate"
version = "1.0.0"
source = "registry+https://github.com/rust-lang/crates.io-index"

[[package]]
name = "ambiguous-crate"
version = "2.0.0"
source = "registry+https://github.com/rust-lang/crates.io-index"

[[package]]
name = "local-thing"
version = "0.0.1"
"""


def _repo(tmp_path: Path) -> Path:
    (tmp_path / "Cargo.toml").write_text(CARGO_TOML)
    (tmp_path / "Cargo.lock").write_text(CARGO_LOCK)
    return tmp_path


# ---------------------------------------------------------------------------
# Manifest / lockfile parsing
# ---------------------------------------------------------------------------


class TestDirectDeps:
    def test_sections_renames_and_workspace_table(self, tmp_path):
        deps = _direct_deps(_repo(tmp_path))
        assert deps == {"anyhow", "real-crate", "tempfile", "serde"}

    def test_path_only_deps_are_skipped(self, tmp_path):
        assert "local-thing" not in _direct_deps(_repo(tmp_path))

    def test_missing_manifest_is_empty(self, tmp_path):
        assert _direct_deps(tmp_path) == set()


class TestLockedVersions:
    def test_registry_entries_only(self, tmp_path):
        locked = _locked_versions(_repo(tmp_path))
        assert locked["anyhow"] == "1.0.102"
        assert "local-thing" not in locked  # no registry source

    def test_multi_version_crate_is_dropped(self, tmp_path):
        locked = _locked_versions(_repo(tmp_path))
        assert "ambiguous-crate" not in locked  # an ambiguous current is not a candidate


# ---------------------------------------------------------------------------
# Sparse index
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, text: str, status: int = 200):
        self.text = text
        self._status = status

    def raise_for_status(self):
        if self._status >= 400:
            raise RuntimeError(f"HTTP {self._status}")


class TestIndex:
    def test_prefix_rules(self):
        assert _index_prefix("a") == "1"
        assert _index_prefix("ab") == "2"
        assert _index_prefix("abc") == "3/a"
        assert _index_prefix("anyhow") == "an/yh"

    def test_latest_is_max_non_yanked(self, monkeypatch):
        monkeypatch.setattr(cargo_mod.httpx, "get", lambda url, **kw: _FakeResponse(INDEX_FIXTURE))
        assert _index_latest("anyhow", timeout=5) == "1.0.103"

    def test_yanked_highest_and_prereleases_are_skipped(self, monkeypatch):
        # Synthetic edge lines the live capture lacks: a yanked HIGHEST version and a
        # prerelease published after it — neither may become "latest".
        text = INDEX_FIXTURE + "\n".join(
            [
                json.dumps({"name": "anyhow", "vers": "1.0.104", "yanked": True}),
                json.dumps({"name": "anyhow", "vers": "2.0.0-beta.1", "yanked": False}),
            ]
        )
        monkeypatch.setattr(cargo_mod.httpx, "get", lambda url, **kw: _FakeResponse(text))
        assert _index_latest("anyhow", timeout=5) == "1.0.103"

    def test_fetch_failure_is_none_never_fabricated(self, monkeypatch):
        def boom(url, **kw):
            raise OSError("network down")

        monkeypatch.setattr(cargo_mod.httpx, "get", boom)
        assert _index_latest("anyhow", timeout=5) is None
        monkeypatch.setattr(cargo_mod.httpx, "get", lambda url, **kw: _FakeResponse("", status=404))
        assert _index_latest("ghost-crate", timeout=5) is None


# ---------------------------------------------------------------------------
# scan_outdated — integration over real manifest+lock with a faked index
# ---------------------------------------------------------------------------


class TestScanOutdated:
    def test_candidates_from_manifest_lock_and_index(self, tmp_path, monkeypatch):
        repo = _repo(tmp_path)
        latest = {
            "anyhow": "1.0.103",  # patch ahead
            "serde": "1.0.219",  # current == latest -> not a candidate
            "real-crate": "1.0.0",  # major ahead
            "tempfile": None,  # index fetch failed -> dropped, never fabricated
        }
        monkeypatch.setattr(cargo_mod, "_index_latest", lambda crate, timeout: latest.get(crate))
        candidates = CargoEcosystem().scan_outdated(repo)
        by_name = {c.name: c for c in candidates}
        assert set(by_name) == {"anyhow", "real-crate"}
        assert by_name["anyhow"].delta == "patch"
        assert by_name["real-crate"].delta == "major"
        assert [c.delta for c in candidates] == ["patch", "major"]  # patch-first sort


# ---------------------------------------------------------------------------
# apply_bump
# ---------------------------------------------------------------------------


def _candidate(name="anyhow", current="1.0.102", latest="1.0.103", delta="patch"):
    return BumpCandidate(name=name, current=current, latest=latest, delta=delta)


class TestApplyBump:
    def test_precise_pkgid_update(self, monkeypatch, tmp_path):
        commands = []

        def fake_run(cmd, **kw):
            commands.append(cmd)
            return CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr(cargo_mod.subprocess, "run", fake_run)
        monkeypatch.setattr(cargo_mod, "get_changed_files", lambda repo: ["Cargo.lock"])
        changed = CargoEcosystem().apply_bump(tmp_path, _candidate())
        assert changed == ["Cargo.lock"]
        assert commands[0] == ["cargo", "update", "-p", "anyhow@1.0.102"]

    def test_failure_reverts_and_raises(self, monkeypatch, tmp_path):
        reverted = []
        monkeypatch.setattr(
            cargo_mod.subprocess,
            "run",
            lambda cmd, **kw: CompletedProcess(cmd, 101, stdout="", stderr="bad pkgid"),
        )
        monkeypatch.setattr(cargo_mod, "revert_changes", lambda repo: reverted.append(repo))
        with pytest.raises(BumpError, match="bad pkgid"):
            CargoEcosystem().apply_bump(tmp_path, _candidate())
        assert reverted


# ---------------------------------------------------------------------------
# lockfile_delta — shared TOML-block parser with a Cargo.lock header
# ---------------------------------------------------------------------------


# Realistic unified-diff shape: `name` lines are CONTEXT (unchanged), only `version`
# lines carry +/- — exactly how cargo rewrites the lock (and how uv.lock diffs read).
CARGO_LOCK_DIFF = """\
diff --git a/Cargo.lock b/Cargo.lock
--- a/Cargo.lock
+++ b/Cargo.lock
@@ -10,7 +10,7 @@
 [[package]]
 name = "anyhow"
-version = "1.0.102"
+version = "1.0.103"
 [[package]]
 name = "same-crate"
-version = "2.0.0"
+version = "2.0.0"
diff --git a/uv.lock b/uv.lock
--- a/uv.lock
+++ b/uv.lock
@@ -1 +1 @@
 name = "red-herring"
-version = "1.0.0"
+version = "2.0.0"
"""


class TestLockDelta:
    def test_pairs_from_cargo_lock_hunks_only(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cargo_mod, "working_diff", lambda repo: CARGO_LOCK_DIFF)
        deltas = CargoEcosystem().lockfile_delta(tmp_path)
        assert "anyhow 1.0.102->1.0.103" in deltas
        # Same-version rewrites (checksum churn) and other lockfiles never leak in.
        assert not any("same-crate" in d for d in deltas)
        assert not any("red-herring" in d for d in deltas)
        assert len(deltas) == 1


# ---------------------------------------------------------------------------
# run_audit — cargo-audit primary, osv-scanner fallback, fail-closed floor
# ---------------------------------------------------------------------------


class TestRunAudit:
    def test_rustsec_findings_with_range_lower_bounds(self, monkeypatch, tmp_path):
        seen = {}

        def fake_run(cmd, **kw):
            seen["cmd"] = cmd
            return CompletedProcess(cmd, 1, stdout=AUDIT_FIXTURE, stderr="")

        monkeypatch.setattr(cargo_mod.subprocess, "run", fake_run)
        findings = CargoEcosystem().run_audit(tmp_path)
        assert seen["cmd"] == ["cargo-audit", "audit", "--json"]
        by_id = {f.vuln_id: f for f in findings}
        assert set(by_id) == {"RUSTSEC-2026-0204", "RUSTSEC-2026-0187"}
        assert by_id["RUSTSEC-2026-0204"].package == "crossbeam-epoch"
        assert by_id["RUSTSEC-2026-0204"].fix_versions == ["0.9.20"]

    def test_unexpected_exit_raises(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            cargo_mod.subprocess,
            "run",
            lambda cmd, **kw: CompletedProcess(cmd, 2, stdout="", stderr="db fetch failed"),
        )
        with pytest.raises(AuditError, match="db fetch failed"):
            CargoEcosystem().run_audit(tmp_path)

    def test_missing_cargo_audit_falls_back_to_osv_scanner(self, monkeypatch, tmp_path):
        def fake_run(cmd, **kw):
            if cmd[0] == "cargo-audit":
                raise FileNotFoundError("cargo-audit")
            assert cmd == ["osv-scanner", "--format", "json", "--lockfile", "Cargo.lock"]
            return CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr(cargo_mod.subprocess, "run", fake_run)
        assert CargoEcosystem().run_audit(tmp_path) == []

    def test_neither_scanner_fails_closed(self, monkeypatch, tmp_path):
        def fake_run(cmd, **kw):
            raise FileNotFoundError(cmd[0])

        monkeypatch.setattr(cargo_mod.subprocess, "run", fake_run)
        with pytest.raises(AuditError, match="cargo-audit nor osv-scanner"):
            CargoEcosystem().run_audit(tmp_path)


# ---------------------------------------------------------------------------
# collect_evidence + policy — the demote-not-block contract
# ---------------------------------------------------------------------------


class TestEvidence:
    def test_findings_split_and_demotion_reason(self, tmp_path):
        findings = [
            AuditFinding(
                package="crossbeam-epoch", vuln_id="RUSTSEC-FIXED", fix_versions=["0.9.20"]
            ),
            AuditFinding(
                package="crossbeam-epoch", vuln_id="RUSTSEC-REMAINS", fix_versions=["0.9.30"]
            ),
        ]
        candidate = _candidate(name="crossbeam-epoch", current="0.9.18", latest="0.9.20")
        bundle = CargoEcosystem().collect_evidence(
            candidate, findings, ["crossbeam-epoch 0.9.18->0.9.20"], repo_root=tmp_path
        )
        assert [f.vuln_id for f in bundle.findings_current] == ["RUSTSEC-FIXED"]
        assert [f.vuln_id for f in bundle.findings_target] == ["RUSTSEC-REMAINS"]
        eligible, reason = auto_eligible(bundle)
        assert eligible is False
        assert "advisory track" in reason and "crates.io" in reason
