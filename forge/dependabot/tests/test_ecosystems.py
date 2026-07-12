"""Ecosystem detection, resolution, and the uv adapter's pure-delegation contract."""

from __future__ import annotations

import pytest

from forge.dependabot.ecosystems import (
    EcosystemError,
    detect_ecosystem,
    present_ecosystems,
    resolve_ecosystem,
)
from forge.dependabot.ecosystems import uv as uv_mod
from forge.dependabot.ecosystems.golang import GoEcosystem
from forge.dependabot.ecosystems.pnpm import PnpmEcosystem
from forge.dependabot.ecosystems.uv import UvEcosystem
from forge.dependabot.models import BumpCandidate

# ---------------------------------------------------------------------------
# detect_ecosystem
# ---------------------------------------------------------------------------


class TestDetection:
    def test_uv_lock_selects_uv(self, tmp_path):
        (tmp_path / "uv.lock").touch()
        assert detect_ecosystem(tmp_path) == "uv"

    def test_go_mod_selects_go(self, tmp_path):
        (tmp_path / "go.mod").touch()
        assert detect_ecosystem(tmp_path) == "go"

    def test_both_manifests_is_an_error_naming_both(self, tmp_path):
        (tmp_path / "uv.lock").touch()
        (tmp_path / "go.mod").touch()
        with pytest.raises(EcosystemError, match="uv, go"):
            detect_ecosystem(tmp_path)

    def test_ambiguity_error_names_the_escape_hatch(self, tmp_path):
        (tmp_path / "uv.lock").touch()
        (tmp_path / "go.mod").touch()
        with pytest.raises(EcosystemError, match="--ecosystem"):
            detect_ecosystem(tmp_path)

    def test_no_manifest_is_an_error_listing_markers(self, tmp_path):
        with pytest.raises(EcosystemError, match="uv.lock"):
            detect_ecosystem(tmp_path)

    def test_override_resolves_ambiguity(self, tmp_path):
        (tmp_path / "uv.lock").touch()
        (tmp_path / "go.mod").touch()
        assert detect_ecosystem(tmp_path, override="go") == "go"
        assert detect_ecosystem(tmp_path, override="uv") == "uv"

    def test_override_without_its_manifest_is_an_error(self, tmp_path):
        (tmp_path / "uv.lock").touch()
        with pytest.raises(EcosystemError, match="go.mod"):
            detect_ecosystem(tmp_path, override="go")

    def test_unknown_override_names_supported_ecosystems(self, tmp_path):
        with pytest.raises(EcosystemError, match="cargo, go, pnpm, uv"):
            detect_ecosystem(tmp_path, override="maven")

    def test_pnpm_lock_selects_pnpm(self, tmp_path):
        (tmp_path / "pnpm-lock.yaml").touch()
        assert detect_ecosystem(tmp_path) == "pnpm"

    def test_cargo_lock_selects_cargo(self, tmp_path):
        (tmp_path / "Cargo.lock").touch()
        assert detect_ecosystem(tmp_path) == "cargo"

    def test_cargo_toml_alone_is_not_enough(self, tmp_path):
        # A library that doesn't commit its lockfile has nothing meaningful to bump.
        (tmp_path / "Cargo.toml").touch()
        with pytest.raises(EcosystemError, match="no supported"):
            detect_ecosystem(tmp_path)


class TestPresentEcosystems:
    def test_lists_all_markers_in_order(self, tmp_path):
        (tmp_path / "pnpm-lock.yaml").touch()
        (tmp_path / "uv.lock").touch()
        assert present_ecosystems(tmp_path) == ["uv", "pnpm"]

    def test_empty_when_no_markers(self, tmp_path):
        assert present_ecosystems(tmp_path) == []


# ---------------------------------------------------------------------------
# resolve_ecosystem
# ---------------------------------------------------------------------------


class TestResolve:
    def test_uv_repo_resolves_to_uv_adapter(self, tmp_path):
        (tmp_path / "uv.lock").touch()
        eco = resolve_ecosystem(tmp_path)
        assert isinstance(eco, UvEcosystem) and eco.name == "uv"

    def test_go_repo_resolves_to_go_adapter(self, tmp_path):
        (tmp_path / "go.mod").touch()
        eco = resolve_ecosystem(tmp_path)
        assert isinstance(eco, GoEcosystem) and eco.name == "go"

    def test_pnpm_repo_resolves_to_pnpm_adapter(self, tmp_path):
        (tmp_path / "pnpm-lock.yaml").touch()
        eco = resolve_ecosystem(tmp_path)
        assert isinstance(eco, PnpmEcosystem) and eco.name == "pnpm"

    def test_cargo_repo_resolves_to_cargo_adapter(self, tmp_path):
        from forge.dependabot.ecosystems.cargo import CargoEcosystem

        (tmp_path / "Cargo.lock").touch()
        eco = resolve_ecosystem(tmp_path)
        assert isinstance(eco, CargoEcosystem) and eco.name == "cargo"


# ---------------------------------------------------------------------------
# UvEcosystem — pure delegation (the port must not change uv behavior)
# ---------------------------------------------------------------------------


class TestUvDelegation:
    def test_every_method_delegates_to_the_original_module(self, monkeypatch, tmp_path):
        calls: list[str] = []
        candidate = BumpCandidate(name="idna", current="3.11", latest="3.15", delta="minor")

        monkeypatch.setattr(
            uv_mod, "scan_outdated", lambda repo: calls.append("scan") or [candidate]
        )
        monkeypatch.setattr(
            uv_mod, "apply_bump", lambda repo, c: calls.append("apply") or ["uv.lock"]
        )
        monkeypatch.setattr(uv_mod, "lockfile_delta", lambda repo: calls.append("delta") or [])
        monkeypatch.setattr(uv_mod, "run_audit", lambda repo: calls.append("audit") or [])
        monkeypatch.setattr(
            uv_mod,
            "collect_evidence",
            lambda c, f, d, repo_root: calls.append(f"evidence:{repo_root}") or "BUNDLE",
        )

        eco = UvEcosystem()
        assert eco.scan_outdated(tmp_path) == [candidate]
        assert eco.apply_bump(tmp_path, candidate) == ["uv.lock"]
        assert eco.lockfile_delta(tmp_path) == []
        assert eco.run_audit(tmp_path) == []
        assert eco.collect_evidence(candidate, [], [], repo_root=tmp_path) == "BUNDLE"
        assert calls == ["scan", "apply", "delta", "audit", f"evidence:{tmp_path}"]
