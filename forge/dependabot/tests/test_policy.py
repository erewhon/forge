"""Truth table for the auto-merge risk policy + the rendered evidence contract."""

from __future__ import annotations

from forge.dependabot.models import AuditFinding, BumpCandidate, EvidenceBundle
from forge.dependabot.policy import auto_eligible
from forge.dependabot.prompts import SUPPLY_CHAIN_SIGNOFF, render_evidence


def _evidence(**over) -> EvidenceBundle:
    """A fully-clean, auto-eligible baseline; tests override one condition at a time."""
    candidate = BumpCandidate(
        name=over.pop("name", "idna"),
        current="3.11",
        latest="3.15",
        delta=over.pop("delta", "minor"),
    )
    base = dict(
        candidate=candidate,
        target_yanked=False,
        package_age_days=30,
        changelog_url="https://example.com/CHANGES",
        lockfile_changes=["idna 3.11->3.15"],
        complete=True,
    )
    base.update(over)
    return EvidenceBundle(**base)


class TestAutoEligible:
    def test_clean_patch_or_minor_with_complete_evidence_is_eligible(self):
        for delta in ("patch", "minor"):
            eligible, reason = auto_eligible(_evidence(delta=delta))
            assert eligible and reason == ""

    def test_major_is_never_eligible_even_clean(self):
        eligible, reason = auto_eligible(_evidence(delta="major"))
        assert not eligible
        assert "major" in reason

    def test_unknown_delta_is_never_eligible(self):
        eligible, reason = auto_eligible(_evidence(delta="unknown"))
        assert not eligible
        assert "unknown" in reason

    def test_incomplete_evidence_is_ineligible(self):
        eligible, reason = auto_eligible(_evidence(complete=False))
        assert not eligible
        assert "incomplete" in reason

    def test_undeterminable_yanked_is_ineligible(self):
        eligible, reason = auto_eligible(_evidence(target_yanked=None))
        assert not eligible
        assert "yanked" in reason

    def test_yanked_target_is_ineligible(self):
        eligible, reason = auto_eligible(_evidence(target_yanked=True))
        assert not eligible
        assert "yanked" in reason

    def test_residual_findings_are_ineligible(self):
        finding = AuditFinding(package="idna", vuln_id="PYSEC-2026-999")
        eligible, reason = auto_eligible(_evidence(findings_target=[finding]))
        assert not eligible
        assert "PYSEC-2026-999" in reason

    def test_typosquat_suspect_is_ineligible(self):
        eligible, reason = auto_eligible(_evidence(name="reqeusts", typosquat_suspect="requests"))
        assert not eligible
        assert "typosquat" in reason

    def test_fixed_vulns_do_not_block(self):
        # A vuln on the CURRENT version fixed by the bump is the best reason to bump.
        finding = AuditFinding(package="idna", vuln_id="PYSEC-2026-215", fix_versions=["3.15"])
        eligible, _ = auto_eligible(_evidence(findings_current=[finding]))
        assert eligible

    def test_maintainer_change_is_ineligible(self):
        eligible, reason = auto_eligible(_evidence(maintainer_changed=True))
        assert not eligible
        assert "maintainer" in reason

    def test_new_install_scripts_are_ineligible(self):
        eligible, reason = auto_eligible(_evidence(new_install_scripts=True))
        assert not eligible
        assert "install/build scripts" in reason

    def test_undeterminable_v2_signals_do_not_block(self):
        # None = best-effort signal unavailable; by contract it must NOT become a block.
        eligible, _ = auto_eligible(_evidence(maintainer_changed=None, new_install_scripts=None))
        assert eligible


class TestPrompt:
    def test_prompt_carries_the_verdict_contract(self):
        assert '{"approve": true|false, "blockers": ["..."], "notes": "..."}' in (
            SUPPLY_CHAIN_SIGNOFF
        )

    def test_prompt_refuses_incomplete_evidence(self):
        assert "Incomplete evidence is an automatic block" in SUPPLY_CHAIN_SIGNOFF

    def test_prompt_admits_it_cannot_see_source(self):
        assert "METADATA" in SUPPLY_CHAIN_SIGNOFF
        assert "cannot see" in SUPPLY_CHAIN_SIGNOFF


class TestRenderEvidence:
    def test_every_signal_present(self):
        finding = AuditFinding(package="idna", vuln_id="PYSEC-2026-215", fix_versions=["3.15"])
        out = render_evidence(_evidence(findings_current=[finding]))
        assert "idna 3.11 -> 3.15 (delta: minor)" in out
        assert "fixed by this bump: PYSEC-2026-215" in out
        assert "REMAINING at target: none" in out
        assert "target yanked on PyPI: no" in out
        assert "30 day(s)" in out
        assert "https://example.com/CHANGES" in out
        assert "typosquat signal: none" in out
        assert "maintainer identity changed across the bump: unknown" in out
        assert "new install/build scripts at target: unknown" in out
        assert "idna 3.11->3.15" in out
        assert "evidence complete: yes" in out

    def test_incomplete_is_loud(self):
        out = render_evidence(_evidence(complete=False, target_yanked=None))
        assert "evidence complete: NO" in out
        assert "target yanked on PyPI: unknown" in out
