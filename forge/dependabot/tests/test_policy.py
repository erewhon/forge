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
        target_attested=True,
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

    def test_attested_true_passes_with_require_attestation(self):
        eligible, _ = auto_eligible(_evidence(target_attested=True))
        assert eligible

    def test_attested_false_blocks_when_require_attestation(self):
        eligible, reason = auto_eligible(_evidence(target_attested=False))
        assert not eligible
        assert "no PEP 740 attestations" in reason
        assert "not attested by posture" in reason

    def test_attested_none_blocks_when_require_attestation(self):
        eligible, reason = auto_eligible(_evidence(target_attested=None))
        assert not eligible
        assert "attestation status undeterminable" in reason
        assert "treated as unattested by posture" in reason

    def test_require_attestation_false_restores_passthrough_for_false(self):
        eligible, _ = auto_eligible(_evidence(target_attested=False), require_attestation=False)
        assert eligible

    def test_require_attestation_false_restores_passthrough_for_none(self):
        eligible, _ = auto_eligible(_evidence(target_attested=None), require_attestation=False)
        assert eligible


class TestScorecardFloor:
    def test_score_below_floor_blocks(self):
        eligible, reason = auto_eligible(_evidence(scorecard_score=4.0, scorecard_repo="pypa/pip"))
        assert not eligible
        assert "4.0" in reason
        assert "pip" in reason
        assert "floor 5.0" in reason

    def test_score_at_floor_passes(self):
        eligible, _ = auto_eligible(_evidence(scorecard_score=5.0, scorecard_repo="pypa/pip"))
        assert eligible

    def test_score_above_floor_passes(self):
        eligible, _ = auto_eligible(_evidence(scorecard_score=9.3, scorecard_repo="pypa/pip"))
        assert eligible

    def test_none_score_passes(self):
        eligible, _ = auto_eligible(_evidence(scorecard_score=None, scorecard_repo=None))
        assert eligible

    def test_earlier_failing_condition_wins_reason(self):
        # major delta is checked before scorecard — its reason must win
        eligible, reason = auto_eligible(
            _evidence(delta="major", scorecard_score=1.0, scorecard_repo="pypa/pip")
        )
        assert not eligible
        assert "major" in reason
        assert "OpenSSF" not in reason

    def test_render_evidence_includes_scorecard_line(self):
        out = render_evidence(_evidence(scorecard_score=7.2, scorecard_repo="pypa/pip"))
        assert "OpenSSF Scorecard: 7.2 (pypa/pip)" in out

    def test_render_evidence_scorecard_unavailable_when_none(self):
        out = render_evidence(_evidence(scorecard_score=None, scorecard_repo=None))
        assert "OpenSSF Scorecard: unavailable (no source repo mapped)" in out


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
        ev = _evidence(
            findings_current=[finding],
            scorecard_score=7.2,
            scorecard_repo="pypa/idna",
        )
        out = render_evidence(ev)
        assert "idna 3.11 -> 3.15 (delta: minor)" in out
        assert "fixed by this bump: PYSEC-2026-215" in out
        assert "REMAINING at target: none" in out
        assert "target yanked on PyPI: no" in out
        assert "30 day(s)" in out
        assert "https://example.com/CHANGES" in out
        assert "typosquat signal: none" in out
        assert "maintainer identity changed across the bump: unknown" in out
        assert "new install/build scripts at target: unknown" in out
        assert "target PEP 740 attested on PyPI: YES" in out
        assert "OpenSSF Scorecard: 7.2 (pypa/idna)" in out
        assert "idna 3.11->3.15" in out
        assert "evidence complete: yes" in out

    def test_incomplete_is_loud(self):
        out = render_evidence(_evidence(complete=False, target_yanked=None))
        assert "evidence complete: NO" in out
        assert "target yanked on PyPI: unknown" in out
