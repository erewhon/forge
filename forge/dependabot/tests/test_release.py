"""Release proposal — pure rendering, human-gated, never executes anything."""

from __future__ import annotations

from datetime import date

from forge.dependabot import release
from forge.dependabot.models import AuditFinding, BumpCandidate, EvidenceBundle
from forge.dependabot.release import render_release_proposal, should_propose_release


def _evidence(current=None, target=None) -> EvidenceBundle:
    return EvidenceBundle(
        candidate=BumpCandidate(name="idna", current="3.11", latest="3.15", delta="minor"),
        findings_current=current or [],
        findings_target=target or [],
        complete=True,
    )


def _finding(vuln_id: str) -> AuditFinding:
    return AuditFinding(package="idna", vuln_id=vuln_id, fix_versions=["3.15"])


class TestShouldPropose:
    def test_fixed_cve_proposes(self):
        assert should_propose_release(_evidence(current=[_finding("PYSEC-2026-215")]))

    def test_no_findings_does_not_propose(self):
        assert not should_propose_release(_evidence())

    def test_residual_findings_do_not_propose(self):
        assert not should_propose_release(
            _evidence(current=[_finding("PYSEC-1")], target=[_finding("PYSEC-2")])
        )


class TestRender:
    def test_proposal_contents(self):
        out = render_release_proposal(
            _evidence(current=[_finding("PYSEC-2026-215")]),
            merged_change_id="abc123",
            on=date(2026, 7, 6),
        )
        assert "PYSEC-2026-215" in out
        assert "v2026.07.06" in out
        assert "abc123" in out
        assert "human" in out
        assert "NOT executed" in out

    def test_module_never_touches_subprocess(self):
        # The human-gate is structural: no way to execute anything from this module.
        import inspect

        source = inspect.getsource(release)
        assert "import subprocess" not in source
        assert "os.system" not in source
        assert not hasattr(release, "subprocess")
