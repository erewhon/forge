"""The deterministic auto-merge risk policy — the dial the brief asked the framing to set.

``auto_eligible`` decides auto-track vs advisory-track BEFORE any LLM sees the bump. It is
deliberately conservative for v1: patch/minor only, complete evidence only, no residual vulns,
no yanked target, no typosquat suspicion. The full-quorum sign-off can still block an eligible
bump; nothing can promote an ineligible one. Missing evidence (``complete=False``, or a yanked
flag we couldn't determine) is ineligibility, not a judgment call — fail-closed by construction.
"""

from __future__ import annotations

from forge.dependabot.models import EvidenceBundle


def auto_eligible(
    evidence: EvidenceBundle,
    *,
    require_attestation: bool = True,
) -> tuple[bool, str]:
    """(eligible, reason). The reason names the FIRST failing condition — it becomes the
    advisory Forge task's headline, so it must say something a human can act on.

    Attestation posture diverges from the best-effort v2 signals: when enabled, only bumps whose
    target is provably attested stay auto-eligible. Unattested or undeterminable targets route to
    the advisory track. An integrity-API outage therefore pauses the auto track — bumps still
    flow as advisory tasks, which is the correct failure mode.
    """
    c = evidence.candidate
    if c.delta not in ("patch", "minor"):
        return False, f"version delta is {c.delta} — only patch/minor bumps auto-merge in v1"
    if not evidence.complete:
        return False, "evidence incomplete (a PyPI fetch or the lockfile delta is missing)"
    if evidence.target_yanked is None:
        return False, "yanked status undeterminable — treated as incomplete evidence"
    if evidence.target_yanked:
        return False, f"target version {c.latest} is yanked on PyPI"
    if evidence.findings_target:
        ids = ", ".join(f.vuln_id for f in evidence.findings_target[:3])
        return False, f"known vulnerabilities remain at {c.latest}: {ids}"
    if evidence.typosquat_suspect:
        return False, (
            f"name {c.name!r} is one edit from popular package "
            f"{evidence.typosquat_suspect!r} — possible typosquat"
        )
    # v2 provenance signals: block only when provably True — None (undeterminable) passes,
    # by contract these are best-effort and must not turn missing data into a block.
    if evidence.maintainer_changed:
        return False, (
            f"maintainer/author identity changed between {c.current} and {c.latest} — "
            "possible package takeover; needs human eyes"
        )
    if evidence.new_install_scripts:
        return False, (
            f"target {c.latest} introduces new install/build scripts or changes the build "
            "backend — install-time code surface grew"
        )
    # Attestation posture gate: when enabled, target_attested must be True. Under this
    # posture None (undeterminable) is treated as unattested — the auto track pauses
    # rather than guessing, and the bump still flows as an advisory task.
    if require_attestation and evidence.target_attested is not True:
        if evidence.target_attested is False:
            return False, (
                f"target {c.latest} publishes no PEP 740 attestations on PyPI — "
                "not attested by posture"
            )
        return False, (
            f"attestation status undeterminable for {c.latest} — treated as unattested by posture"
        )
    return True, ""
