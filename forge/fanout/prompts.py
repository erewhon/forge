"""Prompt rendering: the profile's schema wrapper for every seat, and the judge kickoff prompt."""

from __future__ import annotations

from pathlib import Path

from forge.fanout.models import SweepManifest
from forge.fanout.profiles import Profile, load_profile


def wrap_sweep_prompt(user_prompt: str, target: Path, profile: Profile | None = None) -> str:
    """Wrap the user's question with the profile's findings schema so seat outputs collate."""
    profile = profile or load_profile(None)
    return profile.sweep_template.format(target=target, prompt=user_prompt.strip())


def build_judge_prompt(manifest: SweepManifest) -> str:
    """Render the interactive judge kickoff prompt from a sweep manifest (its profile's text)."""
    with_findings = [s for s in manifest.seats if s.findings_path is not None]
    failed = [s for s in manifest.seats if s.findings_path is None]
    findings_list = "\n".join(
        f"- {s.label} ({s.display}): {s.findings_path}" for s in with_findings
    )
    if not with_findings:
        findings_list = "- (no seat produced findings)"
    failed_block = ""
    if failed:
        lines = "\n".join(
            f"- {s.label} ({s.display}): {s.status} — {s.error_message}" for s in failed
        )
        failed_block = f"\nSeats that produced nothing (ignore, but note the gap):\n{lines}\n"
    return manifest.profile.judge_template.format(
        target=manifest.target,
        n=len(manifest.seats),
        prompt_path=manifest.prompt_path,
        user_prompt_path=manifest.user_prompt_path,
        findings_list=findings_list,
        failed_block=failed_block,
        manifest_path=manifest.out_dir / "manifest.json",
        collated_path=manifest.out_dir / "collated.md",
    )
