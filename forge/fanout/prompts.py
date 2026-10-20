"""Prompt templates: the findings-schema wrapper every seat gets, and the judge kickoff prompt."""

from __future__ import annotations

from pathlib import Path

from forge.fanout.models import SweepManifest

_SWEEP_HEADER = """\
You are one of several independent analysts examining the repository at {repo}. Other analysts
are answering the same question in parallel; a reviewer will later verify every claim against
the code and collate the results, so precision and evidence matter more than volume.

Rules:
- Read-only. Do not create, edit, or delete files. Explore with the read/search tools you have.
- Every finding must cite a concrete location (path:line) and quote or paraphrase the evidence.
- Do not pad. If you checked something and it is fine, say so under "Not found".
- Rate your own confidence honestly; a low-confidence lead is useful, a confident guess is not.

Question:

{prompt}

When you are done exploring, reply with your findings as markdown in exactly this shape. The
reply itself is your deliverable; there is no file to write.

# Findings

## F1: <short title>
- **Where:** <path:line, or several>
- **Severity:** critical | high | medium | low
- **Confidence:** high | medium | low
- **Evidence:** <what the code does, quoting the relevant lines>
- **Why it matters:** <the consequence>
- **Suggested fix:** <one or two sentences; optional>

## F2: ...

# Not found
<what you checked and ruled out, one line each>

# Summary
<three to five sentences>
"""


def wrap_sweep_prompt(user_prompt: str, repo: Path) -> str:
    """Wrap the user's question with the fixed findings schema so seat outputs collate."""
    return _SWEEP_HEADER.format(repo=repo, prompt=user_prompt.strip())


_JUDGE_TEMPLATE = """\
You are the judge for a multi-model analysis sweep of the repository at {repo}.

{n} independent seats were given the same question (the wrapped prompt is at {prompt_path};
the question as I wrote it is at {user_prompt_path}). Their findings are in:

{findings_list}
{failed_block}
Sweep manifest (models, timings, exit codes): {manifest_path}

Your job:
1. Read the question and every findings file in full.
2. For each distinct claim, verify it against the code here: open the cited lines, follow the
   call sites, and decide. Label each one CONFIRMED, FALSE, UNVERIFIABLE, or DUPLICATE-of-<id>.
   Note which seats raised it and whether they agree on severity.
3. Show me a compact table (claim, label, severity, seats, one-line reason) before writing
   anything, and flag anything you could not settle so I can weigh in.
4. Then write {collated_path}: confirmed findings grouped by severity with seat attribution and
   your verified evidence, a short list of rejected claims with the reason, and gaps — things the
   question asked about that no seat covered well.

Do not edit anything else in the repository. Start with step 1.
"""


def build_judge_prompt(manifest: SweepManifest) -> str:
    """Render the interactive judge kickoff prompt from a sweep manifest."""
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
    return _JUDGE_TEMPLATE.format(
        repo=manifest.repo_path,
        n=len(manifest.seats),
        prompt_path=manifest.prompt_path,
        user_prompt_path=manifest.user_prompt_path,
        findings_list=findings_list,
        failed_block=failed_block,
        manifest_path=manifest.out_dir / "manifest.json",
        collated_path=manifest.out_dir / "collated.md",
    )
