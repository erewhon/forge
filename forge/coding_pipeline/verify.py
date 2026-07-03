"""Wave verification — the gates between waves (design: "Verification cadence").

Two gates, per dry-run design input #2 (gates are mandatory regardless of doer tier):

- **Hard**: the whole-project suite runs in the dx container (``task_worker.tester.run_tests``).
  Red means the wave does not advance; the failure tail goes into the ``WaveReport`` for replan.
- **Advisory**: a cross-family review of the wave diff. Each active pr_review provider
  independently lists findings (structured JSON, not prose — the same move autotest made for its
  sign-off), then every candidate finding faces a **confirm vote** across the same roster: it is
  ``confirmed`` only when a strict majority of *responding* providers judge it real. Zero
  responders = unconfirmed (advisory stays advisory; only confirmed findings become fix-up
  leaves, so the action path fails closed).

Dedup is deterministic, not an LLM stage: findings collapse on a stable slug derived from
(file, summary head) — the same slug keys the fix-up leaf's external_ref
(``pipeline:{epic}:fix:{slug}``), so re-discovered findings dedup at emission across replans
too. Differently-worded duplicates can survive to the vote; the emission cap and idempotent
refs bound the damage. (Candidate refinement: an LLM consolidation pass like
``recipe.discover_dedup_verify`` uses.)
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from pydantic import BaseModel

from agents.coding_pipeline.config import settings
from agents.coding_pipeline.models import ReviewFinding, SuiteResult, WaveReport
from agents.pr_review_ensemble.providers import build_reviewer_slots
from agents.shared.automerge import slugify
from agents.shared.panel import PanelMember, PanelResult, run_member_panel, verify_each
from agents.task_worker.tester import run_tests
from agents.task_worker.vcs import VCSError, detect_vcs

FINDINGS_SYSTEM = """You are reviewing the accumulated diff of one WAVE of automated coding work.
List only REAL problems this diff introduces: bugs, broken interactions between the changes,
safety-path regressions, misleading tests. Style nits and pre-existing issues do not count.
An empty list is a perfectly good answer. At most 5 findings, most severe first.

Respond with ONLY a JSON object:
{"findings": [{"summary": str, "file": str|null, "severity": "critical"|"high"|"medium"|"low"}]}"""

CONFIRM_SYSTEM = """You are a skeptic judging ONE review finding against the diff that prompted
it. Default to NOT real unless the diff clearly shows the problem: a finding that becomes a task
costs real work. Pre-existing issues and matters of taste are NOT real.

Respond with ONLY a JSON object: {"real": true|false, "reason": str}"""


class RawFinding(BaseModel):
    summary: str
    file: str | None = None
    severity: str = "medium"


class FindingsEnvelope(BaseModel):
    findings: list[RawFinding] = []


# --- VCS helpers ----------------------------------------------------------------


def _run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=60, cwd=cwd)


def current_change_id(repo: Path) -> str:
    """The working-copy change id (jj) / HEAD sha (git) — recorded as the wave start."""
    vcs = detect_vcs(repo)
    if vcs == "jj":
        res = _run(["jj", "log", "--no-graph", "-r", "@", "-T", "change_id.short()"], repo)
    elif vcs == "git":
        res = _run(["git", "rev-parse", "--short", "HEAD"], repo)
    else:
        raise VCSError(f"No VCS detected in {repo}")
    if res.returncode != 0:
        raise VCSError(f"current_change_id failed: {res.stderr.strip()}")
    return res.stdout.strip()


def wave_diff(repo: Path, from_change: str) -> str:
    """Unified git-format diff of everything the wave landed: ``from_change`` → working copy."""
    vcs = detect_vcs(repo)
    if vcs == "jj":
        res = _run(["jj", "diff", "--no-pager", "--git", "--from", from_change, "--to", "@"], repo)
    elif vcs == "git":
        res = _run(["git", "diff", from_change, "HEAD"], repo)
    else:
        raise VCSError(f"No VCS detected in {repo}")
    if res.returncode != 0:
        raise VCSError(f"wave diff failed: {res.stderr.strip()}")
    return res.stdout


def _diff_stat(diff: str) -> str:
    files = sum(1 for line in diff.splitlines() if line.startswith("diff --git"))
    plus = sum(1 for line in diff.splitlines() if line.startswith("+") and line[:3] != "+++")
    minus = sum(1 for line in diff.splitlines() if line.startswith("-") and line[:3] != "---")
    return f"{files} file(s), +{plus}/-{minus}"


# --- advisory review ---------------------------------------------------------------


def stable_slug(file: str | None, summary: str) -> str:
    """Deterministic finding identity — keys fix-up refs, so it must survive re-discovery."""
    return slugify(f"{file or 'general'}-{summary}", max_len=60)


def _roster_members(system: str) -> list[PanelMember]:
    return [
        PanelMember(executor=slot.pool.executors[0], system=system, label=slot.provider)
        for slot in build_reviewer_slots()
        if slot.active
    ]


def collect_findings(diff: str) -> list[ReviewFinding]:
    """Each active provider independently lists findings; flatten + slug-dedup + cap."""
    members = _roster_members(FINDINGS_SYSTEM)
    if not members:
        return []
    panel = run_member_panel(
        members=members,
        user=f"Wave diff:\n\n{diff}",
        floor=1,
        max_tokens=settings.review_max_tokens,
        timeout=settings.review_timeout,
    )
    findings: dict[str, ReviewFinding] = {}
    for response in panel.responses:
        try:
            envelope = FindingsEnvelope.model_validate(response)
        except ValueError:
            continue  # one provider's malformed envelope drops, the rest stand
        for raw in envelope.findings:
            known = {"critical", "high", "medium", "low"}
            severity = raw.severity if raw.severity in known else "medium"
            slug = stable_slug(raw.file, raw.summary)
            findings.setdefault(
                slug,
                ReviewFinding(slug=slug, summary=raw.summary, severity=severity, file=raw.file),
            )
    ordered = sorted(
        findings.values(),
        key=lambda f: ["critical", "high", "medium", "low"].index(f.severity),
    )
    return ordered[: settings.review_max_findings]


def _majority_real(_finding: ReviewFinding, panel: PanelResult) -> bool:
    """Strict majority of RESPONDING skeptics; zero responders fails closed."""
    real_votes = sum(1 for r in panel.responses if r.get("real") is True)
    return len(panel.responses) > 0 and real_votes * 2 > len(panel.responses)


def confirm_findings(diff: str, findings: list[ReviewFinding]) -> list[ReviewFinding]:
    """The confirm vote: each finding faces the roster; ``confirmed`` set by strict majority."""
    if not findings:
        return []
    members = _roster_members(CONFIRM_SYSTEM)
    if not members:
        return findings  # nobody to vote: everything stays unconfirmed (advisory only)
    verdicts = verify_each(
        findings,
        members=members,
        make_user=lambda f: (
            f"Finding: {f.summary}\nFile: {f.file or 'unspecified'} "
            f"(severity claimed: {f.severity})\n\nWave diff:\n\n{diff}"
        ),
        aggregate=_majority_real,
        floor=1,
        concurrency=settings.confirm_concurrency,
        max_tokens=1024,
        timeout=settings.review_timeout,
    )
    for verdict in verdicts:
        verdict.item.confirmed = bool(verdict.verdict)
    return [v.item for v in verdicts]


# --- the wave gate ---------------------------------------------------------------


def verify_wave(
    repo: Path,
    *,
    wave: int,
    from_change: str,
    skip_review: bool = False,
) -> WaveReport:
    """Run both wave gates and assemble the ``WaveReport`` replan consumes.

    The suite is the hard gate (``report.suite_green``); review findings ride along with their
    ``confirmed`` flags — only confirmed ones may become fix-up leaves downstream. The
    orchestrator fills ``report.outcomes`` from dispatch. An empty wave diff skips the review
    (nothing to judge).
    """
    passed, output = run_tests(repo)
    suite = SuiteResult(passed=passed, output_tail=output[-2000:])

    diff = wave_diff(repo, from_change)
    findings: list[ReviewFinding] = []
    if diff.strip() and not skip_review:
        findings = confirm_findings(diff, collect_findings(diff))

    return WaveReport(
        wave=wave,
        suite=suite,
        findings=findings,
        diff_stat=_diff_stat(diff) if diff.strip() else "empty diff",
    )
