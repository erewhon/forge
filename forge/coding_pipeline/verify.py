"""Wave verification — the gates between waves (design: "Verification cadence").

Two gates, per dry-run design input #2 (gates are mandatory regardless of doer tier):

- **Hard**: the whole-project suite runs in the dx container (``task_worker.tester.run_tests``).
  Red means the wave does not advance; the failure tail goes into the ``WaveReport`` for replan.
- **Advisory**: a cross-family review of the wave diff. Each active pr_review provider
  independently lists findings (structured JSON, not prose — the same move autotest made for its
  sign-off); a **consolidation pass** merges semantically-equivalent candidates (cross-provider
  paraphrase is the norm — the dry-run's epic gate phrased one blocker three ways in a single
  verdict) and drops candidates already covered by the epic's OPEN fix-up leaves; then every
  canonical finding faces a **confirm vote** across the roster: it is ``confirmed`` only when a
  strict majority of *responding* providers judge it real. Zero responders = unconfirmed
  (advisory stays advisory; only confirmed findings become fix-up leaves, so the action path
  fails closed).

Dedup layers (dry-run Q2): the deterministic stable slug catches near-verbatim twins and keys
the fix-up leaf's external_ref (``pipeline:{epic}:fix:{finding_slug}``) for exact cross-replan
dedup; the consolidator (the ``recipe.discover_dedup_verify`` dedup stage, same fail-open
semantics) catches paraphrase and re-worded re-discovery. Consolidation failure or a suspicious
output (more canonical findings than raw) passes the raw findings through unchanged — the
consolidator can reduce work, never invent it, and never blocks the wave.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from pydantic import BaseModel

from forge.coding_pipeline.config import settings
from forge.coding_pipeline.models import ReviewFinding, SuiteResult, WaveReport
from forge.pr_review_ensemble.providers import build_reviewer_slots, rotation_pool
from forge.shared.automerge import slugify
from forge.shared.panel import PanelMember, PanelResult, run_member_panel, structured, verify_each
from forge.task_worker.tester import run_tests
from forge.task_worker.vcs import VCSError, detect_vcs

FINDINGS_SYSTEM = """You are reviewing the accumulated diff of one WAVE of automated coding work.
List only REAL problems this diff introduces: bugs, broken interactions between the changes,
safety-path regressions, misleading tests. Style nits and pre-existing issues do not count.

Method: read the diff against its own stated contracts, not just for smells. The highest-value
defects in machine-written code are QUIET — the code parses, its tests pass, and one
load-bearing detail is wrong. Hunt these classes hardest:
- Contract collapse: code implementing a simpler shape than its docstring or documented layout
  promises (a loader walking ONE directory level where the layout has two; a lookup keyed on
  the wrong field). Check every walk/loop and path constant against the documented shape.
- Dropped interpolation: a path, name, or key built WITHOUT the variable that makes it unique,
  so distinct things silently share one slot.
- Bogus defaults: absolute paths or endpoints that exist on no host — defaults that only work
  because every caller (and every test, via tmp dirs) overrides them.
- Wrong destination: output written next to its inputs or inside the package instead of the
  configured output dir; successive runs overwriting each other.
- Inverted wiring: a dispatch table or mapping pairing correct pieces backwards — each function
  is right, the wiring is reversed.
- Tests that assert the bug: new tests whose expected values encode the defect. When code and
  test agree, check both against domain truth, not against each other.
- Swallowed failures: except-pass on an action path; success counts that over-report after
  errors.
- Boundary skips: loops that silently miss the first or last element.

Only emit a finding you can trace to exact hunks, and put the trace IN the summary — name the
file, the constant/loop/line shape, and what it contradicts ("X builds the path without the
model name, so every model shares baselines/.json") — so a skeptical verifier can confirm it
by reading. Vague findings ("error handling could be improved") are noise.

The hunt list says where to LOOK, not what to find: most wave diffs are clean. An empty list
is a perfectly good answer — for a clean diff respond with exactly {"findings": []}, never
prose. At most 5 findings, most severe first.

Respond with ONLY a JSON object:
{"findings": [{"summary": str, "file": str|null, "severity": "critical"|"high"|"medium"|"low"}]}"""

CONFIRM_SYSTEM = """You are a skeptic judging ONE review finding against the diff that prompted
it. Default to NOT real unless the diff clearly shows the problem: a finding that becomes a task
costs real work. Pre-existing issues and matters of taste are NOT real.

Judge by TRACING the hunks against the claim, not by plausibility:
- When the finding claims the code contradicts its documented contract (walks fewer levels than
  the stated layout, drops the distinguishing variable from a path or key, writes output to the
  wrong root), verify by tracing the exact lines: find the loop or constant and check it against
  the docstring/documented shape. If the trace CONFIRMS the claim, it is real — even though the
  code parses, looks tidy, and its tests pass. Green tests are weak evidence: tests that never
  exercise the claimed path prove nothing about it.
- Reject when the trace refutes the claim (the guard or base case said to be missing is right
  there in the hunk), when the issue predates this diff, or when it is style, taste, or a
  hypothetical ("could be slow", "might want jitter").
- When the referenced file's CURRENT CONTENT is provided below the diff, it is ground truth and
  outranks the diff: a diff shows hunks, not the whole file, so a claim that something is
  "missing" (an import, a guard, a fixture) is only real if it is absent from the current
  content — not merely absent from the hunks. A claim the current content refutes is NOT real.

Respond with ONLY a JSON object: {"real": true|false, "reason": str}"""


_GROUND_TRUTH_CAP = 6000  # chars of the referenced file shown to the confirm seat


def _file_ground_truth(repo: Path | None, file: str | None) -> str:
    """The referenced file's current content, for the confirm prompt.

    A diff-only confirm seat cannot refute "X is missing from F" claims — the hunks
    legitimately don't show the rest of F (deps-v2: a 'missing import' phantom on a file
    whose line 5 was that import survived confirmation and burned eight waves). Returns
    an explicit does-not-exist note for dangling references, and empty string when there
    is nothing useful to add.
    """
    if repo is None or not file:
        return ""
    path = repo / file
    if not path.is_file():
        return f"\n\nReferenced file {file} does NOT currently exist in the tree."
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""
    clipped = content[:_GROUND_TRUTH_CAP]
    suffix = "\n[... truncated ...]" if len(content) > _GROUND_TRUTH_CAP else ""
    return (
        f"\n\nCurrent content of {file} (ground truth — outranks the diff for "
        f'"missing X" claims):\n\n```\n{clipped}{suffix}\n```'
    )


class RawFinding(BaseModel):
    summary: str
    file: str | None = None
    severity: str = "medium"


class FindingsEnvelope(BaseModel):
    findings: list[RawFinding] = []


# --- VCS helpers ----------------------------------------------------------------


def _run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=60, cwd=cwd)


def wave_start_rev(repo: Path) -> str:
    """The pre-wave tip, recorded as the basis for the accumulated wave diff.

    jj: @ itself BECOMES the first landed commit when the worker commits
    (describe-in-place + ``jj new``), so recording @'s change id made every
    wave diff empty — the review gate never saw a diff (e2e dry-run finding).
    Record @-'s commit id instead: the last commit before the wave. A merge
    working copy has several parents; the first is the mainline one.

    git: HEAD, which is already the pre-wave tip (commits move HEAD forward).
    """
    vcs = detect_vcs(repo)
    if vcs == "jj":
        res = _run(
            ["jj", "log", "--no-graph", "-r", "@-", "-T", 'commit_id.short() ++ "\\n"'], repo
        )
    elif vcs == "git":
        res = _run(["git", "rev-parse", "--short", "HEAD"], repo)
    else:
        raise VCSError(f"No VCS detected in {repo}")
    if res.returncode != 0:
        raise VCSError(f"wave_start_rev failed: {res.stderr.strip()}")
    lines = [line.strip() for line in res.stdout.splitlines() if line.strip()]
    if not lines:
        raise VCSError(f"wave_start_rev found no parent revision in {repo}")
    return lines[0]


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


# --- consolidation (the recipe's dedup stage) ----------------------------------------

CONSOLIDATE_SYSTEM = """You are consolidating code-review findings from SEVERAL independent
reviewers of ONE diff. Different reviewers phrase the same underlying problem differently;
your job is to merge duplicates, never to review.

Rules:
- Group candidates that describe the SAME underlying problem (same defect, even if worded
  differently or one omits the file). Emit ONE canonical finding per group: the clearest
  summary, the most specific file, the highest severity among members.
- List every merged candidate's slug in merged_slugs, including the canonical's own source.
- If a candidate is ALREADY COVERED by one of the existing open fix-up tasks listed, do not
  emit it — record it under covered instead.
- Never invent findings, never split one candidate into several, never editorialize summaries
  beyond picking the clearest existing phrasing.

Respond with ONLY a JSON object:
{"findings": [{"summary": str, "file": str|null,
               "severity": "critical"|"high"|"medium"|"low", "merged_slugs": [str]}],
 "covered": [{"slug": str, "by_task": str}]}"""


class CanonicalFinding(BaseModel):
    summary: str
    file: str | None = None
    severity: str = "medium"
    merged_slugs: list[str] = []


class CoveredFinding(BaseModel):
    slug: str
    by_task: str


class ConsolidationEnvelope(BaseModel):
    findings: list[CanonicalFinding] = []
    covered: list[CoveredFinding] = []


def consolidate_findings(
    findings: list[ReviewFinding],
    existing_fixups: list[str] | None = None,
) -> tuple[list[ReviewFinding], bool, list[str]]:
    """Merge paraphrase twins and drop candidates covered by open fix-up leaves.

    Returns ``(canonical, ok, dropped_covered)``. Fail-open by contract: a failed call
    or a suspicious envelope (more canonical findings than raw — a consolidator may
    reduce work, never invent it) returns the input unchanged with ``ok=False``. With
    at most one candidate and no open fix-ups there is nothing to merge — no LLM call.
    """
    existing = existing_fixups or []
    if len(findings) <= 1 and not existing:
        return findings, True, []

    parts = ["## Candidate findings (one per line: slug | severity | file | summary)"]
    parts += [f"- {f.slug} | {f.severity} | {f.file or '-'} | {f.summary}" for f in findings]
    if existing:
        parts.append("\n## Existing OPEN fix-up tasks for this epic (already-known work)")
        parts += [f"- {title}" for title in existing]

    result = structured(
        pool=rotation_pool(build_reviewer_slots(), role="review:consolidate"),
        schema=ConsolidationEnvelope,
        system=CONSOLIDATE_SYSTEM,
        user="\n".join(parts),
        max_tokens=settings.review_max_tokens,
        timeout=settings.review_timeout,
    )
    if result.value is None or len(result.value.findings) > len(findings):
        return findings, False, []

    known = {"critical", "high", "medium", "low"}
    canonical: list[ReviewFinding] = []
    for c in result.value.findings:
        severity = c.severity if c.severity in known else "medium"
        canonical.append(
            ReviewFinding(
                slug=stable_slug(c.file, c.summary),
                summary=c.summary,
                severity=severity,
                file=c.file,
            )
        )
    dropped = [f"{c.slug} (covered by: {c.by_task})" for c in result.value.covered]
    return canonical, True, dropped


def _majority_real(_finding: ReviewFinding, panel: PanelResult) -> bool:
    """Strict majority of RESPONDING skeptics; zero responders fails closed."""
    real_votes = sum(1 for r in panel.responses if r.get("real") is True)
    return len(panel.responses) > 0 and real_votes * 2 > len(panel.responses)


def confirm_findings(
    diff: str, findings: list[ReviewFinding], repo: Path | None = None
) -> list[ReviewFinding]:
    """The confirm vote: each finding faces the roster; ``confirmed`` set by strict majority.

    ``repo`` lets the seat see the referenced file's current content alongside the diff —
    ground truth for "missing X" claims the hunks can neither prove nor refute.
    """
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
            f"{_file_ground_truth(repo, f.file)}"
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
    existing_fixups: list[str] | None = None,
) -> WaveReport:
    """Run both wave gates and assemble the ``WaveReport`` replan consumes.

    The suite is the hard gate (``report.suite_green``); review findings flow
    collect → consolidate → confirm, and ride along with their ``confirmed`` flags —
    only confirmed ones may become fix-up leaves downstream. ``existing_fixups`` (the
    epic's open fix-up leaf titles) lets the consolidator drop re-discovered issues.
    The orchestrator fills ``report.outcomes`` from dispatch. An empty wave diff skips
    the review (nothing to judge).
    """
    passed, output = run_tests(repo)
    suite = SuiteResult(passed=passed, output_tail=output[-2000:])

    diff = wave_diff(repo, from_change)
    findings: list[ReviewFinding] = []
    raw_count = 0
    consolidation_ok = True
    dropped: list[str] = []
    if diff.strip() and not skip_review:
        raw = collect_findings(diff)
        raw_count = len(raw)
        canonical, consolidation_ok, dropped = consolidate_findings(raw, existing_fixups)
        findings = confirm_findings(diff, canonical, repo=repo)

    return WaveReport(
        wave=wave,
        suite=suite,
        findings=findings,
        raw_findings=raw_count,
        consolidation_ok=consolidation_ok,
        dropped_covered=dropped,
        diff_stat=_diff_stat(diff) if diff.strip() else "empty diff",
    )
