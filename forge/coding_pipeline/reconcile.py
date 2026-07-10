"""Serial reconcile barrier — integrate concurrent workspace commits one at a time.

jj records conflicts as first-class objects: a rebase always SUCCEEDS and leaves a
conflicted commit detectable from a template field — integration is "check a field",
never "catch a merge exception", which is what makes optimistic concurrency tractable.
Each landed leaf commit is rebased serially onto the accumulating epic head, in dispatch
order; a conflicted result is abandoned and its task demoted back to Ready so the existing
replan/attempt-cap machinery owns the retry. Correctness rests entirely here — the
file-scope batch picker only reduces wasted parallel work.

All jj calls run against the MAIN repo path: workspace commits already live in the shared
repo store, and the worker workspaces are forgotten by the dispatcher afterwards. Leaves
are addressed by change id (stable across rebases) — run_one's jj path returns exactly
that; commit shas go stale the moment the rebase rewrites them.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel

from forge.coding_pipeline.models import LeafOutcome, SuiteResult, WaveReport
from forge.shared.workspaces import JJError, _run_jj


class ReconcileError(RuntimeError):
    """A repo-level reconcile step failed (distinct from a per-leaf demotion)."""


def _change_id(repo: Path, rev: str) -> str:
    out = _run_jj(
        ["log", "-r", rev, "--no-graph", "-T", "change_id.short()", "--limit", "1"], cwd=repo
    )
    change = out.stdout.strip()
    if not change:
        raise JJError(f"could not resolve {rev!r} to a change id")
    return change


def _is_conflicted(repo: Path, rev: str) -> bool:
    out = _run_jj(
        ["log", "-r", rev, "--no-graph", "-T", 'if(conflict, "true", "false")'], cwd=repo
    )
    return out.stdout.strip() == "true"


def _conflicted_files(repo: Path, rev: str) -> str:
    """Best-effort conflicted-path list for the demotion note — never raises."""
    try:
        out = _run_jj(["resolve", "--list", "-r", rev], cwd=repo, check=False)
    except OSError:
        return ""
    return out.stdout.strip()


def _demote(outcome: LeafOutcome, reason: str, note: str, on_demote, log) -> LeafOutcome:
    """Rewrite one leaf's outcome to failed and flip its task back to Ready.

    A failing demotion write must not sink the batch — but it leaves the Forge row Done
    while the commit is gone, so the mismatch is shouted into the reason and the log."""
    try:
        on_demote(outcome.leaf, note)
    except Exception as e:  # noqa: BLE001 — fail closed per leaf, keep batch-mates alive
        reason += f" [DEMOTION WRITE FAILED: {e} — task still shows Done in Forge; fix manually]"
        log(f"  reconcile: demotion write FAILED for {outcome.leaf}: {e}")
    return outcome.model_copy(update={"status": "failed", "reason": reason, "commit_id": None})


def reconcile_wave(
    repo: Path,
    base_rev: str,
    landed: list[tuple[LeafOutcome, str]],
    *,
    on_demote: Callable[[str, str], None],
    log: Callable[[str], None] = print,
) -> list[LeafOutcome]:
    """Serially integrate ``landed`` (outcome, revision) pairs onto ``base_rev``; return one
    outcome per input, in order, with demoted leaves rewritten to ``failed``.

    Per-leaf, fail closed: a conflicted rebase abandons the commit; any other jj failure
    skips the leaf WITHOUT abandoning (an infra error must not destroy work — the dangling
    side head is named in the note for forensics). Either way ``on_demote(title, note)``
    flips the task back to Ready and the loop continues with the batch-mates. The main
    working copy is repositioned onto the final head only when at least one leaf landed;
    a failure THERE is repo-level and raises :class:`ReconcileError`.
    """
    head = base_rev
    results: list[LeafOutcome] = []
    for outcome, rev in landed:
        try:
            change = _change_id(repo, rev)
            _run_jj(["rebase", "-r", change, "-d", head], cwd=repo)
            if _is_conflicted(repo, change):
                files = _conflicted_files(repo, change)  # must read BEFORE the abandon
                _run_jj(["abandon", change], cwd=repo)
                note = (
                    "Demoted by the reconcile barrier: integration conflict with previously "
                    "landed wave sibling(s). The leaf was green in its own workspace; its "
                    "commit was abandoned and the leaf re-opened for a fresh attempt on the "
                    "updated head.\n\nConflicted paths:\n" + (files or "(unavailable)")
                )
                log(f"  reconcile: CONFLICT — abandoned {outcome.leaf}")
                results.append(
                    _demote(
                        outcome,
                        "integration conflict with previously landed wave sibling(s)",
                        note,
                        on_demote,
                        log,
                    )
                )
                continue
        except JJError as e:
            note = (
                f"Demoted by the reconcile barrier: jj failed while integrating this leaf: "
                f"{e}\n\nThe leaf's commit (revision {rev}) was NOT abandoned — it dangles "
                "as a side head for forensics."
            )
            log(f"  reconcile: jj failure on {outcome.leaf} — demoting ({e})")
            results.append(
                _demote(outcome, f"reconcile jj failure: {e}", note, on_demote, log)
            )
            continue
        head = change
        results.append(outcome)
        log(f"  reconcile: landed {outcome.leaf} -> {change}")

    if head != base_rev:
        try:
            _run_jj(["new", head], cwd=repo)
        except JJError as e:
            raise ReconcileError(
                f"integrated chain built at {head} but repositioning the working copy "
                f"failed: {e}"
            ) from e
    return results


# --- bisect-on-red (semantic conflicts: green in isolation, red combined) -------------


class BisectResult(BaseModel):
    """What bisect attributed and what state it left the wave in."""

    offender: str  # leaf title backed out of the chain
    repaired_green: bool  # the suite verdict AFTER the back-out
    repaired_tail: str = ""


def bisect_red(
    repo: Path,
    base_rev: str,
    chain: list[tuple[str, str]],
    run_suite: Callable[[], tuple[bool, str]],
    *,
    on_demote: Callable[[str, str], None],
    log: Callable[[str], None] = print,
) -> BisectResult | None:
    """Locate the first red point along the integrated ``chain`` (title, change_id pairs in
    integration order), back that leaf out, demote it, and re-verify once.

    Serial dispatch catches an integration break at the leaf that introduces it (each leaf's
    suite runs on the accumulated state); concurrent dispatch tests leaves in isolation, so
    a semantic conflict first surfaces at the wave's batch suite gate with no attribution.
    This walk restores parity: position the working copy on each chain point, run the suite,
    stop at the first red. Linear on purpose — wave_size is 4; a true bisect buys nothing.

    The offender is demoted even when the repaired state is still red: first-red-at-X is
    confirmed evidence against X, and restoring X would leave Forge and the chain agreeing
    on a state the suite already condemned. Remaining redness flows to replan exactly like
    a red serial wave. Returns None (fail open, VCS untouched beyond repositioning to the
    chain tip) when every point is green — a flaky suite, not a semantic conflict — or when
    any jj step fails.
    """
    offender_i: int | None = None
    try:
        for i, (title, change) in enumerate(chain):
            _run_jj(["new", change], cwd=repo)
            passed, _ = run_suite()
            log(f"  bisect: {title} -> {'green' if passed else 'RED'}")
            if not passed:
                offender_i = i
                break
        if offender_i is None:
            # Every point green — the wave-level red was flake. @ already sits on the tip.
            log("  bisect: every chain point green — no attribution (flaky suite?)")
            return None

        title, change = chain[offender_i]
        parent = base_rev if offender_i == 0 else chain[offender_i - 1][1]
        if offender_i < len(chain) - 1:
            # Move the rest of the chain off the offender before abandoning it.
            _run_jj(["rebase", "-s", chain[offender_i + 1][1], "-d", parent], cwd=repo)
        _run_jj(["abandon", change], cwd=repo)
        new_tip = chain[-1][1] if offender_i < len(chain) - 1 else parent
        _run_jj(["new", new_tip], cwd=repo)
    except JJError as e:
        log(f"  bisect: jj failed mid-walk — no attribution ({e})")
        return None

    repaired_green, tail = run_suite()
    note = (
        "Demoted by bisect-on-red: this leaf was green in its own workspace but is the "
        "first point where the integrated wave suite goes red — a semantic conflict with "
        "an earlier wave sibling. Its commit was backed out of the epic chain; retry "
        "against the updated head.\n\n"
        f"Suite after the back-out: {'green' if repaired_green else 'still RED'}."
    )
    try:
        on_demote(title, note)
    except Exception as e:  # noqa: BLE001 — attribution stands even if the write fails
        log(f"  bisect: demotion write FAILED for {title}: {e} — Forge still shows Done")
    log(f"  bisect: backed out {title}; repaired suite {'green' if repaired_green else 'RED'}")
    return BisectResult(offender=title, repaired_green=repaired_green, repaired_tail=tail)


def apply_bisect(
    report: WaveReport,
    *,
    repo: Path,
    base_rev: str,
    concurrency: int,
    run_suite: Callable[[], tuple[bool, str]],
    on_demote: Callable[[str, str], None],
    log: Callable[[str], None] = print,
) -> BisectResult | None:
    """The orchestrator's one-line hook: attribute a red CONCURRENT wave, mutate ``report``.

    No-op unless the suite is red, the wave ran concurrently, and more than one leaf
    landed (a single landed leaf IS the attribution; serial waves already attribute).
    On attribution the offender's outcome is rewritten to failed (commit gone) and
    ``report.suite`` is replaced with the repaired verdict, so replan sees a demoted
    leaf instead of a mystery-red wave.
    """
    if report.suite_green or concurrency <= 1:
        return None
    chain = [(o.leaf, o.commit_id) for o in report.landed if o.commit_id]
    if len(chain) < 2:
        return None
    result = bisect_red(repo, base_rev, chain, run_suite, on_demote=on_demote, log=log)
    if result is None:
        return None
    report.outcomes = [
        o.model_copy(
            update={
                "status": "failed",
                "reason": "semantic integration conflict (bisect-on-red)",
                "commit_id": None,
            }
        )
        if o.leaf == result.offender
        else o
        for o in report.outcomes
    ]
    report.suite = SuiteResult(
        passed=result.repaired_green, output_tail=result.repaired_tail[-2000:]
    )
    return result
