"""Supply-chain audit pass: deterministic pre-scan, then a focused ensemble audit.

The pre-scan (``diffscan``) flags the supply-chain-relevant surface — dependency/lockfile, install/
build hook, CI, binary, and obfuscation/network/secret patterns — and collects just those files'
diffs. If it finds nothing, the pass returns a deterministic CLEAR with no model spend. Otherwise
the fan-out review machinery is reused with the supply-chain prompts, auditing the flagged slices
plus a signal summary (which also keeps the audit small on huge PRs).
"""

from __future__ import annotations

from datetime import UTC, datetime

from agents.pr_review_ensemble.diffscan import scan_supply_chain
from agents.pr_review_ensemble.models import EnsembleResult, SupplyChainResult, SupplyChainScan
from agents.pr_review_ensemble.prompts import SUPPLY_CHAIN_AGG_PROMPT, SUPPLY_CHAIN_SYSTEM_PROMPT
from agents.pr_review_ensemble.providers import ReviewerSlot
from agents.pr_review_ensemble.runner import run_ensemble
from agents.shared.ensemble import Combiner


def signals_preamble(scan: SupplyChainScan) -> str:
    """A compact markdown summary of the pre-scan signals to ground the auditors."""
    lines = [
        "## Supply-chain pre-scan signals",
        "",
        f"A deterministic scan flagged {len(scan.signals)} signal(s) across "
        f"{len(scan.relevant_files)} file(s) (of a {scan.full_diff_lines}-line diff). "
        "The diff below is just those files. Assess each signal and look for anything mechanical "
        "scanning would miss.",
        "",
    ]
    for s in scan.signals:
        note = f" — {s.note}" if s.note else ""
        lines.append(f"- **{s.category}** `{s.file}`: `{s.evidence}`{note}")
    lines.append("")
    return "\n".join(lines) + "\n"


async def run_supply_chain_audit(
    *,
    diff_text: str,
    pr_ref: str,
    slots: list[ReviewerSlot] | None = None,
    aggregator: Combiner | None = None,
) -> SupplyChainResult:
    """Pre-scan, then (only if there's a surface) run the focused ensemble audit."""
    timestamp = datetime.now(UTC)
    scan = scan_supply_chain(diff_text)

    if not scan.has_signals:
        return SupplyChainResult(pr_ref=pr_ref, timestamp=timestamp, scan=scan, ensemble=None)

    ensemble: EnsembleResult = await run_ensemble(
        diff_text=scan.relevant_diff,
        pr_ref=pr_ref,
        slots=slots,
        aggregator=aggregator,
        system_prompt=SUPPLY_CHAIN_SYSTEM_PROMPT,
        aggregator_system=SUPPLY_CHAIN_AGG_PROMPT,
        aggregator_noun="audits",
        user_preamble=signals_preamble(scan),
    )
    return SupplyChainResult(pr_ref=pr_ref, timestamp=timestamp, scan=scan, ensemble=ensemble)
