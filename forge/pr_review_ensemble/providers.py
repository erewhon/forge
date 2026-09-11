"""Reviewer roster: each seat is a failover ``Pool`` — a primary model plus backups pulled in only
when the primary is down. Every model resolves on the local LLM router (which holds all provider
creds server-side), so the whole roster is one endpoint + one key.

The roster is the single source of reviewers shared by the PR-review ensemble, the coding-pipeline
epic gate, wave-verify, the testing ensemble, and the Dependabot bumper — reconfigure it here and
every reviewer changes at once. Default: three diverse primary seats, two of them local — Claude
Sonnet (the frontier anchor), Ling 3 flash on hekaton (seated 2026-08-22; qualeval v2 code_review
0.91, the best local reviewer ever measured), and Gemma 4 26B on the talos B70 card 1 (seated
2026-08-22; board #1 composite 0.93, code_review 0.84, the analyst seat: adversarial 0.90 /
research 0.92, and the fleet's only Google-family model). Lightning (NVIDIA, code_review 0.84,
~100 t/s) is the fast local backup on both local seats and holds a primary seat in the all-local
roster. gpt-oss-120b (code_review 0.64, FP-heavy) lost its review seat on the 2026-08-22 board
but stays reachable as a backup; cheaper cloud backups: GLM, MiniMax M3, Kimi. Diversity
(distinct model families) over count. A disabled seat becomes a ``SkipExecutor`` slot:
attempted-but-never-ok for quorum accounting, without a doomed network call.

``build_local_reviewer_slots`` is the all-local roster (Ling 3 / Gemma 4 / Lightning — three
families, Ant/Google/NVIDIA, zero cloud seats), serving the ROUTINE lane and the ``shadow``
pass run side by side with production to decide whether sonnet can be unseated.

Serving contract for the Gemma seat: its 16k completion budget cannot be enforced server-side,
so reviewers must send generous max_tokens — ``review_max_tokens`` (16384) covers it; do not
lower it below 8192 or the v1 truncate-mid-flight trait returns. Ling 3 decodes on hekaton with
T4 offload (~16 t/s, 128K slot since 2026-09-05; was CPU-only ~7 t/s on a 32K slot). Prefill is
the budget item, not decode: it falls from ~285 t/s at 2K depth to ~120-160 t/s at 35K, and
reasoning scales with the diff (~800 tok on qualeval probes, the full 3000-token budget on a 35K
diff). Against the 300s per-provider timeout that fits ~15K-token diffs comfortably and ~25K
marginally; past that the seat fails over to Lightning by design (depth probes 2026-09-05).
Keep thinking ON for this seat: the 2026-09-05 A/B (code_review, 5 probes x3) scored 0.89 with
thinking vs 0.82 without, 1 vs 3 false positives in 15 runs — thinking-off asserts defects it
has not reasoned through, the expensive failure for an ensemble member. Thinking-off is a triage
mode only, and Lightning already fills that niche.
"""

from __future__ import annotations

from dataclasses import dataclass

from forge.pr_review_ensemble.config import settings
from forge.shared.ensemble import (
    ApiExecutor,
    ExecResult,
    ExecStatus,
    Executor,
    FailureClass,
    Pool,
    Prompt,
)
from forge.shared.privacy import PrivacyTier


@dataclass
class ReviewerSlot:
    """One seat in the ensemble: its identity plus the failover Pool that runs it (primary first,
    then backups). ``model`` is the primary model's alias, for display/accounting."""

    provider: str
    model: str
    pool: Pool
    skipped_reason: str | None = None  # set when the slot is a no-op skip (disabled / no creds)

    @property
    def active(self) -> bool:
        return self.skipped_reason is None


class SkipExecutor:
    """An Executor that never calls out — represents a disabled/unconfigured seat.

    Returns a SKIPPED / TERMINAL ExecResult so ``fanout`` counts the slot as attempted but not
    successful (matching the MVP, where a skipped provider still occupied a quorum slot) without
    spending a request that would only 401/“disabled”.
    """

    def __init__(self, label: str, reason: str) -> None:
        self.label = label
        self._reason = reason

    async def run(self, prompt: Prompt, *, timeout: float) -> ExecResult:
        return ExecResult(
            executor=self.label,
            status=ExecStatus.SKIPPED,
            error=self._reason,
            failure_class=FailureClass.TERMINAL,
        )


def _router_executor(label: str, model: str, *, privacy: PrivacyTier) -> ApiExecutor:
    """An OpenAI-compat executor pointed at the local LLM router (holds every provider's creds
    server-side, so a bare alias like ``glm``/``m3``/``kimi``/``coder`` just works). ``privacy``
    is the X-Router-Privacy tier the seat sends — the roster's, so every seat and every backup
    in a roster is held to the same tier."""
    return ApiExecutor(
        label=label,
        kind="openai",
        model=model,
        privacy=privacy,
        base_url=settings.local_base_url,
        api_key=settings.local_api_key,
    )


def _anthropic_primary(*, privacy: PrivacyTier) -> ApiExecutor:
    """The premium Claude seat's primary executor: routed through the router by default (no
    per-shell ANTHROPIC_API_KEY), or the native SDK when ``anthropic_base_url`` is cleared. The
    native SDK bypasses the router, so it can only serve a roster whose tier is ``any`` —
    ApiExecutor refuses anything stricter at construction."""
    model = settings.anthropic_model
    label = f"sonnet:{model}"
    if settings.anthropic_base_url:
        return ApiExecutor(
            label=label,
            kind="openai",
            model=model,
            privacy=privacy,
            base_url=settings.anthropic_base_url,
            api_key=settings.anthropic_api_key,
        )
    return ApiExecutor(label=label, kind="anthropic", model=model, privacy=privacy)


def _failover_slot(
    provider: str,
    primary: Executor,
    model: str,
    backups: list[str],
    *,
    privacy: PrivacyTier,
) -> ReviewerSlot:
    """A seat whose Pool tries ``primary`` first, then each backup alias (via the router)."""
    executors: list[Executor] = [primary]
    executors += [_router_executor(f"{provider}:backup:{m}", m, privacy=privacy) for m in backups]
    return ReviewerSlot(provider, model, Pool(role=f"review:{provider}", executors=executors))


def _sonnet_slot(*, privacy: PrivacyTier) -> ReviewerSlot:
    """Premium Claude seat: sonnet primary, local ``coder`` break-glass backup. Honors the
    ``anthropic_enabled`` toggle — flip it off to drop the seat during an Anthropic outage."""
    if not settings.anthropic_enabled:
        pool = Pool(
            role="review:sonnet-5", executors=[SkipExecutor("sonnet-5", "disabled in config")]
        )
        return ReviewerSlot(
            "sonnet-5", settings.anthropic_model, pool, skipped_reason="disabled in config"
        )
    return _failover_slot(
        "sonnet-5",
        _anthropic_primary(privacy=privacy),
        settings.anthropic_model,
        ["coder"],
        privacy=privacy,
    )


def _ling3_slot(*, privacy: PrivacyTier) -> ReviewerSlot:
    """Local Ant/Bailing seat: primary is Ling-3.0-flash on hekaton (router alias `ling`,
    llama-server-ling3 :5393, T4 offload, 128K slot, no drafter since 2026-09-05 — this seat's
    diffs stay in the homelab). qualeval v2 0.92 composite, code_review 0.91 — the best local
    reviewer ever measured (seated 2026-08-22, displacing gpt-oss from the review seat); the
    2026-09-05 T4-path re-measure (cr 0.89) matched within noise. Its known flaw (tu-07:
    fabricates instead of surfacing tool errors) never fires in a review seat — reviews use no
    tools. ~16 t/s decode with depth-sensitive prefill (see module docstring): diffs past ~25K
    tokens exceed the 300s timeout, so Lightning (local, fast, review 0.84) is the first backup,
    then kimi."""
    primary = _router_executor("ling3", "ling", privacy=privacy)
    return _failover_slot("ling3", primary, "ling", ["lightning", "kimi"], privacy=privacy)


def _gemma_slot(*, privacy: PrivacyTier) -> ReviewerSlot:
    """Local Google seat: primary is Gemma 4 26B on the talos B70 card 1 (router alias `gemma`,
    gemma-server :5392, ~62 t/s Vulkan — diffs stay in the homelab). qualeval v2 board #1
    (0.93): code_review 0.84 (ties Lightning), adversarial 0.90 / research 0.92 — the panel's
    analyst. Sampling params are held server-side; the 16k completion budget is NOT — callers
    must keep max_tokens generous (review_max_tokens 16384 does). gpt-oss (local, the demoted
    executor) is the first backup, then glm (family-diverse cloud)."""
    primary = _router_executor("gemma", "gemma", privacy=privacy)
    return _failover_slot("gemma", primary, "gemma", ["gpt-oss", "glm"], privacy=privacy)


def _lightning_slot(*, privacy: PrivacyTier) -> ReviewerSlot:
    """Local NVIDIA seat: primary is Nemotron 3.5 Lightning on the talos B70 (router alias
    `lightning`, ~100 t/s — this seat's diffs stay in the homelab). The serving side carries the
    load-bearing flags (reasoning-budget 3000, temp 0.6, MTP), so a bare alias gets the tuned
    0.91/0.84-review config. Zen-hosted MiniMax m3 then kimi as cloud backups (family-diverse)."""
    primary = _router_executor("lightning", "lightning", privacy=privacy)
    return _failover_slot("lightning", primary, "lightning", ["m3", "kimi"], privacy=privacy)


def build_reviewer_slots(*, privacy: PrivacyTier | None = None) -> list[ReviewerSlot]:
    """The frontier roster, in a stable order: sonnet (anchor) plus the two strongest local
    reviewers on the 2026-08-22 board — Ling 3 (review 0.91) and Gemma 4 (review 0.84 + the
    analyst profile). Each seat is a router-backed failover chain; Lightning covers the ling3
    seat's failover so a hekaton outage degrades to a fast local reviewer, not a cloud seat.
    Every seat sends ``privacy`` (default ``settings.frontier_privacy``) as X-Router-Privacy."""
    tier = privacy or settings.frontier_privacy
    return [_sonnet_slot(privacy=tier), _ling3_slot(privacy=tier), _gemma_slot(privacy=tier)]


def build_local_reviewer_slots(*, privacy: PrivacyTier | None = None) -> list[ReviewerSlot]:
    """The all-local roster: Ling 3 (Ant), Gemma 4 (Google), Lightning (NVIDIA) — three
    distinct families with zero cloud seats, every diff staying in the homelab, and three
    complementary failure modes (slow-but-thorough, few-FP analyst, fast-and-disciplined).
    Serves the ROUTINE lane (see ``roster_for_lane``) and the ``shadow`` pass; the frontier
    roster keeps the gates where sonnet's unique catches clustered in the shadow trial.
    Rewired 2026-08-22 from Lightning/gpt-oss/coder-next on the qualeval v2 board: gpt-oss
    review 0.64 and coder-next 0.67 were the weakest links; both stay reachable as backups.
    Every seat sends ``privacy`` (default ``settings.local_privacy``, i.e. ``local``): the
    cloud backups stay listed but the router refuses them under ``local``, so "every diff
    staying in the homelab" is enforced on the wire, not just by the primaries' choice."""
    tier = privacy or settings.local_privacy
    return [_ling3_slot(privacy=tier), _gemma_slot(privacy=tier), _lightning_slot(privacy=tier)]


def roster_for_lane(lane: str) -> list[ReviewerSlot]:
    """Lane split (2026-08-18, from the 7-run shadow trial): ``"local"`` — the all-local trio,
    for routine work (dependency bumps, config moves, mechanical passes) where it reviewed at
    parity and costs nothing; anything else — the frontier roster, for the merge-blocking gates
    and security-adjacent diffs where sonnet's unique catches clustered. Unknown values fall
    through to frontier on purpose: the expensive-but-safer roster is the fail-closed default."""
    return build_local_reviewer_slots() if lane == "local" else build_reviewer_slots()


# Capability-ordered rotation for the aggregator/digest failover pool. All seats route through the
# router, so ordering is by synthesis capability; a `preferred` seat is promoted to the front.
# After sonnet: Lightning first among locals for synthesis duty (instruction 0.99, disciplined,
# ~100 t/s), then Ling 3 (instruction 1.00, best reviewer, but slow hekaton CPU decode), then
# Gemma (instruction 1.00 but the most verbose thinker — fine work, slow synthesis). Only
# providers actually present in the given slots are used, so this order spans both rosters.
ROTATION_ORDER = ("sonnet-5", "lightning", "ling3", "gemma")


def rotation_pool(slots: list[ReviewerSlot], *, role: str, preferred: str | None = None) -> Pool:
    """A failover Pool over the *active* seats in capability-rotation order.

    Shared by the aggregator (synthesize N reviews) and the digest (one resilient pass). Inactive
    (skipped) seats are excluded; `preferred` (if active) leads, then ROTATION_ORDER fills in. Each
    seat's primary executor is reused (ApiExecutor is stateless), so the pool rotates over the same
    models the ensemble reviewed with.
    """
    active = {s.provider: s for s in slots if s.active}
    order = [preferred] if preferred in active else []
    order += [p for p in ROTATION_ORDER if p in active and p not in order]
    return Pool(role=role, executors=[active[p].pool.executors[0] for p in order])
