"""The discover → dedup → verify recipe — the dynamic-workflow audit shape as a reusable function.

Composes the harness primitives into one orchestration: fan out blind **finders** (``discover``),
merge their overlapping findings into a canonical list via one **consolidator** (``structured``),
then run a perspective-diverse **skeptic panel per finding** and vote (``verify_each``). A consumer
supplies the prompts, the Pydantic schemas, and small accessor callbacks; the orchestration,
failover, and bounded concurrency are handled here. The code-audit ensemble is the first consumer;
testing/refactoring ensembles are thin specializations (different finder prompts + schemas).

Dedup is non-fatal: if the consolidator is down the raw findings are verified directly, so a run
still produces verdicts.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from pydantic import BaseModel

from forge.shared.ensemble import Pool
from forge.shared.panel import (
    Finder,
    ItemVerdict,
    PanelMember,
    PanelResult,
    discover,
    structured,
    verify_each,
)


@dataclass
class RecipeResult[F, V]:
    """The full recipe outcome: every raw finding, the canonical (deduped) list, and the
    per-finding verdicts. ``dedup_ok`` is False when the consolidator failed and ``canonical``
    fell back to raw."""

    raw: list[F]
    canonical: list[F]
    verdicts: list[ItemVerdict[F, V]]
    dedup_ok: bool


def _emit(log: Callable[[str], None] | None, message: str) -> None:
    if log is not None:
        log(message)


def discover_dedup_verify[E: BaseModel, C: BaseModel, F, V](
    *,
    finders: Sequence[Finder],
    finder_pool: Pool,
    finding_schema: type[E],
    findings_of: Callable[[E], list[F]],
    dedup_pool: Pool,
    dedup_system: str,
    build_dedup_user: Callable[[list[F]], str],
    canonical_schema: type[C],
    canonical_of: Callable[[C], list[F]],
    verify_members: Sequence[PanelMember],
    verify_make_user: Callable[[F], str],
    verify_aggregate: Callable[[F, PanelResult], V],
    verify_floor: int = 1,
    concurrency: int = 4,
    max_tokens: int = 4096,
    timeout: float = 120.0,
    log: Callable[[str], None] | None = None,
) -> RecipeResult[F, V]:
    """Run the find → dedup → verify-each pipeline.

    ``finders`` fan out through ``finder_pool`` returning ``finding_schema`` envelopes that
    ``findings_of`` flattens into findings. One ``structured`` consolidator (``dedup_pool`` +
    ``canonical_schema``) merges them and ``canonical_of`` pulls the canonical list out. Then
    ``verify_each`` runs the ``verify_members`` skeptic panel over every canonical finding and
    ``verify_aggregate`` votes each panel into a verdict. Returns everything for the caller to
    render.
    """
    envelopes = discover(
        finders,
        pool=finder_pool,
        schema=finding_schema,
        concurrency=concurrency,
        max_tokens=max_tokens,
        timeout=timeout,
    )
    raw = [item for env in envelopes for item in findings_of(env)]
    _emit(log, f"discover: {len(raw)} raw from {len(envelopes)}/{len(finders)} finders")
    if not raw:
        return RecipeResult(raw=[], canonical=[], verdicts=[], dedup_ok=False)

    dedup = structured(
        pool=dedup_pool,
        schema=canonical_schema,
        system=dedup_system,
        user=build_dedup_user(raw),
        max_tokens=max_tokens,
        timeout=timeout,
    )
    if dedup.value is not None:
        canonical = canonical_of(dedup.value)
        dedup_ok = True
    else:
        canonical = raw  # consolidator down → verify the raw findings directly rather than abort
        dedup_ok = False
    _emit(log, f"dedup: {len(canonical)} canonical finding(s) (dedup_ok={dedup_ok})")

    verdicts = verify_each(
        canonical,
        members=verify_members,
        make_user=verify_make_user,
        aggregate=verify_aggregate,
        floor=verify_floor,
        concurrency=concurrency,
        max_tokens=max_tokens,
        timeout=timeout,
    )
    _emit(log, f"verify: {len(verdicts)} finding(s) verified")
    return RecipeResult(raw=raw, canonical=canonical, verdicts=verdicts, dedup_ok=dedup_ok)
