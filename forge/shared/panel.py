"""A structured-output verification panel built on the ensemble harness.

Fan a JSON prompt across N diverse models concurrently and return each member's parsed response
plus a quorum flag; the caller aggregates the responses however it likes (e.g. median-score an
adversarial research-verification panel). This is the shared piece that makes the researchers
harness consumers — they reuse `Pool` / `ApiExecutor` instead of re-implementing fan-out + parse +
quorum. Synchronous (wraps the async harness in ``asyncio.run``) so the existing synchronous
research loops can call it directly.

Two shapes:
- **Uniform** (`run_panel`): every member gets the *same* system prompt — N independent graders,
  median-aggregated to kill single-model bias.
- **Perspective-diverse** (`run_member_panel` + `build_lens_members`): each member gets a *distinct
  lens* system prompt, so the union of their challenges covers orthogonal failure modes instead of
  redundantly flagging the same obvious gap. Diversity catches what redundancy can't.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from pydantic import BaseModel, ValidationError

from agents.shared.ensemble import ApiExecutor, ExecResult, Executor, Pool, Prompt, map_items
from agents.shared.llm import extract_json


@dataclass
class PanelResult:
    responses: list[dict] = field(default_factory=list)  # parsed JSON, one per member that answered
    member_labels: list[str] = field(default_factory=list)  # labels aligned with `responses`
    attempted: int = 0
    quorum_met: bool = False
    # (label, reason) per member that produced no usable response — transport error, timeout, or
    # unparseable JSON. Callers rendering a quorum miss can say WHY a seat is absent.
    failures: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class PanelMember:
    """One panel seat: an executor paired with the (possibly lens-specialised) system prompt it
    runs. ``label`` is for reporting only — it defaults to the executor's own label."""

    executor: Executor
    system: str
    label: str | None = None


def build_router_executors(
    models: Sequence[str], *, base_url: str, api_key: str
) -> list[ApiExecutor]:
    """One OpenAI-compatible (router) executor per model name — the panel members."""
    return [
        ApiExecutor(label=f"router:{m}", kind="openai", model=m, base_url=base_url, api_key=api_key)
        for m in models
    ]


def build_lens_members(
    lenses: Sequence[tuple[str, str]],
    models: Sequence[str],
    *,
    base_url: str,
    api_key: str,
    base_system: str,
) -> list[PanelMember]:
    """One panel member per lens — each gets ``base_system`` followed by its lens directive, with
    router models assigned round-robin so the panel is diverse in *both* viewpoint and vendor.

    With 5 lenses and 3 models the members cycle models[0], models[1], models[2], models[0],
    models[1] — every lens still scores all dimensions, but each hunts its own failure mode hardest.
    """
    if not models:
        return []
    members: list[PanelMember] = []
    for i, (name, directive) in enumerate(lenses):
        model = models[i % len(models)]
        executor = ApiExecutor(
            label=f"router:{model}", kind="openai", model=model, base_url=base_url, api_key=api_key
        )
        system = f"{base_system}\n\n{directive}".strip() if directive else base_system
        members.append(PanelMember(executor=executor, system=system, label=f"{model}/{name}"))
    return members


async def _run_member_panel(
    members: Sequence[PanelMember], user: str, *, floor: int, max_tokens: int, timeout: float
) -> PanelResult:
    async def _one(member: PanelMember) -> ExecResult:
        role = f"panel:{member.label or member.executor.label}"
        pool = Pool(role=role, executors=[member.executor])
        prompt = Prompt(system=member.system, user=user, max_tokens=max_tokens)
        # Parse failure is retryable, same as structured(): a member that answers with prose
        # gets one more try before it costs the panel a seat. (extract_json returns a falsy {}
        # on a miss, never None.)
        return await pool.run(
            prompt, timeout=timeout, validate=lambda text: bool(extract_json(text))
        )

    results = await asyncio.gather(*(_one(m) for m in members))
    responses: list[dict] = []
    labels: list[str] = []
    failures: list[tuple[str, str]] = []
    for member, result in zip(members, results):
        label = member.label or result.executor
        if not result.ok:
            reason = result.error or "no response"
            if reason == "output failed validation":  # Pool's demotion label for a parse miss
                reason = "responded but returned no parseable JSON (retried)"
            failures.append((label, reason))
            continue
        data = extract_json(result.output)
        if data:  # transport-ok but unparseable JSON is dropped, like a failed member
            responses.append(data)
            labels.append(label)
        else:
            failures.append((label, "responded but returned no parseable JSON"))
    return PanelResult(
        responses=responses,
        member_labels=labels,
        attempted=len(members),
        quorum_met=len(responses) >= floor,
        failures=failures,
    )


def run_member_panel(
    *,
    members: Sequence[PanelMember],
    user: str,
    floor: int,
    max_tokens: int = 4096,
    timeout: float = 120.0,
) -> PanelResult:
    """Fan a per-member (lens-specialised) prompt set across the panel; return parsed responses +
    quorum. Each member runs its own system prompt against the shared ``user`` message. Members
    that error or return unparseable JSON are dropped; ``quorum_met`` is whether at least ``floor``
    members produced a usable response.
    """
    return asyncio.run(
        _run_member_panel(members, user, floor=floor, max_tokens=max_tokens, timeout=timeout)
    )


def run_panel(
    *,
    executors: Sequence[Executor],
    system: str,
    user: str,
    floor: int,
    max_tokens: int = 4096,
    timeout: float = 120.0,
) -> PanelResult:
    """Uniform panel: every member runs the same ``system`` prompt. Thin wrapper over
    ``run_member_panel`` for the N-independent-graders case."""
    members = [PanelMember(executor=e, system=system) for e in executors]
    return run_member_panel(
        members=members, user=user, floor=floor, max_tokens=max_tokens, timeout=timeout
    )


# --- single-pool structured output --------------------------------------------------------------
# A panel is N structured calls aggregated; this is the one-pool primitive underneath. It runs a
# failover Pool and returns a *validated Pydantic model*, turning a caller's hand-rolled
# extract_json + .get() into a typed payload — the schema itself acts as Pool.run's validator.


@dataclass
class StructuredResult[T: BaseModel]:
    """The outcome of a :func:`structured` call: the validated model in ``value``, or ``None`` if
    the pool was exhausted without a parseable, schema-valid response. ``result`` is the underlying
    ExecResult (label, attempts, error) for logging and fallback messaging."""

    value: T | None
    result: ExecResult

    @property
    def ok(self) -> bool:
        return self.value is not None

    @property
    def error(self) -> str | None:
        return self.result.error

    @property
    def raw(self) -> str:
        """The model's last raw output — including a payload that failed schema validation.

        ``value`` is None when nothing parsed; ``raw`` is what the model actually emitted, so a
        caller can log *why* it failed. ``Pool.run`` preserves the last attempt's text when it
        demotes an unparseable-but-transport-OK response, so this survives pool exhaustion."""
        return self.result.output


def _parse_model[T: BaseModel](
    schema: type[T], text: str, predicate: Callable[[T], bool] | None
) -> T | None:
    """extract → validate against ``schema`` → optional semantic ``predicate``. None on any miss."""
    data = extract_json(text)
    if not data:
        return None
    try:
        model = schema.model_validate(data)
    except ValidationError:
        return None
    if predicate is not None and not predicate(model):
        return None
    return model


async def _structured[T: BaseModel](
    pool: Pool,
    schema: type[T],
    prompt: Prompt,
    *,
    predicate: Callable[[T], bool] | None,
    timeout: float,
) -> StructuredResult[T]:
    result = await pool.run(
        prompt,
        timeout=timeout,
        validate=lambda text: _parse_model(schema, text, predicate) is not None,
    )
    value = _parse_model(schema, result.output, predicate) if result.ok else None
    return StructuredResult(value=value, result=result)


def structured[T: BaseModel](
    *,
    pool: Pool,
    schema: type[T],
    system: str,
    user: str,
    max_tokens: int = 4096,
    timeout: float = 120.0,
    predicate: Callable[[T], bool] | None = None,
) -> StructuredResult[T]:
    """Run a failover ``Pool`` and parse its output into a validated ``schema`` instance.

    The schema (plus an optional ``predicate`` for semantic checks a schema can't express, e.g. "the
    index is in range") *is* the validator: output that doesn't extract + validate is demoted to a
    transient failure inside :meth:`Pool.run`, so the same model is retried and then failed over —
    exactly like a 5xx — until it yields a usable payload. Returns a :class:`StructuredResult` whose
    ``value`` is the parsed model, or ``None`` if the whole pool was exhausted. Synchronous (wraps
    the async harness in ``asyncio.run``) to match ``run_panel`` and the synchronous agent loops.
    """
    prompt = Prompt(system=system, user=user, max_tokens=max_tokens)
    return asyncio.run(_structured(pool, schema, prompt, predicate=predicate, timeout=timeout))


# --- per-item adversarial verification ----------------------------------------------------------
# verify_each maps a skeptic *panel* over a runtime-discovered list — each item gets its own
# perspective-diverse panel — then an aggregate/vote turns each panel into a verdict. This is the
# discover→verify-each half of the dynamic-workflow recipe; `aggregate` is the vote (count "real"
# votes → confirmed/tentative/rejected, median scores, etc.). Its first production consumer is the
# next ensemble agent (testing/refactoring); for now it stands on tests + a smoke.


@dataclass
class ItemVerdict[I, V]:
    """One item paired with the skeptic panel that judged it and the verdict the panel voted to."""

    item: I
    panel: PanelResult
    verdict: V


async def _verify_each[I, V](
    items: Sequence[I],
    *,
    members: Sequence[PanelMember],
    make_user: Callable[[I], str],
    aggregate: Callable[[I, PanelResult], V],
    floor: int,
    concurrency: int,
    max_tokens: int,
    timeout: float,
) -> list[ItemVerdict[I, V]]:
    async def _one(item: I) -> ItemVerdict[I, V]:
        panel = await _run_member_panel(
            members, make_user(item), floor=floor, max_tokens=max_tokens, timeout=timeout
        )
        return ItemVerdict(item=item, panel=panel, verdict=aggregate(item, panel))

    return await map_items(items, _one, concurrency=concurrency)


def verify_each[I, V](
    items: Sequence[I],
    *,
    members: Sequence[PanelMember],
    make_user: Callable[[I], str],
    aggregate: Callable[[I, PanelResult], V],
    floor: int = 1,
    concurrency: int = 4,
    max_tokens: int = 4096,
    timeout: float = 120.0,
) -> list[ItemVerdict[I, V]]:
    """Run the skeptic ``members`` panel over each item independently — at most ``concurrency``
    items in flight — then ``aggregate`` each panel into a verdict.

    The same ``members`` are reused across items (executors are stateless); ``make_user`` builds
    each item's prompt and ``aggregate`` is the vote. Results stay aligned to ``items``. Synchronous
    (wraps ``asyncio.run``) to match ``run_panel``. Each item fans out to ``len(members)`` calls, so
    the live call count is up to ``concurrency × len(members)`` — size ``concurrency`` accordingly.
    """
    return asyncio.run(
        _verify_each(
            items,
            members=members,
            make_user=make_user,
            aggregate=aggregate,
            floor=floor,
            concurrency=concurrency,
            max_tokens=max_tokens,
            timeout=timeout,
        )
    )


# --- typed multi-finder discovery ---------------------------------------------------------------
# discover fans out blind *finders* (each its own system+user prompt, sharing a failover pool) and
# returns each finder's validated envelope — the find stage of the discover→dedup→verify recipe.


@dataclass
class Finder:
    """One discovery seat: a labelled (system, user) prompt run through the shared finder pool."""

    label: str
    system: str
    user: str


async def _discover[E: BaseModel](
    finders: Sequence[Finder],
    *,
    pool: Pool,
    schema: type[E],
    concurrency: int,
    max_tokens: int,
    timeout: float,
) -> list[E]:
    async def _one(finder: Finder) -> E | None:
        res = await _structured(
            pool,
            schema,
            Prompt(system=finder.system, user=finder.user, max_tokens=max_tokens),
            predicate=None,
            timeout=timeout,
        )
        return res.value

    results = await map_items(finders, _one, concurrency=concurrency)
    return [r for r in results if r is not None]


def discover[E: BaseModel](
    finders: Sequence[Finder],
    *,
    pool: Pool,
    schema: type[E],
    concurrency: int = 4,
    max_tokens: int = 4096,
    timeout: float = 120.0,
) -> list[E]:
    """Fan out blind finders (each its own prompt, all through the shared failover ``pool``) and
    return the validated envelope from each finder that produced one — bounded to ``concurrency`` in
    flight. The find stage of the discover→dedup→verify recipe; the caller flattens the envelopes'
    finding lists. A finder that produced nothing usable is simply absent from the result.
    """
    return asyncio.run(
        _discover(
            finders,
            pool=pool,
            schema=schema,
            concurrency=concurrency,
            max_tokens=max_tokens,
            timeout=timeout,
        )
    )
