"""Weekly synthesis — the brain. Turns the accumulated candidate feed into radar movement and a
"what moved and why" digest.

This is where signal-vs-noise is actually decided. The scanners are a dumb wide net; the synthesis
is a structured LLM judge that reads the whole week's feed against *this* stack (the euclid LLM
router, OpenCode, gaol sandboxes, the ensemble / coding-pipeline harnesses) and, per candidate,
decides:

- **keep or drop** — is this relevant to how *we* build, or generic AI news?
- a clean **entity name** — extracted from the headline (a Hacker News title is not an entity), so
  a blip is "Structured tool-calling", not "Show HN: I added tool calling to my app";
- a **quadrant** and a **ring**, with new/unproven things defaulting to Assess;
- a **stack-personal rationale** — "worth trialing on the euclid router because Y", not a summary.

Placements are then applied through the movement discipline (:mod:`forge.radar.movement`): a
candidate matching an existing blip is a disciplined :func:`~forge.radar.movement.propose_move`
(evidence-gated, thrash-guarded); a new entity is placed directly at its judged ring. Every judged
candidate is then consumed from the feed. The digest reports what moved and why.

The judge uses the shared router-backed :func:`forge.shared.llm.complete` (the same primitive the
ensembles and graders use), not the heavyweight :mod:`forge.general_researcher` harness — that
harness produces prose, and this needs reliable structured placements. ``general_researcher`` is
instead wired as an opt-in **deep-dive** (:func:`deep_dive`): for candidates the judge wants to move
to Trial/Adopt, a real research pass gathers evidence before the move is committed.
"""

from __future__ import annotations

import os
from collections.abc import Callable

from pydantic import BaseModel, Field, ValidationError

from forge.radar.candidates import CandidateFeed, FeedEntry
from forge.radar.models import Blip, Evidence, Quadrant, Radar, Ring
from forge.radar.movement import DEFAULT_COOLDOWN_DAYS, propose_move
from forge.shared.llm import LLMConfig, extract_json

#: The LLM ``complete`` shape, injectable so the judge is tested without a live router.
CompleteFn = Callable[..., str]

#: Injectable deep-dive: ``(name, rationale) -> evidence note`` (or "" when it finds nothing).
DeepDiveFn = Callable[[str, str], str]

#: How many candidates to judge in one pass. The feed is bounded (synthesis consumes it), but this
#: caps a pathological backlog so one call stays in-context.
DEFAULT_MAX_CANDIDATES = 60


# ---------------------------------------------------------------------------
# LLM config
# ---------------------------------------------------------------------------


def default_llm() -> tuple[LLMConfig, str]:
    """The router config + model alias for the judge, env-overridable. Defaults to a strong,
    instruction-following judgment model: ``glm`` reliably emits the batch JSON (live-checked
    2026-07-22 — ``research`` rambled and returned no JSON at scale, ``glm``/``kimi-k2.7`` parsed
    25/25). Point ``RADAR_ROUTER_URL`` at the LiteLLM router (e.g. http://euclid:4010/v1)."""
    cfg = LLMConfig(
        backend="openai",
        openai_base_url=os.environ.get("RADAR_ROUTER_URL", "http://localhost:4000/v1"),
        openai_api_key=os.environ.get("RADAR_ROUTER_KEY", "sk-litellm-master"),
    )
    model = os.environ.get("RADAR_SYNTH_MODEL", "glm")
    return cfg, model


# ---------------------------------------------------------------------------
# The judge
# ---------------------------------------------------------------------------

_STACK = (
    "The stack this radar is personal to: a self-hosted LLM router on the 'euclid' host (LiteLLM, "
    "OpenAI-compatible, model aliases); OpenCode as the local coding agent; 'gaol' Linux "
    "container/VM sandboxes; and a suite of multi-model 'ensemble' + coding-pipeline harnesses "
    "(forge). Version control is jj (Jujutsu); notes/tasks live in Nous. The radar centers on what "
    "changes how *this* stack builds — local models, agent frameworks, harness/eval techniques, "
    "inference/routing/tooling — not generic AI news."
)

_QUADRANTS = (
    "Quadrants: 'Models' (LLMs, weights, quants); 'Agents & Frameworks' (agent frameworks, SDKs, "
    "MCP servers, orchestration); 'Techniques' (prompting, RAG, eval/harness methods, "
    "fine-tuning); 'Infra/Tooling' (inference servers, routers, gateways, dev tooling)."
)

_RINGS = (
    "Rings: 'Adopt' (proven on our stack, a default); 'Trial' (worth a real hands-on trial on our "
    "stack now, with a concrete reason); 'Assess' (worth understanding / watching, not yet "
    "trialing); 'Hold' (avoid / not worth pursuing / superseded). New or unproven things "
    "default to Assess — only propose Trial or Adopt with a concrete stack-specific reason."
)


def build_judge_messages(entries: list[FeedEntry], radar: Radar) -> tuple[str, str]:
    """The (system, user) messages for one judge pass."""
    system = (
        "You curate a personal AI Technology Radar (ThoughtWorks-style: quadrants × Adopt/Trial/"
        "Assess/Hold). "
        + _STACK
        + " "
        + _QUADRANTS
        + " "
        + _RINGS
        + " You are given the week's raw candidate signals (noisy — headlines, model listings, "
        "repos, papers) and the radar's current blips. For each candidate decide whether it "
        "belongs on THIS radar. Drop generic AI news, duplicates of existing blips (unless a ring "
        "change is warranted), and anything irrelevant to the stack. For each candidate you keep, "
        "give a clean entity name (the technology itself, not the headline), a quadrant, a ring, "
        "and a rationale "
        "phrased as a stack-personal recommendation ('worth trialing on the euclid router because "
        "…'), plus a short concrete action if any. Respond with STRICT JSON and nothing else: "
        '{"placements": [{"key": "<the candidate key verbatim>", "keep": true|false, '
        '"name": "<entity name>", "quadrant": "<quadrant>", "ring": "<ring>", '
        '"rationale": "<why, stack-personal>", "action": "<optional next step>"}]}. '
        "Include one object per candidate, echoing its key exactly. For keep=false, name/quadrant/"
        "ring may be empty."
    )

    current = (
        "\n".join(f"- {b.name} [{b.quadrant.value} / {b.ring.value}]" for b in radar.blips)
        or "(none yet)"
    )

    cand_lines = []
    for e in entries:
        bits = [f"key={e.key}", f"source={e.source}", f"title={e.title!r}"]
        if e.summary:
            bits.append(f"summary={e.summary[:240]!r}")
        if e.score is not None:
            bits.append(f"score={e.score:g}")
        if e.quadrant_hint:
            bits.append(f"hint={e.quadrant_hint.value}")
        cand_lines.append("- " + ", ".join(bits))

    user = (
        f"CURRENT BLIPS:\n{current}\n\n"
        f"CANDIDATES ({len(entries)}):\n" + "\n".join(cand_lines) + "\n\n"
        "Return the JSON object now."
    )
    return system, user


class Placement(BaseModel):
    """One judged candidate — the structured decision the judge returns."""

    key: str  #: Echoes the candidate's ``source:external_id``.
    keep: bool
    name: str = ""
    quadrant: Quadrant | None = None
    ring: Ring | None = None
    rationale: str = ""
    action: str = ""

    def is_actionable(self) -> bool:
        """A kept placement with enough to create/move a blip."""
        return self.keep and bool(self.name) and self.quadrant is not None and self.ring is not None


def _coerce_enum(enum_cls, value):
    """Match a label to an enum case-insensitively; ``None`` when it doesn't match — a hallucinated
    quadrant/ring drops the placement rather than raising."""
    if not value:
        return None
    for member in enum_cls:
        if member.value.lower() == str(value).strip().lower():
            return member
    return None


def parse_placements(raw: dict, entries: list[FeedEntry]) -> list[Placement]:
    """Validate the judge's JSON into placements, keeping only those whose key matches a fed
    candidate. Tolerant of hallucinated enums and missing fields."""
    keys = {e.key for e in entries}
    out: list[Placement] = []
    for item in raw.get("placements", []):
        if not isinstance(item, dict):
            continue
        key = str(item.get("key", "")).strip()
        if key not in keys:
            continue
        try:
            out.append(
                Placement(
                    key=key,
                    keep=bool(item.get("keep", False)),
                    name=str(item.get("name", "")).strip(),
                    quadrant=_coerce_enum(Quadrant, item.get("quadrant")),
                    ring=_coerce_enum(Ring, item.get("ring")),
                    rationale=str(item.get("rationale", "")).strip(),
                    action=str(item.get("action", "")).strip(),
                )
            )
        except ValidationError:
            continue
    return out


#: Candidates per judge call. The response carries a full rationale per candidate, so a large batch
#: overruns ``max_tokens`` and the JSON truncates (live-observed at ~30) — chunk so each call's
#: output stays whole. Each chunk still sees the current blips, so dedup-against-blips holds across
#: chunks.
DEFAULT_CHUNK_SIZE = 15


def judge_candidates(
    entries: list[FeedEntry],
    radar: Radar,
    *,
    complete_fn: CompleteFn,
    cfg: LLMConfig,
    model: str,
    max_tokens: int = 8192,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> list[Placement]:
    """Judge *entries* in chunks of ``chunk_size`` (one model call each, so a big feed can't
    truncate the JSON), returning all placements. ``complete_fn`` is injectable (the shared
    :func:`forge.shared.llm.complete` in production, a fake in tests)."""
    placements: list[Placement] = []
    for start in range(0, len(entries), max(1, chunk_size)):
        chunk = entries[start : start + max(1, chunk_size)]
        system, user = build_judge_messages(chunk, radar)
        text = complete_fn(
            cfg, system=system, user_message=user, model=model, max_tokens=max_tokens
        )
        placements.extend(parse_placements(_extract_json_object(text), chunk))
    return placements


def _extract_json_object(text: str) -> dict:
    """Robustly pull the first balanced ``{...}`` object out of *text*, tolerating leading reasoning
    and trailing chatter (some router models append prose after the JSON — live-observed). Scans
    brace depth while respecting string literals/escapes; falls back to the shared
    :func:`~forge.shared.llm.extract_json` if no balanced object is found."""
    start = text.find("{")
    if start != -1:
        depth = 0
        in_str = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    import json as _json

                    try:
                        return _json.loads(text[start : i + 1])
                    except _json.JSONDecodeError:
                        break
    return extract_json(text)


# ---------------------------------------------------------------------------
# Applying placements through the movement discipline
# ---------------------------------------------------------------------------


class Change(BaseModel):
    """One applied (or refused) placement — the digest's unit."""

    kind: str  #: "create" | "promote" | "demote" | "hold" | "update" | "refused" | "drop" | "noop"
    name: str
    quadrant: Quadrant | None = None
    from_ring: Ring | None = None
    to_ring: Ring | None = None
    rationale: str = ""
    note: str = ""  #: e.g. the refusal reason from the movement guard.


class SynthesisResult(BaseModel):
    judged: int = 0
    kept: int = 0
    changes: list[Change] = Field(default_factory=list)
    deep_dived: list[str] = Field(default_factory=list)


def _entry_for(placements_key: str, entries: list[FeedEntry]) -> FeedEntry | None:
    for e in entries:
        if e.key == placements_key:
            return e
    return None


def apply_placements(
    radar: Radar,
    placements: list[Placement],
    entries: list[FeedEntry],
    *,
    today,
    cooldown_days: int = DEFAULT_COOLDOWN_DAYS,
) -> list[Change]:
    """Apply each placement to *radar* (mutated in place). Existing blips move through the
    disciplined :func:`propose_move`; new entities are placed directly at their judged ring. Returns
    a :class:`Change` per placement for the digest."""
    stamp = today.isoformat()
    changes: list[Change] = []

    for p in placements:
        if not p.keep:
            changes.append(Change(kind="drop", name=p.name or p.key, rationale=p.rationale))
            continue
        if not p.is_actionable():
            changes.append(Change(kind="drop", name=p.name or p.key, note="incomplete placement"))
            continue

        existing = radar.get(p.name)
        if existing is None:
            entry = _entry_for(p.key, entries)
            links = [entry.url] if entry and entry.url else []
            blip = Blip(
                name=p.name,
                quadrant=p.quadrant,
                ring=p.ring,
                first_seen=stamp,
                last_seen=stamp,
                rationale=p.rationale,
                action=p.action,
                links=links,
                evidence=[
                    Evidence(
                        date=stamp,
                        note=f"Placed at {p.ring.value}: {p.rationale}",
                        source="synthesis",
                    )
                ],
            )
            radar.upsert(blip)
            changes.append(
                Change(
                    kind="create",
                    name=p.name,
                    quadrant=p.quadrant,
                    to_ring=p.ring,
                    rationale=p.rationale,
                )
            )
            continue

        if existing.ring == p.ring:
            # Already there — curate in place: refresh the rationale/action and accrete an evidence
            # note, but record it as a no-move update.
            existing.rationale = p.rationale or existing.rationale
            existing.action = p.action or existing.action
            existing.last_seen = stamp
            changes.append(
                Change(
                    kind="update",
                    name=existing.name,
                    quadrant=existing.quadrant,
                    to_ring=existing.ring,
                    rationale=p.rationale,
                )
            )
            continue

        decision = propose_move(
            radar,
            existing.name,
            p.ring,
            p.rationale or "synthesis move",
            today=today,
            source="synthesis",
            cooldown_days=cooldown_days,
        )
        if decision.applied:
            changes.append(
                Change(
                    kind=decision.kind,
                    name=existing.name,
                    quadrant=existing.quadrant,
                    from_ring=decision.from_ring,
                    to_ring=decision.to_ring,
                    rationale=p.rationale,
                )
            )
        else:
            changes.append(
                Change(
                    kind="refused",
                    name=existing.name,
                    quadrant=existing.quadrant,
                    from_ring=decision.from_ring,
                    to_ring=p.ring,
                    rationale=p.rationale,
                    note=decision.reason,
                )
            )
    return changes


# ---------------------------------------------------------------------------
# Digest
# ---------------------------------------------------------------------------

_DIGEST_SECTIONS = [
    ("create", "New blips"),
    ("promote", "Promoted"),
    ("demote", "Demoted"),
    ("hold", "Moved to Hold"),
    ("update", "Reaffirmed"),
    ("refused", "Deferred (movement guard)"),
    ("drop", "Dropped"),
]


def render_digest(result: SynthesisResult, *, today: str) -> str:
    """The weekly "what moved and why" digest, in markdown."""
    lines = [f"# Radar synthesis — {today}", ""]
    lines.append(
        f"Judged {result.judged} candidate(s); kept {result.kept}. "
        f"{sum(1 for c in result.changes if c.kind in ('create', 'promote', 'demote', 'hold'))} "
        "radar change(s)."
    )
    by_kind: dict[str, list[Change]] = {}
    for c in result.changes:
        by_kind.setdefault(c.kind, []).append(c)

    for kind, heading in _DIGEST_SECTIONS:
        items = by_kind.get(kind, [])
        if not items:
            continue
        lines += ["", f"## {heading}"]
        for c in items:
            ring = c.to_ring.value if c.to_ring else ""
            if c.kind in ("promote", "demote", "hold"):
                arrow = f"{c.from_ring.value if c.from_ring else '?'} → {c.to_ring.value}"
                head = f"**{c.name}** ({arrow})"
            elif c.kind == "create":
                head = f"**{c.name}** — {c.quadrant.value if c.quadrant else '?'} / {ring}"
            else:
                head = f"**{c.name}**"
            detail = c.rationale or c.note
            lines.append(f"- {head}" + (f" — {detail}" if detail else ""))
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def deep_dive(name: str, rationale: str, *, max_sprints: int = 1) -> str:
    """Best-effort deep research on a Trial/Adopt-bound candidate via
    :mod:`forge.general_researcher`. Returns a short evidence excerpt, or "" on any failure. Runs
    the real (slow, file-writing) harness — only invoked under ``--deep``."""
    try:
        from forge.general_researcher.main import _topic_dir, run
        from forge.general_researcher.models import TopicConfig

        topic = TopicConfig(
            question=(
                f"Is {name} worth trialing or adopting for our stack (euclid LLM router, OpenCode, "
                f"gaol sandboxes, forge ensemble/coding-pipeline harnesses)? {rationale}"
            ),
            context=_STACK,
        )
        run(topic, max_sprints=max_sprints)
        synth = _topic_dir(topic) / "synthesis.md"
        if synth.is_file():
            return synth.read_text().strip()[:1200]
    except Exception:
        return ""
    return ""


def synthesize(
    radar: Radar,
    feed: CandidateFeed,
    *,
    today,
    complete_fn: CompleteFn,
    cfg: LLMConfig,
    model: str,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    deep: bool = False,
    deep_dive_fn: DeepDiveFn | None = None,
    cooldown_days: int = DEFAULT_COOLDOWN_DAYS,
) -> tuple[SynthesisResult, str]:
    """One synthesis pass over the feed: judge → (optional deep-dive) → apply → prune. Mutates
    *radar* and *feed* in place; returns the result and the rendered digest. Persisting them is the
    caller's job (so ``--dry-run`` can skip it)."""
    entries = feed.entries[:max_candidates]
    stamp = today.isoformat()
    if not entries:
        return SynthesisResult(), render_digest(SynthesisResult(), today=stamp)

    placements = judge_candidates(entries, radar, complete_fn=complete_fn, cfg=cfg, model=model)

    dived: list[str] = []
    if deep:
        dive = deep_dive_fn or deep_dive
        for p in placements:
            if p.is_actionable() and p.ring in (Ring.TRIAL, Ring.ADOPT):
                note = dive(p.name, p.rationale)
                if note:
                    p.rationale = f"{p.rationale} [deep-dive: {note[:280]}]"
                    dived.append(p.name)

    changes = apply_placements(radar, placements, entries, today=today, cooldown_days=cooldown_days)

    # Every judged candidate is consumed from the feed — kept ones became/updated blips, dropped
    # ones were judged out. Only un-judged overflow (beyond max_candidates) remains.
    judged_keys = {p.key for p in placements}
    feed.entries = [e for e in feed.entries if e.key not in judged_keys]

    result = SynthesisResult(
        judged=len(placements),
        kept=sum(1 for p in placements if p.keep),
        changes=changes,
        deep_dived=dived,
    )
    return result, render_digest(result, today=stamp)
