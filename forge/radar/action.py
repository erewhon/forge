"""The action loop — the piece that makes the radar *cause* work instead of just informing.

When a blip sits at **Trial**, the stack has decided it's worth a real hands-on evaluation. This
turns that verdict into a concrete, pre-filled Forge task ("Trial X in a gaol sandbox: …") so it
actually gets scheduled rather than forgotten. That Forge tie-in is what makes the radar more than a
newsletter.

Two deliberate constraints, matching the spec:

- **Suggestion / manual confirm.** By default ``forge radar act`` only *lists* what it would file —
  it writes nothing. Filing happens on an explicit ``--file`` (optionally ``--blip NAME`` for one).
  The emitted task itself is doubly gated: :func:`forge.shared.forge_emit.emit_task` defaults it to
  ``status='Spec Needed'`` + ``execution_mode='Manual'``, so even once filed a human flips it to
  Ready before the autonomous worker can touch it. No backlog noise is auto-created.
- **Idempotent, with a visible back-link.** Each trial has a stable ``external_ref``
  (``radar:trial:{slug}``); ``emit_task`` skips a ref that already exists, so re-running never
  double-files — the Forge task DB is the source of truth for "already actioned", not a flag on the
  blip. After filing, the blip's ``action`` column is set to a back-link so the radar shows which
  trials are in flight.
"""

from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel, Field

from forge.radar.models import Blip, Evidence, Radar, Ring

#: Where radar-spawned trials land by default. A dedicated project keeps exploratory trials out of
#: real product backlogs; override with ``--project``.
DEFAULT_TRIAL_PROJECT = "Radar Trials"

#: Feature grouping for the emitted tasks.
TRIAL_FEATURE = "AI Tech Radar"

#: Prefix of the back-link written to a blip's ``action`` once its trial task is filed. Used to
#: detect an already-back-linked blip so re-runs don't rewrite it.
_FILED_PREFIX = "Trial task filed"


def external_ref(blip: Blip) -> str:
    """The stable dedup key for *blip*'s trial task. One per blip, so filing is idempotent."""
    return f"radar:trial:{blip.slug}"


class TrialSuggestion(BaseModel):
    """A proposed Forge task for one Trial-ring blip."""

    blip: Blip
    project: str
    title: str
    content: str
    ref: str

    def preview(self) -> str:
        return f"- {self.title}\n    ref: {self.ref}  → project «{self.project}»"


def _trial_content(blip: Blip) -> str:
    lines = [
        f"**Trial candidate from the AI Radar** — {blip.name} "
        f"({blip.quadrant.value}, ring: {blip.ring.value})",
        "",
        blip.rationale or "_No rationale recorded._",
    ]
    if blip.action:
        lines += ["", f"**Suggested trial:** {blip.action}"]
    if blip.links:
        lines += ["", "**Links:**", *[f"- {link}" for link in blip.links]]
    if blip.evidence:
        lines += ["", "**Evidence:**", *[f"- {e.date}: {e.note}" for e in blip.evidence]]
    lines += [
        "",
        "---",
        "_Filed by the AI Radar action loop. This is a suggestion: flip to **Ready** to schedule "
        "the trial (e.g. stand it up in a gaol sandbox and evaluate on the stack)._",
    ]
    return "\n".join(lines)


def build_suggestion(blip: Blip, *, project: str = DEFAULT_TRIAL_PROJECT) -> TrialSuggestion:
    """The proposed trial task for *blip*, pre-filled from rationale, action, links, evidence."""
    return TrialSuggestion(
        blip=blip,
        project=project,
        title=f"Trial {blip.name} in a gaol sandbox",
        content=_trial_content(blip),
        ref=external_ref(blip),
    )


def pending_trials(radar: Radar, existing_refs: set[str]) -> list[Blip]:
    """Trial-ring blips whose trial task hasn't been filed yet (its ``external_ref`` is absent from
    the Forge task DB). Sorted by name for stable output."""
    return sorted(
        (b for b in radar.blips if b.ring == Ring.TRIAL and external_ref(b) not in existing_refs),
        key=lambda b: b.name.lower(),
    )


class FiledTrial(BaseModel):
    """The outcome of acting on one suggestion."""

    name: str
    title: str
    ref: str
    status: str  #: "created" | "skipped" | "dry-run"
    detail: str = ""


class ActionResult(BaseModel):
    project: str
    dry_run: bool = True
    filed: list[FiledTrial] = Field(default_factory=list)

    @property
    def created(self) -> int:
        return sum(1 for f in self.filed if f.status == "created")

    def render(self) -> str:
        if not self.filed:
            return "No Trial blips awaiting a task — nothing to file."
        if self.dry_run:
            lines = [
                f"{len(self.filed)} Trial blip(s) would be filed into «{self.project}» "
                "(run with --file to file them):"
            ]
            lines += [f"  {f.title}\n    ref {f.ref}" for f in self.filed]
            return "\n".join(lines)
        lines = [f"{self.created} trial task(s) filed into «{self.project}»:"]
        lines += [f"  [{f.status}] {f.title} ({f.ref})" for f in self.filed]
        return "\n".join(lines)


#: The emit callable's shape (``forge.shared.forge_emit.emit_task``), injectable for tests.
EmitFn = Callable[..., object]


def act(
    radar: Radar,
    *,
    project: str = DEFAULT_TRIAL_PROJECT,
    today,
    only: str | None = None,
    dry_run: bool = True,
    emit_fn: EmitFn | None = None,
    ensure_project_fn: Callable[[str], None] | None = None,
    existing_refs_fn: Callable[[], set[str]] | None = None,
) -> ActionResult:
    """File Forge trial tasks for pending Trial blips (default ``dry_run`` — list only, no writes).

    Mutates *radar* in place on a real run: each filed blip's ``action`` becomes a back-link and an
    evidence entry records the filing. The ``*_fn`` hooks default to the real
    :mod:`forge.shared.forge_emit` functions; tests inject fakes so no Forge/daemon is touched.
    """
    if emit_fn is None or ensure_project_fn is None or existing_refs_fn is None:
        from forge.shared.forge_emit import emit_task, ensure_project, existing_external_refs

        emit_fn = emit_fn or emit_task
        ensure_project_fn = ensure_project_fn or ensure_project
        existing_refs_fn = existing_refs_fn or existing_external_refs

    refs = existing_refs_fn()
    trials = sorted(
        (b for b in radar.blips if b.ring == Ring.TRIAL), key=lambda b: b.name.lower()
    )
    if only is not None:
        want = only.strip().lower()
        trials = [b for b in trials if b.name.lower() == want or b.slug == want]

    result = ActionResult(project=project, dry_run=dry_run)

    # Dry run: list only the trials that would be *newly* filed (ref not yet in Forge).
    if dry_run:
        for blip in trials:
            if external_ref(blip) not in refs:
                sug = build_suggestion(blip, project=project)
                result.filed.append(
                    FiledTrial(name=blip.name, title=sug.title, ref=sug.ref, status="dry-run")
                )
        return result

    if not trials:
        return result
    ensure_project_fn(project)

    stamp = today.isoformat()
    for blip in trials:
        sug = build_suggestion(blip, project=project)
        if sug.ref in refs:
            status, detail = "skipped", "already filed"
        else:
            outcome = emit_fn(
                project=project,
                title=sug.title,
                content=sug.content,
                external_ref=sug.ref,
                feature=TRIAL_FEATURE,
                tags="radar,trial",
                task_type="chore",
                priority=6,
                dry_run=False,
                existing_refs=refs,
            )
            status = getattr(outcome, "action", "created")
            detail = getattr(outcome, "detail", "")

        # Back-link every filed trial (created now or previously) that lacks the marker — so the
        # radar shows which trials are in flight, and a task filed on an earlier run self-heals.
        if status in ("created", "skipped") and not blip.action.startswith(_FILED_PREFIX):
            blip.action = f"{_FILED_PREFIX}: {sug.title}"
            blip.evidence.append(
                Evidence(date=stamp, note=f"Trial task filed ({sug.ref})", source="action-loop")
            )

        result.filed.append(
            FiledTrial(name=blip.name, title=sug.title, ref=sug.ref, status=status, detail=detail)
        )

    return result
