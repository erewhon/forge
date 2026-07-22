"""``forge radar`` — a living **AI Technology Radar** (ThoughtWorks-style: quadrants ×
Adopt/Trial/Assess/Hold rings) kept in Nous, personal to this stack.

The research engine already exists (:mod:`forge.general_researcher`), the scheduled-task pattern is
proven, and Nous is the surface. So the *fetching* half is mostly assembly. **The hard part is
curation over time** — a radar is a living artifact whose value is the *movement* of a blip and the
*why*, not a fresh weekly dump. That curation is a state problem, and this package is its spine.

The spine, built first (this module), mirrors the switcheroo **baton** primitive — the durable state
layer everything else hangs off:

- :mod:`forge.radar.models` — the domain: :class:`~forge.radar.models.Blip`,
  :class:`~forge.radar.models.Quadrant`, :class:`~forge.radar.models.Ring`, and the
  :class:`~forge.radar.models.Radar` container.
- :mod:`forge.radar.movement` — the **curation discipline** (pure functions). Two mechanisms the
  design deliberately separates:

  1. *Scanners accumulate candidates* — noisy and frequent.
     :func:`~forge.radar.movement.integrate_candidate` folds a re-surfaced candidate into its
     existing blip (refresh, accrete evidence/links) instead of spawning a duplicate. It never
     moves a ring.
  2. *The weekly synthesis moves blips* — rare and evidence-gated.
     :func:`~forge.radar.movement.propose_move` is the only path that changes a ring, and it
     enforces the anti-thrash rules: **movement needs evidence, blips must not thrash** (cooldown +
     reversal guard), and promotions step one ring at a time toward the centre (Assess → Trial →
     Adopt).

- :mod:`forge.radar.store` — persistence behind a small :class:`~forge.radar.store.RadarStore`
  protocol. :class:`~forge.radar.store.JsonRadarStore` is the durable local store (and the test
  backend); :class:`~forge.radar.store.NousRadarStore` projects the radar into the "AI Radar" Nous
  notebook's blip database.

Downstream workstreams (separate tasks under feature "AI Tech Radar") layer on this spine: source
scanners feed :func:`~forge.radar.movement.integrate_candidate`; the weekly
:mod:`forge.general_researcher`-backed synthesis calls :func:`~forge.radar.movement.propose_move`
with stack-personal rationale; a renderer draws the SVG radar; and the action loop turns a fresh
**Trial** promotion into a Forge task suggestion.

Two discipline rules are borrowed straight from :mod:`forge.shared.baton`, because a radar has the
same "never silently lose the *why*" hazard:

- **Accretion.** ``rationale`` and ``action`` are *current state* and may be rewritten every cycle;
  ``evidence`` and ``links`` **accrete** (union on write) so the trail of why a blip moved is never
  dropped.
- **An explicit override, not a silent one.** The anti-thrash guards refuse a too-soon or
  reversing move unless the caller passes ``force=True`` — the "yes, I really mean it" escape hatch.
"""
