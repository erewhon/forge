"""``forge switcheroo`` — the manual-gated Claude-outage failover orchestrator.

When Claude is down, don't improvise open-ended work on weaker local models — **drain the
worker-shaped queue**. The coding pipeline already decomposes work into *routine leaves* an
auto-tier model can run, the task worker already stands each leaf up under OpenCode against the
local router, and the :mod:`forge.shared.baton` primitive already makes the interactive session's
state durable. Switcheroo is the thin orchestrator that brackets a drain of those leaves between a
baton refresh and a **failover journal**, so a later switch-back can reconcile.

The design center is a **manual gate** (``forge switcheroo now``): on a 529 the human checks
status.claude.com and decides — false-positive auto-bailing on a transient blip is worse than
waiting. Auto-detection scoped to unattended waves is a separate, later task.

Two deliberate scope calls:

- **Leaves, not the thread.** The fleet only picks up tasks the worker gate already blesses (Ready
  ∧ Auto-OK/Auto-Preferred ∧ unblocked) — well-specified, self-contained work. The open-ended
  conversation waits for Claude to come back; that is what the baton is *for* (switch-back), not
  what the fleet resumes. So the baton is the failover's **anchor and record**, and its
  :func:`~forge.shared.baton.render_baton_preamble` seeds the *other* two consumers (overnight
  resume, A/B), not these leaf workers.
- **Cross-repo by nature.** Drained leaves resolve their own checkouts, so a window spans many
  repos. The baton + journal live in one "home" ``.forge/`` (where the session was); the home-repo
  ``jj diff`` only sees home-repo work, so the **journal's per-leaf commit ids** are what let
  switch-back find what the fleet landed elsewhere.
"""
