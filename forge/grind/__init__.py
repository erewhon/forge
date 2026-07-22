"""`forge grind` — a diagnostic iterate-on-a-goal loop.

Grind runs a user-defined *experiment cycle* (reset → load → run → check) over and over against a
day's goal, letting a model adjust the code between turns — **without ever committing**. It is the
diagnostic sibling of the coding-pipeline wave loop: same loop machinery (a no-progress guard,
lessons, a journal) but none of the VCS/commit/gate spine. Commit-less checkpointing and rollback
ride on the **jj operation log**; a machine-checkable done-signal doubles as a fitness score, so the
loop hill-climbs — keep the best iteration, ``jj op restore`` a regression.
"""
