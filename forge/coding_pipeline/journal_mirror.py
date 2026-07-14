"""Mirror the epic's run dir into the target repo under ``refs/pipeline/<epic-slug>``.

The pipeline's decision history — ``pipeline-runs/<epic>/journal.jsonl`` plus wave records,
framing, and tree — lives in the orchestrator machine's checkout, stranded from the repo it
describes. This mirrors it INTO the target repo as an **append-only commit chain**: each wave
appends one commit whose tree snapshots the run dir, parented on the previous tip. That is
git-bug's op-log-under-a-custom-ref pattern — ``git log refs/pipeline/<epic>`` IS the epic
timeline, it travels with every clone, and sprinkles can render it. The three deps-v2 pipeline
fixes were all diagnosed from journal.jsonl alone; this keeps that debugging substrate alive
beyond the orchestrator box, and makes resume-from-a-fresh-clone possible (see hydrate).

Pure git plumbing (``hash-object`` / ``mktree`` / ``commit-tree`` / ``update-ref``): no working
copy checkout and no interaction with jj's view of the repo — safe in a jj-colocated repo, same
assumption the gate/leaf notes already make. Best-effort: a mirror failure warns and is skipped,
never failing a wave. The local run dir stays write-primary; the ref is a one-way mirror (the
single-orchestrator dispatch lock means there is never a second writer to reconcile).
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable
from pathlib import Path

#: Custom-ref hierarchy for the mirrored decision history, pushed alongside refs/notes/pipeline/*.
PIPELINE_REF_PREFIX = "refs/pipeline"

_TIMEOUT = 30

# commit-tree needs a committer identity; a jj-colocated repo often leaves git's user.* unset
# (jj owns identity), so supply a stable pipeline identity via the environment. Commit dates are
# git-native audit metadata (the timeline), not the deterministic JSON-payload timestamps the
# notes plumbing deliberately avoids generating.
_IDENT_ENV = {
    "GIT_AUTHOR_NAME": "forge-pipeline",
    "GIT_AUTHOR_EMAIL": "pipeline@forge.local",
    "GIT_COMMITTER_NAME": "forge-pipeline",
    "GIT_COMMITTER_EMAIL": "pipeline@forge.local",
}


class MirrorError(RuntimeError):
    """A git plumbing step failed while mirroring or hydrating the run dir."""


def epic_ref(epic_slug: str) -> str:
    """The mirror ref for an epic. ``epic_slug`` is a validated ``[a-z0-9-]`` slug, so it is a
    safe ref path component."""
    return f"{PIPELINE_REF_PREFIX}/{epic_slug}"


def _git(repo: Path, *args: str, check: bool = True, stdin: str | None = None) -> str:
    res = subprocess.run(
        ["git", *args],
        cwd=str(repo),
        capture_output=True,
        text=True,
        input=stdin,
        timeout=_TIMEOUT,
        env={**os.environ, **_IDENT_ENV},
    )
    if check and res.returncode != 0:
        raise MirrorError(
            f"git {' '.join(args)} failed (exit {res.returncode}): {res.stderr.strip()}"
        )
    return res.stdout.strip()


def _git_bytes(repo: Path, *args: str) -> bytes:
    res = subprocess.run(["git", *args], cwd=str(repo), capture_output=True, timeout=_TIMEOUT)
    if res.returncode != 0:
        raise MirrorError(
            f"git {' '.join(args)} failed: {res.stderr.decode(errors='replace').strip()}"
        )
    return res.stdout


def mirror_run_dir(
    repo: Path,
    run_dir: Path,
    epic_slug: str,
    *,
    message: str,
    log: Callable[[str], None] = print,
) -> str | None:
    """Append one commit snapshotting *run_dir*'s top-level files to ``refs/pipeline/<epic>``.

    Returns the new commit sha, or ``None`` when nothing was mirrored (no run dir, no files) or a
    plumbing step failed — best-effort, never raises. The commit parents the current ref tip, so
    the chain is append-only; subdirectories are skipped (the decision history is the flat set of
    journal/wave/framing/tree files).
    """
    if not run_dir.is_dir():
        return None
    files = sorted(p for p in run_dir.iterdir() if p.is_file())
    if not files:
        return None
    ref = epic_ref(epic_slug)
    try:
        tree_lines = []
        for path in files:
            blob = _git(repo, "hash-object", "-w", "--", str(path))
            tree_lines.append(f"100644 blob {blob}\t{path.name}")
        tree = _git(repo, "mktree", stdin="\n".join(tree_lines) + "\n")
        parent = _git(repo, "rev-parse", "--verify", "--quiet", ref, check=False)
        commit_args = ["commit-tree", tree, "-m", message]
        if parent:
            commit_args += ["-p", parent]
        commit = _git(repo, *commit_args)
        # CAS update: guard against a lost update (the single-orchestrator invariant already
        # holds, but "" = must-not-exist / parent = must-still-be-tip keeps the chain honest).
        _git(repo, "update-ref", ref, commit, parent)
        return commit
    except (MirrorError, OSError) as exc:
        log(f"warning: mirroring run dir to {ref} failed (will retry next wave): {exc}")
        return None


def hydrate_run_dir(
    repo: Path,
    run_dir: Path,
    epic_slug: str,
    *,
    log: Callable[[str], None] = print,
) -> bool:
    """Materialize *run_dir* from ``refs/pipeline/<epic>`` when the local run dir is absent —
    enabling resume from a fresh clone that carries the ref but no ``pipeline-runs`` dir.

    Returns ``True`` when files were written. A no-op (returns ``False``) when the local run dir
    already holds a framing (local is write-primary — never overwrite live state) or when the ref
    does not exist. Best-effort: a plumbing failure warns and returns ``False``.
    """
    if (run_dir / "framing.json").exists():
        return False  # local state present; the ref is only a fallback
    ref = epic_ref(epic_slug)
    try:
        tip = _git(repo, "rev-parse", "--verify", "--quiet", ref, check=False)
        if not tip:
            return False
        names = [n for n in _git(repo, "ls-tree", "--name-only", ref).splitlines() if n.strip()]
        if not names:
            return False
        run_dir.mkdir(parents=True, exist_ok=True)
        for name in names:
            (run_dir / name).write_bytes(_git_bytes(repo, "cat-file", "blob", f"{ref}:{name}"))
    except (MirrorError, OSError) as exc:
        log(f"warning: hydrating run dir from {ref} failed: {exc}")
        return False
    log(f"hydrated run dir for {epic_slug} from {ref} ({len(names)} file(s))")
    return True
