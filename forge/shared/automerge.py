"""Reusable gates + VCS actions for auto-merging low-blast-radius changes.

The testing ensemble's auto-merge loop and (later) the Dependabot bumper share these primitives:
a *tests-only* classifier (blocks anything that isn't a test file, or that touches a dependency
manifest), the branch / advance-main VCS actions for jj and git, a slug helper, and a JSONL
decision log. Everything here is **fail-closed**: an unclassifiable change is not tests-only, and
a VCS the module doesn't understand raises rather than guessing.

The classifier and the log are VCS-agnostic; the branch/merge actions dispatch on ``detect_vcs``
and reuse ``forge.task_worker.vcs`` for the existing detect/commit primitives.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from forge.task_worker.vcs import VCSError, commit, detect_vcs, get_changed_files

_TIMEOUT = 30

# Dependency manifests / lockfiles. A "tests-only" change that touches one of these is NOT safe
# to auto-merge — it can pull in a new external dependency (the roadmap's explicit risk for
# auto-merged test PRs), so the classifier blocks on any of them, matched by basename.
MANIFEST_FILENAMES = frozenset(
    {
        "pyproject.toml",
        "setup.py",
        "setup.cfg",
        "requirements.txt",
        "requirements-dev.txt",
        "uv.lock",
        "poetry.lock",
        "Pipfile",
        "Pipfile.lock",
        "package.json",
        "package-lock.json",
        "pnpm-lock.yaml",
        "yarn.lock",
        "Cargo.toml",
        "Cargo.lock",
        "go.mod",
        "go.sum",
        "Gemfile",
        "Gemfile.lock",
    }
)

_TEST_DIR_PARTS = {"tests", "test", "__tests__", "spec"}


def is_test_path(path: str | Path) -> bool:
    """True if *path* is a test file by name or by living under a test directory.

    Mirrors the testing ensemble's own discovery classifier so a generated change is judged the
    same way its gaps were found. Recognizes pytest (``test_*.py`` / ``*_test.py`` /
    ``conftest.py``), Rust/Go (``*_test.rs`` / ``*_test.go`` / files under ``tests/``), and JS/TS
    (``*.test.*`` / ``*.spec.*``).
    """
    p = Path(path)
    name = p.name
    lname = name.lower()
    if name.startswith("test_") or name == "conftest.py":
        return True
    if name.endswith(("_test.py", "_test.rs", "_test.go")):
        return True
    if ".test." in lname or ".spec." in lname:
        return True
    return any(part.lower() in _TEST_DIR_PARTS for part in p.parts)


def is_manifest_path(path: str | Path) -> bool:
    """True if *path* is a dependency manifest or lockfile (matched by basename)."""
    return Path(path).name in MANIFEST_FILENAMES


@dataclass(frozen=True)
class TestsOnlyVerdict:
    """Result of the tests-only gate. ``ok`` is the only thing callers must check to proceed."""

    ok: bool
    changed: list[str]
    non_test: list[str] = field(default_factory=list)
    manifests: list[str] = field(default_factory=list)
    reason: str = ""


def classify_tests_only(repo_path: Path, *, changed: list[str] | None = None) -> TestsOnlyVerdict:
    """Decide whether the working-copy changes are safe *tests-only* edits.

    Blocks (``ok=False``) if there are no changes, if any changed file is a dependency
    manifest/lockfile, or if any changed file is not a test file. Pass ``changed`` to classify a
    known file list (e.g. from a captured diff) instead of reading the working copy.
    """
    files = [
        f for f in (changed if changed is not None else get_changed_files(repo_path)) if f.strip()
    ]
    if not files:
        return TestsOnlyVerdict(ok=False, changed=[], reason="no changes to classify")

    manifests = [f for f in files if is_manifest_path(f)]
    if manifests:
        return TestsOnlyVerdict(
            ok=False,
            changed=files,
            non_test=[f for f in files if not is_test_path(f)],
            manifests=manifests,
            reason=f"touches dependency manifest(s): {', '.join(manifests)}",
        )
    non_test = [f for f in files if not is_test_path(f)]
    if non_test:
        return TestsOnlyVerdict(
            ok=False,
            changed=files,
            non_test=non_test,
            reason=f"non-test file(s) changed: {', '.join(non_test)}",
        )
    return TestsOnlyVerdict(ok=True, changed=files)


@dataclass(frozen=True)
class ManifestOnlyVerdict:
    """Result of the manifest-only gate. ``ok`` is the only thing callers must check to proceed."""

    ok: bool
    changed: list[str]
    non_manifest: list[str] = field(default_factory=list)
    reason: str = ""


def classify_manifest_only(
    repo_path: Path, *, changed: list[str] | None = None
) -> ManifestOnlyVerdict:
    """Decide whether the working-copy changes are a pure dependency bump — manifests/lockfiles
    and NOTHING else.

    The Dependabot bumper's first gate, the inverse twin of :func:`classify_tests_only`: a bump
    that needs a source change is by definition not a clean bump and must fall through to
    advisory. Blocks (``ok=False``) if there are no changes or if any changed file is not a
    dependency manifest/lockfile. Pass ``changed`` to classify a known file list instead of
    reading the working copy. Fail-closed: an unclassifiable change is not manifest-only.
    """
    files = [
        f for f in (changed if changed is not None else get_changed_files(repo_path)) if f.strip()
    ]
    if not files:
        return ManifestOnlyVerdict(ok=False, changed=[], reason="no changes to classify")

    non_manifest = [f for f in files if not is_manifest_path(f)]
    if non_manifest:
        return ManifestOnlyVerdict(
            ok=False,
            changed=files,
            non_manifest=non_manifest,
            reason=f"non-manifest file(s) changed: {', '.join(non_manifest)}",
        )
    return ManifestOnlyVerdict(ok=True, changed=files)


_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def slugify(text: str, *, max_len: int = 40) -> str:
    """Lowercase, hyphenate, and truncate *text* into a branch-name-safe slug."""
    s = _NON_ALNUM.sub("-", text.strip().lower()).strip("-")
    if len(s) > max_len:
        s = s[:max_len].rstrip("-")
    return s or "change"


# ---------------------------------------------------------------------------
# VCS actions — branch push and advance-main (the loaded auto-merge step)
# ---------------------------------------------------------------------------


def _run(cmd: list[str], cwd: Path, timeout: int = _TIMEOUT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd)


@dataclass(frozen=True)
class PushResult:
    """Outcome of a branch push or a main advance."""

    vcs: str
    branch: str
    change_id: str
    pushed: bool
    merged_to_main: bool = False
    detail: str = ""


def push_branch(repo_path: Path, branch: str, message: str, *, push: bool = True) -> PushResult:
    """Commit the working copy onto *branch* and push it to origin (the 'auto-PR' step).

    jj: the working-copy change is described, a bookmark is set at it, and pushed (jj 0.42+
    pushes new bookmarks by default — ``--allow-new`` no longer exists). git: a branch is
    created at a fresh commit and pushed with ``-u``. Pass
    ``push=False`` to stop at the local branch (used by tests without a remote). Returns the new
    change/commit id, which ``advance_main`` can later fast-forward ``main`` to.
    """
    vcs = detect_vcs(repo_path)
    if vcs == "jj":
        return _jj_push_branch(repo_path, branch, message, push=push)
    if vcs == "git":
        return _git_push_branch(repo_path, branch, message, push=push)
    raise VCSError(f"No VCS detected in {repo_path}")


def advance_main(
    repo_path: Path, change_id: str, *, main: str = "main", push: bool = True
) -> PushResult:
    """Move the ``main`` bookmark/branch to *change_id* and push it — the auto-merge action.

    Call only after every gate has passed. jj sets the bookmark and ``jj git push``es it; git
    force-updates the branch ref (must not be the checked-out branch) and pushes it.
    """
    vcs = detect_vcs(repo_path)
    if vcs == "jj":
        return _jj_advance_main(repo_path, change_id, main=main, push=push)
    if vcs == "git":
        return _git_advance_main(repo_path, change_id, main=main, push=push)
    raise VCSError(f"No VCS detected in {repo_path}")


def _jj_push_branch(repo_path: Path, branch: str, message: str, *, push: bool) -> PushResult:
    change_id = commit(repo_path, message)  # describes @, then advances with `jj new`
    if not change_id:
        raise VCSError("jj commit did not return a change id")
    # --allow-backwards: automation branches are force-moved on re-runs. A stale bookmark from
    # an earlier run may sit on a sibling lineage (live finding: a pre-merge branched run's
    # bookmark blocked the first post-merge auto-merge with "refusing to move sideways").
    set_bm = _run(
        ["jj", "bookmark", "set", branch, "-r", change_id, "--allow-backwards"], repo_path
    )
    if set_bm.returncode != 0:
        raise VCSError(f"jj bookmark set {branch} failed: {set_bm.stderr.strip()}")
    detail, pushed = "", False
    if push:
        res = _run(["jj", "git", "push", "--bookmark", branch], repo_path)
        if res.returncode != 0:
            raise VCSError(f"jj git push {branch} failed: {res.stderr.strip()}")
        pushed = True
        detail = (res.stderr.strip() or res.stdout.strip())[:500]
    return PushResult(vcs="jj", branch=branch, change_id=change_id, pushed=pushed, detail=detail)


def _jj_advance_main(repo_path: Path, change_id: str, *, main: str, push: bool) -> PushResult:
    set_bm = _run(["jj", "bookmark", "set", main, "-r", change_id], repo_path)
    if set_bm.returncode != 0:
        raise VCSError(f"jj bookmark set {main} failed: {set_bm.stderr.strip()}")
    detail, pushed = "", False
    if push:
        res = _run(["jj", "git", "push", "--bookmark", main], repo_path)
        if res.returncode != 0:
            raise VCSError(f"jj git push {main} failed: {res.stderr.strip()}")
        pushed = True
        detail = (res.stderr.strip() or res.stdout.strip())[:500]
    return PushResult(
        vcs="jj",
        branch=main,
        change_id=change_id,
        pushed=pushed,
        merged_to_main=True,
        detail=detail,
    )


def _git_current_branch(repo_path: Path) -> str:
    res = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], repo_path)
    return res.stdout.strip() if res.returncode == 0 else ""


def _git_push_branch(repo_path: Path, branch: str, message: str, *, push: bool) -> PushResult:
    checkout = _run(["git", "checkout", "-b", branch], repo_path)
    if checkout.returncode != 0:
        raise VCSError(f"git checkout -b {branch} failed: {checkout.stderr.strip()}")
    change_id = commit(repo_path, message)  # git add -A && git commit; returns short sha
    if not change_id:
        raise VCSError("git commit did not return a sha")
    detail, pushed = "", False
    if push:
        res = _run(["git", "push", "-u", "origin", branch], repo_path)
        if res.returncode != 0:
            raise VCSError(f"git push {branch} failed: {res.stderr.strip()}")
        pushed = True
        detail = (res.stderr.strip() or res.stdout.strip())[:500]
    return PushResult(vcs="git", branch=branch, change_id=change_id, pushed=pushed, detail=detail)


def _git_advance_main(repo_path: Path, change_id: str, *, main: str, push: bool) -> PushResult:
    if _git_current_branch(repo_path) == main:
        raise VCSError(f"refusing to force-update {main} while it is checked out")
    force = _run(["git", "branch", "-f", main, change_id], repo_path)
    if force.returncode != 0:
        raise VCSError(f"git branch -f {main} failed: {force.stderr.strip()}")
    detail, pushed = "", False
    if push:
        res = _run(["git", "push", "origin", main], repo_path)
        if res.returncode != 0:
            raise VCSError(f"git push {main} failed: {res.stderr.strip()}")
        pushed = True
        detail = (res.stderr.strip() or res.stdout.strip())[:500]
    return PushResult(
        vcs="git",
        branch=main,
        change_id=change_id,
        pushed=pushed,
        merged_to_main=True,
        detail=detail,
    )


def working_copy_base(repo_path: Path) -> str:
    """The revision the working copy currently sits on — capture BEFORE a loop mutates the
    working copy so :func:`repark_working_copy` can return there after a branch push.
    jj: the ``@-`` commit id; git: the current branch name."""
    vcs = detect_vcs(repo_path)
    if vcs == "jj":
        res = _run(["jj", "log", "--no-graph", "-r", "@-", "-T", "commit_id"], repo_path)
        if res.returncode != 0:
            raise VCSError(f"jj log @- failed: {res.stderr.strip()}")
        return res.stdout.strip()
    if vcs == "git":
        branch = _git_current_branch(repo_path)
        if not branch:
            raise VCSError("git: could not determine current branch")
        return branch
    raise VCSError(f"No VCS detected in {repo_path}")


def repark_working_copy(repo_path: Path, base: str) -> None:
    """Return the working copy to *base* after :func:`push_branch`, so the pushed branch stays
    a side head instead of becoming the mainline's parent. Without this, every advisory or
    branched outcome stacks its bump commit under all subsequent work (live-smoke finding:
    a failed advisory push left a dep bump inside an epic branch's lineage).
    jj: ``jj new <base>``; git: ``git checkout <base>``."""
    vcs = detect_vcs(repo_path)
    if vcs == "jj":
        res = _run(["jj", "new", base], repo_path)
    elif vcs == "git":
        res = _run(["git", "checkout", base], repo_path)
    else:
        raise VCSError(f"No VCS detected in {repo_path}")
    if res.returncode != 0:
        raise VCSError(f"repark onto {base} failed: {res.stderr.strip()}")


def working_diff(repo_path: Path) -> str:
    """Unified (git-format) diff of the working copy vs its parent/HEAD, including new files.

    Feeds the sign-off reviewers. For git, newly written test files are marked intent-to-add so
    they appear in the diff (``git add -N`` is undone by ``revert_changes`` on a blocked gate).
    """
    vcs = detect_vcs(repo_path)
    if vcs == "jj":
        res = _run(["jj", "diff", "--no-pager", "--git"], repo_path)
        if res.returncode != 0:
            raise VCSError(f"jj diff failed: {res.stderr.strip()}")
        return res.stdout
    if vcs == "git":
        _run(["git", "add", "-N", "."], repo_path)
        res = _run(["git", "diff", "HEAD"], repo_path)
        if res.returncode != 0:
            raise VCSError(f"git diff failed: {res.stderr.strip()}")
        return res.stdout
    raise VCSError(f"No VCS detected in {repo_path}")


def find_repo_root(start: Path) -> Path | None:
    """Walk up from *start* to the nearest dir containing ``.jj`` or ``.git``; None if none."""
    start = start.resolve()
    for d in (start, *start.parents):
        if (d / ".jj").is_dir() or (d / ".git").exists():
            return d
    return None


def log_decision(record: dict, log_path: Path) -> Path:
    """Append one JSONL decision record (mirrors the pr_review run log). Returns the log path."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=str) + "\n")
    return log_path
