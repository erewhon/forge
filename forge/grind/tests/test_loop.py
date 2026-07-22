"""The grind loop over a real toy jj repo, with the model edit injected.

The toy "code" is a single file `value.txt` holding a number. The check passes when the value
reaches the target and prints `SCORE=<value>` so hill-climbing has a fitness. The injected edit_fn
stands in for OpenCode — it just writes `value.txt` — so we can drive every branch.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

from forge.grind.jj import current_op, restore_op
from forge.grind.loop import grind
from forge.grind.models import GrindConfig

pytestmark = pytest.mark.skipif(shutil.which("jj") is None, reason="jj not installed")


def _init_repo(tmp_path):
    subprocess.run(["jj", "git", "init"], cwd=tmp_path, check=True, capture_output=True, text=True)
    (tmp_path / "value.txt").write_text("0\n")
    subprocess.run(["jj", "status"], cwd=tmp_path, check=True, capture_output=True, text=True)
    return tmp_path


def _cfg(*, target: int, score: bool, window: int = 3, max_iter: int = 10) -> GrindConfig:
    check = f'v=$(cat value.txt); echo SCORE=$v; test "$v" -ge {target}'
    return GrindConfig(
        goal="reach the target",
        steps=[{"name": "run", "run": "cat value.txt"}],
        check={"run": check, "score_regex": "SCORE=([0-9]+)" if score else None},
        max_iterations=max_iter,
        no_progress_window=window,
    )


def _set_value(repo, value):
    (repo / "value.txt").write_text(f"{value}\n")


def test_reaches_goal_and_never_commits(tmp_path):
    repo = _init_repo(tmp_path)
    state = {"v": 0}

    def edit_fn(_repo, _spec, _model, _timeout):
        state["v"] += 1
        _set_value(repo, state["v"])
        return True, "", False

    outcome = grind(
        _cfg(target=3, score=True),
        repo,
        model="fake",
        run_dir=repo / "runs",
        log=lambda *_: None,
        edit_fn=edit_fn,
    )
    assert outcome.status == "done"
    assert outcome.iterations == 3
    assert (repo / "value.txt").read_text().strip() == "3"

    # No commits were made — the only change is in the working copy (@), history is just root+@.
    log = subprocess.run(
        ["jj", "log", "--no-pager", "-T", "change_id.short()", "--no-graph"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    assert len([ln for ln in log.stdout.splitlines() if ln.strip()]) <= 2

    journal = (repo / "runs" / "journal.jsonl").read_text().splitlines()
    assert len(journal) == 3
    assert journal[-1].count('"passed":true') == 1


def test_hill_climb_rolls_back_a_regression(tmp_path):
    repo = _init_repo(tmp_path)
    # turn 1 → value 5 (improves, kept); turn 2 → value 2 (regression, must roll back to 5);
    # target 99 so it never "passes" — we're testing the keep/restore behavior, then it exhausts.
    plan = iter([5, 2])

    def edit_fn(_repo, _spec, _model, _timeout):
        _set_value(repo, next(plan))
        return True, "", False

    outcome = grind(
        _cfg(target=99, score=True, max_iter=2),
        repo,
        model="fake",
        run_dir=repo / "runs",
        log=lambda *_: None,
        edit_fn=edit_fn,
    )
    assert outcome.status == "exhausted"
    assert outcome.best_score == 5.0
    # After the regression was rolled back and the run kept the best, value.txt is 5, not 2.
    assert (repo / "value.txt").read_text().strip() == "5"


def test_no_progress_guard_trips(tmp_path):
    repo = _init_repo(tmp_path)

    def edit_fn(_repo, _spec, _model, _timeout):
        return True, "", False  # model makes NO change → identical failure every turn

    outcome = grind(
        _cfg(target=3, score=False, window=2),
        repo,
        model="fake",
        run_dir=repo / "runs",
        log=lambda *_: None,
        edit_fn=edit_fn,
    )
    assert outcome.status == "stuck"
    assert outcome.iterations == 2
    assert (repo / "runs" / "lessons.proposed.md").is_file()  # a lesson was proposed


def test_blocked_rolls_back_and_stops(tmp_path):
    repo = _init_repo(tmp_path)

    def edit_fn(repo_, _spec, _model, _timeout):
        _set_value(repo_, 7)  # a partial edit that must be undone on BLOCKED
        return False, "BLOCKED: cannot reach the schema", True

    outcome = grind(
        _cfg(target=3, score=True),
        repo,
        model="fake",
        run_dir=repo / "runs",
        log=lambda *_: None,
        edit_fn=edit_fn,
    )
    assert outcome.status == "blocked"
    assert (repo / "value.txt").read_text().strip() == "0"  # rolled back to pre-edit


def test_already_done_short_circuits(tmp_path):
    repo = _init_repo(tmp_path)
    _set_value(repo, 5)  # baseline already >= target

    def edit_fn(*_):  # pragma: no cover - must never be called
        raise AssertionError("edit_fn called though baseline already passed")

    outcome = grind(
        _cfg(target=3, score=True),
        repo,
        model="fake",
        run_dir=repo / "runs",
        log=lambda *_: None,
        edit_fn=edit_fn,
    )
    assert outcome.status == "already-done"
    assert outcome.iterations == 0


def test_jj_op_roundtrip_reverts_working_copy(tmp_path):
    """Direct check of the checkpoint primitive the loop leans on."""
    repo = _init_repo(tmp_path)
    op = current_op(repo)
    _set_value(repo, 42)
    current_op(repo)  # snapshot the edit
    restore_op(repo, op)
    assert (repo / "value.txt").read_text().strip() == "0"
