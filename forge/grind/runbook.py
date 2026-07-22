"""Runbook execution — running the experiment cycle and scoring it.

Steps are plain shell commands run in the repo root (the user's world: gradle, scripts, psql,
whatever). Deliberately deterministic and harness-owned: the model never runs the cycle, so the
observation, the done-check, and the fitness score are captured by us, not narrated by the model.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from forge.grind.models import Check, CycleResult, GrindConfig, StepResult

_OUTPUT_TAIL = 4000  # chars of stdout+stderr kept per step (the model reads a tail, not a firehose)


def _tail(text: str, n: int = _OUTPUT_TAIL) -> str:
    if len(text) <= n:
        return text
    cut = text[-n:]
    nl = cut.find("\n")
    return cut[nl + 1 :] if 0 <= nl < len(cut) - 1 else cut


def run_step(name: str, command: str, cwd: Path, timeout: int) -> StepResult:
    """Run one shell command in *cwd*, capturing a tail of its combined output."""
    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or "") if isinstance(e.stdout, str) else ""
        err = (e.stderr or "") if isinstance(e.stderr, str) else ""
        return StepResult(
            name=name,
            exit_code=124,
            output=_tail(f"TIMEOUT after {timeout}s\n{out}\n{err}"),
            timed_out=True,
        )
    combined = (
        (proc.stdout or "") + ("\n" if proc.stdout and proc.stderr else "") + (proc.stderr or "")
    )
    return StepResult(name=name, exit_code=proc.returncode, output=_tail(combined))


def _parse_score(check: Check, stdout_tail: str) -> float | None:
    """Extract the fitness number from the check's output, if a score_regex is configured."""
    if not check.score_regex:
        return None
    m = re.search(check.score_regex, stdout_tail)
    if not m:
        return None
    try:
        return float(m.group(1))
    except (ValueError, IndexError):
        return None


def _observation(cfg: GrindConfig, steps: list[StepResult], check: StepResult) -> str:
    """The block of text the model reads next turn: the observed steps' output, labelled."""
    by_name = {s.name: s for s in steps} | {"check": check}
    parts: list[str] = []
    for name in cfg.resolved_observe():
        r = by_name.get(name)
        if r is None:
            continue
        status = "TIMEOUT" if r.timed_out else f"exit {r.exit_code}"
        parts.append(f"### {name} ({status})\n{r.output.strip()}")
    return "\n\n".join(parts).strip()


def run_cycle(cfg: GrindConfig, repo: Path) -> CycleResult:
    """Run every step then the check, in order. Short-circuits: if a step fails, later steps and
    the check still run only if earlier ones were clean is NOT assumed — we run the full cycle so
    the observation is complete, but `passed` requires every step + the check to be clean."""
    step_results: list[StepResult] = [
        run_step(s.name, s.run, repo, cfg.step_timeout) for s in cfg.steps
    ]
    check_result = run_step("check", cfg.check.run, repo, cfg.step_timeout)
    score = _parse_score(cfg.check, check_result.output)
    observation = _observation(cfg, step_results, check_result)
    return CycleResult(steps=step_results, check=check_result, observation=observation, score=score)


def score_improves(new: float | None, best: float | None, goal: str) -> bool:
    """Is *new* a strictly better fitness than *best* under the goal direction? A missing best is
    always improved upon; a missing new never improves."""
    if new is None:
        return False
    if best is None:
        return True
    return new > best if goal == "max" else new < best
