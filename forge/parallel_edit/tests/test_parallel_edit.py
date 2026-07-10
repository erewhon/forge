"""Unit tests for the parallel_edit core: candidate parsing, the N-way judge verdict parsing,
the dynamic judge prompt, diff-noise exclusion, and per-candidate opencode isolation.

No network and no jj/opencode subprocesses — these cover the pure logic that the live smokes
exercise end-to-end but don't pin down (label generation, winner validation, legacy-spelling
normalization, the loose-host XDG_DATA_HOME seeding).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from forge.parallel_edit.config import settings
from forge.parallel_edit.judge import (
    _extract_json,
    _legacy_verdict_to_best,
    _parse_verdict,
)
from forge.parallel_edit.main import _MAX_CANDIDATES, _parse_candidates, _parse_one_candidate
from forge.parallel_edit.models import CandidateSpec
from forge.parallel_edit.prompts import build_judge_system_prompt
from forge.parallel_edit.runner import _opencode_host_state, _should_sandbox
from forge.parallel_edit.workspaces import _diff_exclude_fileset

# --------------------------------------------------------------------------- candidate parsing


def test_parse_one_candidate_bare_is_claude():
    spec = _parse_one_candidate("claude-opus-4-8", "A")
    assert spec == CandidateSpec(
        label="A", kind="claude", model="claude-opus-4-8", display="claude-opus-4-8"
    )


def test_parse_one_candidate_explicit_claude_prefix():
    spec = _parse_one_candidate("claude:claude-sonnet-4-6", "B")
    assert spec.kind == "claude"
    assert spec.model == "claude-sonnet-4-6"
    assert spec.display == "claude-sonnet-4-6"


def test_parse_one_candidate_opencode_adds_llm_prefix():
    spec = _parse_one_candidate("opencode:glm-5.1", "A")
    assert spec.kind == "opencode"
    assert spec.model == "llm/glm-5.1"  # provider prefix added
    assert spec.display == "opencode:llm/glm-5.1"


def test_parse_one_candidate_opencode_keeps_existing_provider():
    spec = _parse_one_candidate("opencode:llm/glm-5.1", "A")
    assert spec.model == "llm/glm-5.1"  # not double-prefixed


def test_parse_candidates_three_way_labels_abc():
    specs = _parse_candidates("opencode:glm-5.1,opencode:kimi,opencode:deepseek")
    assert [s.label for s in specs] == ["A", "B", "C"]
    assert [s.kind for s in specs] == ["opencode"] * 3


def test_parse_candidates_default_is_open_fleet_trio():
    specs = _parse_candidates(None)  # falls back to default_candidate_models
    assert len(specs) == len(settings.default_candidate_models)
    assert all(s.kind == "opencode" for s in specs)
    assert [s.label for s in specs] == list("ABC")[: len(specs)]


def test_parse_candidates_strips_whitespace_and_blanks():
    specs = _parse_candidates(" claude-opus-4-8 , , opencode:kimi ")
    assert [s.label for s in specs] == ["A", "B"]
    assert specs[0].model == "claude-opus-4-8"
    assert specs[1].model == "llm/kimi"


def test_parse_candidates_rejects_single():
    with pytest.raises(SystemExit):
        _parse_candidates("only-one")


def test_parse_candidates_rejects_over_max():
    too_many = ",".join(["m"] * (_MAX_CANDIDATES + 1))
    with pytest.raises(SystemExit):
        _parse_candidates(too_many)


def test_parse_candidates_max_uses_label_z():
    specs = _parse_candidates(",".join(["m"] * _MAX_CANDIDATES))
    assert len(specs) == _MAX_CANDIDATES
    assert specs[-1].label == "Z"


# --------------------------------------------------------------------------- _extract_json


def test_extract_json_plain():
    assert _extract_json('{"winner": "A"}') == {"winner": "A"}


def test_extract_json_fenced():
    assert _extract_json('```json\n{"winner": "B"}\n```') == {"winner": "B"}


def test_extract_json_from_surrounding_prose():
    text = 'Here is my verdict:\n{"winner": "tie"}\nThanks!'
    assert _extract_json(text) == {"winner": "tie"}


def test_extract_json_garbage_returns_empty():
    assert _extract_json("no json here at all") == {}


# --------------------------------------------------------------------------- legacy verdict map


@pytest.mark.parametrize(
    ("verdict", "expected"),
    [
        ("A better", "A"),
        ("B better", "B"),
        ("A only", "A"),
        ("B only", "B"),
        ("equivalent", "equivalent"),
        ("", "equivalent"),
    ],
)
def test_legacy_verdict_to_best(verdict, expected):
    assert _legacy_verdict_to_best(verdict) == expected


# --------------------------------------------------------------------------- verdict parsing

_SCORE = {
    "prompt_fidelity": 8,
    "correctness": 7,
    "scope_discipline": 9,
    "code_quality": 8,
    "completeness": 7,
}


def _verdict_payload(winner="C", labels=("A", "B", "C")):
    return {
        "winner": winner,
        "scores": {label: dict(_SCORE) for label in labels},
        "per_file_notes": [{"file": "x.py", "best": "C", "note": "C cleanest"}],
        "summary": "C wins",
        "recommendation": "merge C",
    }


def test_parse_verdict_three_way():
    v = _parse_verdict(_verdict_payload(), ["A", "B", "C"])
    assert v is not None
    assert v.winner == "C"
    assert set(v.scores) == {"A", "B", "C"}
    assert v.per_file_notes[0].best == "C"


@pytest.mark.parametrize("winner", ["A", "B", "C", "tie", "all_flawed"])
def test_parse_verdict_accepts_valid_winners(winner):
    v = _parse_verdict(_verdict_payload(winner=winner), ["A", "B", "C"])
    assert v is not None and v.winner == winner


def test_parse_verdict_normalizes_legacy_both_flawed():
    v = _parse_verdict(_verdict_payload(winner="both_flawed"), ["A", "B", "C"])
    assert v is not None and v.winner == "all_flawed"


def test_parse_verdict_rejects_unknown_winner_token():
    assert _parse_verdict(_verdict_payload(winner="best one"), ["A", "B", "C"]) is None


def test_parse_verdict_rejects_label_outside_run_set():
    # "D" is a valid-looking label but wasn't one of the candidates compared.
    assert _parse_verdict(_verdict_payload(winner="D"), ["A", "B", "C"]) is None


def test_parse_verdict_rejects_missing_scores():
    payload = _verdict_payload()
    payload["scores"] = {}
    assert _parse_verdict(payload, ["A", "B", "C"]) is None


def test_parse_verdict_allows_partial_scores():
    # A judge that only scored a subset still yields a usable verdict for those it scored.
    payload = _verdict_payload()
    payload["scores"] = {"A": dict(_SCORE), "B": dict(_SCORE)}
    v = _parse_verdict(payload, ["A", "B", "C"])
    assert v is not None and set(v.scores) == {"A", "B"}


def test_parse_verdict_maps_legacy_per_file_verdict_field():
    payload = _verdict_payload()
    payload["per_file_notes"] = [{"file": "y.py", "verdict": "A better", "note": "n"}]
    v = _parse_verdict(payload, ["A", "B", "C"])
    assert v is not None and v.per_file_notes[0].best == "A"


def test_parse_verdict_skips_malformed_per_file_entries():
    payload = _verdict_payload()
    payload["per_file_notes"] = ["not a dict", {"file": "z.py", "best": "B", "note": "ok"}]
    v = _parse_verdict(payload, ["A", "B", "C"])
    assert v is not None and [n.file for n in v.per_file_notes] == ["z.py"]


def test_parse_verdict_rejects_non_dict():
    assert _parse_verdict(["not", "a", "dict"], ["A", "B"]) is None


# --------------------------------------------------------------------------- judge prompt builder


def test_build_judge_system_prompt_enumerates_labels():
    prompt = build_judge_system_prompt(["A", "B", "C"])
    assert "3 independent attempts" in prompt
    assert '"A" | "B" | "C" | "tie" | "all_flawed"' in prompt
    # one scores entry per label
    for label in ("A", "B", "C"):
        assert f'"{label}": {{' in prompt


def test_build_judge_system_prompt_two_way():
    prompt = build_judge_system_prompt(["A", "B"])
    assert "2 independent attempts" in prompt
    assert '"A" | "B" | "tie" | "all_flawed"' in prompt
    assert '"C"' not in prompt


# --------------------------------------------------------------------------- diff-noise exclusion


def test_diff_exclude_fileset_covers_cruft():
    fileset = _diff_exclude_fileset()
    assert len(fileset) == 1
    term = fileset[0]
    assert '~".open-mem"' in term
    assert '~glob:"**/__pycache__/**"' in term
    assert '~glob:"**/*.pyc"' in term
    # negated-and-ANDed
    assert " & " in term


# --------------------------------------------------------------------------- sandbox gating


def test_should_sandbox_off_by_default(monkeypatch):
    monkeypatch.setattr(settings, "sandbox", False)
    oc = _parse_one_candidate("opencode:glm-5.1", "A")
    assert _should_sandbox(oc) is False


def test_should_sandbox_isolates_opencode_exempts_claude(monkeypatch):
    monkeypatch.setattr(settings, "sandbox", True)
    monkeypatch.setattr(settings, "sandbox_exempt_kinds", ["claude"])
    oc = _parse_one_candidate("opencode:glm-5.1", "A")
    cl = _parse_one_candidate("claude-opus-4-8", "B")
    assert _should_sandbox(oc) is True
    assert _should_sandbox(cl) is False  # claude trusted -> loose on host


# --------------------------------------------------------------------- loose-host opencode state


def test_opencode_host_state_seeds_private_data_dir(monkeypatch, tmp_path):
    fake_home = tmp_path / "home"
    auth_dir = fake_home / ".local" / "share" / "opencode"
    auth_dir.mkdir(parents=True)
    (auth_dir / "auth.json").write_text('{"llm": {}}', encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: fake_home)

    env, state = _opencode_host_state()
    try:
        assert state is not None
        assert env["XDG_DATA_HOME"] == str(state)
        # auth.json seeded into <state>/opencode/ so the llm/ provider still resolves
        seeded = state / "opencode" / "auth.json"
        assert seeded.is_file()
        assert seeded.read_text(encoding="utf-8") == '{"llm": {}}'
    finally:
        if state is not None:
            shutil.rmtree(state, ignore_errors=True)


def test_opencode_host_state_noop_without_host_auth(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "empty-home")
    env, state = _opencode_host_state()
    assert env == {}
    assert state is None
