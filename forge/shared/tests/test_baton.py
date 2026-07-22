"""Session baton — round-trip, decision-accretion drift discipline, VCS anchoring, and the resume
preamble. Mirrors the care in test_lessons.py: a durable ``.forge/`` file that must survive hand
edits and never silently lose rationale."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from forge.shared.baton import (
    Baton,
    baton_path,
    capture_vcs_anchor,
    parse_baton,
    read_baton,
    record_decision,
    render_baton,
    render_baton_preamble,
    write_baton,
)


def _sample() -> Baton:
    return Baton(
        goal="Ship the baton primitive",
        next_action="Write the round-trip tests",
        plan=["write tests", "wire into switcheroo", "document"],
        working_set=["forge/shared/baton.py", "forge/shared/tests/test_baton.py"],
        decisions=["baton lives in shared/, not switcheroo/", "frontmatter holds only the anchor"],
        notes="highest-leverage piece of the epic",
    )


# --- read/write basics ------------------------------------------------------


def test_read_absent_is_none(tmp_path: Path):
    assert read_baton(tmp_path) is None


def test_write_creates_forge_dir_and_file(tmp_path: Path):
    write_baton(tmp_path, _sample())
    assert baton_path(tmp_path) == tmp_path / ".forge" / "baton.md"
    assert baton_path(tmp_path).is_file()


def test_round_trip_preserves_all_content_fields(tmp_path: Path):
    written = write_baton(tmp_path, _sample())
    back = read_baton(tmp_path)
    assert back is not None
    for field in ("goal", "next_action", "plan", "working_set", "decisions", "notes"):
        assert getattr(back, field) == getattr(written, field)


def test_parse_render_is_stable(tmp_path: Path):
    written = write_baton(tmp_path, _sample())
    # render → parse → render is a fixed point on content.
    assert parse_baton(render_baton(written)).model_dump() == written.model_dump()


def test_empty_sections_are_omitted_from_file(tmp_path: Path):
    write_baton(tmp_path, Baton(goal="just a goal", next_action="do the thing"))
    text = baton_path(tmp_path).read_text()
    assert "## Goal" in text
    assert "## Plan" not in text  # empty list → no heading
    assert "## Notes" not in text


# --- drift discipline: decisions accrete ------------------------------------


def test_decisions_accrete_across_rewrites(tmp_path: Path):
    write_baton(tmp_path, Baton(goal="g", decisions=["A", "B"]))
    # A later write that "forgets" the earlier decisions must not drop them.
    written = write_baton(tmp_path, Baton(goal="g moved on", decisions=["C"]))
    assert written.decisions == ["A", "B", "C"]
    assert read_baton(tmp_path).decisions == ["A", "B", "C"]


def test_decisions_dedupe_on_merge(tmp_path: Path):
    write_baton(tmp_path, Baton(goal="g", decisions=["A", "B"]))
    written = write_baton(tmp_path, Baton(goal="g", decisions=["B", "C"]))
    assert written.decisions == ["A", "B", "C"]


def test_allow_prune_lets_decisions_be_rewritten(tmp_path: Path):
    write_baton(tmp_path, Baton(goal="g", decisions=["A", "B"]))
    written = write_baton(tmp_path, Baton(goal="g", decisions=["C"]), allow_prune=True)
    assert written.decisions == ["C"]


def test_record_decision_appends_and_dedupes(tmp_path: Path):
    write_baton(tmp_path, Baton(goal="g", next_action="n"))
    assert record_decision(tmp_path, "first choice") is True
    assert record_decision(tmp_path, "first choice") is False  # already present
    back = read_baton(tmp_path)
    assert back.decisions == ["first choice"]
    assert back.goal == "g"  # other state untouched


def test_record_decision_on_absent_baton_creates_one(tmp_path: Path):
    assert record_decision(tmp_path, "bootstrap decision") is True
    assert read_baton(tmp_path).decisions == ["bootstrap decision"]


# --- timestamps -------------------------------------------------------------


def test_created_at_preserved_updated_at_advances(tmp_path: Path):
    first = write_baton(tmp_path, Baton(goal="g"))
    second = write_baton(tmp_path, Baton(goal="g2"))
    assert second.created_at == first.created_at
    assert second.updated_at >= first.updated_at


# --- forgiving parse of hand-edited files -----------------------------------


def test_parse_hand_edited_without_frontmatter(tmp_path: Path):
    text = (
        "# Session baton\n\n"
        "## Goal\nHand-written goal\n\n"
        "## Plan\n- step one\n- step two\n\n"
        "## Decisions\n- keep it simple\n"
    )
    baton = parse_baton(text)
    assert baton.goal == "Hand-written goal"
    assert baton.plan == ["step one", "step two"]
    assert baton.decisions == ["keep it simple"]
    assert baton.change_id is None  # no frontmatter → no anchor, no crash


def test_parse_ignores_unknown_sections_and_order(tmp_path: Path):
    text = "## Decisions\n- d1\n\n## Bogus\n- ignored\n\n## Goal\nlate goal\n"
    baton = parse_baton(text)
    assert baton.goal == "late goal"
    assert baton.decisions == ["d1"]


def test_malformed_frontmatter_does_not_crash():
    text = "---\n: not: valid: yaml\n---\n## Goal\nstill parses\n"
    assert parse_baton(text).goal == "still parses"


# --- preamble ---------------------------------------------------------------


def test_preamble_empty_for_none_and_empty():
    assert render_baton_preamble(None) == ""
    assert render_baton_preamble(Baton()) == ""


def test_preamble_includes_state_and_separator():
    out = render_baton_preamble(_sample())
    assert "READ FIRST" in out
    assert "Ship the baton primitive" in out
    assert "Write the round-trip tests" in out
    assert "baton lives in shared/, not switcheroo/" in out
    assert out.rstrip().endswith("---")  # composes with the next prompt block


# --- VCS anchoring ----------------------------------------------------------


def test_capture_anchor_no_vcs_is_all_none(tmp_path: Path):
    assert capture_vcs_anchor(tmp_path) == {"vcs": None, "branch": None, "change_id": None}


def _has_jj() -> bool:
    try:
        return subprocess.run(["jj", "--version"], capture_output=True).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


@pytest.mark.skipif(not _has_jj(), reason="jj not installed")
def test_write_anchors_to_jj_working_copy(tmp_path: Path):
    subprocess.run(["jj", "git", "init"], cwd=tmp_path, capture_output=True, check=True)
    written = write_baton(tmp_path, Baton(goal="anchored"))
    assert written.vcs == "jj"
    assert written.change_id  # a real change id was captured
