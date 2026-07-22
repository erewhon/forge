"""The action loop: Trial-blip → Forge task suggestion, idempotent filing by external_ref, and the
back-link — all with injected fakes (no Forge/daemon)."""

from __future__ import annotations

from datetime import date

from forge.radar.action import (
    DEFAULT_TRIAL_PROJECT,
    act,
    build_suggestion,
    external_ref,
    pending_trials,
)
from forge.radar.models import Blip, Quadrant, Radar, Ring

D = date(2026, 7, 22)


def _blip(name: str, ring: Ring, **kw) -> Blip:
    kw.setdefault("quadrant", Quadrant.AGENTS)
    return Blip(name=name, ring=ring, first_seen="2026-07-01", last_seen="2026-07-01", **kw)


class _Outcome:
    """Mirrors forge_emit.EmitOutcome — the status field is named ``action``."""

    def __init__(self, action, detail=""):
        self.action = action
        self.detail = detail


def _fake_emit(created_refs: set[str]):
    """An emit_task stand-in: idempotent by external_ref, records what it created."""

    def emit(*, external_ref, dry_run, existing_refs, **kw):
        if external_ref in existing_refs:
            return _Outcome("skipped", "external_ref exists")
        existing_refs.add(external_ref)
        created_refs.add(external_ref)
        return _Outcome("created", "pg123")

    return emit


def _radar() -> Radar:
    return Radar(
        blips=[
            _blip(
                "LangGraph",
                Ring.TRIAL,
                rationale="stateful agent orchestration",
                links=["http://x"],
            ),
            _blip("llama.cpp", Ring.TRIAL, quadrant=Quadrant.INFRA),
            _blip("OpenCode", Ring.ADOPT),  # not Trial → never suggested
            _blip("Some Model", Ring.ASSESS),  # not Trial
        ]
    )


# --- suggestion building -----------------------------------------------------


def test_external_ref_is_stable_per_blip():
    assert external_ref(_blip("Qwen3-Coder 30B", Ring.TRIAL)) == "radar:trial:qwen3-coder-30b"


def test_build_suggestion_prefills_from_the_blip():
    blip = _blip(
        "LangGraph",
        Ring.TRIAL,
        rationale="orchestration",
        action="try it on forge",
        links=["http://a"],
    )
    sug = build_suggestion(blip)
    assert sug.title == "Trial LangGraph in a gaol sandbox"
    assert sug.ref == "radar:trial:langgraph"
    assert "orchestration" in sug.content and "try it on forge" in sug.content
    assert "http://a" in sug.content


def test_pending_trials_only_unfiled_trial_blips():
    radar = _radar()
    pending = pending_trials(radar, existing_refs=set())
    assert [b.name for b in pending] == ["LangGraph", "llama.cpp"]  # Adopt/Assess excluded, sorted

    # Once LangGraph's ref exists, it drops out.
    pending2 = pending_trials(radar, existing_refs={"radar:trial:langgraph"})
    assert [b.name for b in pending2] == ["llama.cpp"]


# --- act: suggest (dry-run) --------------------------------------------------


def test_act_dry_run_writes_nothing():
    radar = _radar()
    created: set[str] = set()
    result = act(
        radar,
        today=D,
        dry_run=True,
        emit_fn=_fake_emit(created),
        ensure_project_fn=lambda p: None,
        existing_refs_fn=lambda: set(),
    )
    assert result.dry_run is True
    assert {f.name for f in result.filed} == {"LangGraph", "llama.cpp"}
    assert all(f.status == "dry-run" for f in result.filed)
    assert created == set()  # nothing filed
    assert all(b.action == "" for b in radar.blips)  # no back-links written
    assert "would be filed" in result.render()


# --- act: file ---------------------------------------------------------------


def test_act_file_creates_tasks_and_backlinks_blips():
    radar = _radar()
    created: set[str] = set()
    ensured: list[str] = []
    result = act(
        radar,
        today=D,
        dry_run=False,
        project="Radar Trials",
        emit_fn=_fake_emit(created),
        ensure_project_fn=ensured.append,
        existing_refs_fn=lambda: set(),
    )
    assert ensured == ["Radar Trials"]  # project ensured before emitting
    assert result.created == 2
    assert created == {"radar:trial:langgraph", "radar:trial:llama-cpp"}

    lang = radar.get("LangGraph")
    assert lang.action == "Trial task filed: Trial LangGraph in a gaol sandbox"
    assert any("Trial task filed" in e.note and e.source == "action-loop" for e in lang.evidence)


def test_act_does_not_refile_an_existing_ref_but_self_heals_the_backlink():
    radar = _radar()
    created: set[str] = set()
    # LangGraph already has a task on file, but its blip has no back-link yet.
    result = act(
        radar,
        today=D,
        dry_run=False,
        emit_fn=_fake_emit(created),
        ensure_project_fn=lambda p: None,
        existing_refs_fn=lambda: {"radar:trial:langgraph"},
    )
    by_name = {f.name: f for f in result.filed}
    assert by_name["LangGraph"].status == "skipped"  # not re-filed
    assert by_name["llama.cpp"].status == "created"
    assert created == {"radar:trial:llama-cpp"}  # only the unfiled one was emitted
    # The already-filed blip gets its missing back-link written (self-heal).
    assert radar.get("LangGraph").action == "Trial task filed: Trial LangGraph in a gaol sandbox"


def test_act_does_not_rewrite_an_existing_backlink():
    radar = _radar()
    radar.get("LangGraph").action = "Trial task filed: Trial LangGraph in a gaol sandbox"
    result = act(
        radar,
        today=D,
        dry_run=False,
        emit_fn=_fake_emit(set()),
        ensure_project_fn=lambda p: None,
        existing_refs_fn=lambda: {"radar:trial:langgraph"},
    )
    # LangGraph already back-linked → no new evidence entry appended.
    assert not any(e.source == "action-loop" for e in radar.get("LangGraph").evidence)
    assert {f.name for f in result.filed} == {"LangGraph", "llama.cpp"}


def test_act_only_targets_one_blip():
    radar = _radar()
    created: set[str] = set()
    result = act(
        radar,
        today=D,
        dry_run=False,
        only="llama.cpp",
        emit_fn=_fake_emit(created),
        ensure_project_fn=lambda p: None,
        existing_refs_fn=lambda: set(),
    )
    assert {f.name for f in result.filed} == {"llama.cpp"}
    assert created == {"radar:trial:llama-cpp"}


def test_act_no_pending_trials_is_a_clean_noop():
    radar = Radar(blips=[_blip("OpenCode", Ring.ADOPT)])
    result = act(
        radar,
        today=D,
        dry_run=False,
        emit_fn=_fake_emit(set()),
        ensure_project_fn=lambda p: None,
        existing_refs_fn=lambda: set(),
    )
    assert result.filed == []
    assert "nothing to file" in result.render()


def test_default_project_constant():
    assert DEFAULT_TRIAL_PROJECT == "Radar Trials"
