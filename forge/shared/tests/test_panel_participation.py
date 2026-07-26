"""describe_participation: a dropped panel seat must be visible, with its reason.

Regression cover for a live finding — every verification in a research run reported "2/5 lenses"
and nothing else. Three seats were timing out at 120s (CPU-resident models), so whole perspectives
(claim-verification, counter-narrative, actionability) went ungraded and the two survivors were the
*same* model, making the median an average of one model with itself. The panel had the reasons in
``failures`` the entire time and never rendered them.
"""

from __future__ import annotations

from forge.shared.panel import PanelResult, describe_participation


def test_full_panel_names_its_graders():
    panel = PanelResult(
        responses=[{}, {}],
        member_labels=["coder/depth", "glm/counter-narrative"],
        attempted=2,
        quorum_met=True,
    )
    out = describe_participation(panel)
    assert out.startswith("2/2 lenses")
    assert "coder/depth" in out and "glm/counter-narrative" in out
    assert "Absent" not in out


def test_dropped_seats_are_named_with_reasons():
    panel = PanelResult(
        responses=[{}, {}],
        member_labels=["coder/source-quality", "coder/depth"],
        attempted=5,
        quorum_met=True,  # floor was met, but the panel is NOT what it claims to be
        failures=[
            ("gptoss/claim-verification", "timed out after 120s"),
            ("m2.7-local/counter-narrative", "timed out after 120s"),
            ("gptoss/actionability", "timed out after 120s"),
        ],
    )
    out = describe_participation(panel)
    assert "2/5 lenses" in out
    # the survivors are named, so "both seats are the same model" is visible
    assert "coder/source-quality" in out and "coder/depth" in out
    # every absent seat and its reason is reported
    assert "gptoss/claim-verification — timed out after 120s" in out
    assert "m2.7-local/counter-narrative" in out
    assert "gptoss/actionability" in out
    # quorum was met, so it must NOT be mislabelled as below-floor
    assert "BELOW FLOOR" not in out


def test_below_floor_is_called_out():
    panel = PanelResult(
        responses=[{}],
        member_labels=["coder/depth"],
        attempted=5,
        quorum_met=False,
        failures=[("gptoss/actionability", "responded with empty output")],
    )
    out = describe_participation(panel)
    assert "BELOW FLOOR" in out
    assert "responded with empty output" in out


def test_total_wipeout_is_legible():
    panel = PanelResult(
        attempted=3, quorum_met=False, failures=[("a", "x"), ("b", "y"), ("c", "z")]
    )
    out = describe_participation(panel)
    assert "0/3 lenses" in out
    assert "graded by: none" in out
    assert "BELOW FLOOR" in out
