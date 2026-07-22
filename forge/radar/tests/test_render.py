"""Rendering: blip placement geometry, the palette-themed SVG fence, the legend, and the page."""

from __future__ import annotations

from forge.radar.models import Blip, Quadrant, Radar, Ring
from forge.radar.render import (
    place_blips,
    publish_radar,
    render_legend,
    render_radar_page,
    render_radar_svg,
    render_radar_svg_fence,
)


def _blip(name: str, quadrant: Quadrant, ring: Ring, **kw) -> Blip:
    return Blip(
        name=name,
        quadrant=quadrant,
        ring=ring,
        first_seen="2026-07-01",
        last_seen="2026-07-01",
        **kw,
    )


def _radar() -> Radar:
    return Radar(
        blips=[
            _blip("OpenCode", Quadrant.AGENTS, Ring.ADOPT, rationale="the stack's coding agent"),
            _blip("llama.cpp", Quadrant.INFRA, Ring.TRIAL, rationale="serve quants"),
            _blip("LangChain", Quadrant.AGENTS, Ring.ASSESS),
            _blip("Some Paper", Quadrant.TECHNIQUES, Ring.ASSESS),
        ]
    )


def test_place_blips_numbers_stably_and_stays_in_bounds():
    placed = place_blips(_radar())
    assert [p.number for p in placed] == [1, 2, 3, 4]
    # Numbering follows quadrant order (Agents before Techniques before Infra) then ring.
    names = [p.blip.name for p in placed]
    assert names.index("OpenCode") < names.index("LangChain") < names.index("Some Paper")
    assert names.index("Some Paper") < names.index("llama.cpp")
    # All within the 720×720 viewBox.
    assert all(0 <= p.x <= 720 and 0 <= p.y <= 720 for p in placed)


def test_placement_radius_reflects_ring():
    # An Adopt blip sits nearer the centre than a Hold blip in the same quadrant.
    radar = Radar(
        blips=[
            _blip("A", Quadrant.MODELS, Ring.ADOPT),
            _blip("H", Quadrant.MODELS, Ring.HOLD),
        ]
    )
    placed = {p.blip.name: p for p in place_blips(radar)}

    def dist(p):
        return ((p.x - 360) ** 2 + (p.y - 360) ** 2) ** 0.5

    assert dist(placed["A"]) < dist(placed["H"])


def test_svg_uses_palette_vars_and_has_no_hardcoded_hex_fill():
    svg = render_radar_svg(_radar())
    assert svg.startswith("<svg") and svg.endswith("</svg>")
    assert "var(--accent" in svg and "var(--border" in svg and "var(--text" in svg
    # Ring + quadrant labels present (the ampersand quadrant is HTML-escaped).
    for label in ("Adopt", "Trial", "Assess", "Hold", "Infra/Tooling", "Techniques"):
        assert label in svg
    assert "Agents &amp; Frameworks" in svg
    # One accent dot per blip.
    assert svg.count("fill:var(--accent") == 4


def test_svg_escapes_blip_text():
    radar = Radar(blips=[_blip("A & B <x>", Quadrant.MODELS, Ring.ADOPT)])
    # The SVG shows blip numbers, not names, but any injected text must be escaped.
    svg = render_radar_svg(radar)
    assert "<x>" not in svg  # nothing raw-injected


def test_fence_wraps_animation():
    fence = render_radar_svg_fence(_radar())
    assert fence.startswith("```animation\n") and fence.rstrip().endswith("```")
    assert "<svg" in fence


def test_legend_groups_and_numbers_match_the_chart():
    radar = _radar()
    placed = {p.blip.name: p.number for p in place_blips(radar)}
    legend = render_legend(radar)
    assert "### Agents & Frameworks" in legend and "### Infra/Tooling" in legend
    assert f"**{placed['OpenCode']}. OpenCode**" in legend
    assert "the stack's coding agent" in legend  # rationale carried through


def test_page_has_chart_legend_db_pointer_and_digest():
    page = render_radar_page(_radar(), digest="## Promoted\n- X", updated="2026-07-22")
    assert "# AI Technology Radar" in page
    assert "```animation" in page
    assert "## Legend" in page
    assert '"Radar Blips"' in page  # database pointer
    assert "## This week" in page and "Promoted" in page
    assert "4 blips" in page


def test_page_without_digest_says_so():
    page = render_radar_page(_radar(), updated="2026-07-22")
    assert "No synthesis run yet" in page


# --- publish (fake daemon) ---------------------------------------------------


class FakePageDaemon:
    def __init__(self, notebooks=None):
        self.notebooks = notebooks or [{"id": "nb1", "name": "AI Radar"}]
        self.pages: dict[str, dict] = {}
        self.updated: list[str] = []
        self._n = 0

    def list_notebooks(self):
        return list(self.notebooks)

    def create_notebook(self, name, *, notebook_type=None):
        nb = {"id": f"nb{len(self.notebooks) + 1}", "name": name}
        self.notebooks.append(nb)
        return nb

    def resolve_page(self, notebook_id, title_or_id):
        for p in self.pages.values():
            if p["title"] == title_or_id:
                return p
        raise ValueError("not found")

    def create_page(self, notebook_id, title, *, blocks=None, tags=None, **kw):
        self._n += 1
        page = {"id": f"pg{self._n}", "title": title, "blocks": blocks}
        self.pages[page["id"]] = page
        return page

    def update_page(self, notebook_id, page_id, *, blocks=None, tags=None, **kw):
        self.pages[page_id]["blocks"] = blocks
        self.updated.append(page_id)
        return self.pages[page_id]


def _fake_to_blocks(md: str):
    """Stand-in for nous_mcp.markdown.markdown_to_blocks: promotes fenced blocks to code blocks so
    the animation chart is verifiably a block, not flattened to a paragraph."""
    blocks = []
    for chunk in md.split("```animation\n"):
        if chunk.startswith("<svg") or "</svg>" in chunk.split("\n```", 1)[0]:
            code, _, rest = chunk.partition("\n```")
            blocks.append({"type": "code", "data": {"language": "animation", "code": code}})
            chunk = rest
        for line in chunk.splitlines():
            if line.strip():
                blocks.append({"type": "paragraph", "data": {"text": line}})
    return blocks


def test_publish_writes_the_animation_as_a_code_block_not_a_paragraph():
    daemon = FakePageDaemon()
    r1 = publish_radar(
        daemon, _radar(), notebook_name="AI Radar", updated="2026-07-22", to_blocks=_fake_to_blocks
    )
    assert r1["created"] is True
    blocks = daemon.pages[r1["page_id"]]["blocks"]
    anim = [b for b in blocks if b["type"] == "code" and b["data"]["language"] == "animation"]
    assert len(anim) == 1
    assert "<svg" in anim[0]["data"]["code"]  # real SVG markup, not stripped text


def test_publish_updates_the_same_page_in_place():
    daemon = FakePageDaemon()
    r1 = publish_radar(
        daemon, _radar(), notebook_name="AI Radar", updated="2026-07-22", to_blocks=_fake_to_blocks
    )
    r2 = publish_radar(
        daemon, _radar(), notebook_name="AI Radar", updated="2026-07-23", to_blocks=_fake_to_blocks
    )
    assert r2["created"] is False and r2["page_id"] == r1["page_id"]
    assert daemon.updated == [r1["page_id"]]  # updated in place, no second page


def test_publish_creates_notebook_when_absent():
    daemon = FakePageDaemon(notebooks=[])
    publish_radar(
        daemon, _radar(), notebook_name="AI Radar", updated="2026-07-22", to_blocks=_fake_to_blocks
    )
    assert [nb["name"] for nb in daemon.notebooks] == ["AI Radar"]
