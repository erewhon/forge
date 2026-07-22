"""Surface the radar in Nous: the SVG radar chart, an in-page legend, and the page that ties them to
the blip database and the weekly digest.

The chart is a ThoughtWorks-style radar — four quadrants, four concentric Adopt/Trial/Assess/Hold
rings, blips placed by quadrant angle + ring radius. Mermaid can't draw a radar, so it goes in a
nous-diagrams ``animation`` fence. But a radar doesn't move: this emits a **static SVG** whose color
is the page palette via ``style="fill:var(--accent)"``. CSS ``var()`` re-evaluates when the reader
flips the theme, so the chart re-themes with zero JavaScript — which also satisfies the
reduced-motion contract by construction (there is nothing to reduce). Every color is a palette
variable with a fallback, so the SVG is also legible previewed outside Nous.

The pure builders (``render_radar_svg``/``render_radar_page``) produce markdown and are fully
tested; :func:`publish_radar` writes the page into the "AI Radar" notebook.
"""

from __future__ import annotations

import math

from pydantic import BaseModel

from forge.radar.models import RING_ORDER, Blip, Quadrant, Radar, Ring

# --- geometry ---------------------------------------------------------------

_SIZE = 720
_C = _SIZE / 2  # center x/y

#: Ring outer radii, centre → edge (Adopt innermost). The band for a ring is (prev, this].
_RING_OUTER: dict[Ring, float] = {
    Ring.ADOPT: 70,
    Ring.TRIAL: 140,
    Ring.ASSESS: 210,
    Ring.HOLD: 280,
}

#: Quadrant → its 90° sector, as (start_deg, end_deg) in an up-positive convention
#: (x = C + r·cos θ, y = C − r·sin θ, so 0°=right, 90°=up).
_QUAD_SECTOR: dict[Quadrant, tuple[float, float]] = {
    Quadrant.MODELS: (0, 90),  # top-right
    Quadrant.AGENTS: (90, 180),  # top-left
    Quadrant.TECHNIQUES: (180, 270),  # bottom-left
    Quadrant.INFRA: (270, 360),  # bottom-right
}

#: Stable draw/number order.
_QUAD_ORDER: list[Quadrant] = [
    Quadrant.MODELS,
    Quadrant.AGENTS,
    Quadrant.TECHNIQUES,
    Quadrant.INFRA,
]

_ANGLE_INSET = 12  # degrees kept clear of each quadrant's axis edges


def _xy(radius: float, angle_deg: float) -> tuple[float, float]:
    a = math.radians(angle_deg)
    return _C + radius * math.cos(a), _C - radius * math.sin(a)


class PlacedBlip(BaseModel):
    blip: Blip
    number: int
    x: float
    y: float


def place_blips(radar: Radar) -> list[PlacedBlip]:
    """Assign each blip a stable number and an (x, y) inside its quadrant sector and ring band.
    Blips in the same quadrant+ring are spread evenly across the sector's angular range."""
    placed: list[PlacedBlip] = []
    number = 0
    for quad in _QUAD_ORDER:
        start, end = _QUAD_SECTOR[quad]
        for ring in RING_ORDER:
            group = sorted(
                (b for b in radar.blips if b.quadrant == quad and b.ring == ring),
                key=lambda b: b.name.lower(),
            )
            if not group:
                continue
            inner = (
                0.0 if ring == Ring.ADOPT else _RING_OUTER[RING_ORDER[RING_ORDER.index(ring) - 1]]
            )
            mid_r = (inner + _RING_OUTER[ring]) / 2
            lo, hi = start + _ANGLE_INSET, end - _ANGLE_INSET
            for i, blip in enumerate(group):
                number += 1
                frac = 0.5 if len(group) == 1 else i / (len(group) - 1)
                angle = lo + frac * (hi - lo)
                # A small radial zig-zag so a crowded band doesn't collapse onto one arc.
                r = mid_r + (6 if i % 2 else -6) * (len(group) > 3)
                x, y = _xy(r, angle)
                placed.append(PlacedBlip(blip=blip, number=number, x=x, y=y))
    return placed


# --- SVG --------------------------------------------------------------------

_HALO = "paint-order:stroke;stroke:var(--bg,#fff);stroke-width:3px;stroke-linejoin:round"


def _svg_text(
    x: float,
    y: float,
    text: str,
    *,
    size: int,
    color: str,
    weight: str = "normal",
    anchor: str = "middle",
    halo: bool = True,
) -> str:
    style = f"fill:var({color});font:{weight} {size}px sans-serif"
    if halo:
        style += ";" + _HALO
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" dominant-baseline="central" '
        f'style="{style}">{_escape(text)}</text>'
    )


def _escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render_radar_svg(radar: Radar) -> str:
    """The radar as a self-contained, palette-themed SVG string."""
    parts: list[str] = [
        f'<svg viewBox="0 0 {_SIZE} {_SIZE}" xmlns="http://www.w3.org/2000/svg" '
        'style="width:100%;height:100%;display:block" role="img" '
        'aria-label="AI Technology Radar">'
    ]

    # Alternating ring bands (draw largest → smallest so inner overwrites), then boundary strokes.
    band_fill = {
        Ring.HOLD: "var(--panel,#f0f0f3)",
        Ring.ASSESS: "var(--bg,#fff)",
        Ring.TRIAL: "var(--panel,#f0f0f3)",
        Ring.ADOPT: "var(--bg,#fff)",
    }
    for ring in RING_ORDER[::-1]:  # Hold outermost first
        parts.append(
            f'<circle cx="{_C}" cy="{_C}" r="{_RING_OUTER[ring]}" '
            f'style="fill:{band_fill[ring]};fill-opacity:0.6"/>'
        )
    for ring in RING_ORDER:
        parts.append(
            f'<circle cx="{_C}" cy="{_C}" r="{_RING_OUTER[ring]}" '
            'style="fill:none;stroke:var(--border,#d0d0d5);stroke-width:1"/>'
        )

    # Axes.
    r_out = _RING_OUTER[Ring.HOLD]
    parts.append(
        f'<line x1="{_C - r_out}" y1="{_C}" x2="{_C + r_out}" y2="{_C}" '
        'style="stroke:var(--border,#d0d0d5);stroke-width:1"/>'
    )
    parts.append(
        f'<line x1="{_C}" y1="{_C - r_out}" x2="{_C}" y2="{_C + r_out}" '
        'style="stroke:var(--border,#d0d0d5);stroke-width:1"/>'
    )

    # Ring labels up the vertical axis, at each band's mid-radius.
    prev = 0.0
    for ring in RING_ORDER:
        mid = (prev + _RING_OUTER[ring]) / 2
        parts.append(_svg_text(_C, _C - mid, ring.value, size=11, color="--muted,#888"))
        prev = _RING_OUTER[ring]

    # Quadrant labels at each sector's mid-angle, just outside the rings.
    for quad in _QUAD_ORDER:
        start, end = _QUAD_SECTOR[quad]
        x, y = _xy(r_out + 22, (start + end) / 2)
        parts.append(_svg_text(x, y, quad.value, size=15, color="--text,#222", weight="600"))

    # Blips.
    for pb in place_blips(radar):
        parts.append(
            f'<circle cx="{pb.x:.1f}" cy="{pb.y:.1f}" r="11" style="fill:var(--accent,#7c3aed)"/>'
        )
        parts.append(
            _svg_text(
                pb.x,
                pb.y + 0.5,
                str(pb.number),
                size=11,
                color="--bg,#fff",
                weight="600",
                halo=False,
            )
        )

    parts.append("</svg>")
    return "".join(parts)


def render_radar_svg_fence(radar: Radar) -> str:
    """The radar SVG wrapped in a nous-diagrams ``animation`` fence."""
    return "```animation\n" + render_radar_svg(radar) + "\n```"


# --- page -------------------------------------------------------------------


def render_legend(radar: Radar) -> str:
    """A markdown legend keyed to the chart's blip numbers, grouped quadrant → ring, each blip with
    its ring and one-line rationale — the browsable index that pairs with the numbered dots."""
    placed = {pb.blip.slug: pb.number for pb in place_blips(radar)}
    lines: list[str] = []
    for quad in _QUAD_ORDER:
        quad_blips = [b for b in radar.blips if b.quadrant == quad]
        if not quad_blips:
            continue
        lines.append(f"### {quad.value}")
        for ring in RING_ORDER:
            ring_blips = sorted(
                (b for b in quad_blips if b.ring == ring), key=lambda b: b.name.lower()
            )
            for b in ring_blips:
                num = placed.get(b.slug, "?")
                why = f" — {b.rationale}" if b.rationale else ""
                lines.append(f"- **{num}. {b.name}** · _{ring.value}_{why}")
        lines.append("")
    return "\n".join(lines).rstrip()


def render_radar_page(
    radar: Radar,
    *,
    digest: str | None = None,
    updated: str,
    database_title: str = "Radar Blips",
) -> str:
    """The full "AI Technology Radar" page markdown: intro, chart, legend, database pointer, and the
    latest weekly digest."""
    counts = radar.counts()
    per_quad = ", ".join(
        f"{q.value} {sum(counts[q].values())}" for q in _QUAD_ORDER if sum(counts[q].values())
    )
    intro = (
        f"_Updated {updated}. {len(radar.blips)} blips"
        + (f" — {per_quad}." if per_quad else ".")
        + " Rings inner→outer: Adopt · Trial · Assess · Hold._"
    )

    body = [
        "# AI Technology Radar",
        "",
        intro,
        "",
        render_radar_svg_fence(radar),
        "",
        "## Legend",
        "",
        render_legend(radar) or "_No blips yet._",
        "",
        "## The blips",
        "",
        f'The full, filterable table lives in the **"{database_title}"** database in this '
        "notebook — browse by quadrant, ring, or recently-moved, and open a blip for its evidence "
        "trail.",
    ]

    body += ["", "## This week", "", (digest.strip() if digest else "_No synthesis run yet._")]
    return "\n".join(body) + "\n"


# --- publish ----------------------------------------------------------------

RADAR_PAGE_TITLE = "AI Technology Radar"


def publish_radar(
    client,
    radar: Radar,
    *,
    notebook_name: str,
    digest: str | None = None,
    page_title: str = RADAR_PAGE_TITLE,
    updated: str,
    to_blocks=None,
) -> dict:
    """Create-or-update the radar page in the *notebook_name* notebook, returning
    ``{page_id, created}``. Idempotent: the same page is rewritten each render.

    The page is written as **blocks**, not ``content`` markdown: the daemon's markdown-``content``
    parser silently drops fenced code blocks (the ``animation`` chart would flatten to a paragraph),
    so the markdown is converted with Nous's own :func:`nous_mcp.markdown.markdown_to_blocks` —
    which turns the fence into a live ``{type: code, language: animation}`` block. ``to_blocks`` is
    injectable for tests; production resolves the real converter lazily."""
    from forge.radar.store import _find_by_name

    if to_blocks is None:
        from nous_mcp.markdown import markdown_to_blocks as to_blocks

    notebook = _find_by_name(client.list_notebooks(), notebook_name, key="name")
    if notebook is None:
        notebook = client.create_notebook(notebook_name)
    notebook_id = notebook["id"]

    blocks = to_blocks(render_radar_page(radar, digest=digest, updated=updated))

    existing = None
    try:
        existing = client.resolve_page(notebook_id, page_title)
    except Exception:
        existing = None

    if existing and existing.get("id"):
        client.update_page(notebook_id, existing["id"], blocks=blocks, tags=["radar"])
        return {"page_id": existing["id"], "created": False}
    page = client.create_page(notebook_id, page_title, blocks=blocks, tags=["radar"])
    return {"page_id": page.get("id"), "created": True}
