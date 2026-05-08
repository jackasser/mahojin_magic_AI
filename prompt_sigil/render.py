"""Render the prompt as a magic-circle sigil dominated by vector trajectories.

A vector has a direction and a magnitude; a sequence of vectors is a path; a
path that crosses itself produces a pattern. This renderer treats the
document as a *time-ordered* sequence of vectors and draws the path through
2D PCA space — the dominant inscribed figure is the trajectory itself.

Mapping (visual element ↔ real data property):
  outer double boundary       ↔ frame
  continuous token tick band  ↔ every token of the prompt as one tick
  equal angular slots         ↔ section count (semantic order)
  radial dividers             ↔ slot boundaries
  vertex ornaments            ↔ subsection / bullet / code-block counts
  section trajectory arrows   ↔ document-order path through h1 sections
                                (drawn around the inner ring; arrows in the
                                reading direction; crossings are real)
  sentence trajectory line    ↔ document-order path through every sentence
                                in PCA-2 space; segment hue = parent section
  inter-section jump arrows   ↔ where the sentence path leaves one section
                                for another (highlighted with arrowheads)
  k-NN polygon (faint)        ↔ pairwise cosine similarity (background)
  central seal                ↔ minimal centre frame
  centre title                ↔ source filename only

Text is intentionally minimised; the figure is the data.
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np

from .analyze import Layout, compute_layout
from .parse import Section

SVG_SIZE = 1600
CX = SVG_SIZE / 2
CY = SVG_SIZE / 2

R_BOUND_OUT = 740
R_BOUND_IN = 730
R_TOKEN_OUT = 720
R_TOKEN_IN = 692
R_DIV_IN = 684
R_VERTEX_OUT = 620
R_VERTEX = 560
R_CLOUD = 480
R_SEAL_OUT = 80
R_SEAL_IN = 70

BG = "#f5f3ec"
INK = "#101010"


def _polar(r: float, theta: float) -> Tuple[float, float]:
    return CX + r * math.cos(theta), CY + r * math.sin(theta)


def _hue_from_angle(theta: float) -> float:
    return (math.degrees(theta) + 360.0) % 360.0


def _xml_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _arrow(
    x1: float, y1: float, x2: float, y2: float,
    stroke: str, width: float, head_size: float, opacity: float,
) -> str:
    """Line + filled triangle arrowhead at (x2, y2)."""
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    if length < 1e-3:
        return ""
    angle = math.atan2(dy, dx)
    # Shorten line so head sits cleanly on the endpoint.
    back = min(head_size * 0.6, length * 0.9)
    bx = x2 - back * math.cos(angle)
    by = y2 - back * math.sin(angle)
    line = (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{bx:.1f}" y2="{by:.1f}" '
        f'stroke="{stroke}" stroke-width="{width:.2f}" '
        f'stroke-opacity="{opacity:.2f}" stroke-linecap="round"/>'
    )
    a1 = angle + math.pi - 0.45
    a2 = angle + math.pi + 0.45
    px1 = x2 + head_size * math.cos(a1)
    py1 = y2 + head_size * math.sin(a1)
    px2 = x2 + head_size * math.cos(a2)
    py2 = y2 + head_size * math.sin(a2)
    head = (
        f'<polygon points="{x2:.1f},{y2:.1f} {px1:.1f},{py1:.1f} {px2:.1f},{py2:.1f}" '
        f'fill="{stroke}" fill-opacity="{opacity:.2f}"/>'
    )
    return line + head


def _knn_edges(
    sim: np.ndarray, k: int, min_sim: float
) -> Dict[Tuple[int, int], float]:
    n = sim.shape[0]
    edges: Dict[Tuple[int, int], float] = {}
    for i in range(n):
        row = sim[i].copy()
        row[i] = -np.inf
        order = np.argsort(row)[::-1]
        kept = 0
        for j in order:
            if kept >= k or row[j] < min_sim:
                break
            a, b = (i, int(j)) if i < j else (int(j), i)
            edges[(a, b)] = max(edges.get((a, b), 0.0), float(row[j]))
            kept += 1
    return edges


def _section_subsection_count(sec: Section) -> int:
    return len(sec.children) + sum(_section_subsection_count(c) for c in sec.children)


def render_sigil(root: Section, title: str = "") -> str:
    layout = compute_layout(root)
    if layout is None or len(layout.sections) == 0:
        return _empty(title)

    sections = layout.sections
    n = len(sections)
    section_2d = layout.section_2d
    sentence_2d = layout.sentence_2d
    sent_idx = layout.sentence_section_idx
    sim = layout.similarity

    raw_angle = np.arctan2(section_2d[:, 1], section_2d[:, 0])
    order = np.argsort(raw_angle)
    slot_index = np.zeros(n, dtype=int)
    for pos, idx in enumerate(order):
        slot_index[idx] = pos

    section_tokens = np.array([s.total_tokens() for s in sections])
    heaviest = int(np.argmax(section_tokens))
    slot_arc = 2.0 * math.pi / n
    base_centre = -math.pi + (slot_index[heaviest] + 0.5) * slot_arc
    rotation = -math.pi / 2 - base_centre

    slot_centre = np.zeros(n)
    slot_start = np.zeros(n)
    for i in range(n):
        c = -math.pi + (slot_index[i] + 0.5) * slot_arc + rotation
        slot_centre[i] = c
        slot_start[i] = c - slot_arc / 2

    section_hues = [_hue_from_angle(slot_centre[i]) for i in range(n)]

    boundary: List[str] = []
    cloud_dots: List[str] = []
    sentence_path: List[str] = []
    section_path: List[str] = []
    polygon_faint: List[str] = []
    inscription: List[str] = []
    dividers: List[str] = []
    vertex_ornaments: List[str] = []
    seal: List[str] = []
    centre: List[str] = []

    # ---- Outer double boundary (the frame) -------------------------------
    boundary.append(
        f'<circle cx="{CX}" cy="{CY}" r="{R_BOUND_OUT}" fill="none" '
        f'stroke="{INK}" stroke-width="1.6" stroke-opacity="0.85"/>'
    )
    boundary.append(
        f'<circle cx="{CX}" cy="{CY}" r="{R_BOUND_IN}" fill="none" '
        f'stroke="{INK}" stroke-width="0.8" stroke-opacity="0.6"/>'
    )
    boundary.append(
        f'<circle cx="{CX}" cy="{CY}" r="{R_DIV_IN}" fill="none" '
        f'stroke="{INK}" stroke-width="0.4" stroke-opacity="0.30"/>'
    )
    boundary.append(
        f'<circle cx="{CX}" cy="{CY}" r="{R_VERTEX_OUT}" fill="none" '
        f'stroke="{INK}" stroke-width="0.4" stroke-opacity="0.25"/>'
    )

    # ---- Continuous token tick band (one tick per token) -----------------
    for i, sec in enumerate(sections):
        tokens = sec.total_tokens()
        if tokens <= 0:
            continue
        a0 = slot_start[i] + slot_arc * 0.04
        a1 = slot_start[i] + slot_arc * 0.96
        n_ticks = int(min(tokens, 240))
        if n_ticks <= 0:
            continue
        density = sec.total_imperatives() / max(1, tokens)
        long_frac = min(0.4, density * 6.0)
        long_count = max(0, int(n_ticks * long_frac))
        hue = section_hues[i]
        for k in range(n_ticks):
            t = (k + 0.5) / n_ticks
            a = a0 + (a1 - a0) * t
            r_in = R_TOKEN_IN
            is_long = (long_count > 0) and (
                (k * long_count) // n_ticks
                != ((k - 1) * long_count) // n_ticks
            )
            r_out = R_TOKEN_OUT + (4 if is_long else 0)
            x1, y1 = _polar(r_in, a)
            x2, y2 = _polar(r_out, a)
            inscription.append(
                f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                f'stroke="hsl({hue:.0f} 65% 35%)" stroke-width="0.7" '
                f'stroke-opacity="0.85"/>'
            )

    # ---- Slot dividers ---------------------------------------------------
    for i in range(n):
        a = slot_start[i]
        x1, y1 = _polar(R_DIV_IN, a)
        x2, y2 = _polar(R_BOUND_IN, a)
        dividers.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{INK}" stroke-width="0.5" stroke-opacity="0.45"/>'
        )

    # ---- Section vertices on R_VERTEX ring -------------------------------
    inner_xy: List[Tuple[float, float]] = []
    for i, sec in enumerate(sections):
        x, y = _polar(R_VERTEX, slot_centre[i])
        inner_xy.append((x, y))

    # ---- k-NN polygon (faint background) ---------------------------------
    edges = _knn_edges(sim, k=4, min_sim=0.08)
    for (a_idx, b_idx), w in sorted(edges.items(), key=lambda kv: kv[1]):
        x1, y1 = inner_xy[a_idx]
        x2, y2 = inner_xy[b_idx]
        polygon_faint.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{INK}" stroke-width="0.4" '
            f'stroke-opacity="{0.08 + w * 0.20:.2f}"/>'
        )

    # ---- Section trajectory: document-order arrows around the inner ring --
    # h1+ sections are visited in document order. Crossings are real and
    # carry the prompt's narrative drift in semantic space.
    for i in range(n - 1):
        x1, y1 = inner_xy[i]
        x2, y2 = inner_xy[i + 1]
        # Soft hue blend along the timeline: start cool, end warm.
        t = i / max(1, n - 2)
        hue = 220 - t * 220  # 220 (cool blue) → 0 (warm red)
        section_path.append(
            _arrow(x1, y1, x2, y2,
                   stroke=f"hsl({hue:.0f} 60% 30%)",
                   width=1.6, head_size=9.0, opacity=0.85)
        )

    # ---- Sentence trajectory: polyline through PCA-2 in document order ---
    if sentence_2d.size:
        radii = np.linalg.norm(sentence_2d, axis=1)
        max_r = float(max(1e-6, np.percentile(radii, 98)))
        cloud_scale = R_CLOUD / max_r
        pts: List[Tuple[float, float, int]] = []
        for i in range(sentence_2d.shape[0]):
            sx, sy = sentence_2d[i]
            si = int(sent_idx[i])
            x = CX + sx * cloud_scale
            y = CY + sy * cloud_scale
            pts.append((x, y, si))

        for i in range(len(pts) - 1):
            x1, y1, s1 = pts[i]
            x2, y2, s2 = pts[i + 1]
            same = s1 == s2
            hue = section_hues[s2]
            if same:
                sentence_path.append(
                    f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                    f'stroke="hsl({hue:.0f} 55% 45%)" stroke-width="0.9" '
                    f'stroke-opacity="0.55" stroke-linecap="round"/>'
                )
            else:
                sentence_path.append(
                    _arrow(x1, y1, x2, y2,
                           stroke=f"hsl({hue:.0f} 65% 35%)",
                           width=1.2, head_size=6.0, opacity=0.80)
                )

        for x, y, si in pts:
            if (x - CX) ** 2 + (y - CY) ** 2 < (R_SEAL_OUT + 4) ** 2:
                continue
            hue = section_hues[si]
            cloud_dots.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.0" '
                f'fill="hsl({hue:.0f} 60% 50%)" fill-opacity="0.7"/>'
            )

    # ---- Vertex ornaments (small runes) ----------------------------------
    for i, sec in enumerate(sections):
        a = slot_centre[i]
        hue = section_hues[i]
        x, y = inner_xy[i]
        size = 4.5 + math.log1p(sec.total_tokens()) * 1.4
        density = sec.total_imperatives() / max(1, sec.total_tokens())
        light = max(28.0, 55.0 - min(0.05, density) / 0.05 * 25.0)
        vertex_ornaments.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{size:.1f}" '
            f'fill="hsl({hue:.0f} 65% {light:.0f}%)" '
            f'stroke="{INK}" stroke-width="0.9"/>'
        )
        sub_n = _section_subsection_count(sec)
        for k in range(sub_n):
            offset = (k - (sub_n - 1) / 2) * 5.5
            ax = math.cos(a + math.pi / 2)
            ay = math.sin(a + math.pi / 2)
            dx = math.cos(a) * 14
            dy = math.sin(a) * 14
            cx = x + ax * offset + dx
            cy = y + ay * offset + dy
            vertex_ornaments.append(
                f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="1.7" '
                f'fill="hsl({hue:.0f} 65% 40%)" fill-opacity="0.85"/>'
            )
        for k in range(sec.bullets):
            spread = (k - (sec.bullets - 1) / 2) * 0.02
            xa, ya = _polar(R_VERTEX - size - 4, a + spread)
            xb, yb = _polar(R_VERTEX - size - 14, a + spread)
            vertex_ornaments.append(
                f'<line x1="{xa:.1f}" y1="{ya:.1f}" x2="{xb:.1f}" y2="{yb:.1f}" '
                f'stroke="{INK}" stroke-width="0.7" stroke-opacity="0.55"/>'
            )
        for k in range(sec.code_blocks):
            spread = (k - (sec.code_blocks - 1) / 2) * 0.02
            xa, ya = _polar(R_VERTEX + size + 4, a + spread)
            xb, yb = _polar(R_VERTEX + size + 14, a + spread)
            vertex_ornaments.append(
                f'<line x1="{xa:.1f}" y1="{ya:.1f}" x2="{xb:.1f}" y2="{yb:.1f}" '
                f'stroke="{INK}" stroke-width="0.9" stroke-opacity="0.7"/>'
            )

    # ---- Central seal (small, mostly silent) -----------------------------
    seal.append(
        f'<circle cx="{CX}" cy="{CY}" r="{R_SEAL_OUT}" fill="{BG}" '
        f'stroke="{INK}" stroke-width="0.9" stroke-opacity="0.8"/>'
    )
    seal.append(
        f'<circle cx="{CX}" cy="{CY}" r="{R_SEAL_IN}" fill="none" '
        f'stroke="{INK}" stroke-width="0.5" stroke-opacity="0.45"/>'
    )

    centre.append(
        f'<text x="{CX}" y="{CY + 5}" text-anchor="middle" '
        f'font-family="ui-monospace, Menlo, Consolas, monospace" '
        f'font-size="11" fill="#444">{_xml_escape(title)}</text>'
    )

    body = "\n".join(
        boundary
        + polygon_faint     # faintest, under everything
        + cloud_dots        # waypoints
        + sentence_path     # primary inner pattern
        + section_path      # inscribed star (timeline order)
        + dividers
        + inscription
        + vertex_ornaments
        + seal
        + centre
    )

    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {SVG_SIZE} {SVG_SIZE}" '
        f'width="{SVG_SIZE}" height="{SVG_SIZE}" '
        f'style="background:{BG};">\n'
        f"<rect width='{SVG_SIZE}' height='{SVG_SIZE}' fill='{BG}'/>\n"
        f"{body}\n"
        "</svg>\n"
    )


def _empty(title: str) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {SVG_SIZE} {SVG_SIZE}" '
        f'width="{SVG_SIZE}" height="{SVG_SIZE}" style="background:{BG};">\n'
        f"<rect width='{SVG_SIZE}' height='{SVG_SIZE}' fill='{BG}'/>\n"
        f'<text x="{CX}" y="{CY}" text-anchor="middle" '
        f'font-family="ui-monospace, monospace" font-size="20" fill="{INK}">'
        f"(empty: {_xml_escape(title)})</text>\n"
        "</svg>\n"
    )
