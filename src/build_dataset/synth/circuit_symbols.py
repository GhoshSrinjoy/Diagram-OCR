"""Logic-gate symbols for the synthetic circuit generator, built from schemdraw.logic.

A gate is placed by its input-pin column and its centre line. Multi-input gates can be stretched
along the signal axis (`sx`); 3-input gates are also stretched across it (SY3) so that all their
pins land on the layout grid. Inversion bubbles of multi-input gates are drawn as separate circles
so that they stay round under that non-uniform scaling. NOT / BUF are two-terminal elements with
equal leads and a uniform scale.

symbol_geom() measures a symbol in the canonical layout frame (signal flows to +x, y points DOWN,
origin = input-pin column on the centre line); body_points() returns the drawn outline of a
placed symbol (leads excluded) in drawing coordinates, which is what the target boxes enclose.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import schemdraw
import schemdraw.logic as logic
from schemdraw.elements import Element
from schemdraw.segments import Segment, SegmentCircle

GRID = 0.25  # layout grid; 2-input pins sit at +-1 grid step, 3-input pins at 0 and +-2
BUBBLE_R = 0.12  # schemdraw.logic.notbubble
SY3 = 0.5 / 0.33  # schemdraw spaces 3 inputs .33 apart; stretch them to +-0.5


@dataclass(frozen=True)
class GateStyle:
    """Per-drawing symbol proportions (local schemdraw units, before scaling)."""

    sx: float = 1.0  # length scale of multi-input gates
    s_not: float = 0.8  # uniform scale of NOT / BUF
    leadin: float = 0.3
    leadout: float = 0.3
    lead_not: float = 0.3


@dataclass(frozen=True)
class SymbolGeom:
    """A symbol measured in the canonical frame (x along the signal, y down), relative to its
    input-pin column (x = 0) and centre line (y = 0)."""

    kind: str
    n_in: int
    length: float  # input pins -> output pin
    pin_offsets: tuple[float, ...]  # y of each input pin, top first
    body: tuple[float, float, float, float]  # x0, y0, x1, y1 of the drawn body (leads excluded)

    @property
    def half_h(self) -> float:
        return max(-self.body[1], self.body[3])


class Polyline(Element):
    """An open polyline given in drawing coordinates (wire with crossing hops)."""

    def __init__(self, points: list[tuple[float, float]], **kwargs):
        super().__init__(**kwargs)
        self.segments.append(Segment(points))
        self.anchors["start"] = points[0]
        self.anchors["end"] = points[-1]


class Bubble(Element):
    """An inversion bubble as an element of its own (stays circular whatever the gate scaling)."""

    def __init__(self, radius: float = BUBBLE_R, **kwargs):
        super().__init__(**kwargs)
        self.segments.append(SegmentCircle((0, 0), radius))
        self.anchors["center"] = (0, 0)


def make_gate(kind: str, n_in: int, gs: GateStyle) -> tuple[Element, tuple[float, float], Optional[float]]:
    """Unplaced schemdraw element, its (sx, sy) zoom, and the local x of a separate bubble."""
    if kind in ("not", "buf"):
        el = logic.Not() if kind == "not" else logic.Buf()
        in_len = 0.65 + (2 * BUBBLE_R if kind == "not" else 0.0)
        el.length((in_len + 2 * gs.lead_not) * gs.s_not)  # two-terminal: equal leads both sides
        return el, (gs.s_not, gs.s_not), None
    bubble = kind in ("nand", "nor", "xnor")
    leadout = gs.leadout + (2 * BUBBLE_R if bubble else 0.0)
    if kind in ("and", "nand"):
        el = logic.And(inputs=n_in, nand=bubble, leadin=gs.leadin, leadout=leadout)
        tip = gs.leadin + 0.65 + 0.5
    else:
        el = logic.Or(inputs=n_in, nor=bubble, xor=kind in ("xor", "xnor"), leadin=gs.leadin, leadout=leadout)
        tip = el.tip
    if bubble:  # drawn separately so that it stays round
        el.segments = [s for s in el.segments if not isinstance(s, SegmentCircle)]
    zoom = (gs.sx * 1.15, SY3) if n_in == 3 else (gs.sx, 1.0)
    return el, zoom, (tip + BUBBLE_R if bubble else None)


def place_gate(d: schemdraw.Drawing, kind: str, n_in: int, gs: GateStyle, at: tuple[float, float],
               theta: float, **style) -> tuple[Element, Optional[Bubble]]:
    """Add a gate to `d` with its input-pin column / centre line at `at` (drawing coords)."""
    el, (sx, sy), bub_x = make_gate(kind, n_in, gs)
    el.scalex(sx).scaley(sy)
    d.add(el.at(at).theta(theta).style(**style))
    bub = None
    if bub_x is not None:
        bub = Bubble(BUBBLE_R * sx)
        d.add(bub.at(el.transform.transform((bub_x, 0))).theta(0).style(**style))
    return el, bub


def pins_of(el: Element, kind: str) -> tuple[list[tuple[float, float]], tuple[float, float]]:
    """Input pin and output pin positions (lead ends) of a placed gate, drawing coordinates."""
    a = el.absanchors
    if kind in ("not", "buf"):
        return [tuple(a["start"])], tuple(a["end"])
    ins = [tuple(v) for k, v in a.items() if k.startswith("in") and k[2:].isdigit()]
    return ins, tuple(a["out"])


def body_points(el: Element, bub: Optional[Bubble] = None, n_circle: int = 48) -> list[tuple[float, float]]:
    """Outline of the drawn symbol without its leads, in drawing coordinates."""
    pts: list[tuple[float, float]] = []
    for elem in (el, bub):
        if elem is None:
            continue
        for s in elem.segments:
            if isinstance(s, Segment):
                path = [p for p in s.path if not (math.isnan(p[0]) or math.isnan(p[1]))]
                if len(path) >= 2 and all(abs(p[1] - path[0][1]) < 1e-9 for p in path):
                    continue  # a straight horizontal (local frame) line is a lead
                pts += [tuple(elem.transform.transform(p)) for p in path]
            elif isinstance(s, SegmentCircle):
                for k in range(n_circle):
                    t = 2 * math.pi * k / n_circle
                    p = (s.center[0] + s.radius * math.cos(t), s.center[1] + s.radius * math.sin(t))
                    pts.append(tuple(elem.transform.transform(p)))
    return pts


def symbol_geom(kind: str, n_in: int, gs: GateStyle) -> SymbolGeom:
    d = schemdraw.Drawing(show=False)
    el, bub = place_gate(d, kind, n_in, gs, (0.0, 0.0), 0.0)
    ins, out = pins_of(el, kind)
    pts = body_points(el, bub)
    # drawing coords are y-up; the canonical frame is y-down
    xs = [p[0] for p in pts]
    ys = [-p[1] for p in pts]
    return SymbolGeom(
        kind=kind,
        n_in=n_in,
        length=out[0],
        pin_offsets=tuple(sorted(-p[1] for p in ins)),
        body=(min(xs), min(ys), max(xs), max(ys)),
    )
