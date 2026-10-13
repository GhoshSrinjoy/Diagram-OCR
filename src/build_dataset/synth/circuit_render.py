"""Draw synthetic logic circuits with schemdraw and derive exact Diagram IR targets.

render_sample(index, out_dir) is the multiprocessing worker (top level, so a spawn-context pool can
import it). It is deterministic in `index`: it draws a netlist (circuit_netlist), lays it out in
the canonical left-to-right frame (circuit_layout), maps that frame to the final signal direction
(one of the 8 axis-preserving symmetries; text is always set upright), renders gates, wires,
junction dots and labels with schemdraw into a matplotlib canvas whose data -> pixel mapping is
fixed by construction, then puts the ink on paper (circuit_augment) and saves the image.

Boxes are computed from geometry, not from pixels: gate outlines come from the placed schemdraw
elements (leads excluded), labels are TextPath glyph outlines; both go through the same pixel
mapping and optional warp as the image, padded by half the stroke width.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

import cv2
import matplotlib
import numpy as np
import schemdraw
import schemdraw.elements as elm
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.patches import PathPatch
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D
from PIL import Image

from . import circuit_augment as aug
from .circuit_layout import Layout, LayoutParams, check_layout, layout_circuit
from .circuit_netlist import Netlist, classic_netlist, random_netlist
from .circuit_symbols import GateStyle, Polyline, body_points, pins_of, place_gate, symbol_geom

DPI = 64  # a power of two keeps width / DPI * DPI exact, so the canvas is exactly W x H
MAX_SIDE = 1600
MIN_SIDE_TARGET = 600
PPU_MIN, PPU_MAX = 44.0, 110.0  # pixels per layout unit (= 2-input gate height)
PPU_HARD_MIN = 30.0
MODEL_PIXELS = 1.25e6  # the model resizes inputs to at most this many pixels
CLASSIC_P = 0.15
HOP_R = 0.1  # radius of a crossing hop (layout units; parallel wires are >= 0.25 apart)
SKETCH_P = 0.25
FONT_DIR = Path("C:/Windows/Fonts")
PRINT_FONTS = ("arial.ttf", "calibri.ttf", "times.ttf", "timesi.ttf", "cambriai.ttf", "consola.ttf",
               "verdana.ttf", "segoeui.ttf", "tahoma.ttf", "georgia.ttf", "trebuc.ttf", "pala.ttf",
               "arialbd.ttf", "GARA.TTF", "lucon.ttf")
HAND_FONTS = ("Inkfree.ttf", "segoepr.ttf", "segoesc.ttf", "comic.ttf", "BRADHITC.TTF", "mvboli.ttf",
              "LHANDW.TTF", "ITCKRIST.TTF")
# Final signal direction -> 2x2 maps (canonical x right / y down -> final frame), natural first.
ORIENT = {
    "LR": (((1, 0), (0, 1)), ((1, 0), (0, -1))),
    "RL": (((-1, 0), (0, 1)), ((-1, 0), (0, -1))),
    "TB": (((0, 1), (1, 0)), ((0, -1), (1, 0))),
    "BT": (((0, 1), (-1, 0)), ((0, -1), (-1, 0))),
}
ROTATION = {"LR": 0, "TB": 90, "RL": 180, "BT": 270}
Pts = list[tuple[float, float]]


@lru_cache(maxsize=1)
def _fonts() -> tuple[list[str], list[str]]:
    def have(names: tuple[str, ...]) -> list[str]:
        return [str(FONT_DIR / n) for n in names if (FONT_DIR / n).exists()]

    dejavu = str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf")
    return have(PRINT_FONTS) or [dejavu], have(HAND_FONTS) or [dejavu]


def _seed(index: int, attempt: int = 0) -> int:
    return int.from_bytes(hashlib.sha256(f"synth_circuits:{index}:{attempt}".encode()).digest()[:8], "little")


@dataclass
class Style:
    labeled: bool
    direction: str
    matrix: tuple[tuple[int, int], tuple[int, int]]
    sketch: bool
    paper: str  # white | scan | grid
    ink: str  # black | sepia | blue | pencil
    font: str
    font_units: float  # label em size in layout units
    lw_rel: float  # stroke width in layout units (pixels = lw_rel * ppu)
    dot_rel: float  # junction dot radius / stroke width
    term_r: float  # open terminal circles at labelled stub ends (0 = none)
    long_side: int  # target longest image side
    margin: float  # margin as a fraction of the drawing's longest side
    square: bool
    warp: bool
    hops: bool  # crossings drawn as little semicircular hops (some LogicBench-1K drawings do)
    jpeg_q: int  # 0 = PNG


def sample_style(rng: random.Random, labeled: bool) -> Style:
    direction = rng.choices(["LR", "TB", "BT", "RL"], weights=[0.42, 0.24, 0.24, 0.10])[0]
    matrix = ORIENT[direction][0 if rng.random() < 0.75 else 1]
    sketch = rng.random() < SKETCH_P
    if labeled:
        paper = rng.choices(["white", "scan", "grid"], weights=[0.45, 0.3, 0.25])[0]
    else:
        paper = rng.choices(["white", "scan", "grid"], weights=[0.2, 0.6, 0.2])[0]
    if paper == "scan" and rng.random() < 0.7:
        ink = "sepia"
    else:
        ink = rng.choices(["black", "blue", "pencil"], weights=[0.62, 0.2, 0.18])[0]
    printed, hand = _fonts()
    use_hand = rng.random() < (0.85 if sketch else 0.12)
    font = rng.choice(hand if use_hand else printed)
    jpeg_q = 0
    if paper == "scan":
        jpeg_q = rng.randint(60, 92)
    elif paper == "grid" and rng.random() < 0.5:
        jpeg_q = rng.randint(75, 95)
    return Style(
        labeled=labeled, direction=direction, matrix=matrix, sketch=sketch, paper=paper, ink=ink,
        font=font, font_units=rng.uniform(0.5, 0.78),
        lw_rel=rng.uniform(0.032, 0.06) if paper == "scan" else rng.uniform(0.024, 0.05),
        dot_rel=rng.uniform(1.2, 2.0),
        term_r=rng.uniform(0.07, 0.11) if labeled and rng.random() < 0.3 else 0.0,
        long_side=rng.randint(MIN_SIDE_TARGET, MAX_SIDE),
        margin=rng.uniform(0.04, 0.12) if paper == "scan" else rng.uniform(0.03, 0.18),
        square=rng.random() < (0.6 if paper == "scan" else 0.25),
        warp=paper != "white" and rng.random() < 0.35, hops=rng.random() < 0.12, jpeg_q=jpeg_q,
    )


def _layout_params(rng: random.Random, st: Style) -> LayoutParams:
    return LayoutParams(
        gate_gap=rng.choice([0.5, 0.75, 1.0, 1.25, 1.5]),
        wire_gap=rng.choice([0.3, 0.4, 0.5, 0.6]),
        wire_pitch=rng.choice([0.25, 0.35, 0.5]),
        track_pitch=rng.uniform(0.28, 0.5),
        channel_margin=rng.uniform(0.35, 0.7),
        align=rng.choice([0.0, 0.0, 0.5]),
        layering=rng.choice(["asap", "alap"]),
        input_mode="edge" if st.labeled else rng.choice(["edge", "near"]),
        output_mode="edge" if st.labeled else rng.choice(["edge", "near"]),
        stub_extra=rng.uniform(0.0, 0.6),
        natural_inputs=st.labeled or rng.random() < 0.5,
        vertical_flow=st.direction in ("TB", "BT"),
        label_gap=rng.uniform(0.1, 0.22),
        term_r=st.term_r,
    )


def _gate_style(rng: random.Random) -> GateStyle:
    return GateStyle(sx=rng.uniform(0.85, 1.3), s_not=rng.uniform(0.65, 0.95), leadin=rng.uniform(0.22, 0.45),
                     leadout=rng.uniform(0.15, 0.45), lead_not=rng.uniform(0.15, 0.35))


@lru_cache(maxsize=4096)
def _glyphs(text: str, font: str) -> tuple:
    """TextPath of `text` at em size 1, translated so its ink box starts at (0, 0); and that box."""
    tp = TextPath((0, 0), text, size=1.0, prop=FontProperties(fname=font))
    ext = tp.get_extents()
    path = Affine2D().translate(-ext.x0, -ext.y0).transform_path(tp)
    return path, ext.width, ext.height


# --------------------------------------------------------------------------- #
# drawing
# --------------------------------------------------------------------------- #
def _hops(lay: Layout, net: str, a: tuple[float, float], b: tuple[float, float]) -> list[float]:
    """x of every clean crossing of the horizontal wire a-b (canonical frame) by another net."""
    if abs(a[1] - b[1]) > 1e-9:
        return []
    y, x1, x2 = a[1], min(a[0], b[0]), max(a[0], b[0])
    xs = []
    for n2, p, q in lay.wires:
        if n2 == net or abs(p[0] - q[0]) > 1e-9:  # same net, or not vertical
            continue
        if x1 + 0.2 <= p[0] <= x2 - 0.2 and min(p[1], q[1]) + 0.2 <= y <= max(p[1], q[1]) - 0.2:
            xs.append(p[0])
    return sorted(xs)


def _render_check(alpha: np.ndarray, slack: float, wires: list[Pts], gate_pts: dict[str, Pts],
                  gate_span: dict[str, Pts], dots: Pts, dot_r: float, boxes: list[Pts],
                  holes: list[tuple[float, float, float]]) -> None:
    """The picture must show exactly the layout: no ink off the expected geometry, and every wire
    and gate outline actually inked (except inside `holes`, the white-filled terminal circles).
    Guards the drawing code against misplaced elements."""
    h, w = alpha.shape
    pad = int(math.ceil(slack))
    expect = np.zeros((h, w), np.uint8)
    for poly in wires:
        cv2.polylines(expect, [np.round(np.array(poly)).astype(np.int32)], False, 1, thickness=2 * pad + 1)
    for pts in list(gate_span.values()) + boxes:
        q = np.array(pts)
        x0, y0 = np.floor(q.min(0)).astype(int) - pad
        x1, y1 = np.ceil(q.max(0)).astype(int) + pad
        cv2.rectangle(expect, (int(x0), int(y0)), (int(x1), int(y1)), 1, -1)
    for c in dots:
        cv2.circle(expect, (int(round(c[0])), int(round(c[1]))), int(math.ceil(dot_r)) + pad, 1, -1)
    stray = int(((alpha > 0.35) & (expect == 0)).sum())
    assert stray <= 25, f"render check: {stray} ink pixels off the layout"
    k = int(math.ceil(slack)) - 1
    near = cv2.dilate(alpha, np.ones((2 * k + 1, 2 * k + 1), np.uint8))

    def inked(pts: np.ndarray) -> float:
        for cx, cy, r in holes:
            pts = pts[np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) > r + 1.5]
        if len(pts) == 0:
            return 1.0
        xi = np.clip(np.rint(pts[:, 0]).astype(int), 0, w - 1)
        yi = np.clip(np.rint(pts[:, 1]).astype(int), 0, h - 1)
        return float((near[yi, xi] > 0.25).mean())

    for poly in wires:
        q = np.array(poly, float)
        samples = [np.linspace(q[i], q[i + 1], max(2, int(np.hypot(*(q[i + 1] - q[i])) / 2)))
                   for i in range(len(q) - 1)]
        assert inked(np.concatenate(samples)) > 0.97, "render check: a wire is not drawn"
    for gid, pts in gate_pts.items():
        assert inked(np.array(pts, float)) > 0.9, f"render check: gate {gid} is not drawn"


def _apply(m: tuple, p: tuple[float, float]) -> tuple[float, float]:
    return (m[0][0] * p[0] + m[0][1] * p[1], m[1][0] * p[0] + m[1][1] * p[1])


def _rect_f(m: tuple, r: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    a, b = _apply(m, (r[0], r[1])), _apply(m, (r[2], r[3]))
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1]))


def _draw(nl: Netlist, lay: Layout, gs: GateStyle, st: Style, rng: random.Random, nrng: np.random.Generator,
          out_base: Path) -> Optional[dict]:
    m = st.matrix
    flow = _apply(m, (1, 0))  # final-frame direction of the signal flow
    theta = math.degrees(math.atan2(-flow[1], flow[0]))  # schemdraw is y-up

    # ---- extent of everything in the final frame (layout units, y down)
    pts: list[tuple[float, float]] = []
    for g in lay.gates:
        b = g.body()
        pts += [_apply(m, (g.x, b[1])), _apply(m, (g.out()[0], b[3]))]
    for _, a, b in lay.wires:
        pts += [_apply(m, a), _apply(m, b)]
    label_f = {t.id: _rect_f(m, t.label_rect) for t in lay.terminals if t.label_rect}
    for r in label_f.values():
        pts += [(r[0], r[1]), (r[2], r[3])]
    pad = 0.2
    u_min = min(p[0] for p in pts) - pad
    v_min = min(p[1] for p in pts) - pad
    ext_w = max(p[0] for p in pts) + pad - u_min
    ext_h = max(p[1] for p in pts) + pad - v_min

    # ---- scale and canvas
    longest = max(ext_w, ext_h)
    # the sampled size, but never below PPU_MIN (big circuits grow up to MAX_SIDE instead)
    ppu = min(PPU_MAX, max(PPU_MIN, st.long_side * (1 - 2 * st.margin) / longest))
    side = longest * ppu * (1 + 2 * st.margin) + 12
    if side > MAX_SIDE:
        ppu *= (MAX_SIDE - 12) / (side - 12)
    if ppu < PPU_HARD_MIN:
        return None
    cw, ch = ext_w * ppu, ext_h * ppu
    mpx = st.margin * max(cw, ch) + 6
    W, H = cw + 2 * mpx, ch + 2 * mpx
    if st.square:  # towards square (LogicBench is 512x512), at most 1.4:1 for long drawings
        lo = max(W, H) / rng.uniform(1.0, 1.4)
        # ...unless that leaves a gate under ~30 px once the model shrinks the image to <= 1.25 MP
        if ppu * min(1.0, math.sqrt(MODEL_PIXELS / (max(W, lo) * max(H, lo)))) >= 30:
            W, H = max(W, lo), max(H, lo)
    if max(W, H) < MIN_SIDE_TARGET:
        k = MIN_SIDE_TARGET / max(W, H)
        W, H = W * k, H * k
    W, H = int(math.ceil(min(W, MAX_SIDE))), int(math.ceil(min(H, MAX_SIDE)))
    u0 = u_min - (W - cw) / 2 / ppu  # final-frame coordinate of the image's left edge
    v0 = v_min - (H - ch) / 2 / ppu

    def to_px(sd: tuple[float, float]) -> tuple[float, float]:
        return ((sd[0] - u0) * ppu, (-sd[1] - v0) * ppu)

    def to_sd(p: tuple[float, float]) -> tuple[float, float]:
        f = _apply(m, p)
        return (f[0], -f[1])

    lw_px = min(5.0, max(1.1, st.lw_rel * ppu))
    lw_pt = lw_px * 72 / DPI
    dot_r = min(max(st.dot_rel * lw_px, 2.2), 0.085 * ppu) / ppu
    # hand-drawn: a gentle wobble (wavelength well above the gate size) plus a smooth rubber-sheet warp
    sketch = None
    if st.sketch:
        amp = max(0.6, ppu * rng.uniform(0.01, 0.022))
        sketch = (amp, ppu * rng.uniform(1.5, 4.0), rng.uniform(1.2, 2.5))

    fig = Figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor="white")
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_axis_off()
    ax.set_xlim(u0, u0 + W / ppu)
    ax.set_ylim(-(v0 + H / ppu), -v0)
    d = schemdraw.Drawing(canvas=ax, show=False)
    style = {"lw": lw_pt, "color": "black"}

    gate_pts: dict[str, list[tuple[float, float]]] = {}
    gate_span: dict[str, list[tuple[float, float]]] = {}  # body + pins, for the render check
    wire_px: list[list[tuple[float, float]]] = []
    dot_px: list[tuple[float, float]] = []
    for g in lay.gates:
        el, bub = place_gate(d, g.kind, g.geom.n_in, gs, to_sd((g.x, g.y)), theta, **style)
        ins, _ = pins_of(el, g.kind)
        want = sorted((round(a, 6), round(b, 6)) for a, b in (to_sd(p) for p in g.pins()))
        got = sorted((round(a, 6), round(b, 6)) for a, b in ins)
        # the schemdraw symbol must put its pins exactly where the router ended the wires
        assert all(abs(a[0] - b[0]) < 1e-6 and abs(a[1] - b[1]) < 1e-6
                   for a, b in zip(want, got, strict=True)), f"{g.id}: pins {got} != layout {want}"
        gate_pts[g.id] = [to_px(p) for p in body_points(el, bub)]
        gate_span[g.id] = gate_pts[g.id] + [to_px(p) for p in ins + [pins_of(el, g.kind)[1]]]
    for net, a, b in lay.wires:
        hops = _hops(lay, net, a, b) if st.hops else []
        if not hops:
            d.add(elm.Line().endpoints(to_sd(a), to_sd(b)).style(**style))
            wire_px.append([to_px(to_sd(a)), to_px(to_sd(b))])
            continue
        path = [a]
        for xc in hops:  # canonical horizontal wire: hop over each crossing vertical
            path += [(xc + HOP_R * math.cos(t), a[1] - HOP_R * math.sin(t))
                     for t in np.linspace(math.pi, 0, 13)]
        path.append(b)
        d.add(Polyline([to_sd(q) for q in path]).at((0, 0)).theta(0).style(**style))
        wire_px.append([to_px(to_sd(q)) for q in path])
    for _, p in lay.dots:
        d.add(elm.Dot(radius=dot_r).at(to_sd(p)).style(**style))
        dot_px.append(to_px(to_sd(p)))
    term_extra: dict[str, list[tuple[float, float]]] = {}
    for t in lay.terminals:
        if st.term_r > 0 and t.label_rect:
            d.add(elm.Dot(radius=st.term_r, open=True).at(to_sd(t.end)).style(**style))
        r = st.term_r if (st.term_r > 0 and t.label_rect) else 0.0
        c = to_sd(t.end)
        term_extra[t.id] = [to_px((c[0] - r, c[1] - r)), to_px((c[0] + r, c[1] + r))]
    with matplotlib.rc_context({"path.sketch": sketch}):
        d.draw(show=False)
    ax.set_xlim(u0, u0 + W / ppu)  # unchanged by schemdraw (user axes), re-asserted
    ax.set_ylim(-(v0 + H / ppu), -v0)

    label_px: dict[str, list[tuple[float, float]]] = {}
    for t in lay.terminals:
        if t.id not in label_f:
            continue
        path, w1, h1 = _glyphs(t.label, st.font)
        r = label_f[t.id]
        cu, cv = (r[0] + r[2]) / 2, (r[1] + r[3]) / 2
        s = st.font_units
        tr = Affine2D().scale(s).translate(cu - w1 * s / 2, -cv - h1 * s / 2)
        ax.add_patch(PathPatch(tr.transform_path(path), facecolor="black", edgecolor="none", lw=0))
        x0, y0 = to_px((cu - w1 * s / 2, -cv + h1 * s / 2))
        x1, y1 = to_px((cu + w1 * s / 2, -cv - h1 * s / 2))
        label_px[t.id] = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())
    assert rgba.shape[:2] == (H, W), (rgba.shape, H, W)
    alpha = 1.0 - rgba[:, :, :3].astype(np.float32).mean(axis=2) / 255.0
    fig.clear()
    slack = lw_px / 2 + (sketch[0] if sketch else 0.0) + 2
    holes = [((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (b[0] - a[0]) / 2) for a, b in term_extra.values()]
    _render_check(alpha, slack, wire_px, gate_pts, gate_span, dot_px, dot_r * ppu,
                  list(label_px.values()) + list(term_extra.values()), [h for h in holes if h[2] > 0])

    # ---- optional geometric effects; boxes follow through exactly the same maps
    ink_yx = np.argwhere(alpha > 0.05)
    (yy0, xx0), (yy1, xx1) = ink_yx.min(0), ink_yx.max(0)
    frame = np.array([[xx0, yy0], [xx1, yy0], [xx1, yy1], [xx0, yy1], [(xx0 + xx1) / 2, yy0],
                      [(xx0 + xx1) / 2, yy1], [xx0, (yy0 + yy1) / 2], [xx1, (yy0 + yy1) / 2]], float)

    def inside(q: np.ndarray) -> bool:
        return bool(q[:, 0].min() > 4 and q[:, 1].min() > 4
                    and q[:, 0].max() < W - 5 and q[:, 1].max() < H - 5)

    field = None
    if st.sketch:
        cand_f = aug.rubber_sheet(nrng, H, W, amp=max(W, H) * float(nrng.uniform(0.003, 0.009)))
        if inside(aug.sheet_points(cand_f, frame)):
            field = cand_f
            alpha = aug.remap(alpha, field)
            frame = aug.sheet_points(field, frame)
    hmat = None
    if st.warp:
        cand = aug.warp_homography(nrng, H, W)
        if inside(aug.apply_points(cand, frame)):
            hmat = cand
            alpha = aug.warp_alpha(alpha, hmat)
    pitch = min(60.0, max(12.0, rng.choice([0.5, 0.75, 1.0]) * ppu))
    img = aug.compose(nrng, alpha, st.paper, st.ink, pitch=pitch)

    half = lw_px / 2 + (sketch[0] if sketch else 0.0)

    def box(points: list[tuple[float, float]], grow: float) -> list[int]:
        q = aug.apply_points(hmat, aug.sheet_points(field, np.array(points, float)))
        x0, y0 = q[:, 0].min() - grow, q[:, 1].min() - grow
        x1, y1 = q[:, 0].max() + grow, q[:, 1].max() + grow
        return [max(0, min(1000, round(1000 * x0 / W))), max(0, min(1000, round(1000 * y0 / H))),
                max(0, min(1000, round(1000 * x1 / W))), max(0, min(1000, round(1000 * y1 / H)))]

    nodes = [[g.id, g.kind, "", box(gate_pts[g.id], half)] for g in lay.gates]
    edges: list[list[str]] = []
    for g in nl.gates:
        for s in g.inputs:
            if s[0] == "g" or st.labeled:
                edges.append([s, g.id])
    if st.labeled:
        for t in lay.terminals:
            nodes.append([t.id, t.kind, t.label, box(label_px[t.id] + term_extra[t.id], half)])
        for j, (_, drv) in enumerate(nl.outputs):
            edges.append([drv, f"o{j}"])

    ext = ".jpg" if st.jpeg_q else ".png"
    path = out_base.with_suffix(ext)
    stale = out_base.with_suffix(".png" if st.jpeg_q else ".jpg")  # left by an older build of this id
    if stale.exists():
        stale.unlink()
    pil = Image.fromarray(img)
    if st.jpeg_q:
        pil.save(path, quality=st.jpeg_q)
    else:
        pil.save(path, optimize=False, compress_level=6)
    return {"image": str(path), "nodes": nodes, "edges": edges, "size": [W, H], "ppu": round(ppu, 2),
            "lw_px": round(lw_px, 2)}


# --------------------------------------------------------------------------- #
# worker
# --------------------------------------------------------------------------- #
def build_layout(index: int, attempt: int, labeled: bool):
    """Netlist, layout and styles for one attempt (None when this attempt does not route)."""
    rng = random.Random(_seed(index, attempt + 1))
    st = sample_style(rng, labeled)
    nl = classic_netlist(rng) if rng.random() < CLASSIC_P else None
    if nl is None:
        nl = random_netlist(rng, n_gates=None if attempt < 6 else rng.randint(2, 8))
        if nl is None:
            return None
    gs = _gate_style(rng)
    geoms = {g.id: symbol_geom(g.kind, len(g.inputs), gs) for g in nl.gates}
    sizes: dict[str, tuple[float, float]] = {}
    if labeled:
        names = {f"i{k}": s for k, s in enumerate(nl.inputs)}
        names.update({f"o{j}": s for j, (s, _) in enumerate(nl.outputs)})
        for tid, text in names.items():
            _, w1, h1 = _glyphs(text, st.font)
            sizes[tid] = (w1 * st.font_units, h1 * st.font_units)
    p = _layout_params(rng, st)
    lay = layout_circuit(nl, geoms, sizes, p)
    if lay is None or check_layout(lay, p.min_sep):
        return None
    return nl, lay, gs, st, rng


def render_sample(index: int, out_dir: str) -> dict:
    """Worker: render sample `index` into out_dir; returns plain data (or an 'error' entry)."""
    try:
        labeled = random.Random(_seed(index)).random() < 0.5
        local_id = f"{'labeled' if labeled else 'bare'}_{index:05d}"
        for attempt in range(12):
            built = build_layout(index, attempt, labeled)
            if built is None:
                continue
            nl, lay, gs, st, rng = built
            nrng = np.random.default_rng(_seed(index, 100 + attempt))
            res = _draw(nl, lay, gs, st, rng, nrng, Path(out_dir) / local_id)
            if res is None:
                continue
            res.update({
                "index": index, "local_id": local_id, "direction": st.direction,
                "meta": {"style": "labeled" if labeled else "bare", "rotation": ROTATION[st.direction],
                         "mirrored": st.matrix != ORIENT[st.direction][0], "n_gates": len(nl.gates),
                         "sketch": st.sketch, "background": st.paper, "ink": st.ink,
                         **({"classic": nl.classic} if nl.classic else {})},
                "attempts": attempt + 1,
            })
            return res
        return {"index": index, "error": "no clean layout in 12 attempts"}
    except Exception as exc:  # report, never kill the pool
        return {"index": index, "error": f"{type(exc).__name__}: {exc}"}
