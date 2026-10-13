"""Draw a laid-out BPMN scene with matplotlib, in a clean tool style or a hand-drawn style.

Both styles share the element drawing code; they differ in the painter:
  * CleanPainter draws exact geometry (bpmn.io / Camunda look: rounded tasks, thin start / thick
    end circles, filled arrowheads, dashed message flows with an open circle and open arrowhead).
  * SketchPainter draws every stroke as a wobbly pen line (low-frequency displacement, slight bow,
    overshooting corners, circles that do not close exactly) and writes text word by word in a
    handwriting font with small size / angle / baseline jitter.

Coordinates: the scene is in layout units; `to_px` maps them to canvas pixels (scale k + offset).
Matplotlib runs at dpi 72, so 1 pt = 1 px for line widths and font sizes.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch, PathPatch, Polygon
from matplotlib.path import Path as MplPath
from matplotlib.textpath import TextPath

from .bpmn_layout import Scene, font_props
from .bpmn_model import El, Flow

Pt = tuple[float, float]

PEN_COLORS = [("#1b1b1f", 3.0), ("#14205e", 2.0), ("#1a3b9c", 1.6), ("#24366e", 1.0), ("#3a3a3a", 0.8),
              ("#5a5a5f", 0.9), ("#2a2a2a", 1.0), ("#0b3d6b", 0.6)]
CLEAN_THEMES = ["bpmnio", "bpmnio", "bpmnio", "camunda", "signavio", "colored", "gray"]
MULTI_PENS = ["#1b1b1f", "#1a3b9c", "#5b2a86", "#1f6f5c", "#9c1f1f", "#0e7c86", "#7a4b12"]


@dataclass
class Theme:
    style: str
    ink: str  # main stroke / text colour
    lw: float  # base stroke width in px (task outline)
    bg: str = "#ffffff"
    fills: dict = field(default_factory=dict)  # kind -> fill colour (clean)
    strokes: dict = field(default_factory=dict)  # kind -> stroke colour (clean)
    pencil: bool = False
    arrow: str = "filled"  # sketch: open / filled
    wobble: float = 1.0
    xor_mark: str = "x"
    hdr_fill: Optional[str] = None
    dot_grid: bool = False
    seq_lw: float = 1.0
    # handwriting (sketch): slant, glyph distortion, size / angle / baseline jitter, stroke boldness
    slant: float = 0.0
    glyph_amp: float = 0.0
    size_jit: float = 0.0
    angle_jit: float = 0.0
    base_jit: float = 0.0
    bold: float = 0.0


def make_theme(style: str, rng: random.Random, k: float) -> Theme:
    if style == "clean":
        name = rng.choice(CLEAN_THEMES)
        ink = "#000000" if name in ("bpmnio", "camunda") else "#222222"
        t = Theme(style, ink, lw=max(1.0, 2.0 * k * rng.uniform(0.75, 1.0)))
        t.seq_lw = t.lw * rng.uniform(0.6, 0.85)
        if name == "signavio":
            t.fills = {"task": "#fffecc", "start": "#ffffff", "end": "#ffffff", "inter": "#ffffff",
                       "gw": "#ffffff", "data": "#ffffff", "store": "#ffffff"}
            t.strokes = {"task": "#bb2a2a" if rng.random() < 0.3 else "#000000"}
            t.hdr_fill = "#f2f2f2"
        elif name == "colored":
            palette = [("#bbdefb", "#0d4372"), ("#c8e6c9", "#205022"), ("#ffe0b2", "#6b3c00"),
                       ("#e1bee7", "#5b176d"), ("#ffcdd2", "#831311")]
            for kind in ("task", "start", "end", "inter", "gw", "data", "store"):
                if rng.random() < 0.6:
                    fill, stroke = rng.choice(palette)
                    t.fills[kind], t.strokes[kind] = fill, stroke
        elif name == "gray":
            t.ink = "#404040"
            t.fills = {"task": "#f5f5f5"}
            t.hdr_fill = "#eeeeee" if rng.random() < 0.5 else None
        if rng.random() < 0.25:
            t.dot_grid = True
        if rng.random() < 0.08:
            t.bg = rng.choice(["#fbfbfb", "#f7f9fc", "#fffdf7"])
        t.xor_mark = "x" if rng.random() < 0.88 else "blank"
        return t
    colors, weights = zip(*PEN_COLORS, strict=True)
    ink = rng.choices(colors, weights=weights)[0]
    pencil = ink in ("#5a5a5f", "#3a3a3a") and rng.random() < 0.6
    t = Theme(style, ink, lw=max(1.2, k * rng.uniform(1.4, 2.6)) * (0.8 if pencil else 1.0), pencil=pencil)
    t.seq_lw = t.lw * rng.uniform(0.85, 1.05)
    t.arrow = "open" if rng.random() < 0.6 else "filled"
    t.wobble = rng.uniform(0.7, 1.6)
    t.xor_mark = "x" if rng.random() < 0.9 else "blank"
    if rng.random() < 0.15:  # several pens: e.g. data objects, events or messages in another colour
        others = [c for c in MULTI_PENS if c != ink]
        for group in (("data", "store"), ("start", "end", "inter"), ("msg",), ("note", "assoc"), ("gw",)):
            if rng.random() < 0.45:
                col = rng.choice(others)
                for kind in group:
                    t.strokes[kind] = col
    t.slant = math.radians(rng.uniform(-8, 16))
    t.glyph_amp = rng.uniform(0.015, 0.05)
    t.size_jit = rng.uniform(0.03, 0.09)
    t.angle_jit = rng.uniform(1.0, 4.0)
    t.base_jit = rng.uniform(0.02, 0.08)
    t.bold = rng.uniform(0.15, 0.55)
    return t


# --------------------------------------------------------------------------- #
# Painters
# --------------------------------------------------------------------------- #
class CleanPainter:
    def __init__(self, ax, theme: Theme, rng: random.Random, font: str):
        self.ax, self.t, self.rng, self.font = ax, theme, rng, font

    def poly(self, pts: list[Pt], lw: float, color: str, dash: Optional[tuple] = None, closed: bool = False,
             z: float = 2, alpha: float = 1.0, cap: str = "butt") -> None:
        if closed:
            pts = list(pts) + [pts[0]]
        xs, ys = zip(*pts, strict=True)
        kw = {"linestyle": (0, dash)} if dash else {}
        self.ax.plot(xs, ys, lw=lw, color=color, zorder=z, alpha=alpha, solid_capstyle=cap,
                     solid_joinstyle="miter", dash_capstyle="butt", **kw)

    def fill(self, pts: list[Pt], color: str, z: float = 3, edge: Optional[str] = None,
             lw: float = 0.0) -> None:
        self.ax.add_patch(Polygon(pts, closed=True, facecolor=color, edgecolor=edge or "none",
                                  lw=lw, zorder=z, joinstyle="miter"))

    def rect(self, x0: float, y0: float, x1: float, y1: float, lw: float, color: str, radius: float = 0.0,
             fill: Optional[str] = None, z: float = 3) -> None:
        if radius > 0:
            self.ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                             boxstyle=f"round,pad=0,rounding_size={radius}",
                                             facecolor=fill or "none", edgecolor=color, lw=lw, zorder=z))
        else:
            pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
            if fill:
                self.fill(pts, fill, z=z - 0.1)
            self.poly(pts, lw, color, closed=True, z=z)

    def circle(self, cx: float, cy: float, r: float, lw: float, color: str, fill: Optional[str] = None,
               z: float = 3) -> None:
        th = np.linspace(0, 2 * np.pi, 90)
        pts = list(zip(cx + r * np.cos(th), cy + r * np.sin(th), strict=True))
        if fill:
            self.fill(pts, fill, z=z - 0.1)
        if lw > 0:
            self.poly(pts, lw, color, z=z)

    def text(self, x: float, y: float, s: str, size: float, color: str, ha: str = "center",
             rotation: float = 0.0, z: float = 5) -> None:
        """y is the vertical centre of the line."""
        prop = font_props(self.font, round(size, 1))
        if rotation:
            self.ax.text(x, y, s, fontproperties=prop, color=color, ha="center", va="center",
                         rotation=rotation, rotation_mode="anchor", zorder=z, parse_math=False)
            return
        self.ax.text(x, y + 0.36 * size, s, fontproperties=prop, color=color, ha=ha, va="baseline",
                     zorder=z, parse_math=False)


class SketchPainter(CleanPainter):
    """Every stroke is a wobbly pen line."""

    def __init__(self, ax, theme: Theme, rng: random.Random, font: str, unit: float):
        super().__init__(ax, theme, rng, font)
        self.unit = unit  # px per F, scales wobble amplitude

    # -- strokes
    def _wobble(self, pts: list[Pt], closed: bool = False, amp: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        rng = self.rng
        P = np.asarray(pts + ([pts[0]] if closed else []), dtype=float)
        seg = np.diff(P, axis=0)
        L = np.hypot(seg[:, 0], seg[:, 1])
        keep = L > 1e-6
        if not keep.any():
            return P[:, 0], P[:, 1]
        step = 3.0
        out_x: list[np.ndarray] = []
        out_y: list[np.ndarray] = []
        a = self.t.wobble * amp * self.unit * 0.045
        lam1, lam2 = self.unit * rng.uniform(3.0, 9.0), self.unit * rng.uniform(1.0, 2.2)
        ph1, ph2 = rng.uniform(0, 6.3), rng.uniform(0, 6.3)
        s0 = 0.0
        for (x0, y0), (dx, dy), length in zip(P[:-1], seg, L, strict=True):
            if length < 1e-6:
                continue
            n = max(2, int(length / step) + 1)
            t = np.linspace(0, 1, n)
            s = s0 + t * length
            nx, ny = -dy / length, dx / length
            bow = rng.uniform(-1, 1) * min(self.unit * 0.25, 0.015 * length) * self.t.wobble * amp
            d = a * (0.9 * np.sin(2 * np.pi * s / lam1 + ph1) + 0.35 * np.sin(2 * np.pi * s / lam2 + ph2))
            d = d + bow * np.sin(np.pi * t)
            out_x.append(x0 + dx * t + nx * d)
            out_y.append(y0 + dy * t + ny * d)
            s0 += length
        return np.concatenate(out_x), np.concatenate(out_y)

    def poly(self, pts: list[Pt], lw: float, color: str, dash: Optional[tuple] = None, closed: bool = False,
             z: float = 2, alpha: float = 1.0, cap: str = "round") -> None:
        xs, ys = self._wobble(list(pts), closed)
        alpha = alpha * (self.rng.uniform(0.82, 0.97) if self.t.pencil else self.rng.uniform(0.9, 1.0))
        lw = lw * self.rng.uniform(0.88, 1.12)
        if dash:
            xs, ys = _dash(xs, ys, dash[0], dash[1], self.rng)
        self.ax.plot(xs, ys, lw=lw, color=color, zorder=z, alpha=alpha, solid_capstyle="round",
                     solid_joinstyle="round")

    def fill(self, pts: list[Pt], color: str, z: float = 3, edge: Optional[str] = None,
             lw: float = 0.0) -> None:
        xs, ys = self._wobble(list(pts), closed=True, amp=0.5)
        self.ax.fill(xs, ys, color=color, zorder=z, alpha=0.92, lw=0)

    def rect(self, x0: float, y0: float, x1: float, y1: float, lw: float, color: str, radius: float = 0.0,
             fill: Optional[str] = None, z: float = 3) -> None:
        rng = self.rng
        if radius > 0 and rng.random() < 0.85:
            pts = _rounded_rect_pts(x0, y0, x1, y1, radius)
            start = rng.randrange(len(pts))
            pts = pts[start:] + pts[:start]
            over = max(2, len(pts) // 25)
            self.poly(pts + pts[:over], lw, color, z=z)
            return
        corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        if rng.random() < 0.5:  # one continuous stroke, slightly overshooting where it closes
            c = rng.randrange(4)
            ordered = corners[c:] + corners[:c] + [corners[c]]
            ex = _extend(ordered[-2], ordered[-1], rng.uniform(0, 4))
            self.poly(ordered[:-1] + [ex], lw, color, z=z)
            return
        for i in range(4):
            p, q = corners[i], corners[(i + 1) % 4]
            o1, o2 = rng.uniform(-1.5, 4.5), rng.uniform(-1.5, 4.5)
            self.poly([_extend(q, p, o1), _extend(p, q, o2)], lw, color, z=z)

    def circle(self, cx: float, cy: float, r: float, lw: float, color: str, fill: Optional[str] = None,
               z: float = 3) -> None:
        rng = self.rng
        th0 = rng.uniform(0, 2 * np.pi)
        over = rng.uniform(0.15, 0.6)
        n = max(40, int(r * 1.2))
        th = np.linspace(th0, th0 + 2 * np.pi + over, n)
        ecc = rng.uniform(0.0, 0.08)
        phi = rng.uniform(0, np.pi)
        drift = rng.uniform(-0.07, 0.07)
        rr = r * (1 + ecc * np.cos(2 * (th - phi)) + drift * (th - th0) / (2 * np.pi)
                  + rng.uniform(0, 0.025) * np.sin(3 * th + rng.uniform(0, 6.3)))
        xs, ys = cx + rr * np.cos(th), cy + rr * np.sin(th)
        if fill:
            self.ax.fill(xs, ys, color=fill, zorder=z - 0.1, lw=0)
        if lw > 0:
            self.ax.plot(xs, ys, lw=lw * rng.uniform(0.9, 1.1), color=color, zorder=z,
                         alpha=rng.uniform(0.88, 1.0), solid_capstyle="round")

    def text(self, x: float, y: float, s: str, size: float, color: str, ha: str = "center",
             rotation: float = 0.0, z: float = 5) -> None:
        """Handwriting. Every letter is its own glyph outline with a little size / angle /
        baseline / spacing jitter; the word is then slanted and gently distorted and drawn as one
        path. Word widths stay at the measured width the layout reserved. (x, y) is the line
        centre (or the left end for left-aligned text)."""
        rng, t = self.rng, self.t
        words = [w for w in s.split(" ") if w]
        if not words:
            return
        sizes = [size * rng.uniform(1 - t.size_jit, 1 + t.size_jit) for _ in words]
        widths = [_text_w(w, self.font, sz) for w, sz in zip(words, sizes, strict=True)]
        space = (_text_w("n n", self.font, size) - _text_w("nn", self.font, size)) * rng.uniform(0.9, 1.4)
        total = sum(widths) + space * (len(words) - 1)
        line_ang = math.radians(rotation + rng.uniform(-1.5, 1.5))
        u = np.array([math.cos(line_ang), -math.sin(line_ang)])  # baseline direction (y down)
        v = np.array([-math.sin(line_ang), -math.cos(line_ang)])  # glyph "up"
        start = 0.0 if ha == "left" and not rotation else -total / 2
        base = np.array([x, y]) + start * u - 0.36 * size * v
        alpha = rng.uniform(0.8, 0.95) if t.pencil else rng.uniform(0.92, 1.0)
        pos = 0.0
        for w, sz, wd in zip(words, sizes, widths, strict=True):
            verts, codes = self._word_glyphs(w, sz, wd)
            if verts is None:
                pos += wd + space
                continue
            gx, gy = verts[:, 0], verts[:, 1]
            gx = gx + gy * math.tan(t.slant + math.radians(rng.uniform(-3, 3)))
            amp = t.glyph_amp * sz
            lam1, lam2 = sz * rng.uniform(0.5, 1.2), sz * rng.uniform(0.6, 1.4)
            gx2 = gx + amp * np.sin(2 * np.pi * gy / lam1 + rng.uniform(0, 6.3))
            gy2 = gy + amp * np.sin(2 * np.pi * gx / lam2 + rng.uniform(0, 6.3))
            d = math.radians(rng.uniform(-t.angle_jit, t.angle_jit) * 0.6)
            rx = gx2 * math.cos(d) - gy2 * math.sin(d)
            ry = gx2 * math.sin(d) + gy2 * math.cos(d)
            o = base + pos * u + rng.uniform(-t.base_jit, t.base_jit) * size * v
            X = o[0] + rx * u[0] + ry * v[0]
            Y = o[1] + rx * u[1] + ry * v[1]
            self.ax.add_patch(PathPatch(MplPath(np.stack([X, Y], axis=1), codes), facecolor=color,
                                        edgecolor=color, lw=t.lw * t.bold * rng.uniform(0.8, 1.2), zorder=z,
                                        alpha=alpha))
            pos += wd + space

    def _word_glyphs(self, word: str, size: float,
                     width: float) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Letter outlines of `word` laid out with per-letter jitter, scaled to `width`.
        Coordinates: x along the baseline from the word start, y up (TextPath convention)."""
        rng, t = self.rng, self.t
        prop = font_props(self.font, 10.0)
        parts: list[tuple[np.ndarray, np.ndarray, float]] = []
        cursor = 0.0
        for ch in word:
            sz = size * rng.uniform(1 - t.size_jit * 0.8, 1 + t.size_jit * 0.8)
            adv = _text_w(ch, self.font, sz)
            tp = TextPath((0, 0), ch, size=sz, prop=prop)
            if len(tp.vertices):
                V = tp.vertices.copy()
                a = math.radians(rng.uniform(-t.angle_jit, t.angle_jit) * 1.4)
                cx, cy = adv / 2, 0.35 * sz
                dx, dy = V[:, 0] - cx, V[:, 1] - cy
                V[:, 0] = cx + dx * math.cos(a) - dy * math.sin(a) + cursor
                lift = rng.uniform(-1, 1) * t.base_jit * 0.6 * sz
                V[:, 1] = cy + dx * math.sin(a) + dy * math.cos(a) + lift
                parts.append((V, tp.codes, cursor))
            cursor += adv * rng.uniform(0.92, 1.08)
        if not parts:
            return None, None
        scale = width * rng.uniform(0.96, 1.03) / max(cursor, 1e-6)
        verts = np.concatenate([p[0] for p in parts])
        verts[:, 0] *= scale
        codes = np.concatenate([p[1] for p in parts])
        return verts, codes


def _text_w(s: str, font: str, size: float) -> float:
    from .bpmn_layout import _renderer

    return _renderer().get_text_width_height_descent(s, font_props(font, round(size, 1)), ismath=False)[0]


def _extend(p: Pt, q: Pt, d: float) -> Pt:
    """q moved by d along the direction p -> q."""
    dx, dy = q[0] - p[0], q[1] - p[1]
    n = math.hypot(dx, dy) or 1.0
    return (q[0] + dx / n * d, q[1] + dy / n * d)


def _dash(xs: np.ndarray, ys: np.ndarray, on: float, off: float,
          rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    """Cut a dense polyline into dashes of arc length `on` (exact, however short the dash)."""
    seg = np.hypot(np.diff(xs), np.diff(ys))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    period = on + off
    ox: list[np.ndarray] = []
    oy: list[np.ndarray] = []
    nan = np.array([np.nan])
    for a in np.arange(-rng.uniform(0, period * 0.5), total, period):
        lo, hi = max(a, 0.0), min(a + on, total)
        if hi - lo < 0.4:
            continue
        ss = np.concatenate([[lo], s[(s > lo) & (s < hi)], [hi]])
        ox += [np.interp(ss, s, xs), nan]
        oy += [np.interp(ss, s, ys), nan]
    if not ox:
        return xs, ys
    return np.concatenate(ox), np.concatenate(oy)


def _rounded_rect_pts(x0: float, y0: float, x1: float, y1: float, r: float) -> list[Pt]:
    r = min(r, (x1 - x0) / 2, (y1 - y0) / 2)
    pts: list[Pt] = []
    corners = ((x1 - r, y0 + r, -90), (x1 - r, y1 - r, 0), (x0 + r, y1 - r, 90), (x0 + r, y0 + r, 180))
    for cx, cy, a0 in corners:
        for a in np.linspace(a0, a0 + 90, 6):
            pts.append((cx + r * math.cos(math.radians(a)), cy + r * math.sin(math.radians(a))))
    return pts


# --------------------------------------------------------------------------- #
# Scene drawing
# --------------------------------------------------------------------------- #
class SceneDrawer:
    def __init__(self, scene: Scene, theme: Theme, k: float, ox: float, oy: float, rng: random.Random):
        self.s, self.t, self.k, self.ox, self.oy, self.rng = scene, theme, k, ox, oy, rng
        self.F = scene.P.F * k  # task text size in px

    def px(self, x: float, y: float) -> Pt:
        return (x * self.k + self.ox, y * self.k + self.oy)

    def render(self, W: int, H: int) -> np.ndarray:
        fig = Figure(figsize=(W / 72, H / 72), dpi=72)
        canvas = FigureCanvasAgg(fig)
        fig.patch.set_facecolor(self.t.bg if self.t.style == "clean" else "#ffffff")
        ax = fig.add_axes((0, 0, 1, 1))
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.axis("off")
        if self.t.style == "clean":
            self.p: CleanPainter = CleanPainter(ax, self.t, self.rng, self.s.P.font)
            if self.t.dot_grid:
                self._dot_grid(ax, W, H)
        else:
            self.p = SketchPainter(ax, self.t, self.rng, self.s.P.font, self.F)
        for b in self.s.bands:
            self._band(b)
        by_id = {e.id: e for e in self.s.els}
        for f in self.s.flows:
            self._flow(f, by_id)
        for e in self.s.els:
            self._element(e)
        canvas.draw()
        img = np.asarray(canvas.buffer_rgba())[:, :, :3].copy()
        fig.clear()
        return img

    def _dot_grid(self, ax, W: int, H: int) -> None:
        step = max(8.0, 10 * self.k * self.rng.uniform(0.9, 1.6))
        xs = np.arange(step / 2, W, step)
        ys = np.arange(step / 2, H, step)
        gx, gy = np.meshgrid(xs, ys)
        ax.scatter(gx.ravel(), gy.ravel(), s=max(0.4, 0.5 * self.k), c="#d8d8d8", marker=".", lw=0, zorder=0)

    # ------------------------------------------------------------- bands
    def _band(self, b) -> None:
        p, t, rng = self.p, self.t, self.rng
        x0, y0 = self.px(b.x0, b.y0)
        x1, y1 = self.px(b.x1, b.y1)
        lw = t.lw * (0.75 if t.style == "clean" else 0.9)
        size = self.s.P.Fh * self.k
        ink = t.ink
        if b.kind == "collapsed":
            p.rect(x0, y0, x1, y1, lw, ink, z=1)
            cx = (x0 + x1) / 2 + (rng.uniform(-0.25, 0.25) * (x1 - x0) if t.style == "sketch" else 0.0)
            self._lines(b.lines, cx, (y0 + y1) / 2, size, ink, "center")
            if b.multi:
                self._multi_marker((x0 + x1) / 2, y1 - 0.45 * self.F, 0.9 * self.F)
            return
        hdr = b.hdr * self.k
        if b.kind == "pool":
            if t.hdr_fill:
                p.fill([(x0, y0), (x0 + hdr, y0), (x0 + hdr, y1), (x0, y1)], t.hdr_fill, z=0.5)
            p.rect(x0, y0, x1, y1, lw, ink, z=1)
            p.poly([(x0 + hdr, y0), (x0 + hdr, y1)], lw, ink, z=1)
            self._rot_lines(b.lines, x0 + hdr / 2, (y0 + y1) / 2, size, ink)
            return
        # lane: in a pool only the separators between lanes are new lines; lanes without a pool
        # are bands of horizontal lines (one line between two lanes, never two)
        siblings = [x for x in self.s.bands if x.kind == "lane" and x.parent == b.parent]
        first, last = siblings[0] is b, siblings[-1] is b
        if t.style == "clean":
            p.rect(x0, y0, x1, y1, lw * 0.9, ink, z=1)
        else:
            if b.parent is None and first:
                p.poly([(x0, y0), (x1, y0)], lw, ink, z=1)
            if b.parent is None or not last:
                p.poly([(x0, y1), (x1, y1)], lw, ink, z=1)
            if b.parent is None and not b.horizontal:
                p.poly([(x0, y0), (x0, y1)], lw, ink, z=1)
        if b.horizontal:
            self._lines(b.lines, x0 + hdr / 2, (y0 + y1) / 2, size, ink, "center")
            return
        p.poly([(x0 + hdr, y0), (x0 + hdr, y1)], lw * 0.9, ink, z=1)
        self._rot_lines(b.lines, x0 + hdr / 2, (y0 + y1) / 2, size, ink)

    def _rot_lines(self, lines: list[str], cx: float, cy: float, size: float, color: str) -> None:
        n = len(lines)
        lh = self.s.P.lh * size
        for i, ln in enumerate(lines):
            self.p.text(cx + (i - (n - 1) / 2) * lh, cy, ln, size, color, rotation=90)

    def _lines(self, lines: list[str], cx: float, cy: float, size: float, color: str, align: str,
               box_x0: Optional[float] = None) -> None:
        n = len(lines)
        lh = self.s.P.lh * size
        for i, ln in enumerate(lines):
            y = cy + (i - (n - 1) / 2) * lh
            if align == "left" and box_x0 is not None:
                self.p.text(box_x0, y, ln, size, color, ha="left")
            else:
                self.p.text(cx, y, ln, size, color, ha="center")

    def _multi_marker(self, cx: float, cy: float, s: float) -> None:
        for dx in (-0.3, 0.0, 0.3):
            self.p.poly([(cx + dx * s, cy - s / 2), (cx + dx * s, cy + s / 2)], self.t.lw * 0.9, self.t.ink,
                        z=4)

    # ------------------------------------------------------------- flows
    def _flow(self, f: Flow, by_id: dict[str, El]) -> None:
        p, t, rng = self.p, self.t, self.rng
        pts = [self.px(x, y) for x, y in f.points]
        ink = t.strokes.get(f.kind, t.ink)
        lw = t.seq_lw
        F = self.F
        ah = F * (0.75 if t.style == "clean" else rng.uniform(0.7, 1.05))
        if f.kind == "seq":
            end = _shorten(pts, ah * 0.6) if (t.style == "clean" or t.arrow == "filled") else pts
            p.poly(end, lw, ink, z=2)
            self._arrow(pts[-2], pts[-1], ah, "filled" if t.style == "clean" else t.arrow, lw, ink)
        elif f.kind == "msg":
            r = F * (0.26 if t.style == "clean" else rng.uniform(0.2, 0.3))
            if t.style == "clean":
                dash = (F * 0.45, F * 0.3)
            else:
                dash = (F * rng.uniform(0.45, 0.9), F * rng.uniform(0.3, 0.5))
            centre = _advance(pts, r)[0]
            body = _shorten(_advance(pts, 2 * r), ah * 0.9)
            p.poly(body, lw * 0.9, ink, dash=dash, z=2)
            p.circle(centre[0], centre[1], r, lw * 0.8, ink, fill="#ffffff" if t.style == "clean" else None,
                     z=3)
            self._arrow(body[-1], pts[-1], ah * 0.9, "hollow", lw * 0.8, ink)
        elif f.kind == "data":
            if t.style == "clean":
                dash = (F * 0.12, F * 0.25)
            else:
                dash = (F * rng.uniform(0.15, 0.4), F * rng.uniform(0.25, 0.4))
            p.poly(_shorten(pts, 1.0), lw * 0.85, ink, dash=dash, z=2)
            self._arrow(pts[-2], pts[-1], ah * 0.8, "open", lw * 0.85, ink)
        else:  # association to a text annotation
            dash = (F * 0.12, F * 0.25) if t.style == "clean" else (F * 0.25, F * 0.3)
            if t.style == "sketch" and rng.random() < 0.4:
                dash = None
            p.poly(pts, lw * 0.8, ink, dash=dash, z=2)
        if f.label_lines and f.label_box:
            self._label_box(f.label_lines, f.label_box, color=ink)

    def _arrow(self, a: Pt, b: Pt, size: float, kind: str, lw: float, ink: str) -> None:
        dx, dy = b[0] - a[0], b[1] - a[1]
        n = math.hypot(dx, dy) or 1.0
        ux, uy = dx / n, dy / n
        ang = math.radians(22 if self.t.style == "clean" else self.rng.uniform(20, 32))
        ca, sa = math.cos(ang), math.sin(ang)
        bx, by = -ux, -uy
        left = (b[0] + size * (bx * ca - by * sa), b[1] + size * (bx * sa + by * ca))
        right = (b[0] + size * (bx * ca + by * sa), b[1] + size * (-bx * sa + by * ca))
        if kind == "filled":
            self.p.fill([left, b, right], ink, z=3.5)
        elif kind == "hollow":
            if self.t.style == "clean":
                self.p.fill([left, b, right], "#ffffff", z=3.4)
            self.p.poly([left, b, right], lw, ink, closed=True, z=3.5)
        else:
            self.p.poly([left, b, right], lw, ink, z=3.5)

    def _label_box(self, lines: list[str], box: tuple[float, float, float, float],
                   color: Optional[str] = None) -> None:
        x0, y0 = self.px(box[0], box[1])
        x1, y1 = self.px(box[2], box[3])
        size = self.s.P.Fl * self.k
        self._lines(lines, (x0 + x1) / 2, (y0 + y1) / 2, size, color or self.t.ink, "center")

    # ------------------------------------------------------------- elements
    def _element(self, e: El) -> None:
        kind = e.kind
        if kind == "task":
            self._task(e)
        elif kind in ("start", "end", "inter"):
            self._event(e)
        elif kind == "gw":
            self._gateway(e)
        elif kind == "data":
            self._data(e)
        elif kind == "store":
            self._store(e)
        elif kind == "note":
            self._note(e)
        if kind != "note" and kind != "task" and e.lines and e.label_box:
            self._label_box(e.lines, e.label_box, color=self._colors(kind)[0])

    def _colors(self, kind: str) -> tuple[str, Optional[str]]:
        t = self.t
        return t.strokes.get(kind, t.ink), t.fills.get(kind, "#ffffff" if t.style == "clean" else None)

    def _task(self, e: El) -> None:
        p, t = self.p, self.t
        x0, y0 = self.px(e.x - e.w / 2, e.y - e.h / 2)
        x1, y1 = self.px(e.x + e.w / 2, e.y + e.h / 2)
        stroke, fill = self._colors("task")
        radius = self.s.P.rounded * self.F
        p.rect(x0, y0, x1, y1, t.lw, stroke, radius=radius, fill=fill, z=3)
        marker_h = 0.85 * self.F if e.marker else 0.0
        icon_h = 0.9 * self.F if e.icon else 0.0
        cy = (y0 + y1) / 2 - marker_h / 2 + icon_h / 2 * 0.6
        text_ink = stroke if (t.style == "sketch" or stroke != "#000000") else t.ink
        if self.s.P.align == "left" and t.style == "sketch":
            self._lines(e.lines, (x0 + x1) / 2, cy, self.F, text_ink, "left", box_x0=x0 + 0.35 * self.F)
        else:
            self._lines(e.lines, (x0 + x1) / 2, cy, self.F, text_ink, "center")
        if e.marker:
            self._task_marker(e.marker, (x0 + x1) / 2, y1 - 0.55 * self.F, 0.62 * self.F, stroke)
        if e.icon:
            self._task_icon(e.icon, x0 + 0.65 * self.F, y0 + 0.6 * self.F, 0.85 * self.F, stroke)

    def _task_marker(self, m: str, cx: float, cy: float, s: float, ink: str) -> None:
        p, lw = self.p, self.t.lw * 0.7
        if m == "sub":
            p.rect(cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2, lw, ink, z=4)
            p.poly([(cx - s * 0.3, cy), (cx + s * 0.3, cy)], lw, ink, z=4)
            p.poly([(cx, cy - s * 0.3), (cx, cy + s * 0.3)], lw, ink, z=4)
        elif m == "loop":
            th = np.linspace(math.radians(-60), math.radians(240), 20)
            pts = list(zip(cx + s * 0.45 * np.cos(th), cy - s * 0.45 * np.sin(th), strict=True))
            p.poly(pts, lw, ink, z=4)
            end = pts[-1]
            head = [(end[0] - s * 0.25, end[1] - s * 0.05), end, (end[0] - s * 0.05, end[1] + s * 0.28)]
            p.poly(head, lw, ink, z=4)
        elif m in ("mi", "mis"):
            for d in (-0.3, 0.0, 0.3):
                if m == "mi":
                    p.poly([(cx + d * s, cy - s / 2), (cx + d * s, cy + s / 2)], lw * 1.3, ink, z=4)
                else:
                    p.poly([(cx - s / 2, cy + d * s), (cx + s / 2, cy + d * s)], lw * 1.3, ink, z=4)

    def _task_icon(self, icon: str, cx: float, cy: float, s: float, ink: str) -> None:
        p, lw = self.p, self.t.lw * 0.6
        if icon in ("send", "receive"):
            self._envelope(cx, cy, s * 0.9, filled=icon == "send", lw=lw)
        elif icon == "user":
            p.circle(cx, cy - s * 0.18, s * 0.2, lw, ink, z=4)
            th = np.linspace(math.pi, 2 * math.pi, 12)
            pts = list(zip(cx + s * 0.4 * np.cos(th), cy + s * 0.45 + s * 0.3 * np.sin(th), strict=True))
            p.poly(pts + [pts[0]], lw, ink, z=4)
        elif icon == "service":
            for i in range(8):
                a = i * math.pi / 4
                p.poly([(cx + s * 0.3 * math.cos(a), cy + s * 0.3 * math.sin(a)),
                        (cx + s * 0.48 * math.cos(a), cy + s * 0.48 * math.sin(a))], lw * 1.6, ink, z=4)
            p.circle(cx, cy, s * 0.32, lw, ink, z=4)
            p.circle(cx, cy, s * 0.12, lw, ink, z=4)
        elif icon == "script":
            p.rect(cx - s * 0.35, cy - s * 0.45, cx + s * 0.35, cy + s * 0.45, lw, ink, z=4)
            for d in (-0.2, 0.0, 0.2):
                p.poly([(cx - s * 0.2, cy + d * s), (cx + s * 0.2, cy + d * s)], lw, ink, z=4)
        elif icon == "rule":
            p.rect(cx - s * 0.45, cy - s * 0.32, cx + s * 0.45, cy + s * 0.32, lw, ink, z=4)
            p.poly([(cx - s * 0.45, cy - s * 0.1), (cx + s * 0.45, cy - s * 0.1)], lw, ink, z=4)
            p.poly([(cx - s * 0.15, cy - s * 0.32), (cx - s * 0.15, cy + s * 0.32)], lw, ink, z=4)
        else:  # manual: a stylised hand
            for d in (-0.25, -0.08, 0.09):
                p.poly([(cx - s * 0.1, cy + d * s), (cx + s * 0.45, cy + d * s)], lw, ink, z=4)
            p.rect(cx - s * 0.45, cy - s * 0.3, cx - s * 0.05, cy + s * 0.35, lw, ink, radius=s * 0.1, z=4)

    def _envelope(self, cx: float, cy: float, s: float, filled: bool, lw: float,
                  ink: Optional[str] = None) -> None:
        p = self.p
        ink = ink or self.t.ink
        w, h = s, s * 0.66
        x0, y0, x1, y1 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        if filled:
            p.fill([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], ink, z=4)
            p.poly([(x0, y0), (cx, cy + h * 0.1), (x1, y0)], lw, "#ffffff", z=4.2)
        else:
            p.rect(x0, y0, x1, y1, lw, ink, z=4)
            p.poly([(x0, y0), (cx, cy + h * 0.1), (x1, y0)], lw, ink, z=4)

    def _event(self, e: El) -> None:
        p, t, rng = self.p, self.t, self.rng
        cx, cy = self.px(e.x, e.y)
        r = e.w / 2 * self.k
        stroke, fill = self._colors(e.kind)
        if e.kind == "start":
            lw = t.lw * (0.9 if t.style == "clean" else rng.uniform(0.8, 1.0))
            p.circle(cx, cy, r, lw, stroke, fill=fill)
        elif e.kind == "end":
            if t.style == "clean":
                p.circle(cx, cy, r - t.lw, t.lw * 2.2, stroke, fill=fill)
            else:  # a bold ring: one heavy stroke or a few strokes traced over each other
                if rng.random() < 0.5:
                    p.circle(cx, cy, r - t.lw, t.lw * rng.uniform(2.3, 3.4), stroke)
                else:
                    for j in range(rng.randint(2, 4)):
                        p.circle(cx, cy, r - t.lw * (0.6 + 0.55 * j), t.lw * 1.5, stroke)
        else:
            gap = max(2.0, r * 0.16)
            p.circle(cx, cy, r, t.lw * 0.8, stroke, fill=fill)
            p.circle(cx, cy, r - gap, t.lw * 0.8, stroke)
        icon_r = r * 0.55
        if e.ev == "message":
            filled = e.kind == "end" or (e.kind == "inter" and e.throw)
            self._envelope(cx, cy, icon_r * 1.55, filled=filled, lw=t.lw * 0.7, ink=stroke)
        elif e.ev == "timer":
            p.circle(cx, cy, icon_r * 1.05, t.lw * 0.7, stroke, z=4)
            if t.style == "clean" or rng.random() < 0.5:
                for i in range(12):
                    a = i * math.pi / 6
                    p.poly([(cx + icon_r * 0.8 * math.cos(a), cy + icon_r * 0.8 * math.sin(a)),
                            (cx + icon_r * 0.98 * math.cos(a), cy + icon_r * 0.98 * math.sin(a))],
                           t.lw * 0.5, stroke, z=4)
            p.poly([(cx, cy), (cx + icon_r * 0.55, cy - icon_r * 0.3)], t.lw * 0.7, stroke, z=4)
            p.poly([(cx, cy), (cx - icon_r * 0.05, cy - icon_r * 0.75)], t.lw * 0.7, stroke, z=4)
        elif e.ev == "terminate":
            p.circle(cx, cy, r * 0.6, 0, stroke, fill=stroke, z=4)
        elif e.ev == "signal":
            tri = [(cx, cy - icon_r), (cx + icon_r * 0.9, cy + icon_r * 0.6),
                   (cx - icon_r * 0.9, cy + icon_r * 0.6)]
            if e.throw:
                p.fill(tri, stroke, z=4)
            else:
                p.poly(tri, t.lw * 0.7, stroke, closed=True, z=4)
        elif e.ev == "conditional":
            p.rect(cx - icon_r * 0.6, cy - icon_r * 0.75, cx + icon_r * 0.6, cy + icon_r * 0.75, t.lw * 0.6,
                   stroke, z=4)
            for d in (-0.4, 0.0, 0.4):
                p.poly([(cx - icon_r * 0.4, cy + d * icon_r), (cx + icon_r * 0.4, cy + d * icon_r)],
                       t.lw * 0.5, stroke, z=4)
        elif e.ev == "escalation":
            pts = [(cx, cy - icon_r), (cx + icon_r * 0.6, cy + icon_r * 0.7), (cx, cy + icon_r * 0.15),
                   (cx - icon_r * 0.6, cy + icon_r * 0.7)]
            p.poly(pts, t.lw * 0.7, stroke, closed=True, z=4)

    def _gateway(self, e: El) -> None:
        p, t, rng = self.p, self.t, self.rng
        cx, cy = self.px(e.x, e.y)
        hw, hh = e.w / 2 * self.k, e.h / 2 * self.k
        stroke, fill = self._colors("gw")
        pts = [(cx, cy - hh), (cx + hw, cy), (cx, cy + hh), (cx - hw, cy)]
        if fill:
            p.fill(pts, fill, z=2.9)
        p.poly(pts, t.lw * (0.9 if t.style == "clean" else 1.0), stroke, closed=True, z=3)
        m = hw * 0.42
        lw = t.lw * (1.8 if t.style == "clean" else rng.uniform(1.0, 1.6))
        if e.gw == "xor":
            if e.marker != "blank" and t.xor_mark != "blank":
                p.poly([(cx - m * 0.8, cy - m * 0.8), (cx + m * 0.8, cy + m * 0.8)], lw, stroke, z=4)
                p.poly([(cx - m * 0.8, cy + m * 0.8), (cx + m * 0.8, cy - m * 0.8)], lw, stroke, z=4)
        elif e.gw == "and":
            p.poly([(cx - m, cy), (cx + m, cy)], lw, stroke, z=4)
            p.poly([(cx, cy - m), (cx, cy + m)], lw, stroke, z=4)
        elif e.gw == "or":
            p.circle(cx, cy, m * 0.85, lw * 0.8, stroke, z=4)
        elif e.gw == "evgw":
            p.circle(cx, cy, m * 1.05, t.lw * 0.6, stroke, z=4)
            p.circle(cx, cy, m * 0.85, t.lw * 0.6, stroke, z=4)
            pent = [(cx + m * 0.55 * math.sin(a), cy - m * 0.55 * math.cos(a))
                    for a in np.linspace(0, 2 * math.pi, 6)[:-1]]
            p.poly(pent, t.lw * 0.6, stroke, closed=True, z=4)

    def _data(self, e: El) -> None:
        p, t = self.p, self.t
        x0, y0 = self.px(e.x - e.w / 2, e.y - e.h / 2)
        x1, y1 = self.px(e.x + e.w / 2, e.y + e.h / 2)
        f = (x1 - x0) * 0.32
        stroke, fill = self._colors("data")
        outline = [(x0, y0), (x1 - f, y0), (x1, y0 + f), (x1, y1), (x0, y1)]
        if fill:
            p.fill(outline, fill, z=2.9)
        p.poly(outline, t.lw * 0.85, stroke, closed=True, z=3)
        p.poly([(x1 - f, y0), (x1 - f, y0 + f), (x1, y0 + f)], t.lw * 0.7, stroke, z=3)

    def _store(self, e: El) -> None:
        p, t = self.p, self.t
        x0, y0 = self.px(e.x - e.w / 2, e.y - e.h / 2)
        x1, y1 = self.px(e.x + e.w / 2, e.y + e.h / 2)
        cx = (x0 + x1) / 2
        rx = (x1 - x0) / 2
        ry = min((y1 - y0) * 0.16, rx * 0.4)
        stroke, fill = self._colors("store")
        th = np.linspace(0, 2 * np.pi, 40)
        top = list(zip(cx + rx * np.cos(th), y0 + ry + ry * np.sin(th), strict=True))
        th2 = np.linspace(0, np.pi, 20)
        bottom = list(zip(cx + rx * np.cos(th2), y1 - ry + ry * np.sin(th2), strict=True))
        if fill:
            p.fill([(x0, y0 + ry)] + bottom[::-1] + [(x1, y0 + ry)], fill, z=2.9)
        lw = t.lw * 0.85
        p.poly(top, lw, stroke, z=3)
        p.poly([(x0, y0 + ry), (x0, y1 - ry)], lw, stroke, z=3)
        p.poly([(x1, y0 + ry), (x1, y1 - ry)], lw, stroke, z=3)
        p.poly(bottom, lw, stroke, z=3)
        if t.style == "clean" or self.rng.random() < 0.5:
            for k in (1, 2):
                ys = y0 + ry + k * ry * 0.55 + ry * np.sin(th2)
                arc = list(zip(cx + rx * np.cos(th2), ys, strict=True))
                p.poly(arc, lw * 0.8, stroke, z=3)

    def _note(self, e: El) -> None:
        p, t = self.p, self.t
        x0, y0 = self.px(e.x - e.w / 2, e.y - e.h / 2)
        x1, y1 = self.px(e.x + e.w / 2, e.y + e.h / 2)
        b = min(self.F * 0.9, (x1 - x0) * 0.25)
        ink = t.strokes.get("note", t.ink)
        p.poly([(x0 + b, y0), (x0, y0), (x0, y1), (x0 + b, y1)], t.lw * 0.8, ink, z=3)
        size = self.s.P.Fl * self.k
        self._lines(e.lines, (x0 + x1) / 2, (y0 + y1) / 2, size, ink, "left", box_x0=x0 + 0.35 * self.F)


def _shorten(pts: list[Pt], d: float) -> list[Pt]:
    """Polyline with the last point pulled back by d (so a filled arrowhead covers the end)."""
    (ax_, ay), (bx, by) = pts[-2], pts[-1]
    n = math.hypot(bx - ax_, by - ay)
    if n <= d + 1e-6:
        return pts
    return pts[:-1] + [(bx - (bx - ax_) / n * d, by - (by - ay) / n * d)]


def _advance(pts: list[Pt], d: float) -> list[Pt]:
    (ax_, ay), (bx, by) = pts[0], pts[1]
    n = math.hypot(bx - ax_, by - ay)
    if n <= d + 1e-6:
        return pts
    return [(ax_ + (bx - ax_) / n * d, ay + (by - ay) / n * d)] + pts[1:]
