"""Layout of a synthetic BPMN process (bpmn_model.Process) in layout units.

1. Block grid: every block of the process tree becomes a rectangle of grid cells. Sequences go
   left to right (sometimes stepping one row up / down), gateway branches are stacked vertically,
   loops get a private corridor row for the back edge. The grid drawing is planar.
2. Lanes are contiguous bands of grid rows, so lane membership follows from the row of a node
   and no flow has to cross a node to reach another lane.
3. Rows / columns get sizes from their contents (event labels below events, reserved space for
   data objects and annotations, gaps wide enough for branch labels).
4. Sequence flows are routed orthogonally from the block structure (split: leave the gateway
   through the top / bottom corner; join: enter the merge from above / below; loop: corridor).
5. Optional things (event / edge / data labels, data objects, annotations, message flows) are
   placed with a collision check against everything placed before; whatever does not fit is
   dropped, so the target only ever lists what is drawn.
"""

from __future__ import annotations

import math
import random
import textwrap
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import pairwise
from typing import Optional

from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.font_manager import FontProperties

from . import bpmn_vocab as V
from .bpmn_model import FLOW_KINDS, El, Flow, Leaf, Loop, Process, ProcessGenerator, Seq, Split, XorEnd

FONT_DIR = "C:/Windows/Fonts/"
HAND_FONTS = ["Inkfree.ttf", "segoepr.ttf", "segoesc.ttf", "comic.ttf", "BRADHITC.TTF", "mvboli.ttf",
              "ITCKRIST.TTF"]
HAND_WEIGHTS = [1.4, 1.2, 0.8, 0.7, 1.0, 1.0, 0.6]
CLEAN_FONTS = ["arial.ttf", "segoeui.ttf", "calibri.ttf", "verdana.ttf", "tahoma.ttf", "Candara.ttf",
               "corbel.ttf"]
CLEAN_WEIGHTS = [3.0, 1.5, 1.0, 0.8, 0.8, 0.4, 0.4]

Rect = tuple[float, float, float, float]


class LayoutError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# Text measurement
# --------------------------------------------------------------------------- #
_RENDERER: Optional[RendererAgg] = None


def _renderer() -> RendererAgg:
    global _RENDERER
    if _RENDERER is None:
        _RENDERER = RendererAgg(8, 8, 72)
    return _RENDERER


@lru_cache(maxsize=256)
def font_props(path: str, size: float) -> FontProperties:
    return FontProperties(fname=path, size=size)


class Measure:
    def __init__(self, font: str, slack: float):
        self.font = font
        self.slack = slack
        self.cache: dict[tuple[str, float], float] = {}

    def width(self, text: str, size: float) -> float:
        key = (text, round(size, 1))
        w = self.cache.get(key)
        if w is None:
            prop = font_props(self.font, key[1])
            w = _renderer().get_text_width_height_descent(text, prop, ismath=False)[0]
            self.cache[key] = w
        return w * self.slack

    def block(self, lines: list[str], size: float, lh: float) -> tuple[float, float]:
        if not lines:
            return 0.0, 0.0
        return max(self.width(t, size) for t in lines), len(lines) * lh * size


def wrap_width(text: str, max_w: float, size: float, meas: Measure) -> list[str]:
    lines: list[str] = []
    cur = ""
    for word in text.split():
        cand = f"{cur} {word}" if cur else word
        if not cur or meas.width(cand, size) <= max_w:
            cur = cand
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def wrap_chars(text: str, n: int, rng: random.Random, hyphen: float = 0.0) -> list[str]:
    words = text.split()
    if hyphen and rng.random() < hyphen:
        long = [i for i, w in enumerate(words) if len(w) >= max(8, n + 2) and w.isalpha()]
        if long:
            i = rng.choice(long)
            cut = rng.randint(4, len(words[i]) - 3)
            words[i:i + 1] = [words[i][:cut] + "-", words[i][cut:]]
    out: list[str] = []
    for line in textwrap.wrap(" ".join(words), max(3, n), break_long_words=False, break_on_hyphens=False):
        out.append(line)
    # "Documenta- tion" must stay "Documenta-\ntion": textwrap may have joined it back on one line
    fixed: list[str] = []
    for line in out:
        parts = line.split("- ")
        for j, part in enumerate(parts):
            fixed.append(part + ("-" if j < len(parts) - 1 else ""))
    return [ln for ln in fixed if ln]


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def inflate(r: Rect, m: float) -> Rect:
    return (r[0] - m, r[1] - m, r[2] + m, r[3] + m)


def overlaps(a: Rect, b: Rect) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def seg_hits_rect(p: tuple[float, float], q: tuple[float, float], r: Rect) -> bool:
    """Liang-Barsky: does segment pq intersect rectangle r?"""
    x0, y0 = p
    dx, dy = q[0] - x0, q[1] - y0
    t0, t1 = 0.0, 1.0
    for pp, qq in ((-dx, x0 - r[0]), (dx, r[2] - x0), (-dy, y0 - r[1]), (dy, r[3] - y0)):
        if abs(pp) < 1e-12:
            if qq < 0:
                return False
        else:
            t = qq / pp
            if pp < 0:
                t0 = max(t0, t)
            else:
                t1 = min(t1, t)
            if t0 > t1:
                return False
    return True


def segs_cross(a1, a2, b1, b2) -> bool:
    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    d1, d2 = orient(b1, b2, a1), orient(b1, b2, a2)
    d3, d4 = orient(a1, a2, b1), orient(a1, a2, b2)
    return (d1 > 0) != (d2 > 0) and (d3 > 0) != (d4 > 0)


class Occ:
    """Everything drawn so far, for collision checks."""

    def __init__(self) -> None:
        self.rects: list[tuple[Rect, str]] = []
        self.segs: list[tuple[tuple[float, float], tuple[float, float], str]] = []

    def add_rect(self, r: Rect, tag: str) -> None:
        self.rects.append((r, tag))

    def add_path(self, pts: list[tuple[float, float]], tag: str) -> None:
        for a, b in pairwise(pts):
            self.segs.append((a, b, tag))

    def rect_free(self, r: Rect, margin: float, ignore: tuple[str, ...] = ()) -> bool:
        rr = inflate(r, margin)
        for o, tag in self.rects:
            if tag not in ignore and overlaps(rr, o):
                return False
        for a, b, tag in self.segs:
            if tag not in ignore and seg_hits_rect(a, b, rr):
                return False
        return True

    def path_free(self, pts: list[tuple[float, float]], margin: float, ignore: tuple[str, ...] = (),
                  max_cross: int = 0) -> bool:
        """No rectangle hit; at most `max_cross` crossings with drawn lines (never collinear)."""
        crossings = 0
        for a, b in pairwise(pts):
            for o, tag in self.rects:
                if tag not in ignore and seg_hits_rect(a, b, inflate(o, margin)):
                    return False
            for c, d, tag in self.segs:
                if tag in ignore:
                    continue
                if _collinear_overlap(a, b, c, d, margin):
                    return False
                if segs_cross(a, b, c, d):
                    crossings += 1
                    if crossings > max_cross:
                        return False
        return True


def _collinear_overlap(a, b, c, d, tol: float) -> bool:
    """Two (near) axis-parallel segments running on top of each other."""
    if abs(a[0] - b[0]) < 1e-6 and abs(c[0] - d[0]) < 1e-6 and abs(a[0] - c[0]) < tol:
        lo1, hi1 = sorted((a[1], b[1]))
        lo2, hi2 = sorted((c[1], d[1]))
        return min(hi1, hi2) - max(lo1, lo2) > 1.0
    if abs(a[1] - b[1]) < 1e-6 and abs(c[1] - d[1]) < 1e-6 and abs(a[1] - c[1]) < tol:
        lo1, hi1 = sorted((a[0], b[0]))
        lo2, hi2 = sorted((c[0], d[0]))
        return min(hi1, hi2) - max(lo1, lo2) > 1.0
    return False


# --------------------------------------------------------------------------- #
# Scene
# --------------------------------------------------------------------------- #
@dataclass
class Params:
    style: str
    font: str
    F: float  # task text size (layout units)
    Fl: float  # external labels
    Fh: float  # pool / lane names
    lh: float  # line height factor
    col_gap: float
    row_gap: float
    lane_pad: float
    hdr: float
    lhdr: float
    pool_gap: float
    collapsed_h: float
    jitter: float
    p_step: float
    margin: float
    task_w: float = 0.0  # clean: uniform task size
    task_h: float = 0.0
    ev_d: float = 0.0
    gw_s: float = 0.0
    wrap_task: int = 12  # sketch: characters per line
    wrap_label: int = 12
    align: str = "center"
    rounded: float = 0.0  # task corner radius as a fraction of F


@dataclass
class Band:
    id: str
    kind: str  # pool lane collapsed
    lines: list[str]
    x0: float
    y0: float
    x1: float
    y1: float
    hdr: float = 0.0
    parent: Optional[str] = None
    members: list[str] = field(default_factory=list)
    multi: bool = False
    horizontal: bool = False  # lane name written horizontally (lanes without a pool)


@dataclass
class Scene:
    style: str
    P: Params
    els: list[El]
    flows: list[Flow]
    bands: list[Band]
    W: float
    H: float
    content: Rect


def make_params(style: str, rng: random.Random) -> Params:
    F = 14.0
    if style == "clean":
        font = FONT_DIR + rng.choices(CLEAN_FONTS, weights=CLEAN_WEIGHTS)[0]
        return Params(
            style=style, font=font, F=F, Fl=F * rng.uniform(0.86, 0.95), Fh=F * rng.uniform(0.95, 1.1),
            lh=1.22, col_gap=F * rng.uniform(2.0, 3.4), row_gap=F * rng.uniform(1.6, 2.8),
            lane_pad=F * rng.uniform(0.9, 1.6), hdr=F * rng.uniform(2.2, 2.6), lhdr=F * rng.uniform(2.2, 2.6),
            pool_gap=F * rng.uniform(3.0, 5.0), collapsed_h=F * rng.uniform(3.0, 4.4), jitter=0.0,
            p_step=rng.uniform(0.0, 0.1), margin=F * rng.uniform(0.8, 3.0), task_w=F * rng.uniform(6.6, 7.8),
            task_h=F * rng.uniform(4.6, 5.8), ev_d=F * rng.uniform(2.7, 3.1), gw_s=F * rng.uniform(3.6, 4.1),
            align="center", rounded=rng.choice([0.7, 0.7, 0.7, 0.35, 0.0]),
        )
    font = FONT_DIR + rng.choices(HAND_FONTS, weights=HAND_WEIGHTS)[0]
    return Params(
        style=style, font=font, F=F, Fl=F * rng.uniform(0.8, 1.0), Fh=F * rng.uniform(0.95, 1.3),
        lh=rng.uniform(1.18, 1.4), col_gap=F * rng.uniform(1.6, 3.2), row_gap=F * rng.uniform(1.3, 2.6),
        lane_pad=F * rng.uniform(0.8, 1.8), hdr=F * rng.uniform(2.0, 3.0), lhdr=F * rng.uniform(2.0, 3.0),
        pool_gap=F * rng.uniform(2.6, 4.6), collapsed_h=F * rng.uniform(2.6, 4.2),
        jitter=F * rng.uniform(0.0, 0.25), p_step=rng.uniform(0.0, 0.14), margin=F * 0.5,
        ev_d=F * rng.uniform(1.9, 2.8), gw_s=F * rng.uniform(2.3, 3.3), wrap_task=rng.randint(7, 15),
        wrap_label=rng.randint(7, 14), align="center" if rng.random() < 0.7 else "left",
        rounded=rng.choice([0.0, 0.0, 0.5, 0.8, 1.0]),
    )


# --------------------------------------------------------------------------- #
# Element sizes
# --------------------------------------------------------------------------- #
def size_element(el: El, P: Params, meas: Measure, rng: random.Random) -> None:
    F = P.F
    sketch = P.style == "sketch"
    if el.kind == "task":
        extra = 0.85 * F if el.marker else 0.0
        if sketch:
            n = max(5, P.wrap_task + rng.randint(-2, 2))
            el.lines = wrap_chars(el.text, n, rng, hyphen=0.05)
            tw, th = meas.block(el.lines, F, P.lh)
            el.w = max(F * 3.4, tw + 2 * F * rng.uniform(0.35, 0.9))
            el.h = max(F * 2.3, th + 2 * F * rng.uniform(0.3, 0.75)) + extra
        else:
            inner = P.task_w - 0.8 * F
            el.lines = wrap_width(el.text, inner, F, meas)
            tw, th = meas.block(el.lines, F, P.lh)
            el.w = max(P.task_w, tw + 0.8 * F)
            el.h = max(P.task_h, th + 1.0 * F + (0.9 * F if el.icon else 0.0)) + extra
    elif el.kind in ("start", "end", "inter"):
        d = P.ev_d * (rng.uniform(0.9, 1.12) if sketch else 1.0)
        if el.kind == "end" and sketch:
            d *= rng.uniform(1.0, 1.15)
        el.w = el.h = d
        el.lines = _label_lines(el.text, P, meas, rng)
    elif el.kind == "gw":
        s = P.gw_s * (rng.uniform(0.9, 1.12) if sketch else 1.0)
        el.w = el.h = s
        el.lines = _label_lines(el.text, P, meas, rng)
    elif el.kind == "data":
        el.w = F * (rng.uniform(1.8, 2.5) if sketch else 2.6)
        el.h = el.w * (rng.uniform(1.25, 1.45) if sketch else 1.38)
        el.lines = _label_lines(el.text, P, meas, rng)
    elif el.kind == "store":
        el.w = F * (rng.uniform(2.5, 3.4) if sketch else 3.4)
        el.h = el.w * (rng.uniform(0.75, 0.95) if sketch else 1.0)
        el.lines = _label_lines(el.text, P, meas, rng)
    elif el.kind == "note":
        if sketch:
            el.lines = wrap_chars(el.text, rng.randint(10, 20), rng)
        else:
            el.lines = wrap_width(el.text, F * rng.uniform(7, 11), P.Fl, meas)
        tw, th = meas.block(el.lines, P.Fl, P.lh)
        el.w = tw + F * 0.9
        el.h = th + F * 0.7


def _label_lines(text: str, P: Params, meas: Measure, rng: random.Random) -> list[str]:
    if not text:
        return []
    if P.style == "sketch":
        return wrap_chars(text, max(5, P.wrap_label + rng.randint(-2, 3)), rng, hyphen=0.03)
    return wrap_width(text, P.F * 6.4, P.Fl, meas)


# --------------------------------------------------------------------------- #
# Block grid
# --------------------------------------------------------------------------- #
@dataclass
class Box:
    w: int
    top: int
    bot: int
    exit_row: int
    cells: list[tuple[El, int, int]]
    corr: list[tuple[Flow, int]]

    def shift(self, dc: int, dr: int) -> tuple[list[tuple[El, int, int]], list[tuple[Flow, int]]]:
        return ([(e, c + dc, r + dr) for e, c, r in self.cells], [(f, r + dr) for f, r in self.corr])


class Grid:
    def __init__(self, rng: random.Random, p_step: float, flows: list[Flow]):
        self.rng = rng
        self.p_step = p_step
        self.by_key = {(f.src, f.dst, f.hint): f for f in flows}

    def place(self, b) -> Box:
        if isinstance(b, Leaf):
            return Box(1, 0, 0, 0, [(b.el, 0, 0)], [])
        if isinstance(b, Seq):
            return self._seq(b)
        if isinstance(b, Split):
            return self._split(b)
        if isinstance(b, Loop):
            return self._loop(b)
        if isinstance(b, XorEnd):
            return self._xorend(b)
        raise TypeError(type(b))

    def _seq(self, b: Seq) -> Box:
        cells: list = []
        corr: list = []
        col = row = top = bot = 0
        for i, it in enumerate(b.items):
            bx = self.place(it)
            off = row
            if i > 0 and self.rng.random() < self.p_step:
                # step to a neighbouring row (another lane, often), but stay within one row of
                # the sequence's first row so the drawing does not drift
                off += self.rng.choice([-1, 1]) if row == 0 else (-1 if row > 0 else 1)
            c, k = bx.shift(col, off)
            cells += c
            corr += k
            top, bot = min(top, off + bx.top), max(bot, off + bx.bot)
            col += bx.w
            row = off + bx.exit_row
        return Box(col, top, bot, row, cells, corr)

    def _split(self, b: Split) -> Box:
        rng = self.rng
        boxes = [self.place(br) if br.items else None for br in b.branches]
        k = len(boxes)
        nonempty = [i for i in range(k) if boxes[i] is not None]
        skips = [i for i in range(k) if boxes[i] is None]
        main = skips[0] if skips and rng.random() < 0.4 else nonempty[0]
        others = [i for i in range(k) if i != main]
        if len(others) == 1:
            sides = [1 if rng.random() < 0.65 else -1]
        else:
            r = rng.random()
            sides = [-1, 1] if r < 0.6 else [1, 1] if r < 0.9 else [-1, -1]
        w_in = max(bx.w for bx in boxes if bx is not None)
        cells: list = [(b.split, 0, 0), (b.merge, w_in + 1, 0)]
        corr: list = []
        top = bot = 0
        if boxes[main] is not None:
            c, kk = boxes[main].shift(1, 0)
            cells += c
            corr += kk
            top, bot = boxes[main].top, boxes[main].bot
        for i, s in zip(others, sides, strict=True):
            bx = boxes[i]
            if bx is None:
                row = bot + 1 if s > 0 else top - 1
                f = self.by_key[(b.split.id, b.merge.id, "skip")]
                f.side = s
                corr.append((f, row))
                top, bot = min(top, row), max(bot, row)
                continue
            off = bot + 1 - bx.top if s > 0 else top - 1 - bx.bot
            c, kk = bx.shift(1, off)
            cells += c
            corr += kk
            top, bot = min(top, off + bx.top), max(bot, off + bx.bot)
        return Box(w_in + 2, top, bot, 0, cells, corr)

    def _loop(self, b: Loop) -> Box:
        bx = self.place(b.body)
        c0 = 1 if b.merge is not None else 0
        cells, corr = bx.shift(c0, 0)
        if b.merge is not None:
            cells.append((b.merge, 0, 0))
        cells.append((b.split, c0 + bx.w, bx.exit_row))
        target = b.merge.id if b.merge is not None else _entry_id(b.body)
        f = self.by_key[(b.split.id, target, "back")]
        row = bx.bot + 1 if b.side > 0 else bx.top - 1
        corr.append((f, row))
        return Box(c0 + bx.w + 1, min(bx.top, row), max(bx.bot, row), bx.exit_row, cells, corr)

    def _xorend(self, b: XorEnd) -> Box:
        cb = self.place(b.cont)
        xb = self.place(b.exit)
        cells, corr = cb.shift(1, 0)
        cells.append((b.split, 0, 0))
        off = cb.bot + 1 - xb.top if b.side > 0 else cb.top - 1 - xb.bot
        c, k = xb.shift(1, off)
        cells += c
        corr += k
        top, bot = min(cb.top, off + xb.top), max(cb.bot, off + xb.bot)
        return Box(1 + max(cb.w, xb.w), top, bot, cb.exit_row, cells, corr)


def _entry_id(b) -> str:
    if isinstance(b, Leaf):
        return b.el.id
    if isinstance(b, Seq):
        return _entry_id(b.items[0])
    if isinstance(b, Split):
        return b.split.id
    if isinstance(b, Loop):
        return b.merge.id if b.merge is not None else _entry_id(b.body)
    if isinstance(b, XorEnd):
        return b.split.id
    raise TypeError(type(b))


# --------------------------------------------------------------------------- #
# Anchors and routes
# --------------------------------------------------------------------------- #
def anchor(el: El, side: str, dx: float = 0.0) -> tuple[float, float]:
    x, y, w, h = el.x, el.y, el.w, el.h
    if el.kind in ("start", "end", "inter"):
        r = w / 2
        if side in ("top", "bottom"):
            dx = max(-0.8 * r, min(0.8 * r, dx))
            dy = math.sqrt(max(r * r - dx * dx, 0.0))
            return (x + dx, y - dy) if side == "top" else (x + dx, y + dy)
        return (x + r, y) if side == "right" else (x - r, y)
    if el.kind == "gw":
        return {"right": (x + w / 2, y), "left": (x - w / 2, y), "top": (x, y - h / 2),
                "bottom": (x, y + h / 2)}[side]
    return {"right": (x + w / 2, y), "left": (x - w / 2, y), "top": (x + dx, y - h / 2),
            "bottom": (x + dx, y + h / 2)}[side]


def el_rect(el: El) -> Rect:
    return (el.x - el.w / 2, el.y - el.h / 2, el.x + el.w / 2, el.y + el.h / 2)


# --------------------------------------------------------------------------- #
# Main entry
# --------------------------------------------------------------------------- #
def layout_process(proc: Process, gen: ProcessGenerator, style: str, rng: random.Random) -> Scene:
    P = make_params(style, rng)
    meas = Measure(P.font, 1.0 if style == "clean" else 1.07)
    L = _Layout(proc, gen, P, meas, rng)
    return L.run()


class _Layout:
    def __init__(self, proc: Process, gen: ProcessGenerator, P: Params, meas: Measure, rng: random.Random):
        self.proc, self.gen, self.P, self.meas, self.rng = proc, gen, P, meas, rng
        self.occ = Occ()
        self.els = proc.els
        self.flows: list[Flow] = list(proc.flows)
        self.bands: list[Band] = []
        self.used_sides: set[tuple[str, str]] = set()

    # ------------------------------------------------------------------ run
    def run(self) -> Scene:
        P, rng = self.P, self.rng
        for el in self.els.values():
            size_element(el, P, self.meas, rng)
        grid = Grid(rng, P.p_step if (len(self.proc.lanes) > 1) else P.p_step * 0.3, self.flows)
        box = grid.place(self.proc.root)
        r0 = box.top
        for el, c, r in box.cells:
            el.col, el.row = c, r - r0
        self.corr_row = {id(f): r - r0 for f, r in box.corr}
        self.n_rows = box.bot - box.top + 1
        self.n_cols = box.w
        self.main = [el for el, _, _ in box.cells]
        self._reserve_extras()
        self._rows_and_lanes()
        self._columns()
        self._route_main()
        self._event_labels(self.main)
        self._flow_labels()
        self._place_data()
        self._place_notes()
        self._main_bands()
        self._partners()
        return self._finish()

    # ------------------------------------------------------------- extras
    def _reserve_extras(self) -> None:
        """Pick a side (above / below the anchor's row) for data objects and annotations and
        reserve room there, so the free-space search later usually succeeds."""
        P, rng = self.P, self.rng
        self.extra_up = [0.0] * self.n_rows
        self.extra_down = [0.0] * self.n_rows
        self.data_side: dict[str, int] = {}
        for d in self.proc.data:
            links = [(s, t) for s, t in self.proc.data_links if d.id in (s, t)]
            task = self.els[links[0][0] if links[0][1] == d.id else links[0][1]]
            side = -1 if rng.random() < 0.5 else 1
            self.data_side[d.id] = side
            _, lab_h = self.meas.block(d.lines, P.Fl, P.lh)
            need = d.h + P.F * 1.0 + lab_h * (1 if side > 0 else 0.0) + P.F * 0.6
            arr = self.extra_down if side > 0 else self.extra_up
            arr[task.row] = max(arr[task.row], need)
        for (eid, nid) in self.proc.note_links:
            el, note = self.els[eid], self.els[nid]
            side = -1 if rng.random() < 0.55 else 1
            self.data_side[nid] = side
            need = note.h + P.F * 1.4
            arr = self.extra_down if side > 0 else self.extra_up
            arr[el.row] = max(arr[el.row], need)

    # ------------------------------------------------------------- rows
    def _rows_and_lanes(self) -> None:
        P, rng = self.P, self.rng
        n = self.n_rows
        up = [P.F * 0.45] * n
        down = [P.F * 0.45] * n
        has_node = [False] * n
        for el in self.main:
            r = el.row
            has_node[r] = True
            u, d = el.h / 2, el.h / 2
            if el.kind in ("start", "end", "inter") and el.lines:
                d += P.F * 0.25 + len(el.lines) * P.lh * P.Fl
            if el.kind == "gw" and el.lines:
                u += P.F * 0.2 + len(el.lines) * P.lh * P.Fl
            up[r] = max(up[r], u)
            down[r] = max(down[r], d)
        for r in range(n):
            up[r] += self.extra_up[r]
            down[r] += self.extra_down[r]
        # lanes: contiguous groups of node rows
        node_rows = [r for r in range(n) if has_node[r]]
        names = list(self.proc.lanes)
        n_l = max(1, min(len(names), len(node_rows))) if names else 1
        if names:
            names = names[:n_l]
            self.proc.lanes = names
        cuts = sorted(rng.sample(range(1, len(node_rows)), n_l - 1)) if n_l > 1 else []
        lane_of_row = [0] * n
        lane = 0
        bounds = [node_rows[c] for c in cuts]
        for r in range(n):
            while lane < len(bounds) and r >= bounds[lane]:
                lane += 1
            lane_of_row[r] = lane
        self.lane_of_row = lane_of_row
        self.n_lanes = n_l
        # lane names: header text must fit the band height
        self.lane_lines: list[list[str]] = []
        need: list[float] = []
        for i in range(n_l):
            if names:
                lines, length = self._fit_name(names[i], P.Fh)
            else:
                lines, length = [], 0.0
            self.lane_lines.append(lines)
            need.append(length + P.F * 1.2)
        # y positions
        yc = [0.0] * n
        self.lane_y: list[tuple[float, float]] = []
        y = 0.0
        for i in range(n_l):
            rows = [r for r in range(n) if lane_of_row[r] == i]
            natural = sum(up[r] + down[r] for r in rows) + P.row_gap * (len(rows) - 1) + 2 * P.lane_pad
            pad = P.lane_pad + max(0.0, need[i] - natural) / 2
            top = y
            y += pad
            for j, r in enumerate(rows):
                y += up[r]
                yc[r] = y
                y += down[r]
                if j < len(rows) - 1:
                    y += P.row_gap * (rng.uniform(0.75, 1.3) if P.style == "sketch" else 1.0)
            y += pad
            self.lane_y.append((top, y))
        # the pool name must fit the whole pool height
        self.pool_lines: list[str] = []
        if self.proc.pool_name:
            self.pool_lines, plen = self._fit_name(self.proc.pool_name, P.Fh)
            short = plen + P.F * 1.2 - (y - self.lane_y[0][0])
            if short > 0:
                self.lane_y[0] = (self.lane_y[0][0] - short / 2, self.lane_y[0][1])
                self.lane_y[-1] = (self.lane_y[-1][0], self.lane_y[-1][1] + short / 2)
        # header strips must hold the (possibly two-line) names
        self.hdr = max(P.hdr, len(self.pool_lines) * P.lh * P.Fh + P.F * 0.6)
        self.lhdr = max([P.lhdr] + [len(ls) * P.lh * P.Fh + P.F * 0.6 for ls in self.lane_lines])
        self.horizontal_names = self.proc.lanes_without_pool and rng.random() < 0.5
        if self.horizontal_names:
            self.lhdr = max(self.meas.width(t, P.Fh) for ls in self.lane_lines for t in ls) + P.F * 1.2
        self.yc = yc
        self.up, self.down = up, down
        for el in self.main:
            el.y = yc[el.row] + (rng.uniform(-1, 1) * P.jitter * 0.6 if P.jitter else 0.0)

    def _fit_name(self, name: str, size: float) -> tuple[list[str], float]:
        """Lines of a lane / pool name and the length they need along the header strip."""
        w = self.meas.width(name, size)
        words = name.split()
        if len(words) >= 2 and self.rng.random() < (0.25 if self.P.style == "sketch" else 0.1):
            mid = (len(words) + 1) // 2
            lines = [" ".join(words[:mid]), " ".join(words[mid:])]
            return lines, max(self.meas.width(t, size) for t in lines)
        return [name], w

    # ------------------------------------------------------------- columns
    def _columns(self) -> None:
        P, rng = self.P, self.rng
        cw = [0.0] * self.n_cols
        for el in self.main:
            cw[el.col] = max(cw[el.col], el.w)
        gap = [P.col_gap * (rng.uniform(0.8, 1.25) if P.style == "sketch" else 1.0) for _ in cw]
        by_id = self.els
        for f in self.flows:
            if f.kind == "seq" and f.label:
                # wrapped once here; _flow_labels places exactly these lines
                f.label_lines = self._edge_label_lines(f.label)
                if f.hint == "back":
                    continue  # placed beside the corridor's vertical part, not in the gap
                w, _ = self.meas.block(f.label_lines, P.Fl, P.lh)
                s = by_id[f.src]
                need = w + P.F * 0.9 - (cw[s.col] - s.w) / 2
                gap[s.col] = max(gap[s.col], need)
        x = 0.0
        xc = [0.0] * self.n_cols
        for c in range(self.n_cols):
            xc[c] = x + cw[c] / 2
            x += cw[c] + gap[c]
        self.xc, self.cw, self.gap = xc, cw, gap
        for el in self.main:
            jx = rng.uniform(-1, 1) * min(P.jitter, gap[el.col] * 0.15) if P.jitter else 0.0
            el.x = xc[el.col] + jx

    def _edge_label_lines(self, text: str) -> list[str]:
        if self.P.style == "sketch":
            return wrap_chars(text, max(5, self.P.wrap_label + self.rng.randint(-2, 2)), self.rng)
        return wrap_width(text, self.P.F * 6.0, self.P.Fl, self.meas)

    # ------------------------------------------------------------- routing
    def _corr_y(self, f: Flow) -> float:
        return self.yc[self.corr_row[id(f)]]

    def _route_main(self) -> None:
        rng = self.rng
        els = self.els
        order = sorted(self.flows, key=lambda f: 0 if f.hint in ("back", "skip") else 1)
        for f in order:
            s, t = els[f.src], els[f.dst]
            if f.hint == "back" or (f.hint == "skip" and id(f) in self.corr_row):
                side = "bottom" if f.side > 0 else "top"
                y = self._corr_y(f)
                f.points = [anchor(s, side), (s.x, y), (t.x, y), anchor(t, side)]
                self.used_sides.add((s.id, side))
                self.used_sides.add((t.id, side))
            elif s.row == t.row:
                f.points = [anchor(s, "right"), anchor(t, "left")]
            elif f.hint == "split":
                side = "bottom" if t.y > s.y else "top"
                f.points = [anchor(s, side), (s.x, t.y), anchor(t, "left")]
                self.used_sides.add((s.id, side))
            elif f.hint == "join":
                side = "top" if s.y < t.y else "bottom"
                f.points = [anchor(s, "right"), (t.x, s.y), anchor(t, side)]
                self.used_sides.add((t.id, side))
            else:
                side = "bottom" if t.y > s.y else "top"
                if s.kind != "gw" and (s.id, side) not in self.used_sides and rng.random() < 0.5:
                    f.points = [anchor(s, side), (s.x, t.y), anchor(t, "left")]
                    self.used_sides.add((s.id, side))
                else:
                    xm = (s.x + s.w / 2 + t.x - t.w / 2) / 2
                    f.points = [anchor(s, "right"), (xm, s.y), (xm, t.y), anchor(t, "left")]
        for el in self.main:
            self.occ.add_rect(el_rect(el), el.id)
        for f in self.flows:
            self.occ.add_path(f.points, "seq")

    # ------------------------------------------------------------- labels
    def _label_box(self, el: El, where: str, lw: float, lh: float) -> Rect:
        F = self.P.F
        x, y, w, h = el.x, el.y, el.w, el.h
        g = F * 0.2
        if where == "below":
            return (x - lw / 2, y + h / 2 + g, x + lw / 2, y + h / 2 + g + lh)
        if where == "above":
            return (x - lw / 2, y - h / 2 - g - lh, x + lw / 2, y - h / 2 - g)
        if where == "below_right":
            return (x + w * 0.2, y + h / 2 + g, x + w * 0.2 + lw, y + h / 2 + g + lh)
        if where == "above_right":
            return (x + w * 0.2, y - h / 2 - g - lh, x + w * 0.2 + lw, y - h / 2 - g)
        if where == "above_left":
            return (x - w * 0.2 - lw, y - h / 2 - g - lh, x - w * 0.2, y - h / 2 - g)
        if where == "below_left":
            return (x - w * 0.2 - lw, y + h / 2 + g, x - w * 0.2, y + h / 2 + g + lh)
        if where == "right":
            return (x + w / 2 + g, y - lh / 2, x + w / 2 + g + lw, y + lh / 2)
        return (x - w / 2 - g - lw, y - lh / 2, x - w / 2 - g, y + lh / 2)

    def _event_labels(self, els: list[El], limits: Optional[tuple[float, float]] = None) -> None:
        """External labels of events / gateways, kept inside the element's lane (or `limits`)."""
        P = self.P
        for el in els:
            if el.kind not in ("start", "end", "inter", "gw") or not el.lines:
                continue
            lw, lh = self.meas.block(el.lines, P.Fl, P.lh)
            lo, hi = limits if limits is not None else self._row_limits(el)
            if el.kind == "gw":
                order = ["above_left", "above", "above_right", "below_left", "below_right"]
            else:
                order = ["below", "above", "below_right", "below_left", "above_right", "right", "left"]
            for where in order:
                box = self._label_box(el, where, lw, lh)
                if box[1] < lo or box[3] > hi:
                    continue
                if self.occ.rect_free(box, P.F * 0.15, ignore=()):
                    el.label_box = box
                    self.occ.add_rect(box, "label")
                    break
            else:
                el.lines = []  # nothing drawn -> no text in the target
                el.text = ""

    def _flow_labels(self) -> None:
        P = self.P
        # branches leaving a gateway through the same corner share the first (vertical) segment:
        # a label beside that trunk could belong to either branch, so it goes after the bend
        trunks: dict[tuple[str, bool], int] = {}
        for f in self.flows:
            if f.kind == "seq" and abs(f.points[0][0] - f.points[1][0]) < 1e-6:
                key = (f.src, f.points[1][1] > f.points[0][1])
                trunks[key] = trunks.get(key, 0) + 1
        for f in self.flows:
            if f.kind != "seq" or not f.label:
                continue
            lines = f.label_lines or self._edge_label_lines(f.label)
            lw, lh = self.meas.block(lines, P.Fl, P.lh)
            vertical = abs(f.points[0][0] - f.points[1][0]) < 1e-6
            shared = vertical and trunks.get((f.src, f.points[1][1] > f.points[0][1]), 0) > 1
            for box in self._flow_label_cands(f.points, lw, lh, shared):
                if self.occ.rect_free(box, P.F * 0.12):
                    f.label_lines, f.label_box = lines, box
                    self.occ.add_rect(box, "label")
                    break
            else:
                f.label, f.label_lines = "", []

    def _flow_label_cands(self, pts: list[tuple[float, float]], lw: float, lh: float,
                          shared: bool = False) -> list[Rect]:
        F = self.P.F
        g = F * 0.3
        (x0, y0), (x1, y1) = pts[0], pts[1]
        out: list[Rect] = []
        if abs(x0 - x1) < 1e-6:  # first segment vertical
            down = y1 > y0
            ys = y0 + F * 0.15 if down else y0 - F * 0.15 - lh
            if not shared:
                out.append((x0 + g, ys, x0 + g + lw, ys + lh))
                out.append((x0 - g - lw, ys, x0 - g, ys + lh))
            if len(pts) > 2:
                (x2, y2) = pts[2]
                right = x2 > x1
                if right:
                    out.append((x1 + g, y1 - F * 0.2 - lh, x1 + g + lw, y1 - F * 0.2))
                    out.append((x1 + g, y1 + F * 0.2, x1 + g + lw, y1 + F * 0.2 + lh))
                else:
                    out.append((x1 - g - lw, y1 - F * 0.2 - lh, x1 - g, y1 - F * 0.2))
                    out.append((x1 - g - lw, y1 + F * 0.2, x1 - g, y1 + F * 0.2 + lh))
            if not shared:
                ym = (y0 + y1) / 2 - lh / 2
                out.append((x0 + g, ym, x0 + g + lw, ym + lh))
        else:
            right = x1 > x0
            xs = x0 + F * 0.25 if right else x0 - F * 0.25 - lw
            out.append((xs, y0 - F * 0.2 - lh, xs + lw, y0 - F * 0.2))
            out.append((xs, y0 + F * 0.2, xs + lw, y0 + F * 0.2 + lh))
            if len(pts) > 2:
                (x2, y2) = pts[2]
                if abs(x1 - x2) < 1e-6:
                    down = y2 > y1
                    ys = y1 + F * 0.15 if down else y1 - F * 0.15 - lh
                    out.append((x1 + g, ys, x1 + g + lw, ys + lh))
                    out.append((x1 - g - lw, ys, x1 - g, ys + lh))
        return out

    # ------------------------------------------------------------- data objects
    def _row_limits(self, el: El) -> tuple[float, float]:
        """y range a data object / note near `el` may use: its lane band."""
        if not self.lane_y:
            return (-1e9, 1e9)
        top, bot = self.lane_y[self.lane_of_row[el.row]]
        return (top + self.P.F * 0.3, bot - self.P.F * 0.3)

    def _place_data(self) -> None:
        P, rng = self.P, self.rng
        keep: list[El] = []
        links_kept: list[tuple[str, str]] = []
        for d in self.proc.data:
            links = [(s, t) for s, t in self.proc.data_links if d.id in (s, t)]
            tasks = [self.els[s if t == d.id else t] for s, t in links]
            a = tasks[0]
            side = self.data_side[d.id]
            lo, hi = self._row_limits(a)
            lw, lh = self.meas.block(d.lines, P.Fl, P.lh)
            xs = [a.x, a.x - a.w * 0.45, a.x + a.w * 0.45, a.x + a.w / 2 + d.w, a.x - a.w / 2 - d.w]
            if len(tasks) > 1:
                xs.insert(0, (a.x + tasks[1].x) / 2)
            placed = False
            for sd in (side, -side):
                for dist in (1.0, 1.6):
                    for x in xs:
                        gap = P.F * dist * rng.uniform(0.8, 1.1)
                        y = a.y + sd * (a.h / 2 + gap + d.h / 2)
                        d.x, d.y = x, y
                        r = el_rect(d)
                        if r[1] < lo or r[3] > hi or not self.occ.rect_free(r, P.F * 0.3):
                            continue
                        paths = [self._assoc_path(d, self.els[s if s != d.id else t], s == d.id)
                                 for s, t in links]
                        tags = tuple(t.id for t in tasks) + (d.id,)
                        if not all(self.occ.path_free(p, P.F * 0.1, ignore=tags, max_cross=1) for p in paths):
                            continue
                        lab = self._free_label(d, lw, lh, ["below", "right", "left", "above"], lo, hi,
                                               extra=paths)
                        if d.lines and lab is None:
                            continue
                        d.label_box = lab
                        self.occ.add_rect(r, d.id)
                        if lab:
                            self.occ.add_rect(lab, "label")
                        for (s, t), p in zip(links, paths, strict=True):
                            self.flows.append(Flow(s, t, kind="data", points=p))
                            self.occ.add_path(p, "data")
                            links_kept.append((s, t))
                        placed = True
                        break
                    if placed:
                        break
                if placed:
                    break
            if placed:
                keep.append(d)
        self.proc.data = keep
        self.proc.data_links = links_kept

    def _assoc_path(self, d: El, t: El, outgoing: bool) -> list[tuple[float, float]]:
        """Straight association between data object d and element t (direction by `outgoing`)."""
        if abs(d.x - t.x) < t.w / 2 - 4 and t.kind == "task":
            x = d.x
            if d.y < t.y:
                p, q = (x, d.y + d.h / 2), (x, t.y - t.h / 2)
            else:
                p, q = (x, d.y - d.h / 2), (x, t.y + t.h / 2)
        else:
            p, q = _border_point(d, t.x, t.y), _border_point(t, d.x, d.y)
        return [p, q] if outgoing else [q, p]

    def _free_label(self, el: El, lw: float, lh: float, order: list[str], lo: float, hi: float,
                    extra: list[list[tuple[float, float]]] = ()) -> Optional[Rect]:
        if not el.lines:
            return None
        for where in order:
            box = self._label_box(el, where, lw, lh)
            if box[1] < lo or box[3] > hi:
                continue
            if not self.occ.rect_free(box, self.P.F * 0.12):
                continue
            if any(seg_hits_rect(a, b, box) for p in extra for a, b in pairwise(p)):
                continue
            return box
        return None

    # ------------------------------------------------------------- annotations
    def _place_notes(self) -> None:
        P, rng = self.P, self.rng
        keep, links = [], []
        for (eid, nid) in self.proc.note_links:
            el, n = self.els[eid], self.els[nid]
            lo, hi = self._row_limits(el)
            side = self.data_side[nid]
            placed = False
            for sd in (side, -side):
                for dx in (0.6, 1.2, 0.0, -0.4, 2.0):
                    x0 = el.x + el.w / 2 + P.F * dx * rng.uniform(0.8, 1.2) - (el.w * 0.5 if dx <= 0 else 0)
                    gap = P.F * rng.uniform(0.9, 1.5)
                    y0 = el.y + el.h / 2 + gap if sd > 0 else el.y - el.h / 2 - gap - n.h
                    n.x, n.y = x0 + n.w / 2, y0 + n.h / 2
                    r = el_rect(n)
                    if r[1] < lo or r[3] > hi or not self.occ.rect_free(r, P.F * 0.3):
                        continue
                    p = _border_point(el, x0, n.y)
                    path = [p, (x0, n.y)]
                    if not self.occ.path_free(path, P.F * 0.1, ignore=(el.id, n.id), max_cross=1):
                        continue
                    self.occ.add_rect(r, n.id)
                    self.occ.add_path(path, "assoc")
                    self.flows.append(Flow(eid, nid, kind="assoc", points=path))
                    placed = True
                    break
                if placed:
                    break
            if placed:
                keep.append(n)
                links.append((eid, nid))
        self.proc.notes = keep
        self.proc.note_links = links

    # ------------------------------------------------------------- bands
    def _content_x(self) -> tuple[float, float]:
        xs: list[float] = []
        for el in self.main + self.proc.data + self.proc.notes:
            r = el_rect(el)
            xs += [r[0], r[2]]
            if el.label_box:
                xs += [el.label_box[0], el.label_box[2]]
        for f in self.flows:
            xs += [p[0] for p in f.points]
            if f.label_box:
                xs += [f.label_box[0], f.label_box[2]]
        return min(xs), max(xs)

    def _main_bands(self) -> None:
        P, rng = self.P, self.rng
        cx0, cx1 = self._content_x()
        pad_l = P.F * rng.uniform(0.8, 2.0)
        pad_r = P.F * rng.uniform(0.8, 2.5)
        proc = self.proc
        self.pool_box: Optional[Rect] = None
        if not proc.pool_name and not proc.lanes_without_pool:
            return
        top, bot = self.lane_y[0][0], self.lane_y[-1][1]
        x1 = cx1 + pad_r
        lanes_here = len(proc.lanes) > 1 or proc.lanes_without_pool
        pool: Optional[Band] = None
        if proc.lanes_without_pool:
            x0 = cx0 - pad_l - self.lhdr
        else:
            x0 = cx0 - pad_l - (self.lhdr if lanes_here else 0.0) - self.hdr
            pool = Band("pool_main", "pool", self.pool_lines, x0, top, x1, bot, hdr=self.hdr)
            self.bands.append(pool)
            self.pool_box = (x0, top, x1, bot)
            self.occ.add_rect((x0, top, x0 + self.hdr, bot), "hdr")
        if lanes_here:
            lx0 = x0 if pool is None else x0 + self.hdr
            for i, (lt, lb) in enumerate(self.lane_y):
                members = [el.id for el in self.main if self.lane_of_row[el.row] == i]
                self.bands.append(Band(f"lane{i}", "lane", self.lane_lines[i], lx0, lt, x1, lb,
                                       hdr=self.lhdr, parent=pool.id if pool else None, members=members,
                                       horizontal=self.horizontal_names))
                self.occ.add_rect((lx0, lt, lx0 + self.lhdr, lb), "hdr")
            if pool is None:
                self.pool_box = (x0, top, x1, bot)
        if pool is not None:
            main_members = [el.id for el in self.main] if not lanes_here else []
            pool.members = main_members + [d.id for d in proc.data] + [n.id for n in proc.notes]
        # pool / lane outlines are obstacles for message-flow labels
        if self.pool_box:
            b = self.pool_box
            self.occ.add_path([(b[0], b[1]), (b[2], b[1]), (b[2], b[3]), (b[0], b[3]), (b[0], b[1])], "pool")
            for lt, _ in self.lane_y[1:]:
                self.occ.add_path([(b[0], lt), (b[2], lt)], "pool")

    # ------------------------------------------------------------- partners
    def _partners(self) -> None:
        if not self.proc.partners or self.pool_box is None:
            self.proc.partners = []
            return
        kept = []
        for p in self.proc.partners:
            ok = self._partner(p)
            if ok:
                kept.append(p)
        self.proc.partners = kept

    def _msg_candidates(self, side: int) -> list[El]:
        """Main elements that can send / receive messages, best first."""
        rng = self.rng
        prio: list[list[El]] = [[], [], []]
        for el in self.main:
            if el.kind == "gw":
                continue
            if el.ev == "message" and el.kind in ("start", "inter", "end"):
                prio[0].append(el)
            elif el.kind == "task":
                prio[1].append(el)
            elif el.kind in ("start", "end") and el.ev == "none" and rng.random() < 0.3:
                prio[2].append(el)
        out: list[El] = []
        for group in prio:
            rng.shuffle(group)
            out += group
        return out

    def _vertical_clear(self, el: El, x: float, y_from: float, y_to: float) -> bool:
        lo, hi = sorted((y_from, y_to))
        line = (x - 1.0, lo, x + 1.0, hi)
        F = self.P.F
        for r, tag in self.occ.rects:
            if tag in (el.id, "hdr"):
                continue
            if overlaps(inflate(r, F * 0.25), line):
                return False
        for a, b, tag in self.occ.segs:
            if tag == "pool":
                continue
            if _collinear_overlap(a, b, (x, lo), (x, hi), F * 0.45):
                return False
            if abs(a[0] - b[0]) > 1e-6 and abs(a[1] - b[1]) > 1e-6:  # diagonal (data / notes)
                if segs_cross(a, b, (x, lo), (x, hi)):
                    return False
        return True

    def _msg_direction(self, el: El) -> int:
        """+1: the main element sends, -1: it receives."""
        if el.kind == "start":
            return -1
        if el.kind == "end":
            return 1
        if el.kind == "inter":
            return 1 if el.throw else -1
        return 1 if self.rng.random() < 0.5 else -1

    def _partner(self, p: dict) -> bool:
        P, rng = self.P, self.rng
        side = p["side"]
        pb = self.pool_box
        assert pb is not None
        edge_main = pb[1] if side < 0 else pb[3]
        a_side = "top" if side < 0 else "bottom"
        links: list[tuple[El, float, int]] = []  # (main element, x, direction: +1 main sends)
        for el in self._msg_candidates(side):
            if len(links) >= p["n_msg"]:
                break
            if (el.id, a_side) in self.used_sides:
                continue
            dxs = [0.0] if el.kind != "task" else [0.0, 0.25, -0.25, 0.38, -0.38]
            for dx in dxs:
                x = el.x + dx * el.w
                if any(abs(x - lx) < P.F * 1.5 for _, lx, _ in links):
                    continue
                a = anchor(el, a_side, dx * el.w)
                if self._vertical_clear(el, x, a[1], edge_main):
                    links.append((el, x, self._msg_direction(el)))
                    self.used_sides.add((el.id, a_side))
                    if el.kind in ("start", "end"):
                        el.ev = "message"  # a start / end event with a message flow is a message event
                    break
        if not links:
            return False
        gid = f"partner{len(self.bands)}"
        lines, _ = self._fit_name(p["name"], P.Fh)
        if p["mode"] == "expanded":
            return self._expanded_partner(p, gid, lines, links, side, edge_main, a_side)
        gap = P.pool_gap
        y1 = pb[1] - gap if side < 0 else pb[3] + gap + P.collapsed_h
        y0 = y1 - P.collapsed_h
        edge_partner = y1 if side < 0 else y0
        jit = P.F * (rng.uniform(-1.5, 1.5) if P.style == "sketch" else 0.0)
        band = Band(gid, "collapsed", lines, pb[0] + jit, y0, pb[2] + jit * 0.5, y1, multi=p["multi"])
        self.bands.append(band)
        self.occ.add_rect((band.x0, band.y0, band.x1, band.y1), gid)
        new_flows = []
        for el, x, direction in links:
            pts = [anchor(el, a_side, x - el.x), (x, edge_partner)]
            f = Flow(el.id, gid, kind="msg") if direction > 0 else Flow(gid, el.id, kind="msg")
            f.points = pts if direction > 0 else pts[::-1]
            self.flows.append(f)
            self.occ.add_path(f.points, "msg")
            new_flows.append((f, x))
        for f, x in new_flows:
            self._msg_label(f, side, x, edge_main, edge_partner)
        return True

    def _msg_label(self, f: Flow, side: int, x: float, edge_main: float, edge_partner: float) -> None:
        """Message name next to the vertical part of a message flow (between the pools if it fits)."""
        P, rng = self.P, self.rng
        if rng.random() > 0.6:
            return
        text = self.gen.label(rng.choice(self.proc.domain["messages"]))
        lines = self._edge_label_lines(text)
        lw, lh = self.meas.block(lines, P.Fl, P.lh)
        ymid = (edge_main + edge_partner) / 2
        g = P.F * 0.3
        cands = [(x + g, ymid - lh / 2, x + g + lw, ymid + lh / 2),
                 (x - g - lw, ymid - lh / 2, x - g, ymid + lh / 2)]
        inner = edge_main + (P.F * 0.4 if side < 0 else -P.F * 0.4 - lh)
        cands += [(x + g, inner, x + g + lw, inner + lh), (x - g - lw, inner, x - g, inner + lh)]
        for box in cands:
            if self.occ.rect_free(box, P.F * 0.12):
                f.label, f.label_lines, f.label_box = text, lines, box
                self.occ.add_rect(box, "label")
                return

    def _expanded_partner(self, p: dict, gid: str, lines: list[str], links: list, side: int,
                          edge_main: float, a_side: str) -> bool:
        """A partner pool with its own short process, aligned under / over its counterparts so that
        most message flows are straight vertical lines (the others jog in the gap between pools)."""
        P, rng = self.P, self.rng
        pb = self.pool_box
        assert pb is not None
        els: list[El] = p["els"]
        mids = els[1:-1]
        links = sorted(links, key=lambda t: t[1])[: len(mids)]
        pairs: dict[str, tuple[El, float, int]] = {}
        for m, (el, x, direction) in zip(mids, links, strict=False):
            pairs[m.id] = (el, x, direction)
            if m.kind == "inter":
                m.throw = direction < 0  # the partner throws what the main element receives
                m.ev = "message"
                m.text = self.gen.label(V.event_name(rng, self.proc.domain, "throw" if m.throw else "catch"))
                m.lines = _label_lines(m.text, P, self.meas, rng)
        row_h = max(e.h for e in els)
        lab_h = max((len(e.lines) * P.lh * P.Fl for e in els if e.kind != "task"), default=0.0)
        pad = P.F * rng.uniform(0.8, 1.4)
        name_len = max(self.meas.width(t, P.Fh) for t in lines) + P.F * 1.2
        pad += max(0.0, name_len - (2 * pad + row_h + P.F * 0.3 + lab_h)) / 2
        height = pad + row_h + P.F * 0.3 + lab_h + pad
        gap = P.pool_gap + P.F * 1.2 * len(links)
        if side < 0:
            y1 = pb[1] - gap
            y0 = y1 - height
        else:
            y0 = pb[3] + gap
            y1 = y0 + height
        cy = y0 + pad + row_h / 2
        x_min = pb[0] + self.hdr + P.F * 1.0

        def half(e: El) -> float:  # half width including a label centred below the element
            if e.kind == "task" or not e.lines:
                return e.w / 2
            return max(e.w / 2, self.meas.block(e.lines, P.Fl, P.lh)[0] / 2 + P.F * 0.3)

        prev: Optional[El] = None
        for e in els:
            want = pairs[e.id][1] if e.id in pairs else -1e9
            lo = x_min + half(e) if prev is None else prev.x + half(prev) + P.col_gap * 0.8 + half(e)
            e.x = max(lo, want)
            e.y = cy
            prev = e
        x1 = max(pb[2], els[-1].x + els[-1].w / 2 + P.F * 1.5)
        band = Band(gid, "pool", lines, pb[0], y0, x1, y1, hdr=self.hdr, members=[e.id for e in els])
        seqs = [Flow(a.id, b.id, kind="seq", points=[anchor(a, "right"), anchor(b, "left")])
                for a, b in pairwise(els)]
        p_side = "bottom" if side < 0 else "top"
        edge_partner = y1 if side < 0 else y0
        sgn = -1 if side < 0 else 1
        trial = Occ()
        for e in els:
            trial.add_rect(el_rect(e), e.id)
        for f in seqs:
            trial.add_path(f.points, "seq")
        msgs: list[tuple[Flow, float]] = []
        jogs = 0
        for m in mids:
            if m.id not in pairs:
                continue
            el, x, direction = pairs[m.id]
            a = anchor(el, a_side, x - el.x)
            b = anchor(m, p_side, 0.0)
            if abs(b[0] - x) < 1.0:
                pts = [a, (x, b[1])]
            else:
                jogs += 1
                yj = edge_main + sgn * P.F * (0.9 + 1.0 * jogs)
                if sgn * (edge_partner - yj) < P.F * 0.5:
                    continue
                pts = [a, (x, yj), (b[0], yj), b]
            outside = [(x, edge_main)] + pts[1:]
            if not trial.path_free(outside, P.F * 0.1, ignore=(m.id,), max_cross=0):
                continue
            if not self.occ.path_free(outside, P.F * 0.1, ignore=("pool", "hdr", m.id), max_cross=0):
                continue
            f = Flow(el.id, m.id, kind="msg") if direction > 0 else Flow(m.id, el.id, kind="msg")
            f.points = pts if direction > 0 else pts[::-1]
            trial.add_path(pts, "msg")
            msgs.append((f, x))
        if not msgs:
            return False  # a partner pool without any message flow is not drawn
        self.bands.append(band)
        self.occ.add_rect((pb[0], y0, pb[0] + self.hdr, y1), "hdr")
        for e in els:
            self.occ.add_rect(el_rect(e), e.id)
        for f in seqs:
            self.flows.append(f)
            self.occ.add_path(f.points, "seq")
        for f, _ in msgs:
            self.flows.append(f)
            self.occ.add_path(f.points, "msg")
        self._event_labels([e for e in els if e.kind != "task"], limits=(y0 + P.F * 0.2, y1 - P.F * 0.2))
        for f, x in msgs:
            self._msg_label(f, side, x, edge_main, edge_partner)
        self.partner_els = getattr(self, "partner_els", []) + els
        return True

    # ------------------------------------------------------------- finish
    def _finish(self) -> Scene:
        P = self.P
        els = self.main + self.proc.data + self.proc.notes + getattr(self, "partner_els", [])
        xs: list[float] = []
        ys: list[float] = []
        for el in els:
            r = el_rect(el)
            xs += [r[0], r[2]]
            ys += [r[1], r[3]]
            if el.label_box:
                xs += [el.label_box[0], el.label_box[2]]
                ys += [el.label_box[1], el.label_box[3]]
        for f in self.flows:
            xs += [q[0] for q in f.points]
            ys += [q[1] for q in f.points]
            if f.label_box:
                xs += [f.label_box[0], f.label_box[2]]
                ys += [f.label_box[1], f.label_box[3]]
        for b in self.bands:
            xs += [b.x0, b.x1]
            ys += [b.y0, b.y1]
        x0, y0 = min(xs) - P.margin, min(ys) - P.margin
        dx, dy = -x0, -y0
        for el in els:
            el.x += dx
            el.y += dy
            if el.label_box:
                b = el.label_box
                el.label_box = (b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy)
        for f in self.flows:
            f.points = [(px + dx, py + dy) for px, py in f.points]
            if f.label_box:
                b = f.label_box
                f.label_box = (b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy)
        for b in self.bands:
            b.x0 += dx
            b.x1 += dx
            b.y0 += dy
            b.y1 += dy
        W = max(xs) + dx + P.margin
        H = max(ys) + dy + P.margin
        content = (P.margin, P.margin, W - P.margin, H - P.margin)
        flows = [f for f in self.flows if f.points]
        return Scene(P.style, P, els, flows, self.bands, W, H, content)


def _border_point(el: El, tx: float, ty: float) -> tuple[float, float]:
    """Point on el's outline in the direction of (tx, ty)."""
    dx, dy = tx - el.x, ty - el.y
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return (el.x, el.y)
    if el.kind in ("start", "end", "inter"):
        d = math.hypot(dx, dy)
        r = el.w / 2
        return (el.x + dx / d * r, el.y + dy / d * r)
    if el.kind == "gw":
        s = (el.w / 2) / (abs(dx) + abs(dy))
        return (el.x + dx * s, el.y + dy * s)
    sx = (el.w / 2) / abs(dx) if abs(dx) > 1e-9 else 1e9
    sy = (el.h / 2) / abs(dy) if abs(dy) > 1e-9 else 1e9
    s = min(sx, sy)
    return (el.x + dx * s, el.y + dy * s)


def main_flow_nodes(proc: Process) -> list[El]:
    return [e for e in proc.els.values() if e.pool == "main" and e.kind in FLOW_KINDS]
