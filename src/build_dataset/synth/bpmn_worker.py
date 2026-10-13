"""One synthetic BPMN sample: index -> process -> layout -> image on disk + Diagram IR.

`make_one(i)` is the multiprocessing entry point (top level, importable under spawn). Everything is
seeded from the index, so a sample is reproducible on its own. The IR follows
sources/hdbpmn.py exactly (see sources/synth_bpmn.py for the mapping summary). Node boxes are the
shapes only (event circle, gateway diamond), computed by mapping outline points through the same
transform as the pixels (scale + offset, then the photo homography for the hand-drawn style).
"""

from __future__ import annotations

import hashlib
import math
import random
import traceback
from collections.abc import Callable
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from diagram_ir import Diagram, Edge, Group, Node, validate
from diagram_ir.model import COORD_MAX

from ..common import extracted_image_path
from . import bpmn_photo as photo
from .bpmn_draw import SceneDrawer, make_theme
from .bpmn_layout import Scene, layout_process
from .bpmn_model import El, generate_process

SOURCE = "synth_bpmn"
SEED_BASE = "synth_bpmn/v1"
MAX_PIXELS = 1254400  # train_v2 resizes images to at most ~1.25 MP
CLEAN_SHARE = 0.4
MIN_EFF_EM = {"clean": 10.5, "sketch": 12.0}  # text em in px after the model's resize
ATTEMPTS = 8

Points = np.ndarray


def seed_of(i: int, tag: str) -> int:
    return int(hashlib.md5(f"{SEED_BASE}/{i}/{tag}".encode()).hexdigest()[:15], 16)


def style_of(i: int) -> str:
    return "clean" if random.Random(seed_of(i, "style")).random() < CLEAN_SHARE else "sketch"


def worker_init() -> None:
    cv2.setNumThreads(1)


def make_one(i: int) -> dict:
    try:
        return _make(i)
    except Exception:  # reported by the caller's SkipLog
        return {"i": i, "error": traceback.format_exc(limit=6)}


def _make(i: int) -> dict:
    style = style_of(i)
    local_id = f"{style}_{i:05d}"
    n_target = int(round(random.Random(seed_of(i, "size")).triangular(4, 25, 11)))
    last_error = "no attempt"
    for attempt in range(ATTEMPTS):
        n = max(4, int(round(n_target * (0.82 ** attempt))))  # smaller after each failure
        try:
            res = _attempt(i, style, local_id, attempt, n)
        except Exception:  # a rare layout corner case: draw again
            last_error = traceback.format_exc(limit=4)
            continue
        if isinstance(res, str):
            last_error = res
            continue
        return res
    return {"i": i, "error": f"no valid sample after {ATTEMPTS} attempts: {last_error}"}


def _attempt(i: int, style: str, local_id: str, attempt: int, n: int):
    rng = random.Random(seed_of(i, f"attempt{attempt}"))
    nrng = np.random.default_rng(seed_of(i, f"np{attempt}"))
    proc, gen = generate_process(rng, style, n)
    scene = layout_process(proc, gen, style, rng)
    out = _render_clean(scene, rng) if style == "clean" else _render_sketch(scene, rng, nrng)
    if isinstance(out, str):
        return out
    img, to_out, ext, quality = out
    d = build_diagram(scene, to_out, img.shape[1], img.shape[0])
    issues = validate(d, strict=True)
    if issues:
        return "; ".join(str(x) for x in issues[:3])
    path = extracted_image_path(SOURCE, f"{style}/{local_id}{ext}")
    pil = Image.fromarray(img)
    if ext == ".jpg":
        pil.save(path, quality=quality, subsampling=2 if quality < 85 else 0, optimize=True)
    else:
        pil.save(path, optimize=True)
    meta = {
        "style": style,
        "n_nodes": len(d.nodes),
        "n_lanes": sum(1 for b in scene.bands if b.kind == "lane") or (1 if proc.pool_name else 0),
        "pools": sum(1 for b in scene.bands if b.kind in ("pool", "collapsed")),
    }
    return {"i": i, "local_id": local_id, "image": str(path), "diagram": d, "meta": meta}


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _scale(rng: random.Random, F: float, w_u: float, h_u: float, em_range: tuple[float, float],
           side: tuple[float, float]) -> float:
    k = rng.uniform(*em_range) / F
    longest = max(w_u, h_u) * k
    if longest > side[1]:
        k *= side[1] / longest
    elif longest < side[0]:
        k *= side[0] / longest
    return k


def _eff_em(F: float, k: float, w: float, h: float) -> float:
    return F * k * min(1.0, math.sqrt(MAX_PIXELS / (w * h)))


def _render_clean(scene: Scene, rng: random.Random):
    F = scene.P.F
    if max(scene.W, scene.H) / min(scene.W, scene.H) > 5.0:
        return "too elongated"  # retried with fewer nodes
    k = _scale(rng, F, scene.W, scene.H, (13.0, 18.0), (1000, 2000))
    W, H = int(round(scene.W * k)), int(round(scene.H * k))
    if _eff_em(F, k, W, H) < MIN_EFF_EM["clean"]:
        return "text too small for the pixel budget"
    theme = make_theme("clean", rng, k)
    img = SceneDrawer(scene, theme, k, 0.0, 0.0, rng).render(W, H)

    def to_out(pts: Points) -> Points:
        return pts * k

    if rng.random() < 0.3:
        return img, to_out, ".jpg", rng.randint(88, 96)
    return img, to_out, ".png", 0


def _render_sketch(scene: Scene, rng: random.Random, nrng: np.random.Generator):
    F = scene.P.F
    ml, mr = F * rng.uniform(1.0, 4.0), F * rng.uniform(1.0, 4.0)
    mt, mb = F * rng.uniform(1.5, 4.5), F * rng.uniform(1.0, 4.5)
    holes = rng.choice(["top", "left", "top", "right"]) if rng.random() < 0.25 else ""
    if holes == "top":
        mt = max(mt, F * 3.8)
    elif holes == "left":
        ml = max(ml, F * 3.8)
    elif holes == "right":
        mr = max(mr, F * 3.8)
    w_u, h_u = scene.W + ml + mr, scene.H + mt + mb
    elongated = max(w_u, h_u) / min(w_u, h_u) > 2.4
    if rng.random() < (0.6 if elongated else 0.3):  # photo of the page: pad towards A4 proportions
        ratio = 1.414
        grow = 1.0 if elongated else 0.6
        if w_u >= h_u:
            want_h = w_u / ratio
            if h_u < want_h:
                add = min(want_h - h_u, grow * h_u)
                extra_b = add * rng.uniform(0.3, 1.0)
                mb += extra_b
                mt += add - extra_b
        else:
            want_w = h_u / ratio
            if w_u < want_w:
                add = min(want_w - w_u, grow * w_u)
                extra_r = add * rng.uniform(0.3, 1.0)
                mr += extra_r
                ml += add - extra_r
        w_u, h_u = scene.W + ml + mr, scene.H + mt + mb
    k = _scale(rng, F, w_u, h_u, (17.0, 30.0), (1100, 1720))
    W, H = int(round(w_u * k)), int(round(h_u * k))
    if _eff_em(F, k, W * 1.08, H * 1.08) < MIN_EFF_EM["sketch"]:
        return "text too small for the pixel budget"
    ox, oy = ml * k, mt * k
    unit = F * k
    theme = make_theme("sketch", rng, k)
    drawing = SceneDrawer(scene, theme, k, ox, oy, rng).render(W, H)
    paper = photo.make_paper(W, H, rng, nrng, unit)
    if holes:
        photo.add_holes(paper, rng, unit, holes)
    comp = photo.composite(paper, drawing, theme.pencil, rng)
    c = scene.content
    content = np.array([[c[0], c[1]], [c[2], c[1]], [c[2], c[3]], [c[0], c[3]]], np.float32) * k
    content += np.array([ox, oy], np.float32)
    scan = rng.random() < 0.3
    res = photo.photo_warp(comp, rng, nrng, content, scan)
    if res is None:
        return "photo warp failed"
    frame, M = res
    frame = photo.photo_effects(frame, rng, nrng, scan, unit)
    img = photo.to_uint8(frame)
    if max(img.shape[:2]) > 2000 or max(img.shape[:2]) < 1000:
        return "image size out of range"

    def to_out(pts: Points) -> Points:
        p = pts * k + np.array([ox, oy])
        return cv2.perspectiveTransform(p.reshape(-1, 1, 2).astype(np.float64), M).reshape(-1, 2)

    return img, to_out, ".jpg", rng.randint(60, 93)


# --------------------------------------------------------------------------- #
# Diagram IR (identical mapping to sources/hdbpmn.py)
# --------------------------------------------------------------------------- #
def outline(el: El) -> Points:
    """Points on the drawn shape outline (layout units); their box is the node box."""
    x, y, w, h = el.x, el.y, el.w, el.h
    if el.kind in ("start", "end", "inter"):
        th = np.linspace(0, 2 * np.pi, 33)[:-1]
        return np.stack([x + w / 2 * np.cos(th), y + h / 2 * np.sin(th)], axis=1)
    if el.kind == "gw":
        return np.array([[x, y - h / 2], [x + w / 2, y], [x, y + h / 2], [x - w / 2, y]])
    return np.array([[x - w / 2, y - h / 2], [x + w / 2, y - h / 2], [x + w / 2, y + h / 2],
                     [x - w / 2, y + h / 2]])


def _bbox(pts: Points, W: int, H: int) -> tuple[int, int, int, int]:
    def nx(v: float) -> int:
        return max(0, min(COORD_MAX, round(v / W * COORD_MAX)))

    def ny(v: float) -> int:
        return max(0, min(COORD_MAX, round(v / H * COORD_MAX)))

    return (nx(pts[:, 0].min()), ny(pts[:, 1].min()), nx(pts[:, 0].max()), ny(pts[:, 1].max()))


def _event_kind(el: El) -> str:
    if el.kind == "start":
        return "initial"
    if el.kind == "end":
        return "final"
    if el.ev == "timer":
        return "time_event"
    if el.ev == "message":
        return "send_signal" if el.throw else "receive_signal"
    return "other"


def build_diagram(scene: Scene, to_out: Callable[[Points], Points], W: int, H: int) -> Diagram:
    nid: dict[str, str] = {}
    nodes: list[Node] = []
    for el in scene.els:
        nid[el.id] = f"n{len(nid) + 1}"
    gid: dict[str, str] = {b.id: f"g{j + 1}" for j, b in enumerate(scene.bands)}
    fan_in: dict[str, int] = {}
    fan_out: dict[str, int] = {}
    for f in scene.flows:
        if f.kind == "seq":
            fan_out[f.src] = fan_out.get(f.src, 0) + 1
            fan_in[f.dst] = fan_in.get(f.dst, 0) + 1
    for el in scene.els:
        stereotype: Optional[str] = None
        if el.kind == "task":
            kind = "action"
        elif el.kind in ("start", "end", "inter"):
            kind = _event_kind(el)
        elif el.kind == "gw":
            splits = fan_out.get(el.id, 0) > 1 or fan_in.get(el.id, 0) <= 1
            if el.gw == "and":
                kind = "fork" if splits else "join"
            else:
                kind = "decision" if splits else "merge"
        elif el.kind in ("data", "store"):
            kind = "object"
            stereotype = "datastore" if el.kind == "store" else None
        else:
            kind = "note"
        drawn = el.kind in ("task", "note") or el.label_box is not None
        text = "\n".join(el.lines) if drawn else ""
        nodes.append(Node(id=nid[el.id], kind=kind, text=text, bbox=_bbox(to_out(outline(el)), W, H),
                          stereotype=stereotype))

    def ref(x: str) -> str:
        return nid.get(x) or gid[x]

    edges: list[Edge] = []
    for f in scene.flows:
        label = "\n".join(f.label_lines) if f.label_box is not None else ""
        if f.kind == "seq":
            edges.append(Edge(source=ref(f.src), target=ref(f.dst), kind="flow", label=label))
        elif f.kind == "msg":
            edges.append(Edge(source=ref(f.src), target=ref(f.dst), kind="flow", label=label, style="dashed"))
        elif f.kind == "data":
            edges.append(Edge(source=ref(f.src), target=ref(f.dst), kind="flow", style="dotted"))
        else:
            edges.append(Edge(source=ref(f.src), target=ref(f.dst), kind="anchor"))
    groups: list[Group] = []
    for b in scene.bands:
        corners = np.array([[b.x0, b.y0], [b.x1, b.y0], [b.x1, b.y1], [b.x0, b.y1]])
        groups.append(Group(id=gid[b.id], kind="lane", text="\n".join(b.lines),
                            members=[nid[m] for m in b.members if m in nid],
                            parent=gid[b.parent] if b.parent else None,
                            bbox=_bbox(to_out(corners), W, H)))
    return Diagram(type="activity", nodes=nodes, edges=edges, groups=groups)
