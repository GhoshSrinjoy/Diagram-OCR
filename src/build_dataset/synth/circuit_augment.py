"""Paper, ink and scan effects for the synthetic circuit renders.

The drawing is rendered once as an ink-coverage map (alpha in 0..1). It is then optionally warped
(small rotation / perspective, returned as a homography so boxes can follow exactly), composited
onto a generated paper and finished with photometric effects:

  white  - clean digital drawing on white
  scan   - aged cream paper like LogicBench-1K: low-frequency illumination, mottling, grain,
           specks, vignette, slight ink bleed, JPEG
  grid   - graph / ruled notebook paper
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

INKS = {
    "black": ((18, 18, 20), (45, 42, 40)),
    "sepia": ((4, 1, 0), (35, 26, 10)),  # LogicBench-1K ink on aged paper
    "blue": ((20, 30, 95), (45, 60, 140)),
    "pencil": ((55, 55, 58), (90, 90, 95)),
}


def ink_color(rng: np.random.Generator, ink: str) -> np.ndarray:
    lo, hi = INKS[ink]
    return np.array([rng.uniform(a, b) for a, b in zip(lo, hi, strict=True)], dtype=np.float32)


def _lowfreq(rng: np.random.Generator, h: int, w: int, cells: int) -> np.ndarray:
    small = rng.normal(0, 1, (cells, cells)).astype(np.float32)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def paper_white(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    v = rng.uniform(246, 255)
    return np.full((h, w, 3), v, dtype=np.float32)


def paper_scan(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    # LogicBench-1K: centre ~(245, 236, 209), corners ~(200, 185, 150) - a brownish vignette
    r0 = rng.uniform(234, 249)
    base = np.array([r0, r0 - rng.uniform(5, 13), r0 - rng.uniform(26, 44)], dtype=np.float32)
    illum = 1 + _lowfreq(rng, h, w, 4) * rng.uniform(0.008, 0.03)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    rad = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2) / np.sqrt(2)
    vig = rng.uniform(0.05, 0.26) * rad ** rng.uniform(1.8, 3.2)
    tint = np.array([1.0, 1.1, 1.35], dtype=np.float32)
    img = base[None, None, :] * illum[..., None] * (1 - vig[..., None] * tint[None, None, :])
    # mid-frequency mottling, slightly brown
    mott = _lowfreq(rng, h, w, int(rng.integers(6, 14))) * rng.uniform(0.4, 1.8)
    img += mott[..., None] * np.array([1.0, 0.95, 0.8], dtype=np.float32)
    # grain
    grain = cv2.GaussianBlur(rng.normal(0, rng.uniform(1.5, 4.0), (h, w)).astype(np.float32), (0, 0), 0.6)
    img += grain[..., None]
    # specks and the odd faint stain
    for _ in range(int(rng.integers(0, 30))):
        cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
        rr = float(rng.uniform(0.6, 2.0))
        col = np.array([rng.uniform(90, 150), rng.uniform(70, 120), rng.uniform(40, 90)])
        _blend_disc(img, cx, cy, rr, col, float(rng.uniform(0.15, 0.6)))
    for _ in range(int(rng.integers(0, 3))):
        cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
        rr = float(rng.uniform(0.02, 0.08) * max(h, w))
        col = np.array([rng.uniform(170, 210), rng.uniform(140, 180), rng.uniform(100, 140)])
        _blend_disc(img, cx, cy, rr, col, float(rng.uniform(0.04, 0.12)), soft=True)
    return img


def _blend_disc(img: np.ndarray, cx: int, cy: int, r: float, col: np.ndarray, a: float,
                soft: bool = False) -> None:
    h, w = img.shape[:2]
    pad = int(np.ceil(r * (3 if soft else 1))) + 2
    x0, x1, y0, y1 = max(0, cx - pad), min(w, cx + pad + 1), max(0, cy - pad), min(h, cy + pad + 1)
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    m = np.exp(-(d / r) ** 2) if soft else np.clip(r + 0.5 - d, 0, 1)
    m = (m * a)[..., None]
    img[y0:y1, x0:x1] = img[y0:y1, x0:x1] * (1 - m) + col[None, None, :] * m


def paper_grid(rng: np.random.Generator, h: int, w: int, pitch: float) -> np.ndarray:
    v = rng.uniform(240, 255)
    tint = np.array([v, v - rng.uniform(0, 6), v - rng.uniform(0, 14)], dtype=np.float32)
    img = np.broadcast_to(tint, (h, w, 3)).copy()
    palette = [[150, 185, 225], [150, 200, 170], [175, 175, 180], [215, 170, 150]]
    color = np.array(palette[int(rng.integers(0, 4))], dtype=np.float32)
    mask = np.zeros((h, w), np.float32)
    ruled = rng.random() < 0.25
    ox, oy = rng.uniform(0, pitch), rng.uniform(0, pitch)
    major = 5 if rng.random() < 0.5 else 0
    a_minor = rng.uniform(0.25, 0.55)
    k = 0
    y = oy
    while y < h:
        cv2.line(mask, (0, int(round(y))), (w - 1, int(round(y))),
                 1.0 if major and k % major == 0 else a_minor, 1, cv2.LINE_AA)
        y += pitch * (1.6 if ruled else 1.0)
        k += 1
    if ruled:
        xm = int(w * rng.uniform(0.06, 0.12))
        cv2.line(mask, (xm, 0), (xm, h - 1), 0.0, 1)
        red = np.zeros((h, w), np.float32)
        cv2.line(red, (xm, 0), (xm, h - 1), 1.0, 2, cv2.LINE_AA)
        img = img * (1 - 0.5 * red[..., None]) + np.array([220, 120, 120], np.float32) * 0.5 * red[..., None]
    else:
        k = 0
        x = ox
        while x < w:
            cv2.line(mask, (int(round(x)), 0), (int(round(x)), h - 1),
                     1.0 if major and k % major == 0 else a_minor, 1, cv2.LINE_AA)
            x += pitch
            k += 1
    m = (np.clip(mask, 0, 1) * rng.uniform(0.5, 0.9))[..., None]
    img = img * (1 - m) + color[None, None, :] * m
    img += rng.normal(0, rng.uniform(0.0, 1.5), (h, w, 1)).astype(np.float32)
    return img


def rubber_sheet(rng: np.random.Generator, h: int, w: int, amp: float) -> tuple[np.ndarray, np.ndarray]:
    """A smooth random displacement field (pixels): straight strokes come out gently curved."""
    cells = int(rng.integers(3, 6))
    return (_lowfreq(rng, h, w, cells) * amp, _lowfreq(rng, h, w, cells) * amp)


def remap(alpha: np.ndarray, field: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """out(p) = in(p + d(p))."""
    h, w = alpha.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return cv2.remap(alpha, xx + field[0], yy + field[1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                     borderValue=0.0)


def sheet_points(field: Optional[tuple[np.ndarray, np.ndarray]], pts: np.ndarray) -> np.ndarray:
    """Where remap() moves the given input points: solve p + d(p) = q by fixed-point iteration."""
    if field is None or len(pts) == 0:
        return pts
    h, w = field[0].shape
    q = np.asarray(pts, dtype=np.float64)
    p = q.copy()
    for _ in range(4):
        xi = np.clip(np.rint(p[:, 0]), 0, w - 1).astype(int)
        yi = np.clip(np.rint(p[:, 1]), 0, h - 1).astype(int)
        p = q - np.stack([field[0][yi, xi], field[1][yi, xi]], axis=1)
    return p


def warp_homography(rng: np.random.Generator, h: int, w: int, max_rot: float = 2.5,
                    max_jitter: float = 0.012) -> np.ndarray:
    """A small rotation + perspective, as a 3x3 homography in pixel coordinates."""
    ang = np.deg2rad(rng.uniform(-max_rot, max_rot))
    c, s = np.cos(ang), np.sin(ang)
    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    ctr = np.array([w / 2, h / 2], dtype=np.float32)
    dst = (src - ctr) @ np.array([[c, s], [-s, c]], dtype=np.float32) + ctr
    dst += rng.uniform(-max_jitter, max_jitter, (4, 2)).astype(np.float32) * np.array([w, h], np.float32)
    return cv2.getPerspectiveTransform(src, dst.astype(np.float32))


def apply_points(hmat: Optional[np.ndarray], pts: np.ndarray) -> np.ndarray:
    if hmat is None or len(pts) == 0:
        return pts
    out = cv2.perspectiveTransform(pts.reshape(-1, 1, 2).astype(np.float64), hmat.astype(np.float64))
    return out.reshape(-1, 2)


def warp_alpha(alpha: np.ndarray, hmat: np.ndarray) -> np.ndarray:
    h, w = alpha.shape
    return cv2.warpPerspective(alpha, hmat, (w, h), flags=cv2.INTER_LINEAR, borderValue=0.0)


def compose(rng: np.random.Generator, alpha: np.ndarray, paper: str, ink: str, pitch: float) -> np.ndarray:
    """Ink coverage -> RGB float image on the chosen paper, with ink / scan effects."""
    h, w = alpha.shape
    if paper == "scan":
        bg = paper_scan(rng, h, w)
        alpha = cv2.GaussianBlur(alpha, (0, 0), float(rng.uniform(0.3, 0.75)))
    elif paper == "grid":
        bg = paper_grid(rng, h, w, pitch)
    else:
        bg = paper_white(rng, h, w)
    if ink == "pencil":
        tex = cv2.GaussianBlur(rng.normal(0, 1, (h, w)).astype(np.float32), (0, 0), 0.8)
        alpha = alpha * np.clip(0.9 + 0.3 * tex, 0.5, 1.0) * rng.uniform(0.85, 1.0)
    else:
        alpha = alpha * rng.uniform(0.88, 1.0)
    a = np.clip(alpha, 0, 1)[..., None]
    img = bg * (1 - a) + ink_color(rng, ink)[None, None, :] * a
    if paper == "scan":
        img = cv2.GaussianBlur(img, (0, 0), float(rng.uniform(0.2, 0.6)))
        img += rng.normal(0, rng.uniform(0.5, 2.5), img.shape).astype(np.float32)
    if paper != "white":
        c, b = rng.uniform(0.93, 1.07), rng.uniform(-8, 6)
        img = (img - 128) * c + 128 + b
    return np.clip(img + 0.5, 0, 255).astype(np.uint8)
