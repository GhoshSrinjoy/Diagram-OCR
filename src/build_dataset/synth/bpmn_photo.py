"""Paper backgrounds and phone-photo / scan degradation for the hand-drawn BPMN style.

The drawing (dark ink on white) is multiplied onto a paper image (plain, squared, lined or dotted,
optionally with punch holes and faint show-through), then warped by a random homography into the
photo frame (rotation, perspective, scale; the desk shows where the paper does not cover) and
degraded (uneven light, shadow, colour cast, blur, noise; JPEG on save). The homography is
returned so that boxes can be mapped exactly.
"""

from __future__ import annotations

import math
import random
from typing import Optional

import cv2
import numpy as np

DESKS = [(150, 105, 70), (120, 85, 55), (60, 60, 62), (25, 25, 28), (200, 200, 195), (95, 75, 60),
         (170, 160, 150), (40, 45, 60)]


def _smooth_noise(h: int, w: int, cells: int, nrng: np.random.Generator) -> np.ndarray:
    """Low-frequency noise in [-1, 1] (bicubic upsampling of a coarse random grid)."""
    gh = max(2, int(cells * h / max(h, w)) + 2)
    gw = max(2, int(cells * w / max(h, w)) + 2)
    g = nrng.uniform(-1, 1, (gh, gw)).astype(np.float32)
    return cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC)


def make_paper(W: int, H: int, rng: random.Random, nrng: np.random.Generator, unit: float) -> np.ndarray:
    """Paper image (H, W, 3) float32 in 0..1. `unit` = px per text em (scales the ruling)."""
    r = rng.uniform(0.93, 1.0)
    warm = rng.uniform(-0.02, 0.06)  # > 0: cream paper, < 0: slightly bluish white
    base = np.clip(np.array([r, r - abs(warm) * 0.4, r - warm], np.float32), 0, 1)
    img = np.ones((H, W, 3), np.float32) * base
    grain = _smooth_noise(H, W, 40, nrng) * 0.012 + nrng.normal(0, 0.006, (H, W)).astype(np.float32)
    img += grain[:, :, None]
    kind = rng.choices(["grid", "lines", "dots", "plain"], weights=[46, 18, 8, 28])[0]
    color = np.array(rng.choice([(0.55, 0.65, 0.85), (0.6, 0.6, 0.62), (0.5, 0.7, 0.8), (0.65, 0.7, 0.85),
                                 (0.55, 0.55, 0.55)]), np.float32)
    strength = rng.uniform(0.15, 0.45)
    pitch = unit * rng.uniform(0.9, 1.6)
    ruling = np.zeros((H, W), np.float32)
    lw = 1 if rng.random() < 0.75 else 2
    ox, oy = rng.uniform(0, pitch), rng.uniform(0, pitch)
    if kind in ("grid", "lines"):
        y = oy
        while y < H:
            cv2.line(ruling, (0, int(y)), (W - 1, int(y)), 1.0, lw, cv2.LINE_AA)
            y += pitch
    if kind == "grid":
        x = ox
        while x < W:
            cv2.line(ruling, (int(x), 0), (int(x), H - 1), 1.0, lw, cv2.LINE_AA)
            x += pitch
    if kind == "dots":
        r = max(1, int(round(unit * 0.08)))
        for yy in np.arange(oy, H, pitch):
            for xx in np.arange(ox, W, pitch):
                cv2.circle(ruling, (int(xx), int(yy)), r, 1.0, -1, cv2.LINE_AA)
    img = img * (1 - strength * ruling[:, :, None] * (1 - color))
    if kind == "lines" and rng.random() < 0.5:  # red margin line
        x = int(W * rng.uniform(0.06, 0.12))
        m = np.zeros((H, W), np.float32)
        cv2.line(m, (x, 0), (x, H - 1), 1.0, max(1, lw + 1), cv2.LINE_AA)
        img = img * (1 - 0.45 * m[:, :, None] * (1 - np.array([0.95, 0.45, 0.45], np.float32)))
    if rng.random() < 0.12:  # faint show-through of the other side of the sheet
        ghost = np.zeros((H, W), np.float32)
        for _ in range(rng.randint(3, 10)):
            x0, y0 = rng.uniform(0, W), rng.uniform(0, H)
            w, h = unit * rng.uniform(3, 10), unit * rng.uniform(2, 5)
            cv2.rectangle(ghost, (int(x0), int(y0)), (int(x0 + w), int(y0 + h)), 1.0, max(1, lw + 1),
                          cv2.LINE_AA)
            if rng.random() < 0.5:
                cv2.line(ghost, (int(x0 + w), int(y0 + h / 2)), (int(x0 + w + unit * 3), int(y0 + h / 2)),
                         1.0,
                         max(1, lw + 1), cv2.LINE_AA)
        ghost = cv2.GaussianBlur(ghost, (0, 0), unit * 0.12)
        img = img * (1 - rng.uniform(0.04, 0.1) * ghost[:, :, None])
    return np.clip(img, 0, 1)


def add_holes(img: np.ndarray, rng: random.Random, unit: float, edge: str) -> None:
    """Binder punch holes along one edge (dark: the desk shows through). The caller keeps a
    margin of at least 3.6 units on that edge, so holes never touch the drawing."""
    H, W = img.shape[:2]
    n = rng.randint(2, 4)
    r = int(unit * rng.uniform(0.5, 0.8))
    color = rng.choice([(0.05, 0.05, 0.05), (0.12, 0.1, 0.08), (0.92, 0.92, 0.92)])
    for i in range(n):
        t = (i + 0.5) / n
        if edge == "top":
            c = (int(W * t), int(unit * rng.uniform(1.0, 1.6)))
        elif edge == "left":
            c = (int(unit * rng.uniform(1.0, 1.6)), int(H * t))
        else:
            c = (W - int(unit * rng.uniform(1.0, 1.6)), int(H * t))
        cv2.circle(img, c, r, color, -1, cv2.LINE_AA)


def composite(paper: np.ndarray, drawing: np.ndarray, pencil: bool, rng: random.Random) -> np.ndarray:
    ink = drawing.astype(np.float32) / 255.0
    if pencil:
        ink = 1 - (1 - ink) * rng.uniform(0.55, 0.8)
    else:
        ink = 1 - (1 - ink) * rng.uniform(0.85, 1.0)
    return paper * ink


def photo_warp(img: np.ndarray, rng: random.Random, nrng: np.random.Generator, content: np.ndarray,
               scan: bool) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Warp the paper into the photo frame. `content` (N, 2) paper px points that must stay inside.
    Returns (frame image float32, 3x3 homography paper px -> frame px) or None if it failed."""
    H, W = img.shape[:2]
    for attempt in range(6):
        damp = 1.0 - attempt * 0.18
        if scan:
            fw, fh = W, H
            theta = math.radians(rng.uniform(-1.5, 1.5) * damp)
            scale = rng.uniform(0.97, 1.0)
            jitter = 0.0
        else:
            grow = rng.uniform(0.94, 1.12)
            fw, fh = int(W * grow), int(H * grow * rng.uniform(0.97, 1.03))
            theta = math.radians(max(-4.0, min(4.0, rng.gauss(0.0, 1.4))) * damp)
            scale = min(fw / W, fh / H) * rng.uniform(0.9, 1.06)
            jitter = rng.uniform(0.0, 0.035) * damp
        cx, cy = fw / 2 + rng.uniform(-0.03, 0.03) * fw * damp, fh / 2 + rng.uniform(-0.03, 0.03) * fh * damp
        src = np.array([[0, 0], [W, 0], [W, H], [0, H]], np.float32)
        dst = []
        c, s = math.cos(theta), math.sin(theta)
        for x, y in src:
            dx, dy = (x - W / 2) * scale, (y - H / 2) * scale
            px = cx + dx * c - dy * s + rng.uniform(-1, 1) * jitter * fw
            py = cy + dx * s + dy * c + rng.uniform(-1, 1) * jitter * fh
            dst.append([px, py])
        M = cv2.getPerspectiveTransform(src, np.array(dst, np.float32))
        pts = cv2.perspectiveTransform(content.reshape(-1, 1, 2).astype(np.float32), M).reshape(-1, 2)
        m = 0.012
        if (pts[:, 0].min() < m * fw or pts[:, 1].min() < m * fh or pts[:, 0].max() > (1 - m) * fw
                or pts[:, 1].max() > (1 - m) * fh):
            continue
        out = cv2.warpPerspective(img, M, (fw, fh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=(0, 0, 0))
        mask = cv2.warpPerspective(np.ones((H, W), np.float32), M, (fw, fh), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        desk = _desk(fh, fw, rng, nrng, scan)
        if not scan:  # soft shadow of the sheet on the desk
            sh = cv2.GaussianBlur(mask, (0, 0), max(2.0, 0.006 * max(fw, fh)))
            desk = desk * (1 - 0.35 * sh[:, :, None])
        out = out * mask[:, :, None] + desk * (1 - mask[:, :, None])
        return out, M
    return None


def _desk(h: int, w: int, rng: random.Random, nrng: np.random.Generator, scan: bool) -> np.ndarray:
    if scan:
        v = rng.uniform(0.75, 1.0)
        return np.ones((h, w, 3), np.float32) * v
    col = np.array(rng.choice(DESKS), np.float32) / 255.0
    desk = np.ones((h, w, 3), np.float32) * col
    streak = _smooth_noise(h, w, 6, nrng)
    if rng.random() < 0.5:  # wood-like streaks
        streak = cv2.resize(_smooth_noise(4, 60, 60, nrng), (w, h), interpolation=cv2.INTER_CUBIC)
    desk *= (1 + 0.12 * streak)[:, :, None]
    return np.clip(desk, 0, 1)


def photo_effects(img: np.ndarray, rng: random.Random, nrng: np.random.Generator, scan: bool,
                  unit: float) -> np.ndarray:
    H, W = img.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    if not scan:
        # uneven light: linear gradient + vignette + optional soft shadow
        a = rng.uniform(0, 2 * math.pi)
        g = ((xx / W - 0.5) * math.cos(a) + (yy / H - 0.5) * math.sin(a))
        light = 1 + rng.uniform(0.0, 0.35) * g
        r2 = ((xx / W - 0.5) ** 2 + (yy / H - 0.5) ** 2) * 2
        light *= 1 - rng.uniform(0.0, 0.22) * r2
        if rng.random() < 0.35:
            sh = np.zeros((H, W), np.float32)
            cx, cy = rng.uniform(0, W), rng.uniform(0, H)
            ax_, ay_ = rng.uniform(0.2, 0.6) * W, rng.uniform(0.2, 0.6) * H
            cv2.ellipse(sh, (int(cx), int(cy)), (int(ax_), int(ay_)), rng.uniform(0, 180), 0, 360, 1.0, -1)
            sh = cv2.GaussianBlur(sh, (0, 0), 0.08 * max(W, H))
            light *= 1 - rng.uniform(0.1, 0.35) * sh
        light *= rng.uniform(0.88, 1.08)
        img = img * light[:, :, None]
        temp = rng.uniform(-0.6, 1.0)  # colour temperature of the light: warm (> 0) or cool
        cast = np.array([1 + 0.06 * temp, 1 + 0.01 * temp, 1 - 0.08 * temp], np.float32)
        img = img * cast
    else:
        gamma = rng.uniform(0.8, 1.15)
        img = np.clip(img, 0, 1) ** gamma
        if rng.random() < 0.4:  # high-contrast copier look
            img = np.clip((img - 0.5) * rng.uniform(1.1, 1.5) + 0.5 + rng.uniform(0.0, 0.08), 0, 1)
    sigma = rng.uniform(0.0, 1.0) * max(0.5, unit / 22)
    if sigma > 0.25:
        img = cv2.GaussianBlur(img, (0, 0), sigma)
    img = img + nrng.normal(0, rng.uniform(0.004, 0.02), img.shape).astype(np.float32)
    return np.clip(img, 0, 1)


def to_uint8(img: np.ndarray) -> np.ndarray:
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)
