"""Helpers shared by several source converters."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from ..common import extracted_image_path

_FENCE = re.compile(r"^\s*```[\w-]*\s*\n?|\n?\s*```\s*$")


def strip_fences(text: str) -> str:
    return _FENCE.sub("", text.strip()).strip()


def hash_split(key: str, val_fraction: float = 0.05) -> str:
    """Deterministic train/val split for sources that only ship a train set."""
    h = int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "val" if h < val_fraction else "train"


def save_image_bytes(source: str, name: str, data: bytes) -> Path:
    """Write embedded image bytes once (skips files already extracted)."""
    path = extracted_image_path(source, name)
    if not path.exists() or path.stat().st_size != len(data):
        path.write_bytes(data)
    return path


def image_ext(data: bytes) -> str:
    if data[:8].startswith(b"\x89PNG"):
        return ".png"
    if data[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    return ".png"
