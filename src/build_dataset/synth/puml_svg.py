"""What a PlantUML render actually shows, read from its SVG (same jar and flags as the PNG render):
the words of every drawn text, and entities drawn inside a container they are not a member of.

PlantUML marks Graphviz-laid-out elements as `<g class="cluster|entity" data-qualified-name="...">`
siblings (members are qualified with their container's name, `Pkg.Class`), so the box of each one
can be measured from the shapes inside its group."""

from __future__ import annotations

import html
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from ..render_plantuml import LIMIT, PLANTUML_JAR, first_block

_TEXT = re.compile(r"<text\b[^>]*>(.*?)</text>", re.S)
_WORD = re.compile(r"[A-Za-z0-9]+")
_GROUP = re.compile(r'<g class="(\w+)"(?: data-qualified-name="([^"]*)")?[^>]*>')
_ELEMENT_COMMENT = re.compile(r"<!--(?:class|entity) [^>]*-->\s*$")
_SHAPE = re.compile(r"<(rect|ellipse|polygon|polyline|path|line)\b([^>]*)>")
_NUM_ATTR = re.compile(r'\b(x|y|width|height|cx|cy|rx|ry|x1|y1|x2|y2)="(-?[\d.]+)"')
_NUM = re.compile(r"-?\d*\.?\d+(?:e-?\d+)?")
Box = tuple[float, float, float, float]


def words(text: str) -> set[str]:
    return {w.lower() for w in _WORD.findall(text)}


def render_svgs(items: list[tuple[str, str]], threads: int) -> dict[str, str]:
    """(key, code) pairs -> SVG text per key (missing when PlantUML wrote none). One JVM."""
    out: dict[str, str] = {}
    if not items:
        return out
    with tempfile.TemporaryDirectory(prefix="puml_svg_") as tmp:
        src, svg = Path(tmp) / "src", Path(tmp) / "svg"
        src.mkdir()
        for key, code in items:
            (src / f"{key}.puml").write_text(first_block(code), encoding="utf-8")
        cmd = ["java", "-Djava.awt.headless=true", f"-DPLANTUML_LIMIT_SIZE={LIMIT}", "-Xmx4g", "-jar",
               str(PLANTUML_JAR), "-tsvg", "-charset", "UTF-8", "-nbthread", str(threads), "-o", str(svg),
               str(src / "*.puml")]
        subprocess.run(cmd, capture_output=True, text=True, timeout=1800, encoding="utf-8", errors="replace")
        for key, _code in items:
            f = svg / f"{key}.svg"
            if f.exists():
                out[key] = f.read_text(encoding="utf-8", errors="replace")
    return out


def drawn_words(svg: str) -> set[str]:
    texts = _TEXT.findall(svg)
    return set().union(*(words(html.unescape(re.sub(r"<[^>]+>", "", t))) for t in texts))


def _path_points(d: str) -> list[tuple[float, float]]:
    """End and control points of an absolute SVG path (arc radii / flags are not points)."""
    pts: list[tuple[float, float]] = []
    cur = (0.0, 0.0)
    for cmd, args in re.findall(r"([MLCQSTAHVZmlcqstahvz])([^MLCQSTAHVZmlcqstahvz]*)", d):
        v = [float(t) for t in _NUM.findall(args)]
        up = cmd.upper()
        if up in "MLTCQS":
            pts += [(v[k], v[k + 1]) for k in range(0, len(v) - 1, 2)]
        elif up == "A":
            pts += [(v[k + 5], v[k + 6]) for k in range(0, len(v) - 6, 7)]
        elif up == "H" and v:
            pts.append((v[-1], cur[1]))
        elif up == "V" and v:
            pts.append((cur[0], v[-1]))
        if pts:
            cur = pts[-1]
    return pts


def _box(chunk: str, filled_only: bool = False) -> Optional[Box]:
    """Box of the shapes in `chunk`. `filled_only` skips lines and unfilled paths: an element's
    group also holds e.g. a note's anchor line, which reaches far outside the note itself."""
    xs: list[float] = []
    ys: list[float] = []
    for name, attrs in _SHAPE.findall(chunk):
        if filled_only and (name in ("line", "polyline") or 'fill="none"' in attrs):
            continue
        a = {k: float(v) for k, v in _NUM_ATTR.findall(attrs)}
        if name == "rect" and {"x", "y", "width", "height"} <= a.keys():
            xs += [a["x"], a["x"] + a["width"]]
            ys += [a["y"], a["y"] + a["height"]]
        elif name == "ellipse" and {"cx", "cy", "rx", "ry"} <= a.keys():
            xs += [a["cx"] - a["rx"], a["cx"] + a["rx"]]
            ys += [a["cy"] - a["ry"], a["cy"] + a["ry"]]
        elif name in ("polygon", "polyline"):
            m = re.search(r'\bpoints="([^"]+)"', attrs)
            v = [float(t) for t in _NUM.findall(m.group(1))] if m else []
            xs += v[0::2]
            ys += v[1::2]
        elif name == "path":
            m = re.search(r'\bd="([^"]+)"', attrs)
            for x, y in _path_points(m.group(1)) if m else []:
                xs.append(x)
                ys.append(y)
    return (min(xs), min(ys), max(xs), max(ys)) if xs else None


def misplaced(svg: str) -> list[tuple[str, str]]:
    """(entity, container) pairs where an entity is drawn mostly inside a container it is not in.
    Notes are not checked: their callout tail reaches into the element they annotate (note groups
    are the entity groups without a `<!--class X-->` / `<!--entity X-->` comment in front)."""
    marks = list(_GROUP.finditer(svg))
    clusters: list[tuple[str, Box]] = []
    entities: list[tuple[str, Box]] = []
    for k, m in enumerate(marks):
        if m.group(1) not in ("cluster", "entity") or m.group(2) is None:
            continue
        if m.group(1) == "entity" and not _ELEMENT_COMMENT.search(svg, max(0, m.start() - 300), m.start()):
            continue
        chunk = svg[m.end():marks[k + 1].start() if k + 1 < len(marks) else len(svg)]
        box = _box(chunk, filled_only=m.group(1) == "entity")
        if box is not None:
            (clusters if m.group(1) == "cluster" else entities).append((m.group(2), box))
    out = []
    for name, b in entities:
        area = max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))
        for cname, c in clusters:
            if name == cname or name.startswith(cname + "."):
                continue
            ix = max(0.0, min(b[2], c[2]) - max(b[0], c[0]))
            iy = max(0.0, min(b[3], c[3]) - max(b[1], c[1]))
            if ix * iy / area > 0.5:
                out.append((name, cname))
    return out
