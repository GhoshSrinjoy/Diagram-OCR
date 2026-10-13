"""Graphviz DOT -> Diagram, using Graphviz itself as the parser (`dot -Tjson`).

Letting Graphviz parse means every valid DOT file works (attributes, escapes, HTML labels,
clusters) and, when asked, the node boxes come from the real layout. Boxes are normalized to the
PNG that `render_png` produces from the same source (same pad), so image and target agree.

Requires the Graphviz `dot` executable on PATH.
"""

from __future__ import annotations

import html
import json
import re
import subprocess
from pathlib import Path
from typing import Optional

from ..model import COORD_MAX, Diagram, Edge, Group, Node, default_edge_kind, edge_kinds

PAD_POINTS = 4.0  # Graphviz default pad (0.0555 in); passed explicitly so layout and PNG agree
RANKDIR = {"TB": "TB", "LR": "LR", "BT": "BT", "RL": "RL"}

# DOT shape -> node kind, per target diagram type. Unlisted shapes fall back to the default.
SHAPES: dict[str, dict[str, str]] = {
    "flowchart": {
        "box": "process", "rect": "process", "rectangle": "process", "square": "process",
        "record": "process", "mrecord": "process", "ellipse": "terminal", "oval": "terminal",
        "diamond": "decision", "parallelogram": "io", "hexagon": "preparation",
        "cylinder": "database", "note": "document", "folder": "document", "box3d": "subroutine",
        "component": "subroutine", "circle": "connector", "doublecircle": "connector",
        "point": "connector", "invtrapezium": "manual_op", "trapezium": "manual_op",
        "house": "offpage", "invhouse": "offpage", "plaintext": "text", "plain": "text",
        "none": "text",
    },
    "graph": {
        "box": "box", "rect": "box", "rectangle": "box", "square": "box", "record": "box",
        "mrecord": "rounded", "ellipse": "ellipse", "oval": "ellipse", "circle": "circle",
        "doublecircle": "circle", "diamond": "diamond", "parallelogram": "parallelogram",
        "hexagon": "hexagon", "cylinder": "cylinder", "point": "point", "box3d": "box3d",
        "component": "component", "folder": "folder", "note": "document", "tab": "folder",
        "plaintext": "text", "plain": "text", "none": "text",
    },
}
DEFAULT_KIND = {"flowchart": "process", "graph": "ellipse"}  # Graphviz's default shape is ellipse

_HTML_TAG = re.compile(r"<[^>]+>")


class DotError(RuntimeError):
    pass


def run_dot(code: str, fmt: str, timeout: int = 30, extra: tuple[str, ...] = ()) -> bytes:
    proc = subprocess.run(
        ["dot", f"-T{fmt}", f"-Gpad={PAD_POINTS / 72:.4f}", *extra],
        input=code.encode("utf-8"), capture_output=True, timeout=timeout,
    )
    if proc.returncode != 0:
        raise DotError(proc.stderr.decode("utf-8", "replace").strip()[:300])
    return proc.stdout


def render_png(code: str, dest: Path, dpi: int = 96, timeout: int = 30) -> None:
    dest.write_bytes(run_dot(code, "png", timeout, (f"-Gdpi={dpi}",)))


def clean_label(raw: Optional[str], name: str) -> str:
    if raw is None or raw == "\\N":
        return name
    text = raw
    if text.startswith("<") and text.endswith(">"):  # HTML-like label
        text = re.sub(r"(?i)<br\s*/?>", "\n", text[1:-1])
        text = html.unescape(_HTML_TAG.sub("", text))
    text = text.replace("\\N", name).replace("\\n", "\n").replace("\\l", "\n").replace("\\r", "\n")
    return text.replace('\\"', '"').strip()


def dot_to_diagram(
    code: str,
    dtype: str = "graph",
    shapes: Optional[dict[str, str]] = None,
    layout: bool = False,
    timeout: int = 30,
) -> Diagram:
    """Parse DOT into a Diagram of type `dtype` (flowchart or graph). `shapes` overrides /
    extends the shape -> kind map (e.g. custom flowchart shape names)."""
    j = json.loads(run_dot(code, "json", timeout))
    shape_map = {**SHAPES.get(dtype, SHAPES["graph"]), **(shapes or {})}
    default_kind = DEFAULT_KIND.get(dtype, "node")

    bb = [float(v) for v in j.get("bb", "0,0,0,0").split(",")]
    width = (bb[2] - bb[0]) + 2 * PAD_POINTS
    height = (bb[3] - bb[1]) + 2 * PAD_POINTS

    def norm_box(cx: float, cy: float, w_pt: float, h_pt: float) -> tuple[int, int, int, int]:
        def nx(x: float) -> int:
            return max(0, min(COORD_MAX, round((x - bb[0] + PAD_POINTS) / width * COORD_MAX)))

        def ny(y: float) -> int:  # Graphviz y grows upwards; image y grows downwards
            return max(0, min(COORD_MAX, round((bb[3] - y + PAD_POINTS) / height * COORD_MAX)))

        return (nx(cx - w_pt / 2), ny(cy + h_pt / 2), nx(cx + w_pt / 2), ny(cy - h_pt / 2))

    objects = j.get("objects", [])
    used: set[str] = set()
    by_gvid = {o["_gvid"]: o for o in objects if "nodes" not in o and "subgraphs" not in o
               and "invis" not in str(o.get("style", ""))}  # invisible nodes are not drawn
    nodes: list[Node] = []
    for gvid in sorted(by_gvid):
        o = by_gvid[gvid]
        shape = str(o.get("shape", "ellipse")).lower()
        if shape == "point":  # Graphviz never draws a label on a point
            label = ""
        else:
            label = clean_label(o.get("label"), o["name"])
        bbox = None
        if layout and "pos" in o:
            cx, cy = (float(v) for v in o["pos"].split(",")[:2])
            bbox = norm_box(cx, cy, float(o.get("width", 0.75)) * 72, float(o.get("height", 0.5)) * 72)
        nodes.append(Node(id=_safe_id(o["name"], gvid, used), kind=shape_map.get(shape, default_kind),
                          text=label, bbox=bbox))
    node_id = {gvid: n.id for gvid, n in zip(sorted(by_gvid), nodes, strict=True)}

    directed = bool(j.get("directed", True))
    kinds = edge_kinds(dtype)
    undirected_kind = "line" if "line" in kinds else default_edge_kind(dtype)
    edges: list[Edge] = []
    for e in j.get("edges", []):
        if e.get("tail") not in node_id or e.get("head") not in node_id:
            continue
        if "invis" in str(e.get("style", "")):
            continue
        dir_attr = str(e.get("dir", "forward" if directed else "none")).lower()
        if e.get("arrowhead") == "none" and dir_attr == "forward":
            dir_attr = "none"
        if dir_attr == "none":
            kind = undirected_kind
        elif dir_attr == "both" and "bidir" in kinds:
            kind = "bidir"
        else:
            kind = default_edge_kind(dtype)
        tail, head = node_id[e["tail"]], node_id[e["head"]]
        if dir_attr == "back":
            tail, head = head, tail
        style = str(e.get("style", "")).lower()
        edges.append(Edge(
            source=tail, target=head, kind=kind,
            label=clean_label(e.get("label") or e.get("xlabel") or "", ""),
            style="dashed" if "dashed" in style else "dotted" if "dotted" in style
            else "thick" if "bold" in style else None,
        ))

    groups: list[Group] = []
    clusters = {o["_gvid"]: o for o in objects if "nodes" in o or "subgraphs" in o}
    parent_of = {c: p for p, o in clusters.items() for c in o.get("subgraphs", [])}
    group_kind = "subgraph" if dtype == "flowchart" else "cluster"
    group_ids = {gvid: _safe_id(o["name"], gvid, used) for gvid, o in clusters.items()
                 if str(o.get("name", "")).startswith("cluster")}  # plain subgraphs are not drawn
    for gvid, o in clusters.items():
        if gvid not in group_ids:
            continue
        nested = {n for c in o.get("subgraphs", []) for n in clusters.get(c, {}).get("nodes", [])}
        bbox = None
        if layout and "bb" in o:
            x1, y1, x2, y2 = (float(v) for v in o["bb"].split(","))
            bbox = norm_box((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)
        parent = parent_of.get(gvid)
        groups.append(Group(
            id=group_ids[gvid], kind=group_kind,
            text=clean_label(o.get("label"), "") if o.get("label") != "\\N" else "",
            members=[node_id[n] for n in o.get("nodes", []) if n not in nested and n in node_id],
            parent=group_ids.get(parent),
            bbox=bbox,
        ))

    return Diagram(type=dtype, nodes=nodes, edges=edges, groups=groups,
                   direction=RANKDIR.get(str(j.get("rankdir", "TB")).upper(), "TB"))


def _safe_id(name: str, gvid: int, used: set[str]) -> str:
    """A valid, unique IR id for a DOT name (any string). canonicalize() renames them anyway."""
    cleaned = re.sub(r"\W", "_", name)
    if not cleaned or not re.match(r"[A-Za-z_]", cleaned) or re.fullmatch(r"e[1-9][0-9]*", cleaned):
        cleaned = f"v_{cleaned}"
    if cleaned in used:
        cleaned = f"{cleaned}_{gvid}"
    used.add(cleaned)
    return cleaned
