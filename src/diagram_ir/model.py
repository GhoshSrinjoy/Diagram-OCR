"""Diagram IR data model, closed vocabularies and validation.

One in-memory object (`Diagram`) represents every supported diagram type. The model
emits it in the compact text form (dsl.py); JSON / YAML / XML are produced from the
same object (convert.py). The normative description lives in SPEC.md - keep the
vocabularies below in sync with it.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import Optional

FORMAT_NAME = "diagram-ir"
VERSION = "1.0"

# Coordinates are integers normalized to the image: 0 = left/top edge, 1000 = right/bottom.
COORD_MAX = 1000
BBox = tuple[int, int, int, int]  # (x1, y1, x2, y2)
Point = tuple[int, int]

DIRECTIONS = ("TB", "BT", "LR", "RL")
EDGE_STYLES = ("solid", "dashed", "dotted", "thick")
MEMBER_KINDS = ("attr", "op", "lit")

# Kinds valid in every diagram type.
COMMON_NODE_KINDS = ("note", "text", "other")
COMMON_EDGE_KINDS = ("anchor",)
COMMON_GROUP_KINDS = ("group",)

# Per diagram type. The FIRST edge kind is the type's default (omitted in the text form).
VOCAB: dict[str, dict[str, tuple[str, ...]]] = {
    "flowchart": {
        "node": (
            "terminal", "process", "decision", "io", "subroutine", "database", "document",
            "manual_input", "manual_op", "preparation", "connector", "offpage", "delay",
            "display", "loop_limit",
        ),
        "edge": ("flow", "line", "bidir"),
        "group": ("subgraph", "lane"),
    },
    "activity": {
        "node": (
            "action", "initial", "final", "flow_final", "decision", "merge", "fork", "join",
            "object", "send_signal", "receive_signal", "time_event",
        ),
        "edge": ("flow",),
        "group": ("lane", "region"),
    },
    "sequence": {
        "node": (
            "participant", "actor", "boundary", "control", "entity", "database",
            "collections", "queue",
        ),
        "edge": ("sync", "async", "reply", "create", "destroy"),
        "group": ("alt", "opt", "loop", "par", "break", "critical", "neg", "ref", "operand", "box"),
    },
    "class": {
        "node": ("class", "abstract", "interface", "enum"),
        "edge": ("assoc", "assoc_dir", "inherit", "realize", "compose", "aggregate", "depend"),
        "group": ("package", "namespace"),
    },
    "state": {
        "node": ("state", "initial", "final", "choice", "fork", "join", "history", "deep_history"),
        "edge": ("transition",),
        "group": ("composite", "region"),
    },
    "er": {
        "node": ("entity", "weak_entity", "relationship", "attribute"),
        "edge": ("rel",),
        "group": (),
    },
    "circuit": {
        "node": (
            # digital logic
            "input", "output", "and", "or", "not", "nand", "nor", "xor", "xnor", "buf",
            "mux", "demux", "dff", "latch", "const", "clock", "junction",
            # electrical / analog
            "resistor", "capacitor", "inductor", "diode", "led", "transistor", "opamp",
            "voltage_source", "current_source", "battery", "ground", "switch", "lamp", "ic",
            "port",
        ),
        "edge": ("wire", "conn"),
        "group": ("block",),
    },
    "graph": {
        "node": (
            "node", "box", "rounded", "ellipse", "circle", "diamond", "parallelogram",
            "hexagon", "cylinder", "point", "image",
            # architecture / deployment shapes
            "box3d", "cloud", "folder", "component", "actor", "document", "queue", "storage",
            "file",
        ),
        "edge": ("arrow", "line", "bidir"),
        "group": ("cluster",),
    },
}
DIAGRAM_TYPES = tuple(VOCAB)

# Edge kinds drawn without an arrowhead. Everything else is directed source -> target.
UNDIRECTED_EDGE_KINDS = frozenset({"line", "anchor", "assoc", "rel", "conn"})
# Node kinds that may carry compartment members (attributes / operations / literals).
MEMBER_NODE_KINDS = frozenset(
    {"class", "abstract", "interface", "enum", "entity", "weak_entity", "object", "state"}
)

ID_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
EDGE_ID_RE = re.compile(r"e[1-9][0-9]*\Z")  # reserved: positional edge ids
KIND_RE = re.compile(r"[a-z][a-z0-9_]*\Z")
ATTR_KEY_RE = re.compile(r"[a-z][a-z0-9_]*\Z")


def node_kinds(dtype: str) -> tuple[str, ...]:
    return VOCAB.get(dtype, VOCAB["graph"])["node"] + COMMON_NODE_KINDS


def edge_kinds(dtype: str) -> tuple[str, ...]:
    return VOCAB.get(dtype, VOCAB["graph"])["edge"] + COMMON_EDGE_KINDS


def group_kinds(dtype: str) -> tuple[str, ...]:
    return VOCAB.get(dtype, VOCAB["graph"])["group"] + COMMON_GROUP_KINDS


def default_edge_kind(dtype: str) -> str:
    return VOCAB.get(dtype, VOCAB["graph"])["edge"][0]


def edge_id(index: int) -> str:
    """Edges have no stored id: the i-th edge (0-based) is always `e{i+1}`."""
    return f"e{index + 1}"


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #
@dataclass
class Member:
    """One compartment line of a class / entity / object node."""

    kind: str  # attr | op | lit
    text: str
    bbox: Optional[BBox] = None


@dataclass
class Node:
    id: str
    kind: str
    text: str = ""
    bbox: Optional[BBox] = None
    stereotype: Optional[str] = None
    members: list[Member] = field(default_factory=list)
    attrs: dict[str, str] = field(default_factory=dict)


@dataclass
class Edge:
    source: str  # node id or group id
    target: str
    kind: str
    label: str = ""
    source_label: Optional[str] = None  # multiplicity / role / cardinality at the source end
    target_label: Optional[str] = None
    source_port: Optional[str] = None  # pin name / index on the source node
    target_port: Optional[str] = None
    style: Optional[str] = None  # None means solid
    path: Optional[list[Point]] = None  # polyline from source to target
    attrs: dict[str, str] = field(default_factory=dict)

    @property
    def directed(self) -> bool:
        return self.kind not in UNDIRECTED_EDGE_KINDS


@dataclass
class Group:
    """A container: swimlane, subgraph, package, composite state, sequence fragment, ..."""

    id: str
    kind: str
    text: str = ""
    members: list[str] = field(default_factory=list)  # node ids and/or edge ids (e1, e2, ...)
    parent: Optional[str] = None  # enclosing group id (nesting is expressed only here)
    bbox: Optional[BBox] = None
    attrs: dict[str, str] = field(default_factory=dict)


@dataclass
class Diagram:
    type: str
    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    groups: list[Group] = field(default_factory=list)
    direction: Optional[str] = None
    title: Optional[str] = None
    attrs: dict[str, str] = field(default_factory=dict)

    def strip_layout(self) -> Diagram:
        """Copy without any coordinates (the 'structure only' output mode)."""
        d = copy.deepcopy(self)
        for n in d.nodes:
            n.bbox = None
            for m in n.members:
                m.bbox = None
        for e in d.edges:
            e.path = None
        for g in d.groups:
            g.bbox = None
        return d

    def has_layout(self) -> bool:
        return bool(self.nodes) and all(n.bbox is not None for n in self.nodes)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Issue:
    level: str  # "error" | "warning"
    where: str  # "diagram", "node n3", "edge e2", "group g1", ...
    message: str

    def __str__(self) -> str:
        return f"{self.level}: {self.where}: {self.message}"


def _check_bbox(bbox: Optional[BBox], where: str, out: list[Issue]) -> None:
    if bbox is None:
        return
    if len(bbox) != 4 or not all(isinstance(v, int) and not isinstance(v, bool) for v in bbox):
        out.append(Issue("error", where, f"bbox must be 4 integers, got {bbox!r}"))
        return
    x1, y1, x2, y2 = bbox
    if not all(0 <= v <= COORD_MAX for v in bbox):
        out.append(Issue("error", where, f"bbox values must be in 0..{COORD_MAX}, got {list(bbox)}"))
    if x1 > x2 or y1 > y2:
        out.append(Issue("error", where, f"bbox must satisfy x1<=x2 and y1<=y2, got {list(bbox)}"))


def _check_attrs(attrs: dict[str, str], where: str, out: list[Issue]) -> None:
    for k, v in attrs.items():
        if not ATTR_KEY_RE.match(k):
            out.append(Issue("error", where, f"bad attribute key {k!r}"))
        if not isinstance(v, str):
            out.append(Issue("error", where, f"attribute {k!r} must be a string"))


def validate(d: Diagram, strict: bool = False) -> list[Issue]:
    """Structural checks are errors; vocabulary mismatches are warnings unless `strict`."""
    out: list[Issue] = []
    vocab_level = "error" if strict else "warning"

    if d.type not in DIAGRAM_TYPES:
        out.append(Issue("error", "diagram", f"unknown diagram type {d.type!r}"))
    if d.direction is not None and d.direction not in DIRECTIONS:
        out.append(Issue("error", "diagram", f"direction must be one of {DIRECTIONS}"))
    _check_attrs(d.attrs, "diagram", out)

    ids: dict[str, str] = {}  # id -> "node" | "group"
    for kind_name, items in (("node", d.nodes), ("group", d.groups)):
        for item in items:
            where = f"{kind_name} {item.id}"
            if not isinstance(item.id, str) or not ID_RE.match(item.id):
                out.append(Issue("error", where, "id must match [A-Za-z_][A-Za-z0-9_]*"))
            elif EDGE_ID_RE.match(item.id):
                out.append(Issue("error", where, "ids of the form e<N> are reserved for edges"))
            if item.id in ids:
                out.append(Issue("error", where, "duplicate id"))
            ids[item.id] = kind_name

    nkinds, ekinds, gkinds = node_kinds(d.type), edge_kinds(d.type), group_kinds(d.type)

    for n in d.nodes:
        where = f"node {n.id}"
        if not KIND_RE.match(n.kind or ""):
            out.append(Issue("error", where, f"bad kind {n.kind!r}"))
        elif n.kind not in nkinds:
            out.append(Issue(vocab_level, where, f"kind {n.kind!r} is not in the {d.type} vocabulary"))
        _check_bbox(n.bbox, where, out)
        _check_attrs(n.attrs, where, out)
        if n.members and n.kind not in MEMBER_NODE_KINDS:
            out.append(Issue("warning", where, f"members on a {n.kind!r} node are unusual"))
        for i, m in enumerate(n.members):
            mwhere = f"{where} member {i + 1}"
            if m.kind not in MEMBER_KINDS:
                out.append(Issue("error", mwhere, f"member kind must be one of {MEMBER_KINDS}"))
            _check_bbox(m.bbox, mwhere, out)

    for i, e in enumerate(d.edges):
        where = f"edge {edge_id(i)}"
        for end, ref in (("source", e.source), ("target", e.target)):
            if ref not in ids:
                out.append(Issue("error", where, f"{end} {ref!r} does not exist"))
        if not KIND_RE.match(e.kind or ""):
            out.append(Issue("error", where, f"bad kind {e.kind!r}"))
        elif e.kind not in ekinds:
            out.append(Issue(vocab_level, where, f"kind {e.kind!r} is not in the {d.type} vocabulary"))
        if e.style is not None and e.style not in EDGE_STYLES:
            out.append(Issue("error", where, f"style must be one of {EDGE_STYLES}"))
        if e.path is not None:
            if len(e.path) < 2:
                out.append(Issue("error", where, "path needs at least 2 points"))
            for p in e.path:
                if len(p) != 2 or not all(isinstance(v, int) and 0 <= v <= COORD_MAX for v in p):
                    out.append(Issue("error", where, f"bad path point {p!r}"))
                    break
        _check_attrs(e.attrs, where, out)

    n_edges = len(d.edges)
    parents = {g.id: g.parent for g in d.groups}
    for g in d.groups:
        where = f"group {g.id}"
        if not KIND_RE.match(g.kind or ""):
            out.append(Issue("error", where, f"bad kind {g.kind!r}"))
        elif g.kind not in gkinds:
            out.append(Issue(vocab_level, where, f"kind {g.kind!r} is not in the {d.type} vocabulary"))
        _check_bbox(g.bbox, where, out)
        _check_attrs(g.attrs, where, out)
        if g.parent is not None:
            if ids.get(g.parent) != "group":
                out.append(Issue("error", where, f"parent {g.parent!r} is not a group"))
            else:
                seen, cur = {g.id}, g.parent
                while cur is not None:
                    if cur in seen:
                        out.append(Issue("error", where, "parent chain forms a cycle"))
                        break
                    seen.add(cur)
                    cur = parents.get(cur)
        for ref in g.members:
            if EDGE_ID_RE.match(ref):
                if int(ref[1:]) > n_edges:
                    out.append(Issue("error", where, f"member {ref!r} refers to a missing edge"))
            elif ids.get(ref) == "group":
                out.append(Issue("error", where, f"member {ref!r} is a group; nest with parent= instead"))
            elif ref not in ids:
                out.append(Issue("error", where, f"member {ref!r} does not exist"))
        if len(set(g.members)) != len(g.members):
            out.append(Issue("warning", where, "duplicate members"))

    return out


def errors(issues: list[Issue]) -> list[Issue]:
    return [i for i in issues if i.level == "error"]
