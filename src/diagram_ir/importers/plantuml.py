"""PlantUML -> Diagram.

`plantuml_to_diagram` is the entry point: it takes the first `@startuml ... @enduml` block,
detects the diagram kind and routes it to the matching parser:

  * class / object        -> type `class`     (plantuml_uml.plantuml_class_to_diagram)
  * sequence              -> type `sequence`  (plantuml_uml.plantuml_sequence_to_diagram)
  * activity, new + legacy -> type `activity` (plantuml_uml.plantuml_activity_to_diagram)
  * state                 -> type `state`     (plantuml_uml.plantuml_state_to_diagram)
  * use case              -> type `graph`     (plantuml_uml.plantuml_usecase_to_diagram)
  * component / deployment -> type `graph`    (`plantuml_graph_to_diagram` below, falling back
                                               to the use case parser, which is a superset)
  * anything else (gantt, mindmap, timing, salt, ...) raises UnsupportedPlantUML.

`plantuml_graph_to_diagram` covers element-style diagrams (deployment / component / use case /
"rectangle" diagrams): element declarations with any of the PlantUML element keywords,
containers (`keyword "Title" as G { ... }`), and links with any head/body/color/direction
syntax. These become the generic `graph` type.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from ..model import Diagram, Edge, Group, Node

# PlantUML element keyword -> graph node kind
ELEMENTS = {
    "node": "box3d", "folder": "folder", "usecase": "ellipse", "cloud": "cloud",
    "rectangle": "box", "database": "cylinder", "component": "component", "artifact": "file",
    "frame": "box", "card": "box", "file": "file", "hexagon": "hexagon", "queue": "queue",
    "stack": "other", "storage": "storage", "agent": "box", "actor": "actor", "person": "actor",
    "boundary": "other", "control": "other", "entity": "other", "circle": "circle",
    "label": "text", "collections": "other", "package": "folder", "interface": "circle",
}
DIRECTIONS = {"left to right direction": "LR", "top to bottom direction": "TB"}

_KW = "|".join(sorted(ELEMENTS, key=len, reverse=True))
_STR = r'"(?:[^"\\]|\\.)*"'
_DECL = re.compile(
    rf"^(?P<kw>{_KW})\s+(?:(?P<label>{_STR})(?:\s+as\s+(?P<alias>[\w.]+))?"
    rf"|(?P<id>[\w.]+)(?:\s+as\s+(?P<label2>{_STR}))?)"
    r"(?P<tail>.*)$",
    re.IGNORECASE,
)
_ENDPOINT = rf'(?:{_STR}|\[[^\]]+\]|\(\)\s*\w+|:[^:]+:|[\w.]+)'
_LINK = re.compile(
    rf"^(?P<a>{_ENDPOINT})\s*"
    r"(?P<lh><\||<<|<|\*|o(?=[-.])|\+|#|\})?"
    r"(?P<body>[-.]+(?:\[[^\]]*\])?(?:(?:up|down|left|right|u|d|l|r)(?=[-.]))?[-.]*)"
    r"(?P<rh>\|>|>>|>|\*|(?<=[-.])o|\+|#|\{)?"
    rf"\s*(?P<b>{_ENDPOINT})\s*(?::\s*(?P<label>.*))?$"
)
_SKIP = re.compile(r"^(skinparam|hide|show|!|scale|title|caption|legend|header|footer|allow"
                   r"mixing|set |left to right|top to bottom|@start|@end|'|$)", re.IGNORECASE)


class PlantUMLError(ValueError):
    pass


class UnsupportedPlantUML(PlantUMLError):
    """The diagram kind (gantt, mindmap, timing, salt, ...) has no Diagram IR mapping."""


def plantuml_to_diagram(code: str) -> Diagram:
    """Parse the first @startuml block of `code`, whatever UML diagram kind it is.

    Raises UnsupportedPlantUML for non-UML kinds and PlantUMLError for input it cannot map
    faithfully. The result always passes `validate(d, strict=True)` without issues."""
    from .plantuml_uml import parse_plantuml  # deferred: plantuml_uml imports this module

    return parse_plantuml(code)


def clean_label(raw: Optional[str]) -> str:
    if raw is None:
        return ""
    t = raw.strip()
    if len(t) >= 2 and t[0] == t[-1] == '"':
        t = t[1:-1]
    t = t.replace("\\n", "\n").replace('\\"', '"')
    t = re.sub(r"<[^>]+>", "", t)  # creole / html markup
    return t.strip()


@dataclass
class _State:
    nodes: dict[str, Node] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)
    groups: dict[str, Group] = field(default_factory=dict)
    stack: list[str] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)

    def endpoint(self, raw: str) -> str:
        raw = raw.strip()
        if raw.startswith("[") and raw.endswith("]"):  # [Component] shorthand
            ident, kind, text = raw[1:-1], "component", raw[1:-1]
        elif raw.startswith(":") and raw.endswith(":"):  # :Actor: shorthand
            ident, kind, text = raw[1:-1], "actor", raw[1:-1]
        elif raw.startswith("()"):
            ident = text = raw[2:].strip()
            kind = "circle"
        elif raw.startswith('"'):
            ident = text = clean_label(raw)
            kind = "box"
        else:
            ident, kind, text = raw, "box", raw
        if ident in self.groups:
            return ident
        if ident not in self.nodes:
            self.nodes[ident] = Node(id=ident, kind=kind, text=text)
            self.order.append(ident)
            if self.stack:
                self.groups[self.stack[-1]].members.append(ident)
        return ident


def _safe(ident: str) -> str:
    out = re.sub(r"\W", "_", ident)
    return out if re.match(r"[A-Za-z_]", out) else f"v_{out}"


def plantuml_graph_to_diagram(code: str) -> Diagram:
    state = _State()
    direction = "TB"
    for raw_line in code.splitlines():
        line = raw_line.strip()
        low = line.lower()
        if low in DIRECTIONS:
            direction = DIRECTIONS[low]
            continue
        if _SKIP.match(line):
            continue
        if line == "}":
            if state.stack:
                state.stack.pop()
            continue
        m = _DECL.match(line)
        if m and not _LINK.match(line):
            kw = m.group("kw").lower()
            if m.group("label") is not None:
                text = clean_label(m.group("label"))
                ident = m.group("alias") or text
            else:
                ident = m.group("id")
                text = clean_label(m.group("label2")) if m.group("label2") else ident
            opens = m.group("tail").rstrip().endswith("{")
            if opens:
                parent = state.stack[-1] if state.stack else None
                state.groups[ident] = Group(id=ident, kind="cluster", text=text, parent=parent)
                state.stack.append(ident)
            elif ident not in state.nodes:
                state.nodes[ident] = Node(id=ident, kind=ELEMENTS[kw], text=text)
                state.order.append(ident)
                if state.stack:
                    state.groups[state.stack[-1]].members.append(ident)
            continue
        m = _LINK.match(line)
        if m:
            a, b = state.endpoint(m.group("a")), state.endpoint(m.group("b"))
            lh, rh, body = m.group("lh"), m.group("rh"), m.group("body")
            if lh and rh:
                kind = "bidir"
            elif lh:  # head on the left: arrow points to `a`
                a, b, kind = b, a, "arrow"
            elif rh:
                kind = "arrow"
            else:
                kind = "line"
            style = "dashed" if "." in re.sub(r"\[[^\]]*\]", "", body) else None
            state.edges.append(Edge(source=a, target=b, kind=kind,
                                    label=clean_label(m.group("label")), style=style))
            continue
        raise PlantUMLError(f"cannot parse line: {line[:60]!r}")

    ids = {old: _safe(old) for old in list(state.nodes) + list(state.groups)}
    nodes = []
    for old in state.order:
        n = state.nodes[old]
        n.id = ids[old]
        nodes.append(n)
    for e in state.edges:
        e.source, e.target = ids[e.source], ids[e.target]
    groups = []
    for g in state.groups.values():
        g.id = ids[g.id]
        g.parent = ids[g.parent] if g.parent else None
        g.members = [ids[m] for m in g.members]
        groups.append(g)
    return Diagram(type="graph", direction=direction, nodes=nodes, edges=state.edges, groups=groups)
