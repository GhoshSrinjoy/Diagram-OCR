"""Mermaid source -> Diagram.

`mermaid_to_diagram` routes on the header keyword:

    graph / flowchart        -> flowchart  (mermaid_flowchart_to_diagram)
    sequenceDiagram          -> sequence   (mermaid_sequence_to_diagram)
    classDiagram             -> class      (mermaid_class_to_diagram)
    stateDiagram[-v2]        -> state      (mermaid_state_to_diagram)
    erDiagram                -> er         (mermaid_er_to_diagram)
    mindmap                  -> graph      (mermaid_mindmap_to_diagram)
    architecture-beta        -> graph      (mermaid_architecture_to_diagram)
    C4Context / C4Container / C4Component / C4Deployment -> graph  (mermaid_c4_to_diagram)

Every other Mermaid diagram type (pie, gantt, gitGraph, journey, block-beta, C4Dynamic, ...)
raises UnsupportedMermaid; malformed input raises MermaidError.

Flowcharts: node shapes (classic bracket syntax and the v11 `id@{ shape: ..., label: ... }` form),
every link type (--> --- -.-> ==> <--> --o --x ~~~, long variants), labels (`-->|x|`, `-- x -->`),
chains (A --> B --> C), `&` fan-out, subgraphs (nested, titled, links to a subgraph), `:::class`
suffixes, node text wrapped over several lines, `%%` comments, front matter; styling is ignored.

The other parsers keep what Mermaid draws: participants / classes / states in first-use order,
messages in time order, fragment guards as rendered (`[guard]`), class members as displayed
(`+total() : float`, `List~T~` -> `List<T>`, attributes before operations), ER cardinalities
normalized to UML multiplicities (`||` -> 1, `o{` -> 0..*). Ids are generated (n1.., g1..);
canonicalize() renames them anyway.
"""

from __future__ import annotations

import html
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Optional

from ..model import Diagram, Edge, Group, Member, Node

DIRECTIONS = {"TB": "TB", "TD": "TB", "BT": "BT", "LR": "LR", "RL": "RL"}

# (opener, closer, kind) - longest openers first; same opener may have several closers.
BRACKET_SHAPES = [
    ("(((", ")))", "connector"),
    ("([", "])", "terminal"),
    ("[[", "]]", "subroutine"),
    ("[(", ")]", "database"),
    ("((", "))", "connector"),
    ("{{", "}}", "preparation"),
    ("[/", "/]", "io"),
    ("[/", "\\]", "manual_op"),
    ("[\\", "\\]", "io"),
    ("[\\", "/]", "manual_op"),
    ("[", "]", "process"),
    ("(", ")", "process"),
    ("{", "}", "decision"),
    (">", "]", "other"),
]
_OPENERS = sorted({o for o, _, _ in BRACKET_SHAPES}, key=len, reverse=True)

# Mermaid v11 named shapes (aliases included) -> flowchart kind.
NAMED_SHAPES = {
    **dict.fromkeys(["rect", "proc", "process", "rectangle", "rounded", "event"], "process"),
    **dict.fromkeys(["stadium", "pill", "terminal", "terminal-point"], "terminal"),
    **dict.fromkeys(["diam", "diamond", "decision", "question"], "decision"),
    **dict.fromkeys(["hex", "hexagon", "prepare"], "preparation"),
    **dict.fromkeys(["lean-r", "lean-right", "in-out", "lean-l", "lean-left", "out-in"], "io"),
    **dict.fromkeys(["trap-b", "trapezoid-bottom", "priority", "trap-t", "trapezoid-top",
                     "manual", "inv-trapezoid"], "manual_op"),
    **dict.fromkeys(["cyl", "cylinder", "database", "db", "h-cyl", "das",
                     "horizontal-cylinder", "lin-cyl", "disk", "lined-cylinder"], "database"),
    **dict.fromkeys(["circle", "circ", "sm-circ", "small-circle", "start", "dbl-circ",
                     "double-circle", "fr-circ", "framed-circle", "stop", "f-circ",
                     "filled-circle", "junction", "cross-circ", "summary"], "connector"),
    **dict.fromkeys(["doc", "document", "docs", "documents", "st-doc", "stacked-document",
                     "lin-doc", "lined-document", "tag-doc", "tagged-document"], "document"),
    **dict.fromkeys(["subproc", "subprocess", "subroutine", "fr-rect", "framed-rectangle"],
                    "subroutine"),
    **dict.fromkeys(["delay", "half-rounded-rectangle"], "delay"),
    **dict.fromkeys(["curv-trap", "curved-trapezoid", "display"], "display"),
    **dict.fromkeys(["sl-rect", "sliced-rectangle", "manual-input"], "manual_input"),
    **dict.fromkeys(["notch-pent", "loop-limit", "notched-pentagon"], "loop_limit"),
    **dict.fromkeys(["text"], "text"),
    **dict.fromkeys(["comment", "brace", "brace-l", "brace-r", "braces"], "note"),
}

_ID = r"\w+(?:-(?![-.>=])\w+)*"  # letters/digits/_ ; '-' only when not part of an arrow
_ID_RE = re.compile(rf"\s*({_ID})")
_PLAIN_ARROW = re.compile(
    r"\s*(?P<left><|o|x)?(?P<body>-{2,}>|={2,}>|-\.+->|-{2,}[ox](?![\w])|={2,}[ox](?![\w])"
    r"|-{3,}|={3,}|-\.+-|~{3,})"
    r"(?:\s*\|(?P<pipe>[^|]*)\|)?"
)
_CLOSE = r"(?P<close>-{2,}>|-{3,}|={2,}>|={3,}|\.-+>|\.-+|-{2,}[ox]|={2,}[ox])"
_TEXT_ARROW = re.compile(  # A -- text --> B, also A--text-->B
    rf"\s*(?P<left><)?(?P<open>--|==|-\.)\s*(?P<text>[^|\s\-=.>][^|]*?)\s*{_CLOSE}"
)
_PIPE_TEXT_ARROW = re.compile(  # A --|text|--> B: not valid Mermaid, but common and unambiguous
    rf"\s*(?P<left><)?(?P<open>--|==|-\.)\s*\|(?P<text>[^|]*)\|\s*{_CLOSE}"
)
_SKIP_PREFIXES = ("style ", "classdef ", "class ", "linkstyle ", "click ", "direction ",
                  "acctitle", "accdescr", "title ")


class MermaidError(ValueError):
    pass


class UnsupportedMermaid(MermaidError):
    pass


def clean_text(raw: str) -> str:
    t = raw.strip()
    if len(t) >= 2 and t[0] == t[-1] == '"':
        t = t[1:-1]
    if len(t) >= 2 and t[0] == t[-1] == "`":  # markdown string
        t = t[1:-1].replace("**", "").replace("*", "")
    t = re.sub(r"#(\d+);", lambda m: chr(int(m.group(1))), t)  # #35; numeric entity codes
    t = re.sub(r"#([a-zA-Z]+);", r"&\1;", t)  # #quot; -> &quot;
    t = re.sub(r"(?i)<br\s*/?>", "\n", t)
    t = re.sub(r"<[^>]+>", "", t)
    t = html.unescape(t).replace("\\n", "\n")
    return t.strip()


# =========================================================================== #
# Flowchart
# =========================================================================== #
@dataclass
class _Ref:
    id: str
    kind: Optional[str] = None
    text: Optional[str] = None


@dataclass
class _State:
    nodes: dict[str, Node] = field(default_factory=dict)
    defined: set[str] = field(default_factory=set)  # ids given an explicit shape/text
    order: list[str] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    groups: dict[str, Group] = field(default_factory=dict)
    member_of: dict[str, str] = field(default_factory=dict)
    stack: list[str] = field(default_factory=list)
    n_anon_groups: int = 0

    def touch(self, ref: _Ref) -> str:
        node = self.nodes.get(ref.id)
        if node is None:
            node = self.nodes[ref.id] = Node(id=ref.id, kind="process", text=ref.id)
            self.order.append(ref.id)
        if ref.kind is not None and ref.id not in self.defined:
            node.kind, node.text = ref.kind, ref.text if ref.text is not None else ref.id
            self.defined.add(ref.id)
        if self.stack and ref.id not in self.member_of:
            self.member_of[ref.id] = self.stack[-1]
        return ref.id


def _scan_line(line: str, buf: list[str], depth: int, quote: bool,
               out: list[str]) -> tuple[int, bool]:
    """Split one line on top-level ';' into `out`, continuing the open statement in `buf`."""
    prev = ""
    for ch in line:
        if ch == '"' and prev != "\\":
            quote = not quote
        elif not quote and ch in "[({":
            depth += 1
        elif not quote and ch in "])}":
            depth = max(0, depth - 1)
        if ch == ";" and not quote and depth == 0:
            out.append("".join(buf).strip())
            buf.clear()
        else:
            buf.append(ch)
        prev = ch
    return depth, quote


def _split_lines(lines: list[str]) -> list[str]:
    out: list[str] = []
    buf: list[str] = []
    depth, quote = 0, False
    open_at: Optional[tuple[int, int]] = None  # (line index, len(out)) where an open statement began
    for i, line in enumerate(lines):
        n_before = len(out)
        if buf:  # node text wrapped onto this line: keep the break
            buf.append("\n")
        depth, quote = _scan_line(line, buf, depth, quote, out)
        if depth or quote:
            if open_at is None:
                open_at = (i, n_before)
            continue
        open_at = None
        if buf:
            out.append("".join(buf).strip())
            buf.clear()
    if open_at is not None:  # never closed: an unbalanced bracket, not wrapped text
        i, n = open_at
        local: list[str] = []
        line_buf: list[str] = []
        _scan_line(lines[i], line_buf, 0, False, local)
        if line_buf:
            local.append("".join(line_buf).strip())
        return out[:n] + local + _split_lines(lines[i + 1:])
    return out


def _split_statements(code: str) -> list[str]:
    lines = [ln.strip() for ln in code.splitlines()]
    out = _split_lines([ln for ln in lines if ln and not ln.startswith("%%")])
    return [s for s in out if s]


def _closing_quote(s: str, start: int, escapes: bool) -> int:
    """Index of the quote closing the string opened at s[start] ('"'), or -1."""
    i = start + 1
    while True:
        j = s.find('"', i)
        if j < 0 or not (escapes and s[j - 1] == "\\"):
            return j
        i = j + 1


def _parse_shape(s: str, pos: int) -> tuple[Optional[str], Optional[str], int]:
    """At s[pos], parse an optional shape. Returns (kind, text, new_pos)."""
    if s.startswith("@{", pos):
        end = s.find("}", pos)
        if end < 0:
            raise MermaidError("unterminated @{ }")
        body = s[pos + 2:end]
        shape = re.search(r"shape\s*:\s*([\w-]+)", body)
        label = re.search(r'label\s*:\s*("(?:[^"\\]|\\.)*"|[^,}]+)', body)
        kind = NAMED_SHAPES.get(shape.group(1).lower(), "other") if shape else "process"
        return kind, clean_text(label.group(1)) if label else None, end + 1
    for opener in _OPENERS:
        if not s.startswith(opener, pos):
            continue
        start = pos + len(opener)
        if start < len(s) and s[start] == '"':
            # Escaped quotes (\") first, then the plain reading; whitespace may precede the closer.
            q_ends = []
            for escapes in (True, False):
                q_end = _closing_quote(s, start, escapes)
                if q_end >= 0 and q_end not in (q for q, _ in q_ends):
                    q_ends.append((q_end, escapes))
            if not q_ends:
                raise MermaidError("unterminated string")
            for q_end, escapes in q_ends:
                after = q_end + 1
                while after < len(s) and s[after] in " \t":
                    after += 1
                for o, closer, kind in BRACKET_SHAPES:
                    if o == opener and s.startswith(closer, after):
                        text = s[start:q_end + 1]
                        if escapes:
                            text = text.replace('\\"', '"')
                        return kind, clean_text(text), after + len(closer)
            raise MermaidError(f"bad closer after {opener!r}")
        best = None  # earliest closer for this opener
        for o, closer, kind in BRACKET_SHAPES:
            if o != opener:
                continue
            idx = s.find(closer, start)
            if idx >= 0 and (best is None or idx < best[0]):
                best = (idx, closer, kind)
        if best is None:
            raise MermaidError(f"unterminated {opener!r}")
        idx, closer, kind = best
        return kind, clean_text(s[start:idx]), idx + len(closer)
    return None, None, pos


def _parse_group(s: str, pos: int) -> tuple[list[_Ref], int]:
    refs = []
    while True:
        m = _ID_RE.match(s, pos)
        if not m or not m.group(1):
            raise MermaidError(f"expected node id at {s[pos:pos + 20]!r}")
        ref = _Ref(id=m.group(1))
        pos = m.end()
        kind, text, pos = _parse_shape(s, pos)
        if kind is not None:
            ref.kind, ref.text = kind, text
        cls = re.match(r":::[\w-]+", s[pos:])
        if cls:
            pos += cls.end()
        refs.append(ref)
        amp = re.match(r"\s*&", s[pos:])
        if not amp:
            return refs, pos
        pos += amp.end()


def _parse_arrow(s: str, pos: int) -> tuple[Optional[dict], int]:
    m = _PLAIN_ARROW.match(s, pos)
    text = None
    if m:
        left, body, text = m.group("left"), m.group("body"), m.group("pipe")
    else:
        m = _TEXT_ARROW.match(s, pos) or _PIPE_TEXT_ARROW.match(s, pos)
        if not m:
            return None, pos
        left, body, text = m.group("left"), m.group("open") + m.group("close"), m.group("text")
    if body.startswith("~"):
        return {"invisible": True}, m.end()
    head = body.endswith((">", "o", "x"))
    if left and head:
        kind = "bidir"
    elif head:
        kind = "flow"
    else:
        kind = "line"
    style = "dotted" if "." in body else "thick" if "=" in body else None
    return {"kind": kind, "style": style, "label": clean_text(text) if text else ""}, m.end()


def _subgraph(state: _State, rest: str) -> None:
    rest = rest.strip()
    m = re.match(r"^([\w-]+)\s*\[(.*)\]$", rest)
    if m:
        gid, title = m.group(1), clean_text(m.group(2))
    elif rest.startswith('"'):
        state.n_anon_groups += 1
        gid, title = f"subgraph_{state.n_anon_groups}", clean_text(rest)
    elif re.fullmatch(r"[\w-]+", rest):
        gid = title = rest
    else:
        state.n_anon_groups += 1
        gid, title = f"subgraph_{state.n_anon_groups}", clean_text(rest)
    parent = state.stack[-1] if state.stack else None
    state.groups[gid] = Group(id=gid, kind="subgraph", text=title, parent=parent)
    state.stack.append(gid)


def mermaid_flowchart_to_diagram(code: str) -> Diagram:
    code = re.sub(r"(?s)^\s*---\n.*?\n---\s*\n", "", code)  # YAML front matter
    code = re.sub(r"(?s)%%\{.*?\}%%", "", code)  # init directives
    stmts = _split_statements(code)
    if not stmts:
        raise MermaidError("empty diagram")
    head = stmts[0].split()
    if head[0].lower() not in ("graph", "flowchart", "flowchart-elk"):
        raise UnsupportedMermaid(f"not a flowchart: {head[0]!r}")
    direction = DIRECTIONS.get(head[1].upper(), "TB") if len(head) > 1 else "TB"
    rest_of_header = stmts[0].split(None, 2)[2] if len(head) > 2 else ""

    state = _State()
    for stmt in ([rest_of_header] if rest_of_header else []) + stmts[1:]:
        low = stmt.lower()
        if low.startswith(_SKIP_PREFIXES):
            continue
        if low == "end":
            if state.stack:
                state.stack.pop()
            continue
        if low.startswith("subgraph"):
            _subgraph(state, stmt[len("subgraph"):])
            continue
        refs, pos = _parse_group(stmt, 0)
        groups, arrows = [refs], []
        while stmt[pos:].strip():
            arrow, new_pos = _parse_arrow(stmt, pos)
            if arrow is None:
                raise MermaidError(f"cannot parse near {stmt[pos:pos + 25]!r}")
            refs, pos = _parse_group(stmt, new_pos)
            arrows.append(arrow)
            groups.append(refs)
        for g in groups:
            for ref in g:
                state.touch(ref)
        for src_refs, arrow, dst_refs in zip(groups, arrows, groups[1:], strict=False):
            if arrow.get("invisible"):
                continue
            for a in src_refs:
                for b in dst_refs:
                    state.edges.append(Edge(source=a.id, target=b.id, kind=arrow["kind"],
                                            label=arrow["label"], style=arrow["style"]))

    # Ids that name a subgraph and were never drawn as a node are links to the subgraph.
    group_refs = {gid for gid in state.groups if gid in state.nodes and gid not in state.defined}
    for gid in group_refs:
        del state.nodes[gid]
        state.order.remove(gid)
        state.member_of.pop(gid, None)
    for nid, gid in state.member_of.items():
        if gid in state.groups and nid in state.nodes:
            state.groups[gid].members.append(nid)

    return Diagram(
        type="flowchart",
        direction=direction,
        nodes=[state.nodes[i] for i in state.order],
        edges=state.edges,
        groups=list(state.groups.values()),
    )


# =========================================================================== #
# Shared helpers for the other diagram types
# =========================================================================== #
_FRONT_MATTER = re.compile(r"(?s)^\s*---[ \t]*\n(.*?)\n---[ \t]*(?:\n|$)")
_ACC_BLOCK = re.compile(r"(?ms)^\s*accDescr\s*\{.*?\}")
_ACC_LINE = re.compile(r"(?i)^acc(?:Title|Descr)\s*:")
_TITLE = re.compile(r"^title(?:\s*:\s*|\s+)(?P<t>.*)$")  # sequence: `title x` / `title: x`
_TITLE_WORD = re.compile(r"^title\s+(?!:)(?P<t>.*)$")  # elsewhere `title: x` could be a member line


def _prepare(code: str) -> tuple[list[str], Optional[str]]:
    """Source lines without front matter, %%{init}%% directives, %% comments, accessibility
    statements and blank lines (indentation kept), plus the front-matter title."""
    code = code.replace("\r\n", "\n").replace("\r", "\n")
    title = None
    m = _FRONT_MATTER.match(code)
    if m:
        t = re.search(r"(?m)^\s*title\s*:\s*(.+?)\s*$", m.group(1))
        title = clean_text(t.group(1)) if t else None
        code = code[m.end():]
    code = _ACC_BLOCK.sub("", re.sub(r"(?s)%%\{.*?\}%%", "", code))
    lines = []
    for ln in code.split("\n"):
        s = ln.strip()
        if s and not s.startswith("%%") and not _ACC_LINE.match(s):
            lines.append(ln.rstrip())
    return lines, title


def _header(lines: list[str], names: tuple[str, ...]) -> str:
    """Check the header keyword (case-insensitive); return the rest of the header line."""
    if not lines:
        raise MermaidError("empty diagram")
    parts = lines[0].split(None, 1)
    if parts[0].lower() not in names:
        raise UnsupportedMermaid(f"expected {names[0]!r}, got {parts[0]!r}")
    return parts[1].strip() if len(parts) > 1 else ""


def _direction(rest: str) -> Optional[str]:
    return DIRECTIONS.get(rest.strip().upper())


class _Ids:
    """Generated ids: n1, n2, ... for nodes and g1, g2, ... for groups."""

    def __init__(self) -> None:
        self.n_nodes = 0
        self.n_groups = 0

    def node(self) -> str:
        self.n_nodes += 1
        return f"n{self.n_nodes}"

    def group(self) -> str:
        self.n_groups += 1
        return f"g{self.n_groups}"


def _bracketed(guard: str) -> str:
    """Fragment / operand guard as Mermaid draws it: `[guard]` (empty stays empty)."""
    return f"[{guard}]" if guard else ""


# =========================================================================== #
# Sequence diagrams
# =========================================================================== #
# Arrow -> (edge kind, style). Solid lines are calls (sync), dashed lines replies; `-)` is the
# async (open) head - dashed `--)` counts as a reply; the cross head (`-x`) is drawn like the UML
# destruction X, so it maps to destroy. Lines without a head (`->`) and the bidirectional
# `<<->>` keep the solid/dashed reading (the vocabulary has no headless or two-way message).
SEQ_ARROWS = {
    "->>": ("sync", None), "->": ("sync", None), "<<->>": ("sync", None),
    "-->>": ("reply", None), "-->": ("reply", None), "<<-->>": ("reply", None),
    "-)": ("async", None), "--)": ("reply", None),
    "-x": ("destroy", None), "--x": ("destroy", "dashed"),
}
SEQ_PARTICIPANT_TYPES = ("participant", "actor", "boundary", "control", "entity", "database",
                         "collections", "queue")
_SEQ_ARROW_RE = "|".join(re.escape(a) for a in sorted(SEQ_ARROWS, key=len, reverse=True))
_SEQ_MSG = re.compile(
    rf"^(?P<a>[^:]+?)\s*(?P<arrow>{_SEQ_ARROW_RE})\s*[+-]?\s*(?P<b>[^:]+?)\s*(?::(?P<text>.*))?$"
)
_SEQ_DECL = re.compile(
    r"^(?P<create>create\s+)?(?P<kw>participant|actor)\s+(?P<name>[^@]+?)\s*(?:@\{(?P<cfg>.*)\})?"
    r"(?:\s+as\s+(?P<alias>.+))?$",
    re.IGNORECASE,
)
_SEQ_NOTE = re.compile(r"^note\s+(?:left\s+of|right\s+of|over)\s+(?P<who>[^:]+?)\s*:(?P<text>.*)$",
                       re.IGNORECASE)
# Block keyword -> group kind; alt / par / critical get one `operand` child per branch.
_SEQ_BLOCKS = {"loop": "loop", "opt": "opt", "alt": "alt", "par": "par", "par_over": "par",
               "critical": "critical", "break": "break", "rect": "group"}
_SEQ_BRANCHES = {"else": "alt", "and": "par", "option": "critical"}
_SEQ_IGNORED = re.compile(r"^(?:activate|deactivate|autonumber|links?|properties|details)(?:\s|:|$)",
                          re.IGNORECASE)
_CSS_COLORS = frozenset("""
aliceblue antiquewhite aqua aquamarine azure beige bisque black blanchedalmond blue blueviolet brown
burlywood cadetblue chartreuse chocolate coral cornflowerblue cornsilk crimson cyan darkblue darkcyan
darkgoldenrod darkgray darkgreen darkgrey darkkhaki darkmagenta darkolivegreen darkorange darkorchid
darkred darksalmon darkseagreen darkslateblue darkslategray darkslategrey darkturquoise darkviolet
deeppink deepskyblue dimgray dimgrey dodgerblue firebrick floralwhite forestgreen fuchsia gainsboro
ghostwhite gold goldenrod gray green greenyellow grey honeydew hotpink indianred indigo ivory khaki
lavender lavenderblush lawngreen lemonchiffon lightblue lightcoral lightcyan lightgoldenrodyellow
lightgray lightgreen lightgrey lightpink lightsalmon lightseagreen lightskyblue lightslategray
lightslategrey lightsteelblue lightyellow lime limegreen linen magenta maroon mediumaquamarine
mediumblue mediumorchid mediumpurple mediumseagreen mediumslateblue mediumspringgreen mediumturquoise
mediumvioletred midnightblue mintcream mistyrose moccasin navajowhite navy oldlace olive olivedrab
orange orangered orchid palegoldenrod palegreen paleturquoise palevioletred papayawhip peachpuff peru
pink plum powderblue purple rebeccapurple red rosybrown royalblue saddlebrown salmon sandybrown
seagreen seashell sienna silver skyblue slateblue slategray slategrey snow springgreen steelblue tan
teal thistle tomato turquoise violet wheat white whitesmoke yellow yellowgreen transparent
""".split())


def _box_title(rest: str) -> str:
    """`box [color] [title]`: drop a leading CSS color, as Mermaid does."""
    m = re.match(r"^((?:rgba?|hsla?)\s*\([^)]*\)|#[0-9a-fA-F]{3,8}(?![\w])|\w+)\s*(.*)$", rest)
    if m and (m.group(1).lower() in _CSS_COLORS or not m.group(1)[0].isalpha()
              or m.group(1).lower().startswith(("rgb", "hsl"))):
        return clean_text(m.group(2))
    return clean_text(rest)


def _seq_statements(lines: list[str]) -> list[str]:
    """`;` separates statements too (except inside entity codes such as #59;)."""
    out = []
    for line in lines:
        protected = re.sub(r"#(\w+);", "#\\1\x00", line.strip())
        out.extend(p.strip().replace("\x00", ";") for p in protected.split(";") if p.strip())
    return out


@dataclass
class _Frame:
    keyword: str  # box, loop, alt, ...
    group: Group
    sections: list[Group] = field(default_factory=list)  # operands of alt / par / critical

    @property
    def current(self) -> Group:
        return self.sections[-1] if self.sections else self.group


class _SequenceBuilder:
    def __init__(self) -> None:
        self.ids = _Ids()
        self.participants: dict[str, Node] = {}
        self.notes: list[Node] = []
        self.edges: list[Edge] = []
        self.groups: list[Group] = []
        self.stack: list[_Frame] = []
        self.creating: set[str] = set()
        self.destroying: set[str] = set()
        self.in_box: set[str] = set()
        self.title: Optional[str] = None

    # -- helpers ----------------------------------------------------------- #
    def _fragment(self) -> Optional[Group]:
        """Innermost open fragment section (boxes hold participants, not messages)."""
        for frame in reversed(self.stack):
            if frame.keyword != "box":
                return frame.current
        return None

    def participant(self, name: str, kind: Optional[str] = None, text: Optional[str] = None) -> str:
        name = name.strip()
        if not name:
            raise MermaidError("empty participant name")
        node = self.participants.get(name)
        if node is None:
            node = self.participants[name] = Node(id=self.ids.node(), kind="participant",
                                                  text=clean_text(name))
        if kind is not None:
            node.kind = kind
        if text is not None:
            node.text = text
        return node.id

    # -- statements -------------------------------------------------------- #
    def declare(self, m: re.Match) -> None:
        kind = m.group("kw").lower()
        alias = clean_text(m.group("alias")) if m.group("alias") else None
        cfg = m.group("cfg") or ""
        t = re.search(r"""["']?type["']?\s*:\s*["']?(\w+)""", cfg)
        if t and t.group(1).lower() in SEQ_PARTICIPANT_TYPES:
            kind = t.group(1).lower()
        a = re.search(r"""["']?alias["']?\s*:\s*["']([^"']*)["']""", cfg)
        if a and alias is None:
            alias = clean_text(a.group(1))
        name = m.group("name").strip()
        nid = self.participant(name, kind, alias if alias is not None else clean_text(name))
        if m.group("create"):
            self.creating.add(name)
        box = next((f for f in reversed(self.stack) if f.keyword == "box"), None)
        if box is not None and name not in self.in_box:
            box.group.members.append(nid)
            self.in_box.add(name)

    def open_block(self, keyword: str, text: str) -> None:
        if keyword == "box":
            if self.stack:
                raise MermaidError("box must be at the top level")
            g = Group(id=self.ids.group(), kind="box", text=_box_title(text))
            self.groups.append(g)
            self.stack.append(_Frame("box", g))
            return
        parent = self._fragment()
        kind = _SEQ_BLOCKS[keyword]
        guard = "" if kind == "group" else _bracketed(clean_text(text))
        g = Group(id=self.ids.group(), kind=kind, parent=parent.id if parent else None)
        self.groups.append(g)
        frame = _Frame(keyword, g)
        if kind in ("alt", "par", "critical"):
            op = Group(id=self.ids.group(), kind="operand", text=guard, parent=g.id)
            self.groups.append(op)
            frame.sections.append(op)
        else:
            g.text = guard
        self.stack.append(frame)

    def branch(self, keyword: str, text: str) -> None:
        frame = self.stack[-1] if self.stack else None
        if frame is None or frame.group.kind != _SEQ_BRANCHES[keyword]:
            raise MermaidError(f"{keyword!r} outside {_SEQ_BRANCHES[keyword]!r}")
        op = Group(id=self.ids.group(), kind="operand", text=_bracketed(clean_text(text)),
                   parent=frame.group.id)
        self.groups.append(op)
        frame.sections.append(op)

    def end(self) -> None:
        if not self.stack:
            raise MermaidError("'end' without an open block")
        frame = self.stack.pop()
        if len(frame.sections) == 1:  # alt / par / critical without branches: no operand group
            op = frame.sections[0]
            frame.group.text, frame.group.members = op.text, op.members
            for g in self.groups:
                if g.parent == op.id:
                    g.parent = frame.group.id
            self.groups.remove(op)

    def message(self, m: re.Match) -> None:
        a, b = m.group("a").strip(), m.group("b").strip()
        src, dst = self.participant(a), self.participant(b)
        kind, style = SEQ_ARROWS[m.group("arrow")]
        if b in self.creating:
            kind = "create"
            self.creating.discard(b)
        elif a in self.destroying or b in self.destroying:
            kind = "destroy"
            self.destroying.discard(a if a in self.destroying else b)
        self._add_edge(Edge(source=src, target=dst, kind=kind, style=style,
                            label=clean_text(m.group("text") or "")))

    def note(self, m: re.Match) -> None:
        who = [w.strip() for w in m.group("who").split(",") if w.strip()]
        if not who:
            raise MermaidError("note without a participant")
        targets = [self.participant(w) for w in who]
        node = Node(id=self.ids.node(), kind="note", text=clean_text(m.group("text")))
        self.notes.append(node)
        frag = self._fragment()
        if frag is not None:
            frag.members.append(node.id)
        for t in dict.fromkeys(targets):
            self._add_edge(Edge(source=node.id, target=t, kind="anchor"), member=False)

    def _add_edge(self, edge: Edge, member: bool = True) -> None:
        self.edges.append(edge)
        frag = self._fragment()
        if member and frag is not None:
            frag.members.append(f"e{len(self.edges)}")

    def statement(self, stmt: str) -> None:
        word = stmt.split(None, 1)
        kw, rest = word[0].lower(), word[1] if len(word) > 1 else ""
        m = _SEQ_DECL.match(stmt)
        if m:
            self.declare(m)
        elif kw == "destroy" and rest:
            self.destroying.add(rest.strip())
        elif kw == "title" or kw.startswith("title:"):
            t = _TITLE.match(stmt)
            self.title = clean_text(t.group("t")) if t else None
        elif _SEQ_IGNORED.match(stmt):
            pass
        elif kw == "box" or kw in _SEQ_BLOCKS:
            self.open_block(kw, rest)
        elif kw in _SEQ_BRANCHES:
            self.branch(kw, rest)
        elif kw == "end" and not rest:
            self.end()
        elif kw == "note" and (m := _SEQ_NOTE.match(stmt)) is not None:
            self.note(m)
        else:
            m = _SEQ_MSG.match(stmt)
            if not m:
                raise MermaidError(f"cannot parse: {stmt[:40]!r}")
            self.message(m)

    def finish(self, title: Optional[str]) -> Diagram:
        if self.stack:
            raise MermaidError(f"unclosed {self.stack[-1].keyword!r} block")
        if not self.participants:
            raise MermaidError("empty diagram")
        return Diagram(type="sequence", nodes=list(self.participants.values()) + self.notes,
                       edges=self.edges, groups=self.groups, title=self.title or title)


def mermaid_sequence_to_diagram(code: str) -> Diagram:
    lines, title = _prepare(code)
    _header(lines, ("sequencediagram",))
    b = _SequenceBuilder()
    for stmt in _seq_statements(lines[1:]):
        b.statement(stmt)
    return b.finish(title)


# =========================================================================== #
# Class diagrams
# =========================================================================== #
_CLS_NAME = r"(?:`[^`]+`|\w[\w$]*(?:[.-]\w[\w$]*)*)"
_CLS_GENERIC = r'(?:~[^\s"]*~)?'
_CLS_REL = re.compile(
    rf'^(?P<a>{_CLS_NAME})(?P<ga>{_CLS_GENERIC})\s*(?:"(?P<ca>[^"]*)")?\s*'
    r"(?P<l><\||\*|o|<|\(\))?(?P<line>--|\.\.)(?P<r>\|>|\*|o(?!\w)|>|\(\))?\s*"
    rf'(?:"(?P<cb>[^"]*)")?\s*(?P<b>{_CLS_NAME})(?P<gb>{_CLS_GENERIC})\s*(?::(?P<label>.*))?$'
)
_CLS_DECL = re.compile(
    rf'^class\s+(?P<name>{_CLS_NAME})(?P<g>{_CLS_GENERIC})\s*'
    r'(?:\[\s*(?P<label>"[^"]*"|[^\]]*)\s*\])?\s*(?P<annot><<[^>]*>>)?\s*(?::::[\w-]+)?\s*'
    r"(?P<brace>\{(?P<rest>.*))?$"
)
_CLS_MEMBER = re.compile(rf"^(?P<name>{_CLS_NAME})(?P<g>{_CLS_GENERIC})\s*:(?P<m>.*)$")
_CLS_ANNOT_LINE = re.compile(rf"^<<(?P<a>[^>]*)>>\s*(?P<name>{_CLS_NAME})\s*$")
_CLS_NOTE = re.compile(rf'^note\s+(?:for\s+(?P<name>{_CLS_NAME})\s+)?(?P<t>".*")\s*$', re.IGNORECASE)
_CLS_SKIP = re.compile(r"^(?:style|cssClass|classDef|click|callback|link)\s")
CLASS_ANNOTATION_KINDS = {"interface": "interface", "abstract": "abstract",
                          "enumeration": "enum", "enum": "enum"}
# Relation end marker -> role; the strongest marker decides the edge kind and direction.
_CLS_MARKERS = {"<|": "tri", "|>": "tri", "*": "comp", "o": "agg", "<": "arrow", ">": "arrow",
                "()": "lolli"}
_CLS_PRECEDENCE = ("tri", "comp", "agg", "arrow", "lolli")


def _generics(s: str) -> str:
    """Mermaid's ~T~ generics -> <T> (nesting allowed: List~List~int~~ -> List<List<int>>)."""
    if "~" not in s:
        return s
    out, depth = [], 0
    for i, ch in enumerate(s):
        if ch != "~":
            out.append(ch)
            continue
        nxt = s[i + 1] if i + 1 < len(s) else ""
        if nxt and (nxt.isalnum() or nxt == "_"):
            out.append("<")
            depth += 1
        elif depth:
            out.append(">")
            depth -= 1
        else:
            return s
    return "".join(out) if depth == 0 else s


def class_member_text(raw: str) -> tuple[bool, str]:
    """(is_operation, text as Mermaid displays it). Mermaid treats any member containing ')' as
    an operation and shows `vis name(params) : return`; `$` / `*` classifiers are not text."""
    s = raw.strip().rstrip(";").strip()
    if s.find(")") > 0:
        m = re.match(r"([#+~-])?(.+)\((.*)\)([\s$*])?(.*)$", s)
        if m:
            vis, name, params, ret = m.group(1) or "", m.group(2).strip(), m.group(3).strip(), m.group(5)
            ret = ret.strip()
            if (m.group(4) or "").strip() == "" and ret[-1:] in ("$", "*"):
                ret = ret[:-1].strip()
            text = f"{vis}{_generics(name)}({_generics(params)})"
            return True, text + (f" : {_generics(ret)}" if ret else "")
    vis = s[0] if s[:1] in ("+", "-", "#", "~") else ""
    body = s[len(vis):]
    if body[-1:] in ("$", "*"):
        body = body[:-1]
    return False, vis + _generics(body.strip())


def _cls_name(raw: str) -> str:
    return raw[1:-1].strip() if raw.startswith("`") else raw


@dataclass
class _ClassInfo:
    node: Node
    generic: str = ""
    label: Optional[str] = None
    annotation: Optional[str] = None
    members: list[tuple[bool, str]] = field(default_factory=list)


class _ClassBuilder:
    def __init__(self) -> None:
        self.ids = _Ids()
        self.classes: dict[str, _ClassInfo] = {}
        self.notes: list[Node] = []
        self.edges: list[Edge] = []
        self.groups: list[Group] = []
        self.namespaces: list[Group] = []  # open namespace stack
        self.direction: Optional[str] = None
        self.title: Optional[str] = None

    def touch(self, raw: str, generic: str = "") -> _ClassInfo:
        name = _cls_name(raw)
        info = self.classes.get(name)
        if info is None:
            info = self.classes[name] = _ClassInfo(Node(id=self.ids.node(), kind="class"))
            if self.namespaces:
                self.namespaces[-1].members.append(info.node.id)
        if generic and not info.generic:
            info.generic = generic[1:-1]
        return info

    def annotate(self, info: _ClassInfo, raw: str) -> None:
        a = raw.strip().strip("<>").strip()
        if a:
            info.annotation = a

    def relation(self, m: re.Match) -> None:
        a = self.touch(m.group("a"), m.group("ga")).node.id
        b = self.touch(m.group("b"), m.group("gb")).node.id
        ca = clean_text(m.group("ca")) if m.group("ca") else None
        cb = clean_text(m.group("cb")) if m.group("cb") else None
        dashed = m.group("line") == ".."
        left = _CLS_MARKERS.get(m.group("l") or "")
        right = _CLS_MARKERS.get(m.group("r") or "")
        style = None
        if (left is None and right is None) or left == right == "arrow":
            kind, marker_at, style = "assoc", None, "dashed" if dashed else None
        else:
            rank = {r: i for i, r in enumerate(_CLS_PRECEDENCE)}
            # the stronger marker wins; on a tie the right-hand (written-last) end
            use_left = right is None or (left is not None and rank[left] < rank[right])
            role, marker_at = (left, "a") if use_left else (right, "b")
            if role == "tri":
                kind = "realize" if dashed else "inherit"
            elif role in ("comp", "agg"):
                kind = "compose" if role == "comp" else "aggregate"
                style = "dashed" if dashed else None
            elif role == "arrow":
                kind = "depend" if dashed else "assoc_dir"
            else:  # lollipop: the end with the circle is the provided interface
                kind = "realize"
        # Source end: whole (diamond) for compose/aggregate; otherwise the end without the head.
        source_is_marker = kind in ("compose", "aggregate")
        reverse = marker_at is not None and ((marker_at == "a") != source_is_marker)
        src, dst, sl, tl = (b, a, cb, ca) if reverse else (a, b, ca, cb)
        self.edges.append(Edge(source=src, target=dst, kind=kind, style=style,
                               label=clean_text(m.group("label") or ""),
                               source_label=sl, target_label=tl))

    def note(self, m: re.Match) -> None:
        node = Node(id=self.ids.node(), kind="note", text=clean_text(m.group("t")))
        self.notes.append(node)
        if m.group("name"):
            self.edges.append(Edge(source=node.id, target=self.touch(m.group("name")).node.id,
                                   kind="anchor"))

    def body_line(self, info: _ClassInfo, line: str) -> None:
        if line.startswith("<<") and line.endswith(">>"):
            self.annotate(info, line)
        elif line:
            info.members.append(class_member_text(line))

    def run(self, lines: list[str]) -> None:
        body: Optional[_ClassInfo] = None  # class whose { ... } body is open
        pending_ns: Optional[Group] = None  # `namespace X` waiting for its `{`
        for line in lines:
            s = line.strip()
            if body is not None:
                if s.endswith("}"):
                    self.body_line(body, s[:-1].strip())
                    body = None
                else:
                    self.body_line(body, s)
                continue
            if pending_ns is not None:
                if s != "{":
                    raise MermaidError("expected '{' after namespace")
                self.namespaces.append(pending_ns)
                pending_ns = None
                continue
            s = s.rstrip(";").strip()
            low = s.lower()
            if low.startswith("direction "):
                self.direction = _direction(s[10:]) or self.direction
            elif _CLS_SKIP.match(s):
                continue
            elif _TITLE_WORD.match(s) and not _CLS_REL.match(s):
                self.title = clean_text(_TITLE_WORD.match(s).group("t"))
            elif low.startswith("namespace "):
                m = re.match(r"^namespace\s+(?P<n>[^{]+?)\s*(?P<b>\{)?\s*$", s)
                if not m:
                    raise MermaidError(f"bad namespace: {s[:40]!r}")
                g = Group(id=self.ids.group(), kind="namespace", text=clean_text(m.group("n")),
                          parent=self.namespaces[-1].id if self.namespaces else None)
                self.groups.append(g)
                if m.group("b"):
                    self.namespaces.append(g)
                else:
                    pending_ns = g
            elif s == "}":
                if not self.namespaces:
                    raise MermaidError("unbalanced '}'")
                self.namespaces.pop()
            elif low.startswith("class "):
                m = _CLS_DECL.match(s)
                if not m:
                    raise MermaidError(f"bad class statement: {s[:40]!r}")
                info = self.touch(m.group("name"), m.group("g"))
                if m.group("label"):
                    info.label = clean_text(m.group("label"))
                if m.group("annot"):
                    self.annotate(info, m.group("annot"))
                if m.group("brace") is not None:
                    rest = m.group("rest").strip()
                    if rest.endswith("}"):  # one-line body
                        for part in rest[:-1].split(";"):
                            self.body_line(info, part.strip())
                    else:
                        if rest:
                            self.body_line(info, rest)
                        body = info
            elif (m := _CLS_NOTE.match(s)) is not None:
                self.note(m)
            elif (m := _CLS_ANNOT_LINE.match(s)) is not None:
                self.annotate(self.touch(m.group("name")), m.group("a"))
            elif (m := _CLS_REL.match(s)) is not None:
                self.relation(m)
            elif (m := _CLS_MEMBER.match(s)) is not None:
                info = self.touch(m.group("name"), m.group("g"))
                self.body_line(info, m.group("m").strip())
            elif re.fullmatch(rf"{_CLS_NAME}{_CLS_GENERIC}", s):
                m = re.match(rf"({_CLS_NAME})({_CLS_GENERIC})", s)
                self.touch(m.group(1), m.group(2))
            else:
                raise MermaidError(f"cannot parse: {s[:40]!r}")
        if body is not None:
            raise MermaidError("unclosed class body")
        if self.namespaces or pending_ns is not None:
            raise MermaidError("unclosed namespace")

    def finish(self, title: Optional[str]) -> Diagram:
        if not self.classes:
            raise MermaidError("empty diagram")
        nodes = []
        for name, info in self.classes.items():
            node = info.node
            annot = (info.annotation or "").lower()
            node.kind = CLASS_ANNOTATION_KINDS.get(annot, "class")
            if info.annotation and annot not in CLASS_ANNOTATION_KINDS:
                node.stereotype = info.annotation
            if info.label is not None:
                node.text = info.label
            else:
                node.text = name + (f"<{_generics(info.generic)}>" if info.generic else "")
            attr_kind = "lit" if node.kind == "enum" else "attr"
            # Mermaid draws the attribute compartment above the operation compartment.
            node.members = [Member(kind=attr_kind, text=t) for is_op, t in info.members if not is_op and t]
            node.members += [Member(kind="op", text=t) for is_op, t in info.members if is_op and t]
            nodes.append(node)
        return Diagram(type="class", nodes=nodes + self.notes, edges=self.edges, groups=self.groups,
                       direction=self.direction, title=self.title or title)


def mermaid_class_to_diagram(code: str) -> Diagram:
    lines, title = _prepare(code)
    rest = _header(lines, ("classdiagram", "classdiagram-v2"))
    b = _ClassBuilder()
    b.run(([rest] if rest else []) + lines[1:])
    return b.finish(title)


# =========================================================================== #
# State diagrams
# =========================================================================== #
_ST_ID = r"[^\s:{}<>\[\]\"(),;-]+(?:-[^\s:{}<>\[\]\"(),;-]+)*"
# `<<fork>>` outside a `state` statement is lexed by Mermaid as a state of its own (drawn "<>").
_ST_STRAY_ANNOT = re.compile(r"<<\w+>>")
_ST_END = rf"(?:\[\*\]|<<\w+>>|{_ST_ID})(?::::[\w-]+)?"
_ST_TRANS = re.compile(rf"^(?P<a>{_ST_END})\s*-->\s*(?P<b>{_ST_END})\s*(?::(?P<label>.*))?$")
_ST_DECL = re.compile(
    rf'^state\s+(?:(?P<desc>"[^"]*")\s+as\s+)?(?P<id>{_ST_ID})(?::::[\w-]+)?\s*'
    r"(?P<annot><<\s*\w+\s*>>)?\s*(?P<brace>\{)?\s*(?::(?P<desc2>.*))?$"
)
_ST_DESC = re.compile(rf"^(?P<id>{_ST_ID})(?::::[\w-]+)?\s*:(?P<desc>.*)$")
_ST_NOTE = re.compile(rf"^note\s+(?:left|right)\s+of\s+(?P<id>{_ST_ID})\s*(?::(?P<text>.*))?$",
                      re.IGNORECASE)
_ST_SKIP = re.compile(r"^(?:classDef|class|style|click|scale|hide)\s")
STATE_ANNOTATION_KINDS = {"choice": "choice", "fork": "fork", "join": "join"}
_PSEUDO = "\x00"  # prefix of internal keys for [*] and notes (never a state name)


def _state_name_text(name: str) -> str:
    """A state drawn under its id. Mermaid sanitizes ids as HTML, so `<<fork>>` shows as `<>`."""
    return re.sub(r"<[A-Za-z/][^>]*>", "", name)


def _split_stray_annotations(s: str) -> list[str]:
    """`A --> B <<fork>>` -> [`A --> B`, `<<fork>>`]; `X <<join>> --> Y` -> [`X`, `<<join>> --> Y`]:
    a stray annotation is an id that ends the statement before it and starts a new one, unless
    a dangling `-->` joins it to its neighbour."""
    parts = [p.strip() for p in _ST_STRAY_ANNOT.split(s)]
    out: list[str] = []
    carry = parts[0]
    for annot, after in zip(_ST_STRAY_ANNOT.findall(s), parts[1:], strict=True):
        if carry.endswith("-->"):
            stmt = f"{carry} {annot}"
        else:
            if carry:
                out.append(carry)
            stmt = annot
        if after.startswith("-->"):
            carry = f"{stmt} {after}"
        else:
            out.append(stmt)
            carry = after
    if carry:
        out.append(carry)
    return out


@dataclass(eq=False)
class _StScope:
    group: Optional[str]  # group id holding the scope's states; None = top level
    initial: Optional[str] = None  # key of this scope's [*] start node
    final: Optional[str] = None


@dataclass
class _StComposite:
    group: Group
    regions: list[_StScope]


class _StateBuilder:
    def __init__(self) -> None:
        self.ids = _Ids()
        self.root = _StScope(group=None)
        self.nodes: dict[str, Node] = {}  # key (state name or internal key) -> node
        self.descs: dict[str, list[str]] = {}
        self.kinds: dict[str, str] = {}
        self.member_of: dict[str, _StScope] = {}
        self.composites: dict[str, Group] = {}
        self.composite_scope: dict[str, _StScope] = {}  # composite name -> scope it was declared in
        self.edges: list[tuple[str, str, str]] = []  # (source key, target key, label | note marker)
        self.stack: list[_StComposite] = []
        self.direction: Optional[str] = None
        self.title: Optional[str] = None
        self.n_internal = 0
        self.regions: list[Group] = []

    @property
    def scope(self) -> _StScope:
        return self.stack[-1].regions[-1] if self.stack else self.root

    def _place(self, key: str) -> None:
        if self.scope is not self.root:
            self.member_of[key] = self.scope  # the innermost composite a state appears in wins

    def state(self, raw: str, as_source: bool = True) -> str:
        name = re.sub(r":::[\w-]+$", "", raw.strip())
        if name == "[*]":  # one start and one end node per scope
            scope = self.scope
            attr = "initial" if as_source else "final"
            key = getattr(scope, attr)
            if key is None:
                self.n_internal += 1
                key = f"{_PSEUDO}{attr}{self.n_internal}"
                setattr(scope, attr, key)
                self.nodes[key] = Node(id=self.ids.node(), kind=attr)
                self._place(key)
            return key
        if name not in self.nodes:
            self.nodes[name] = Node(id=self.ids.node(), kind="state", text=_state_name_text(name))
        self._place(name)
        return name

    def describe(self, name: str, desc: Optional[str]) -> None:
        if desc is not None and clean_text(desc):
            self.descs.setdefault(name, []).append(clean_text(desc))

    def open_composite(self, name: str) -> None:
        g = self.composites.get(name)
        if g is None:
            g = self.composites[name] = Group(id=self.ids.group(), kind="composite")
            self.composite_scope[name] = self.scope
        self.stack.append(_StComposite(g, [_StScope(group=g.id)]))

    def divider(self) -> None:
        if not self.stack:
            raise MermaidError("'--' outside a composite state")
        comp = self.stack[-1]
        if len(comp.regions) == 1:  # first divider: the states so far form region 1
            comp.regions[0].group = self._region(comp).id
        comp.regions.append(_StScope(group=self._region(comp).id))

    def _region(self, comp: _StComposite) -> Group:
        g = Group(id=self.ids.group(), kind="region", parent=comp.group.id)
        self.regions.append(g)
        return g

    def note(self, target: str, text: str) -> None:
        self.n_internal += 1
        key = f"{_PSEUDO}note{self.n_internal}"
        self.nodes[key] = Node(id=self.ids.node(), kind="note", text=clean_text(text))
        self._place(key)
        self.edges.append((key, self.state(target), _PSEUDO))

    def run(self, lines: list[str]) -> None:
        i = 0
        while i < len(lines):
            s = lines[i].strip().rstrip(";").strip()
            i += 1
            m = _ST_NOTE.match(s)
            if m is None:
                self.statement(s)
            elif m.group("text") is not None:
                self.note(m.group("id"), m.group("text"))
            else:  # note ... end note
                text = []
                while i < len(lines) and lines[i].strip().lower() not in ("end note", "endnote"):
                    text.append(lines[i].strip())
                    i += 1
                if i >= len(lines):
                    raise MermaidError("unterminated note")
                i += 1
                self.note(m.group("id"), "\n".join(text))
        if self.stack:
            raise MermaidError("unclosed composite state")

    def statement(self, s: str) -> None:
        low = s.lower()
        if (m := _ST_TRANS.match(s)) is not None:
            a = self.state(m.group("a"), as_source=True)
            b = self.state(m.group("b"), as_source=False)
            self.edges.append((a, b, clean_text(m.group("label") or "")))
        elif (not low.startswith("state ") and _ST_STRAY_ANNOT.search(s)
              and not re.fullmatch(_ST_END, s)):
            for part in _split_stray_annotations(s):
                self.statement(part)
        elif low.startswith("direction "):
            if not self.stack:
                self.direction = _direction(s[10:]) or self.direction
        elif _ST_SKIP.match(s):
            pass
        elif _TITLE_WORD.match(s):
            self.title = clean_text(_TITLE_WORD.match(s).group("t"))
        elif low.startswith("state "):
            m = _ST_DECL.match(s)
            if not m:
                raise MermaidError(f"bad state statement: {s[:40]!r}")
            name = m.group("id")
            first_use = name not in self.nodes
            self.state(name)
            self.describe(name, m.group("desc"))
            self.describe(name, m.group("desc2"))
            annot = (m.group("annot") or "").strip("<> ").lower()
            # Mermaid keeps the shape of a state's first occurrence: a <<choice>> declared after
            # the state was already used is drawn as a plain state.
            if annot in STATE_ANNOTATION_KINDS and first_use:
                self.kinds[name] = STATE_ANNOTATION_KINDS[annot]
            if m.group("brace"):
                self.open_composite(name)
        elif s == "}":
            if not self.stack:
                raise MermaidError("unbalanced '}'")
            self.stack.pop()
        elif s == "--":
            self.divider()
        elif (m := _ST_DESC.match(s)) is not None:
            self.state(m.group("id"))
            self.describe(m.group("id"), m.group("desc"))
        elif re.fullmatch(_ST_END, s):
            self.state(s)
        else:
            raise MermaidError(f"cannot parse: {s[:40]!r}")

    def finish(self, title: Optional[str]) -> Diagram:
        resolve = {name: g.id for name, g in self.composites.items()}
        nodes = []
        for key, node in self.nodes.items():
            if key in self.composites:
                continue
            resolve[key] = node.id
            if key.startswith(_PSEUDO):
                nodes.append(node)
                continue
            node.kind = self.kinds.get(key, "state")
            descs = self.descs.get(key, [])
            if node.kind == "state":
                # The first description replaces the name; more lines go below a divider.
                node.text = descs[0] if descs else _state_name_text(key)
                node.members = [Member(kind="attr", text=d) for d in descs[1:]]
            else:
                node.text = ""  # choice / fork / join carry no text
            nodes.append(node)
        if not nodes and not self.composites:
            raise MermaidError("empty diagram")
        edges = [Edge(source=resolve[a], target=resolve[b], kind="anchor" if lab == _PSEUDO else "transition",
                      label="" if lab == _PSEUDO else lab) for a, b, lab in self.edges]
        groups = []
        by_id = {g.id: g for g in self.regions}
        for name, g in self.composites.items():
            g.text = (self.descs.get(name) or [name])[0]
            g.parent = self.composite_scope[name].group
            groups.append(g)
            by_id[g.id] = g
        for key, scope in self.member_of.items():
            if key not in self.composites and scope.group is not None:
                by_id[scope.group].members.append(resolve[key])
        groups += self.regions
        return Diagram(type="state", nodes=nodes, edges=edges, groups=groups,
                       direction=self.direction, title=self.title or title)


def mermaid_state_to_diagram(code: str) -> Diagram:
    lines, title = _prepare(code)
    rest = _header(lines, ("statediagram", "statediagram-v2"))
    b = _StateBuilder()
    b.run(([rest] if rest else []) + lines[1:])
    return b.finish(title)


# =========================================================================== #
# ER diagrams
# =========================================================================== #
# Crow's-foot end -> UML multiplicity (symbol and word forms, either side).
ER_CARDINALITY = {
    "||": "1", "|o": "0..1", "o|": "0..1", "}|": "1..*", "|{": "1..*", "}o": "0..*", "o{": "0..*",
    "only one": "1", "1": "1", "zero or one": "0..1", "one or zero": "0..1",
    "one or more": "1..*", "one or many": "1..*", "many(1)": "1..*", "1+": "1..*",
    "zero or more": "0..*", "zero or many": "0..*", "many(0)": "0..*", "0+": "0..*",
}
_ER_CARD = "|".join(re.escape(k) for k in sorted(ER_CARDINALITY, key=len, reverse=True))
_ER_ENT = r'(?:"[^"]*"|[\w][\w-]*)'
_ER_REL = re.compile(
    rf"^(?P<a>{_ER_ENT})(?:\[(?P<la>[^\]]*)\])?\s*(?P<l>{_ER_CARD})\s*(?P<line>--|\.\.|optionally\s+to|to)\s*"
    rf"(?P<r>{_ER_CARD})\s*(?P<b>{_ER_ENT})(?:\[(?P<lb>[^\]]*)\])?\s*(?::(?P<label>.*))?$",
    re.IGNORECASE,
)
_ER_ENTITY = re.compile(rf"^(?P<name>{_ER_ENT})\s*(?:\[(?P<label>[^\]]*)\])?\s*(?::::[\w-]+)?\s*"
                        r"(?P<brace>\{(?P<rest>.*))?$")
_ER_SKIP = re.compile(r"^(?:style|classDef|class)\s")


def er_attribute_text(raw: str) -> str:
    """`type name [PK, FK] ["comment"]` -> the attribute row as one line (comment unquoted)."""
    s = raw.strip()
    comment = ""
    m = re.search(r'\s*"([^"]*)"\s*$', s)
    if m:
        comment, s = m.group(1).strip(), s[:m.start()]
    parts = s.split()
    return " ".join(parts + ([comment] if comment else []))


class _ErBuilder:
    def __init__(self) -> None:
        self.ids = _Ids()
        self.entities: dict[str, Node] = {}
        self.edges: list[Edge] = []
        self.direction: Optional[str] = None
        self.title: Optional[str] = None

    def entity(self, raw: str, label: Optional[str] = None) -> Node:
        name = clean_text(raw)
        node = self.entities.get(name)
        if node is None:
            node = self.entities[name] = Node(id=self.ids.node(), kind="entity", text=name)
        if label is not None and clean_text(label):
            node.text = clean_text(label)
        return node

    def run(self, lines: list[str]) -> None:
        body: Optional[Node] = None
        for line in lines:
            s = line.strip()
            if body is not None:
                if s.endswith("}"):
                    s = s[:-1].strip()
                    if s:
                        body.members.append(Member(kind="attr", text=er_attribute_text(s)))
                    body = None
                elif s:
                    body.members.append(Member(kind="attr", text=er_attribute_text(s)))
                continue
            s = s.rstrip(";").strip()
            if s.lower().startswith("direction "):
                self.direction = _direction(s[10:]) or self.direction
            elif _ER_SKIP.match(s):
                continue
            elif (m := _ER_REL.match(s)) is not None:
                a, b = self.entity(m.group("a"), m.group("la")), self.entity(m.group("b"), m.group("lb"))
                dashed = m.group("line") == ".." or m.group("line").lower().startswith("optionally")
                self.edges.append(Edge(
                    source=a.id, target=b.id, kind="rel", label=clean_text(m.group("label") or ""),
                    source_label=ER_CARDINALITY[m.group("l").lower()],
                    target_label=ER_CARDINALITY[m.group("r").lower()],
                    style="dashed" if dashed else None))
            elif _TITLE_WORD.match(s):
                self.title = clean_text(_TITLE_WORD.match(s).group("t"))
            elif (m := _ER_ENTITY.match(s)) is not None:
                node = self.entity(m.group("name"), m.group("label"))
                if m.group("brace") is not None:
                    rest = m.group("rest").strip()
                    if rest.endswith("}"):
                        if rest[:-1].strip():
                            node.members.append(Member(kind="attr", text=er_attribute_text(rest[:-1])))
                    else:
                        if rest:
                            node.members.append(Member(kind="attr", text=er_attribute_text(rest)))
                        body = node
            else:
                raise MermaidError(f"cannot parse: {s[:40]!r}")
        if body is not None:
            raise MermaidError("unclosed entity block")

    def finish(self, title: Optional[str]) -> Diagram:
        if not self.entities:
            raise MermaidError("empty diagram")
        return Diagram(type="er", nodes=list(self.entities.values()), edges=self.edges,
                       direction=self.direction, title=self.title or title)


def mermaid_er_to_diagram(code: str) -> Diagram:
    lines, title = _prepare(code)
    rest = _header(lines, ("erdiagram",))
    b = _ErBuilder()
    b.run(([rest] if rest else []) + lines[1:])
    return b.finish(title)


# =========================================================================== #
# Mindmaps
# =========================================================================== #
# (opener, closer, graph node kind); longer openers first.
MINDMAP_SHAPES = [
    ("((", "))", "circle"), ("))", "((", "other"), ("{{", "}}", "hexagon"),
    (")", "(", "cloud"), ("(", ")", "rounded"), ("[", "]", "box"),
]


def _mindmap_node(s: str) -> tuple[str, str]:
    """(kind, text) of one mindmap line: `id(shape text)` or plain text."""
    m = re.search(r"[(\[{)]", s)
    if m:
        rest = s[m.start():]
        for opener, closer, kind in MINDMAP_SHAPES:
            if (rest.startswith(opener) and rest.endswith(closer)
                    and len(rest) >= len(opener) + len(closer)):
                return kind, clean_text(rest[len(opener):len(rest) - len(closer)])
    return "node", clean_text(s)


def mermaid_mindmap_to_diagram(code: str) -> Diagram:
    lines, title = _prepare(code)
    _header(lines, ("mindmap",))
    ids = _Ids()
    nodes: list[Node] = []
    edges: list[Edge] = []
    stack: list[tuple[int, str]] = []  # (indent, node id) of the open ancestors
    for line in lines[1:]:
        s = line.strip()
        if s.startswith("::icon(") or s.startswith(":::"):  # decorations of the previous node
            continue
        s = re.sub(r"\s*:::[\w\s-]*$", "", s)
        indent = len(line) - len(line.lstrip())
        while stack and stack[-1][0] >= indent:
            stack.pop()
        if not stack and nodes:
            raise MermaidError(f"more than one root: {s[:40]!r}")
        kind, text = _mindmap_node(s)
        node = Node(id=ids.node(), kind=kind, text=text)
        nodes.append(node)
        if stack:
            edges.append(Edge(source=stack[-1][1], target=node.id, kind="line"))
        stack.append((indent, node.id))
    if not nodes:
        raise MermaidError("empty diagram")
    return Diagram(type="graph", nodes=nodes, edges=edges, title=title)


# =========================================================================== #
# Architecture diagrams (architecture-beta) -> graph
# =========================================================================== #
# The five built-in icons; Mermaid draws any other icon name as a "?" tile (kind `other`).
ARCHITECTURE_ICONS = {"server": "box3d", "database": "cylinder", "disk": "storage", "cloud": "cloud",
                      "internet": "image"}
_ARCH_ITEM = re.compile(r"^(?P<kw>service|group|junction)\s+(?P<id>[\w-]+)\s*(?:\((?P<icon>[^)]*)\))?\s*"
                        r"(?:\[(?P<label>[^\]]*)\])?\s*(?:in\s+(?P<parent>[\w-]+))?$")
_ARCH_EDGE = re.compile(
    r"^(?P<a>[\w-]+)(?P<ag>\{group\})?\s*(?::\s*[LRTB])?\s*(?P<lh><)?--(?P<rh>>)?\s*(?:[LRTB]\s*:)?\s*"
    r"(?P<b>[\w-]+)(?P<bg>\{group\})?$"
)


def mermaid_architecture_to_diagram(code: str) -> Diagram:
    """Services / junctions -> nodes, groups -> clusters, edges by arrow heads. Side anchors
    (`:R --> L:`) are layout, not structure, and are dropped; `x{group}` ends at x's group."""
    lines, title = _prepare(code)
    _header(lines, ("architecture-beta", "architecture"))
    ids = _Ids()
    items: dict[str, Node | Group] = {}
    parents: dict[str, str] = {}
    edges: list[tuple[str, bool, str, bool, str]] = []  # (a, a{group}, b, b{group}, kind)
    for line in lines[1:]:
        s = line.strip()
        if s.lower().startswith("title "):
            title = clean_text(s[6:])
        elif (m := _ARCH_ITEM.match(s)) is not None:
            if m.group("id") in items:
                raise MermaidError(f"duplicate id {m.group('id')!r}")
            text = clean_text(m.group("label") or "")
            if m.group("kw") == "group":
                items[m.group("id")] = Group(id=ids.group(), kind="cluster", text=text)
            else:
                icon = (m.group("icon") or "").strip().lower()
                kind = "point" if m.group("kw") == "junction" else ARCHITECTURE_ICONS.get(
                    icon, "other" if icon else "box")
                items[m.group("id")] = Node(id=ids.node(), kind=kind, text=text)
            if m.group("parent"):
                parents[m.group("id")] = m.group("parent")
        elif (m := _ARCH_EDGE.match(s)) is not None:
            kind = "bidir" if m.group("lh") and m.group("rh") else "arrow" if m.group("lh") or m.group("rh") \
                else "line"
            a, b = (m.group("b"), m.group("a")) if m.group("lh") and not m.group("rh") else \
                (m.group("a"), m.group("b"))
            a_group, b_group = (m.group("bg"), m.group("ag")) if a != m.group("a") else \
                (m.group("ag"), m.group("bg"))
            edges.append((a, bool(a_group), b, bool(b_group), kind))
        else:
            raise MermaidError(f"cannot parse: {s[:40]!r}")
    for child, parent in parents.items():
        group = items.get(parent)
        if not isinstance(group, Group):
            raise MermaidError(f"unknown group {parent!r}")
        if isinstance(items[child], Group):
            items[child].parent = group.id
        else:
            group.members.append(items[child].id)

    def end(name: str, to_group: bool) -> str:
        if name not in items:
            raise MermaidError(f"unknown service {name!r}")
        if to_group and name in parents:
            return items[parents[name]].id
        return items[name].id

    nodes = [x for x in items.values() if isinstance(x, Node)]
    if not nodes:
        raise MermaidError("empty diagram")
    return Diagram(type="graph", nodes=nodes,
                   edges=[Edge(source=end(a, ag), target=end(b, bg), kind=k) for a, ag, b, bg, k in edges],
                   groups=[x for x in items.values() if isinstance(x, Group)], title=title)


# =========================================================================== #
# C4 diagrams (C4Context / C4Container / C4Component / C4Deployment) -> graph
# =========================================================================== #
_C4_ELEMENT = re.compile(r"^(?P<base>Person|System|Container|Component)(?P<sub>Db|Queue)?(?P<ext>_Ext)?$")
_C4_BOUNDARY_TYPES = {"Enterprise_Boundary": "ENTERPRISE", "System_Boundary": "SYSTEM",
                      "Container_Boundary": "CONTAINER"}
_C4_NODES = ("Boundary", "Deployment_Node", "Node", "Node_L", "Node_R")  # (alias, label, [type, descr])
_C4_RELS = {"Rel": "arrow", "BiRel": "bidir", "Rel_Back": "back", **dict.fromkeys(
    ["Rel_U", "Rel_Up", "Rel_D", "Rel_Down", "Rel_L", "Rel_Left", "Rel_R", "Rel_Right"], "arrow")}
_C4_IGNORED = re.compile(r"^(?:Update\w+|Add\w+Tag|SHOW_\w+|LAYOUT_\w+|HIDE_\w+)\s*\(")
_C4_CALL = re.compile(r"^(?P<fn>\w+)\s*\((?P<args>.*)\)\s*(?P<brace>\{)?\s*$")


def _c4_args(raw: str) -> list[str]:
    """Positional arguments of a C4 macro call (`$name=...` keyword arguments dropped)."""
    out, buf, quote = [], [], False
    for ch in raw:
        if ch == '"':
            quote = not quote
        if ch == "," and not quote:
            out.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    out.append("".join(buf).strip())
    return [clean_text(a) for a in out if not a.startswith("$")]


def _c4_text(*parts: Optional[str], bracket: int = -1) -> str:
    """Lines as drawn: label, `[technology / type]`, description (empty parts skipped)."""
    lines = [f"[{p}]" if i == bracket else p for i, p in enumerate(parts) if p]
    return "\n".join(lines)


def mermaid_c4_to_diagram(code: str) -> Diagram:
    """Elements -> nodes (stereotype = the drawn `<<type>>`), boundaries / deployment nodes ->
    clusters, Rel* -> edges. Node text is label, [technology] and description, one per line."""
    lines, title = _prepare(code)
    _header(lines, ("c4context", "c4container", "c4component", "c4deployment", "c4dynamic"))
    ids = _Ids()
    items: dict[str, Node | Group] = {}
    stack: list[Group] = []
    rels: list[tuple[str, str, str, str]] = []  # (from alias, to alias, kind, label)
    for line in lines[1:]:
        s = line.strip()
        if s == "}":
            if not stack:
                raise MermaidError("unbalanced '}'")
            stack.pop()
            continue
        if s.lower().startswith("title "):
            title = clean_text(s[6:])
            continue
        if _C4_IGNORED.match(s):
            continue
        m = _C4_CALL.match(s)
        if not m:
            raise MermaidError(f"cannot parse: {s[:40]!r}")
        fn, args = m.group("fn"), _c4_args(m.group("args"))
        args += [""] * (4 - len(args))
        if fn in _C4_RELS:
            if not args[0] or not args[1]:
                raise MermaidError(f"{fn} needs two elements")
            label = _c4_text(args[2], args[3], bracket=1)
            if _C4_RELS[fn] == "back":
                rels.append((args[1], args[0], "arrow", label))
            else:
                rels.append((args[0], args[1], _C4_RELS[fn], label))
            continue
        alias = args[0]
        if not alias:
            raise MermaidError(f"{fn} without an alias")
        parent = stack[-1] if stack else None
        if fn in _C4_BOUNDARY_TYPES or fn in _C4_NODES:
            kind_text = _C4_BOUNDARY_TYPES.get(fn, args[2])
            text = _c4_text(args[1], kind_text, args[3] if fn in _C4_NODES else "", bracket=1)
            if m.group("brace"):
                g = Group(id=ids.group(), kind="cluster", text=text, parent=parent.id if parent else None)
                items[alias] = g
                stack.append(g)
                continue
            node = Node(id=ids.node(), kind="box3d", text=text)
        else:
            e = _C4_ELEMENT.match(fn)
            if e is None:
                raise MermaidError(f"unknown C4 element {fn!r}")
            sub = (e.group("sub") or "").lower()
            if e.group("base") == "Person":
                kind = "actor"
            else:
                kind = {"db": "cylinder", "queue": "queue"}.get(
                    sub, "component" if e.group("base") == "Component" else "box")
            with_techn = e.group("base") in ("Container", "Component")
            text = _c4_text(args[1], args[2] if with_techn else "", args[3] if with_techn else args[2],
                            bracket=1)
            stereo = "external_" if e.group("ext") else ""
            stereo += e.group("base").lower() + (f"_{sub}" if sub else "")
            node = Node(id=ids.node(), kind=kind, text=text, stereotype=stereo)
        items[alias] = node
        if parent is not None:
            parent.members.append(node.id)
    if stack:
        raise MermaidError("unclosed boundary")
    nodes = [x for x in items.values() if isinstance(x, Node)]
    if not nodes:
        raise MermaidError("empty diagram")
    edges = []
    for a, b, kind, label in rels:
        if a not in items or b not in items:
            raise MermaidError(f"relation to an unknown element: {a!r} -> {b!r}")
        edges.append(Edge(source=items[a].id, target=items[b].id, kind=kind, label=label))
    return Diagram(type="graph", nodes=nodes, edges=edges,
                   groups=[x for x in items.values() if isinstance(x, Group)], title=title)


# =========================================================================== #
# Dispatcher
# =========================================================================== #
MERMAID_PARSERS: dict[str, Callable[[str], Diagram]] = {
    "graph": mermaid_flowchart_to_diagram,
    "flowchart": mermaid_flowchart_to_diagram,
    "flowchart-elk": mermaid_flowchart_to_diagram,
    "sequencediagram": mermaid_sequence_to_diagram,
    "classdiagram": mermaid_class_to_diagram,
    "classdiagram-v2": mermaid_class_to_diagram,
    "statediagram": mermaid_state_to_diagram,
    "statediagram-v2": mermaid_state_to_diagram,
    "erdiagram": mermaid_er_to_diagram,
    "mindmap": mermaid_mindmap_to_diagram,
    "architecture-beta": mermaid_architecture_to_diagram,
    "architecture": mermaid_architecture_to_diagram,
    **dict.fromkeys(["c4context", "c4container", "c4component", "c4deployment"], mermaid_c4_to_diagram),
}


def mermaid_header(code: str) -> str:
    """The diagram keyword of Mermaid source (`graph`, `sequenceDiagram`, ...), or ''."""
    lines, _ = _prepare(code)
    m = re.match(r"[^\s;]+", lines[0].strip()) if lines else None
    return m.group(0) if m else ""


def mermaid_to_diagram(code: str) -> Diagram:
    """Parse any supported Mermaid diagram, routing on its header keyword."""
    keyword = mermaid_header(code)
    if not keyword:
        raise MermaidError("empty diagram")
    parser = MERMAID_PARSERS.get(keyword.lower())
    if parser is None:
        raise UnsupportedMermaid(f"unsupported Mermaid diagram type {keyword!r}")
    return parser(code)
