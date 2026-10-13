"""DGM: the compact, line-based text form of the Diagram IR - what the model emits.

One element per line; the first non-comment line is the header:

    diagram flowchart dir=TB
    node n1 terminal "Start" @400,20,600,70
    node n2 decision "Quality Check" @420,470,580,570
    node n3 class "Order"
      attr "-id: int"
      op "+total(): float"
    edge n1 n2
    edge n2 n4 "Pass"
    edge n3 n5 compose sl=1 tl=1..*
    group g1 lane "Sales" @0,0,300,1000 : n1 n2

Grammar (see SPEC.md for the normative version):

    header := "diagram" TYPE attr*
    node   := "node" ID KIND [STRING] [BBOX] attr*
    member := ("attr" | "op" | "lit") STRING [BBOX]           # belongs to the previous node
    edge   := "edge" ID ID [KIND] [STRING] attr*              # KIND omitted = type default
    group  := "group" ID KIND [STRING] [BBOX] attr* [":" ID+]
    BBOX   := "@" INT "," INT "," INT "," INT                  # 0..1000, x1,y1,x2,y2
    attr   := KEY "=" (BARE | STRING)
    STRING := '"' ... '"'   with escapes \\ \" \n \t

Blank lines and lines starting with `#` are ignored. Edge ids are positional (the i-th
edge line is `e{i}`), so groups can list edges as members (sequence fragments).
"""

from __future__ import annotations

import re
from typing import Optional

from .model import (
    ATTR_KEY_RE,
    ID_RE,
    KIND_RE,
    MEMBER_KINDS,
    BBox,
    Diagram,
    Edge,
    Group,
    Member,
    Node,
    default_edge_kind,
)

HEADER = "diagram"

# Short attribute keys in the text form -> dataclass field names.
NODE_KEYS = {"stereo": "stereotype"}
EDGE_KEYS = {
    "sl": "source_label",
    "tl": "target_label",
    "sp": "source_port",
    "tp": "target_port",
    "style": "style",
    "path": "path",
}
GROUP_KEYS = {"parent": "parent"}
HEADER_KEYS = {"dir": "direction", "title": "title"}

_STR = r'"(?:[^"\\\n]|\\.)*"'
_TOKEN_RE = re.compile(
    rf"\s*(?:"
    rf"(?P<str>{_STR})"
    rf"|(?P<kv>[A-Za-z_][A-Za-z0-9_]*=(?:{_STR}|[^\s\"]+))"
    r"|(?P<box>@\S+)"
    r"|(?P<colon>:)"
    r'|(?P<word>[^\s"=:@]+)'
    r"|(?P<bad>\S+)"
    r")"
)
_BBOX_RE = re.compile(r"@(\d+),(\d+),(\d+),(\d+)\Z")
_BARE_VALUE_RE = re.compile(r"[A-Za-z0-9_.*+\-,]+\Z")  # values written without quotes
_ESCAPES = {'"': '"', "\\": "\\", "n": "\n", "t": "\t"}


class DgmSyntaxError(ValueError):
    def __init__(self, line_no: int, message: str, line: str = ""):
        self.line_no = line_no
        self.message = message
        self.line = line
        super().__init__(f"line {line_no}: {message}" + (f"  [{line.strip()}]" if line else ""))


# --------------------------------------------------------------------------- #
# Lexing helpers
# --------------------------------------------------------------------------- #
def _unescape(quoted: str) -> str:
    body = quoted[1:-1]
    out, i = [], 0
    while i < len(body):
        ch = body[i]
        if ch == "\\":
            nxt = body[i + 1] if i + 1 < len(body) else ""
            if nxt not in _ESCAPES:
                raise ValueError(f"unknown escape \\{nxt}")
            out.append(_ESCAPES[nxt])
            i += 2
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def _escape(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\t", "\\t")
    return '"' + re.sub(r"[\x00-\x1f\x7f]", " ", text) + '"'


def _tokenize(line: str) -> list[tuple[str, str]]:
    tokens, pos = [], 0
    while pos < len(line):
        if not line[pos:].strip():
            break
        m = _TOKEN_RE.match(line, pos)
        if not m or m.end() == pos:
            raise ValueError(f"cannot tokenize near {line[pos:pos + 20]!r}")
        kind = m.lastgroup
        value = m.group(kind)
        if kind == "bad":
            raise ValueError(f"unexpected token {value!r}")
        tokens.append((kind, value))
        pos = m.end()
    return tokens


def _parse_bbox(token: str) -> BBox:
    m = _BBOX_RE.match(token)
    if not m:
        raise ValueError(f"bad bbox {token!r}; expected @x1,y1,x2,y2")
    x1, y1, x2, y2 = (int(v) for v in m.groups())
    return (x1, y1, x2, y2)


def _parse_kv(token: str) -> tuple[str, str]:
    key, _, raw = token.partition("=")
    if not ATTR_KEY_RE.match(key):
        raise ValueError(f"bad attribute key {key!r}")
    return key, (_unescape(raw) if raw.startswith('"') else raw)


def _parse_path(raw: str) -> list[tuple[int, int]]:
    try:
        nums = [int(v) for v in raw.split(",")]
    except ValueError:
        raise ValueError(f"bad path {raw!r}; expected x,y,x,y,...") from None
    if len(nums) < 4 or len(nums) % 2:
        raise ValueError(f"path needs an even number (>=4) of integers, got {len(nums)}")
    return [(nums[i], nums[i + 1]) for i in range(0, len(nums), 2)]


def _split_rest(tokens: list[tuple[str, str]], allow_word: bool) -> dict:
    """Parse the optional tail shared by node/edge/group lines."""
    out: dict = {"word": None, "text": None, "bbox": None, "kvs": [], "members": None}
    for i, (kind, value) in enumerate(tokens):
        if kind == "colon":
            members = []
            for mkind, mval in tokens[i + 1:]:
                if mkind != "word" or not ID_RE.match(mval):
                    raise ValueError(f"bad member reference {mval!r}")
                members.append(mval)
            out["members"] = members
            break
        if kind == "word" and allow_word and out["word"] is None and out["text"] is None:
            out["word"] = value
        elif kind == "str" and out["text"] is None:
            out["text"] = _unescape(value)
        elif kind == "box" and out["bbox"] is None:
            out["bbox"] = _parse_bbox(value)
        elif kind == "kv":
            out["kvs"].append(_parse_kv(value))
        else:
            raise ValueError(f"unexpected {value!r}")
    return out


def _id(token: tuple[str, str], what: str) -> str:
    kind, value = token
    if kind != "word" or not ID_RE.match(value):
        raise ValueError(f"expected {what} id, got {value!r}")
    return value


def _kind(token: tuple[str, str], what: str) -> str:
    kind, value = token
    if kind != "word" or not KIND_RE.match(value):
        raise ValueError(f"expected {what} kind, got {value!r}")
    return value


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def _apply_keys(obj, kvs: list[tuple[str, str]], keymap: dict[str, str]) -> None:
    for key, value in kvs:
        fname = keymap.get(key)
        if fname is None:
            obj.attrs[key] = value
        elif fname == "path":
            obj.path = _parse_path(value)
        elif fname == "style" and value == "solid":
            obj.style = None
        else:
            setattr(obj, fname, value)


def _parse_line(d: Diagram, tokens: list[tuple[str, str]], last_node: Optional[Node]) -> Optional[Node]:
    """Apply one body line to `d`. Returns the node that members now attach to."""
    head = tokens[0][1]
    if tokens[0][0] != "word":
        raise ValueError(f"line must start with a keyword, got {head!r}")

    if head == "node":
        if len(tokens) < 3:
            raise ValueError("node needs an id and a kind")
        rest = _split_rest(tokens[3:], allow_word=False)
        if rest["members"] is not None:
            raise ValueError("':' is only valid on group lines")
        node = Node(id=_id(tokens[1], "node"), kind=_kind(tokens[2], "node"),
                    text=rest["text"] or "", bbox=rest["bbox"])
        _apply_keys(node, rest["kvs"], NODE_KEYS)
        d.nodes.append(node)
        return node

    if head in MEMBER_KINDS:
        if last_node is None:
            raise ValueError(f"{head!r} line must follow a node line")
        rest = _split_rest(tokens[1:], allow_word=False)
        if rest["text"] is None or rest["kvs"] or rest["members"] is not None:
            raise ValueError(f"{head!r} line takes a quoted text and an optional bbox only")
        last_node.members.append(Member(kind=head, text=rest["text"], bbox=rest["bbox"]))
        return last_node

    if head == "edge":
        if len(tokens) < 3:
            raise ValueError("edge needs a source and a target")
        rest = _split_rest(tokens[3:], allow_word=True)
        if rest["bbox"] is not None or rest["members"] is not None:
            raise ValueError("edges take no bbox or members (use path=)")
        kind = rest["word"] or default_edge_kind(d.type)
        if not KIND_RE.match(kind):
            raise ValueError(f"bad edge kind {kind!r}")
        edge = Edge(source=_id(tokens[1], "source"), target=_id(tokens[2], "target"),
                    kind=kind, label=rest["text"] or "")
        _apply_keys(edge, rest["kvs"], EDGE_KEYS)
        d.edges.append(edge)
        return None

    if head == "group":
        if len(tokens) < 3:
            raise ValueError("group needs an id and a kind")
        rest = _split_rest(tokens[3:], allow_word=False)
        group = Group(id=_id(tokens[1], "group"), kind=_kind(tokens[2], "group"),
                      text=rest["text"] or "", bbox=rest["bbox"], members=rest["members"] or [])
        _apply_keys(group, rest["kvs"], GROUP_KEYS)
        d.groups.append(group)
        return None

    raise ValueError(f"unknown keyword {head!r}")


def _parse_header(tokens: list[tuple[str, str]]) -> Diagram:
    if len(tokens) < 2 or tokens[0] != ("word", HEADER) or tokens[1][0] != "word":
        raise ValueError(f"first line must be '{HEADER} <type> ...'")
    d = Diagram(type=tokens[1][1])
    for kind, value in tokens[2:]:
        if kind != "kv":
            raise ValueError(f"unexpected {value!r} in header")
        key, val = _parse_kv(value)
        fname = HEADER_KEYS.get(key)
        if fname:
            setattr(d, fname, val)
        else:
            d.attrs[key] = val
    return d


def _body_lines(text: str):
    for line_no, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("```"):
            continue
        yield line_no, line


def parse(text: str) -> tuple[Diagram, list[DgmSyntaxError]]:
    """Lenient parse: skip bad lines and report them. Use on raw model output
    (e.g. a generation truncated mid-line still yields every complete element)."""
    errs: list[DgmSyntaxError] = []
    d: Optional[Diagram] = None
    last_node: Optional[Node] = None
    for line_no, line in _body_lines(text):
        try:
            tokens = _tokenize(line)
            if d is None:
                d = _parse_header(tokens)
                continue
            last_node = _parse_line(d, tokens, last_node)
        except ValueError as exc:
            errs.append(DgmSyntaxError(line_no, str(exc), line))
            if d is None and not line.lstrip().startswith(HEADER):
                d = Diagram(type="graph")  # missing header: keep going with the generic type
                try:
                    last_node = _parse_line(d, _tokenize(line), last_node)
                    errs[-1] = DgmSyntaxError(line_no, "missing 'diagram <type>' header; assumed graph")
                except ValueError:
                    pass
    if d is None:
        errs.append(DgmSyntaxError(0, "empty document"))
        d = Diagram(type="graph")
    return d, errs


def loads(text: str) -> Diagram:
    """Strict parse: raise DgmSyntaxError on the first malformed line."""
    d: Optional[Diagram] = None
    last_node: Optional[Node] = None
    for line_no, line in _body_lines(text):
        try:
            tokens = _tokenize(line)
            if d is None:
                d = _parse_header(tokens)
            else:
                last_node = _parse_line(d, tokens, last_node)
        except ValueError as exc:
            raise DgmSyntaxError(line_no, str(exc), line) from None
    if d is None:
        raise DgmSyntaxError(0, "empty document")
    return d


# --------------------------------------------------------------------------- #
# Serialization
# --------------------------------------------------------------------------- #
def _value(v: str) -> str:
    return v if _BARE_VALUE_RE.match(v) else _escape(v)


def _bbox(b: BBox) -> str:
    return "@" + ",".join(str(v) for v in b)


def _kvs(obj, keymap: dict[str, str]) -> list[str]:
    parts = []
    for key, fname in keymap.items():
        v = getattr(obj, fname)
        if v is None:
            continue
        if fname == "path":
            parts.append(f"path={','.join(str(c) for p in v for c in p)}")
        else:
            parts.append(f"{key}={_value(v)}")
    parts.extend(f"{k}={_value(v)}" for k, v in obj.attrs.items())
    return parts


def dumps(d: Diagram) -> str:
    header = [HEADER, d.type]
    if d.direction:
        header.append(f"dir={_value(d.direction)}")
    if d.title:
        header.append(f"title={_escape(d.title)}")
    header.extend(f"{k}={_value(v)}" for k, v in d.attrs.items())
    lines = [" ".join(header)]

    for n in d.nodes:
        parts = ["node", n.id, n.kind]
        if n.text:
            parts.append(_escape(n.text))
        if n.bbox is not None:
            parts.append(_bbox(n.bbox))
        parts.extend(_kvs(n, NODE_KEYS))
        lines.append(" ".join(parts))
        for m in n.members:
            mparts = ["  " + m.kind, _escape(m.text)]
            if m.bbox is not None:
                mparts.append(_bbox(m.bbox))
            lines.append(" ".join(mparts))

    default_kind = default_edge_kind(d.type)
    for e in d.edges:
        parts = ["edge", e.source, e.target]
        if e.kind != default_kind:
            parts.append(e.kind)
        if e.label:
            parts.append(_escape(e.label))
        parts.extend(_kvs(e, EDGE_KEYS))
        lines.append(" ".join(parts))

    for g in d.groups:
        parts = ["group", g.id, g.kind]
        if g.text:
            parts.append(_escape(g.text))
        if g.bbox is not None:
            parts.append(_bbox(g.bbox))
        parts.extend(_kvs(g, GROUP_KEYS))
        if g.members:
            parts.append(":")
            parts.extend(g.members)
        lines.append(" ".join(parts))

    return "\n".join(lines) + "\n"
