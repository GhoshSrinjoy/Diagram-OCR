"""PlantUML UML diagrams -> Diagram: class, sequence, activity (new and legacy syntax), state and
use case diagrams, plus the preprocessing and kind detection behind `plantuml_to_diagram`.

Preprocessing (`preprocess`): only the first @startuml block is read; `'` and `/' '/` comments are
removed; `!define` / `!definelong` macros and `!$var` variables are expanded, `!ifdef` / `!ifndef`
are honoured, other `!` directives (include, theme, pragma, procedures, ...) are dropped; styling
blocks (skinparam, <style>, legend, header / footer, sprite) are dropped; `title` becomes
Diagram.title and `left to right direction` the direction; multi-line notes and `ref over` blocks
are folded into one line; only the first page (up to `newpage`) is read.

Mapping decisions (target vocabularies: SPEC.md section 2):
  class     `entity` -> class + stereotype "entity"; object / map / annotation / struct / record /
            ... -> class with attrs keyword=<kw>; `circle X` / `() X` lollipops -> interface with
            keyword=circle; `diamond` / `<> X` and allowmixing elements (actor, node, ...) -> other.
            Dotted names create nested packages (PlantUML's default namespace separator) and names
            resolve per package like PlantUML 1.2023+ (see _ClassParser.find). Hidden links are
            dropped. Crow's feet ends become UML multiplicities in source/target_label (`||` 1,
            `|o` 0..1, `}|` 1..*, `}o` 0..*); other ends (`+` nested class, `#`, `x`, `(`) are plain.
  sequence  Boundary messages (`[->`, `->]`, `?->`) are skipped: the IR has no frame gate.
            Fragment guards are kept as drawn, "[guard]". `ref over` -> group kind ref whose
            members are the covered participants. `->x`, `!!` and `destroy X` right after a message
            to X -> destroy edge; `create X` / `**` -> create edge; `return` -> reply to the caller
            of the latest activation, else of the latest message. Notes -> note node + anchor
            edges in time order (not fragment members).
  activity  stop -> final, end -> flow_final, kill / detach -> the flow just ends (PlantUML draws
            no symbol). A merge node is added at endif / endswitch only when two or more flows
            rejoin; while loops loop back into their decision; repeat starts with a merge node (or
            with its `repeat :label;` box). Blocks left open are closed at the end, like PlantUML.
            Swimlanes -> lane groups; partition / group / package / rectangle / card -> region
            groups. Connectors `(A)` -> kind other. Notes -> note + anchor to the previous node.
  state     `[*]` is one initial and one final node per scope (top level, composite, region);
            composite states are groups and transitions into them target the group (a composite's
            own description lines are appended to its text); `--` / `||` split it into regions.
  use case  type graph: actors -> actor, use cases -> ellipse, other elements per ELEMENTS,
            boundaries / packages -> cluster groups (an empty one is drawn as a plain element).
            Bare undeclared names are actors (as drawn).
Display text: `\\n` -> line break, creole / html markup removed, `<<x>>` in labels -> «x».
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from typing import Optional

from ..model import Diagram, Edge, Group, Member, Node, validate
from .plantuml import ELEMENTS, PlantUMLError, UnsupportedPlantUML, plantuml_graph_to_diagram

_QSTR = r'"(?:[^"\\]|\\.)*"'


# --------------------------------------------------------------------------- #
# Text helpers
# --------------------------------------------------------------------------- #
_MARKUP_RE = re.compile(
    r"</?(?:b|i|u|s|w|del|strike|color|size|font|back|img|sub|sup|code|plain|em|strong|tt|big|small)"
    r"\b[^<>]*>|<[&$][\w-]+[^<>]*>",
    re.I,
)
_CREOLE_RE = re.compile(r'(?<!~)(\*\*|//|""|__)(?=\S)(.+?)(?<=\S)\1')
_URL_RE = re.compile(r"\[\[\[?([^\]{\s]*)(?:\{[^}]*\})?(?:\s+([^\]]*))?\]\]\]?")
_URL_TOKEN_RE = re.compile(r"\s*\[\[\[?[^\]{]*(?:\{[^}]*\})?[^\]]*\]\]\]?\s*(?=$|[:{#<]|[-.<>*o])")


def _codepoint(m: re.Match) -> str:
    cp = int(m.group(1), 16)
    return "" if 0xD800 <= cp <= 0xDFFF or cp > 0x10FFFF else chr(cp)


def _text(raw: Optional[str], creole: bool = True) -> str:
    """PlantUML display text -> plain text: unquote, `\\n` -> newline, drop markup."""
    if raw is None:
        return ""
    t = raw.strip()
    if len(t) >= 2 and t[0] == t[-1] == '"':
        t = t[1:-1]
    t = re.sub(r"\\n|%n\(\)|%newline\(\)|<br\s*/?>", "\n", t, flags=re.I)
    t = t.replace("\\t", " ").replace('\\"', '"')
    t = _URL_RE.sub(lambda m: m.group(2) or "", t)
    t = re.sub(r"<U\+([0-9A-Fa-f]{2,6})>", _codepoint, t)
    t = _MARKUP_RE.sub("", t)
    if creole:
        t = re.sub(r"<<\s*(.*?)\s*>>", "«\\1»", t)  # drawn as a guillemet stereotype
        t = _CREOLE_RE.sub(r"\2", t)
        t = re.sub(r"~(?=[*/_\"~<\[\-])", "", t)  # creole escape character
    if "&" in t:
        t = html.unescape(t)
    return "\n".join(ln.strip() for ln in t.split("\n")).strip()


def _split_colon(s: str) -> Optional[tuple[str, str]]:
    """Split at the first ':' outside quotes that is not part of '::'."""
    quoted = False
    for k, c in enumerate(s):
        if c == '"':
            quoted = not quoted
        elif c == ":" and not quoted:
            if s[k + 1:k + 2] == ":" or (k > 0 and s[k - 1] == ":"):
                continue
            return s[:k], s[k + 1:]
    return None


def _split_args(s: str) -> list[str]:
    """Split on top-level commas (outside quotes and brackets)."""
    out, buf, depth, quoted = [], [], 0, False
    for c in s:
        if c == '"':
            quoted = not quoted
        elif not quoted and c in "([{":
            depth += 1
        elif not quoted and c in ")]}":
            depth -= 1
        if c == "," and not quoted and depth == 0:
            out.append("".join(buf).strip())
            buf = []
        else:
            buf.append(c)
    out.append("".join(buf).strip())
    return out


def _direction_free_label(raw: Optional[str]) -> str:
    """Link label without the `<` / `>` reading-direction markers."""
    t = (raw or "").strip()
    t = re.sub(r"^<\s+|^<(?=\w)|\s+>$|(?<=\w)>$|\s+<$|^>\s+", "", t).strip()
    return _text(t)


def _guard(raw: str) -> str:
    t = _text(raw)
    if not t:
        return ""
    return t if t.startswith("[") and t.endswith("]") else f"[{t}]"


class _Build:
    """Collects nodes / edges / groups with ids n1.., g1.. (canonicalize renames them anyway)."""

    def __init__(self, dtype: str):
        self.dtype = dtype
        self.nodes: list[Node] = []
        self.edges: list[Edge] = []
        self.groups: list[Group] = []
        self._n = 0

    def node(self, kind: str, text: str = "", stereotype: Optional[str] = None) -> Node:
        self._n += 1
        n = Node(id=f"n{self._n}", kind=kind, text=text, stereotype=stereotype or None)
        self.nodes.append(n)
        return n

    def group(self, kind: str, text: str = "", parent: Optional[str] = None) -> Group:
        g = Group(id=f"g{len(self.groups) + 1}", kind=kind, text=text, parent=parent)
        self.groups.append(g)
        return g

    def edge(self, source: str, target: str, kind: str, label: str = "", **kw) -> Edge:
        e = Edge(source=source, target=target, kind=kind, label=label, **kw)
        self.edges.append(e)
        return e

    def drop_nodes(self, ids: set[str]) -> None:
        if not ids:
            return
        self.nodes = [n for n in self.nodes if n.id not in ids]
        self.edges = [e for e in self.edges if e.source not in ids and e.target not in ids]
        for g in self.groups:
            g.members = [m for m in g.members if m not in ids]

    def diagram(self, src: Source, direction: Optional[str] = None) -> Diagram:
        for g in self.groups:  # dedupe members, keep order
            g.members = list(dict.fromkeys(g.members))
        return Diagram(type=self.dtype, nodes=self.nodes, edges=self.edges, groups=self.groups,
                       direction=src.direction or direction, title=src.title)


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
@dataclass
class Source:
    """The statements of one @startuml block after preprocessing."""

    lines: list[str] = field(default_factory=list)
    title: Optional[str] = None
    direction: Optional[str] = None
    macro_lines: int = 0  # statements changed by macro / variable expansion (diagnostics)
    directives: int = 0  # `!` preprocessor lines seen (diagnostics)


_START_RE = re.compile(r"^[ \t]*@start(\w*)", re.I | re.M)
_END_RE = re.compile(r"^[ \t]*@end\w*", re.I | re.M)
_BODY_DECL_RE = re.compile(
    r"^[-+#~]?(?:abstract\s+class|abstract|class|interface|enum|annotation|entity|protocol|struct|exception"
    r"|metaclass|stereotype|dataclass|record|object|map|json)\s[^{]*\{\s*$",
    re.I,
)
_NOTE_HEAD_RE = re.compile(r"^(?:floating\s+)?[hr]?note\b", re.I)
_NOTE_END_RE = re.compile(r"^end\s*[hr]?note\b", re.I)
_DROP_RE = re.compile(
    r"^(?:skinparam\b|scale\b|skin\s|allow_?mixing\b|mainframe\b|caption\b|hide\s+footbox\b|page\s+\d+x\d+$"
    r"|show\s+footbox\b|(?:(?:left|right|center)\s+)?(?:header|footer)\b(?!\s*[-.<]))",
    re.I,
)
_DESC_RE = re.compile(  # element with a multi-line [ description ]
    r"^((?:actor|agent|artifact|boundary|card|cloud|collections|component|control|database|entity|file"
    r"|folder|frame|hexagon|interface|label|node|package|person|queue|rectangle|stack|storage|usecase)/?)"
    r"\s+([\w.$@-]+)((?:\s+(?:<<[^>]*>>|#\S+))*)\s*\[$",
    re.I,
)
_DIRECTION_RE = re.compile(r"^(left\s+to\s+right|top\s+to\s+bottom)\s+direction$", re.I)


def _find(raw: list[str], i: int, pattern: re.Pattern) -> int:
    for j in range(i, len(raw)):
        if pattern.match(raw[j]):
            return j
    return -1


def _skip_braces(raw: list[str], i: int, line: str) -> int:
    depth = line.count("{") - line.count("}")
    while depth > 0 and i < len(raw):
        depth += raw[i].count("{") - raw[i].count("}")
        i += 1
    return i


def _note_is_multiline(line: str) -> bool:
    rest = _NOTE_HEAD_RE.sub("", line, count=1).strip()
    return not rest.startswith('"') and _split_colon(rest) is None


def preprocess(code: str) -> Source:
    """First @startuml block of `code` -> statements (see module docstring)."""
    code = code.replace("\r\n", "\n").replace("\r", "\n").replace("\ufeff", "")
    code = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", code)  # stray control characters
    m = _START_RE.search(code)
    if m:
        tag = m.group(1).lower()
        if not tag.startswith("uml"):
            raise UnsupportedPlantUML(f"@start{tag} diagrams are not supported")
        nl = code.find("\n", m.end())
        body = code[nl + 1:] if nl >= 0 else ""
        e = _END_RE.search(body)
        if e:
            body = body[:e.start()]
    else:
        body = "\n".join(ln for ln in code.split("\n") if not ln.strip().startswith(("```", "@end")))
    body = re.sub(r"/'.*?'/", " ", body, flags=re.S)
    body = re.sub(r"(?<!\\)\\[ \t]*\n[ \t]*", " ", body)  # a trailing backslash continues the line
    raw = [ln.strip() for ln in body.split("\n")]

    src = Source()
    out = src.lines
    macros: dict[str, tuple[Optional[list[str]], str]] = {}
    active: list[bool] = []
    i = 0
    while i < len(raw):
        line = raw[i]
        i += 1
        if not line or line.startswith("'"):
            continue
        if line.startswith("!"):
            src.directives += 1
            i = _directive(line, raw, i, macros, active)
            continue
        if not all(active):
            continue
        if macros:
            new = _expand(line, macros)
            if new != line:
                src.macro_lines += 1
                parts = [p.strip() for p in new.split("\n")]
                raw[i:i] = parts[1:]
                line = parts[0]
                if not line:
                    continue
        low = line.lower()
        if low == "newpage" or low.startswith("newpage "):
            break
        j = i  # Allman style: a lone "{" on the next line opens this line's block
        while j < len(raw) and (not raw[j] or raw[j].startswith("'")):
            j += 1
        if j < len(raw) and raw[j] == "{" and not line.endswith("{"):
            line, i = f"{line} {{", j + 1
        if _BODY_DECL_RE.match(line):  # class-like body: members are copied as they are
            out.append(line)
            depth = 1
            is_json = low.startswith("json")
            while i < len(raw):
                ln = raw[i]
                i += 1
                if not ln or ln.startswith("'"):
                    continue
                if macros:
                    ln = _expand(ln, macros).replace("\n", " ")
                out.append(ln)
                if is_json:
                    depth += ln.count("{") - ln.count("}")
                    if depth <= 0:
                        break
                elif ln.startswith("}"):
                    break
            continue
        if low.startswith("<style"):
            if "</style>" not in low:
                j = _find(raw, i, re.compile(r".*</style>", re.I))
                i = j + 1 if j >= 0 else len(raw)
            continue
        if low.startswith(("skinparam", "sprite")) and line.endswith("{"):
            i = _skip_braces(raw, i, line)
            continue
        if re.match(r"^legend\b(?!\s*[-.<:])", low):
            j = _find(raw, i, re.compile(r"^end\s*legend\b", re.I))
            i = j + 1 if j >= 0 else i
            continue
        hm = re.match(r"^(?:(?:left|right|center)\s+)?(header|footer)$", low)
        if hm:
            j = _find(raw, i, re.compile(rf"^end\s*{hm.group(1)}\b", re.I))
            i = j + 1 if j >= 0 else i
            continue
        if low == "title":
            j = _find(raw, i, re.compile(r"^end\s*title\b", re.I))
            if j >= 0:
                src.title = _text("\\n".join(raw[i:j])) or None
                i = j + 1
            continue
        if re.match(r"^title\s*[\s:]", low):
            src.title = _text(line[5:].strip().lstrip(":")) or None
            continue
        dm = _DIRECTION_RE.match(line)
        if dm:
            src.direction = "LR" if dm.group(1).lower().startswith("left") else "TB"
            continue
        if _DROP_RE.match(line) or low.startswith("sprite"):
            continue
        if _NOTE_HEAD_RE.match(line) and _note_is_multiline(line):
            j = _find(raw, i, _NOTE_END_RE)
            if j >= 0:
                text = [t for t in raw[i:j] if not t.startswith("'")]
                out.append(f"{line} : " + "\\n".join(text))
                i = j + 1
            else:
                out.append(f"{line} :")
            continue
        dm = _DESC_RE.match(line)
        if dm:
            j = _find(raw, i, re.compile(r"^\]"))
            if j >= 0:
                text = [t for t in raw[i:j] if t and not re.match(r"^(?:-{2,}|={2,}|\.{2,}|_{2,})$", t)]
                body = "\\n".join(text).replace('"', "'")
                out.append(f'{dm.group(1)} "{body}" as {dm.group(2)}{dm.group(3)}')
                i = j + 1
                continue
        if re.match(r"^ref\s*(?:#\w+\s*)?over\b", low) and _split_colon(line) is None:
            j = _find(raw, i, re.compile(r"^end\s*ref\b", re.I))
            if j >= 0:
                out.append(f"{line} : " + "\\n".join(raw[i:j]))
                i = j + 1
                continue
        out.append(line)
    return src


def _unquote(s: str) -> str:
    s = s.strip()
    return s[1:-1] if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'" else s


def _directive(line: str, raw: list[str], i: int, macros: dict, active: list[bool]) -> int:
    """Handle one `!` preprocessor line; returns the index of the next unread line."""
    on = all(active)
    var = re.match(r"^!\s*(?:global\s+|local\s+)?(\$\w+)\s*\??=\s*(.*)$", line)
    if var:
        if on:
            macros[var.group(1)] = (None, _unquote(var.group(2)))
        return i
    m = re.match(r"^!\s*(\w+)\s*(.*)$", line)
    if not m:
        return i
    word, rest = m.group(1).lower(), m.group(2).strip()
    if word in ("ifdef", "ifndef"):
        val = rest in macros
        active.append(val if word == "ifdef" else not val)
    elif word == "if":
        active.append(True)  # expressions are not evaluated: assume the first branch
    elif word == "elseif":
        if active:
            active[-1] = False
    elif word == "else":
        if active:
            active[-1] = not active[-1]
    elif word == "endif":
        if active:
            active.pop()
    elif word == "define" and on:
        fm = re.match(r"^([\w$]+)\(([^)]*)\)\s*(.*)$", rest)
        if fm:
            params = [p.split("=")[0].strip() for p in fm.group(2).split(",") if p.strip()]
            macros[fm.group(1)] = (params, fm.group(3))
        else:
            om = re.match(r"^([\w$]+)\s*(.*)$", rest)
            if om:
                macros[om.group(1)] = (None, om.group(2))
    elif word == "definelong":
        j = _find(raw, i, re.compile(r"^!\s*enddefinelong\b", re.I))
        end = j if j >= 0 else len(raw)
        fm = re.match(r"^([\w$]+)(?:\(([^)]*)\))?", rest)
        if fm and on:
            params = [p.split("=")[0].strip() for p in (fm.group(2) or "").split(",") if p.strip()]
            macros[fm.group(1)] = (params if fm.group(2) is not None else None,
                                   "\n".join(raw[i:end]))
        return end + 1
    elif word in ("procedure", "function", "unquoted", "final"):
        j = _find(raw, i, re.compile(r"^!\s*end\s*(?:procedure|function)\b", re.I))
        return j + 1 if j >= 0 else len(raw)
    elif word == "undef":
        macros.pop(rest, None)
    return i


def _expand(line: str, macros: dict) -> str:
    for _ in range(8):
        new = line
        for name, (params, body) in macros.items():
            if name not in new:
                continue
            if params is None:
                new = re.sub(rf"(?<![\w$]){re.escape(name)}(?!\w)", lambda _m, b=body: b, new)
            else:
                new = _expand_call(new, name, params, body)
        if new == line:
            break
        line = new
    return line


def _expand_call(line: str, name: str, params: list[str], body: str) -> str:
    pat = re.compile(rf"(?<![\w$]){re.escape(name)}\s*\(")
    out, pos = [], 0
    while True:
        m = pat.search(line, pos)
        if not m:
            break
        depth, j, quoted = 1, m.end(), False
        while j < len(line) and depth:
            c = line[j]
            if c == '"':
                quoted = not quoted
            elif not quoted and c == "(":
                depth += 1
            elif not quoted and c == ")":
                depth -= 1
            j += 1
        if depth:
            break
        args = _split_args(line[m.end():j - 1])
        text = body
        for k, p in enumerate(params):
            val = args[k] if k < len(args) else ""
            text = re.sub(rf"(?<![\w$]){re.escape(p)}(?!\w)", lambda _m, v=val: v, text)
        out.append(line[pos:m.start()])
        out.append(text.replace("##", ""))
        pos = j
    out.append(line[pos:])
    return "".join(out)


# --------------------------------------------------------------------------- #
# Notes (shared by every diagram kind)
# --------------------------------------------------------------------------- #
@dataclass
class _NoteSpec:
    text: str
    pos: str = ""  # left / right / top / bottom / over / across, "" when not given
    targets: list[str] = field(default_factory=list)  # raw element references
    alias: Optional[str] = None
    on_link: bool = False


_NOTE_RE = re.compile(r"^(?:floating\s+)?[hr]?note\b\s*(?P<rest>.*)$", re.I)


def _parse_note(line: str) -> Optional[_NoteSpec]:
    m = _NOTE_RE.match(line)
    if not m:
        return None
    rest = m.group("rest").strip()
    q = re.match(rf"^({_QSTR})\s*(?:as\s+([\w.$@]+))?\s*(?:#\S+\s*)*$", rest)
    if q:
        return _NoteSpec(_text(q.group(1)), alias=q.group(2))
    sh = re.match(r"^(left|right|top|bottom)\s+of\s+(:[^:]+:|\([^()]+\)|\[[^\]]+\])\s*(?:#\S+\s*)?"
                  r"(?::(.*))?$", rest, re.I)
    if sh:
        return _NoteSpec(_text(sh.group(3)), pos=sh.group(1).lower(), targets=[sh.group(2)])
    sp = _split_colon(rest)
    head, body = (sp[0].strip(), sp[1]) if sp else (rest, "")
    head = re.sub(r"(?:^|\s)(?:#[\w#:;.\-/]+|<<[^>]*>>)", " ", head).strip()
    note = _NoteSpec(_text(body))
    a = re.match(r"^as\s+([\w.$@]+)$", head, re.I)
    if a:
        note.alias = a.group(1)
        return note
    if re.match(r"^on\s+link\b", head, re.I):
        note.on_link = True
        return note
    p = re.match(r"^(left|right|top|bottom|over|across)\b\s*(?:of\b)?\s*(.*)$", head, re.I)
    if p:
        note.pos = p.group(1).lower()
        tgt = p.group(2).strip()
        am = re.match(r"^(.*?)\s+as\s+([\w.$@]+)$", tgt)
        if am:
            tgt, note.alias = am.group(1), am.group(2)
        note.targets = [t for t in _split_args(tgt) if t]
        return note
    return note if not head else None


# --------------------------------------------------------------------------- #
# Class diagrams
# --------------------------------------------------------------------------- #
_C_KIND = {"class": "class", "abstract": "abstract", "interface": "interface", "enum": "enum",
           "circle": "interface", "diamond": "other"}
_C_DECL_RE = re.compile(
    r"^[-+#~]?(?P<kw>abstract\s+class|abstract|class|interface|enum|annotation|entity|protocol|struct"
    r"|exception|metaclass|stereotype|dataclass|record|object|map|circle|diamond|json)\s+(?P<rest>.+)$",
    re.I,
)
_C_NAME = r"\.?[\w$@\\]+(?:(?:\.|::)[\w$@\\]+)*"
_C_DNAME = r"\.?(?:[^\s\"<>{}\[\]#,;()|:]|::)*[^\W_](?:[^\s\"<>{}\[\]#,;()|:]|::)*"  # declarations
_C_EP = rf"(?:{_QSTR}|{_C_NAME})"
_HEAD_L = r"<\|\||<\||\|\||\|o|\}\||\}o|<<|<|\^|\*|o|\+|#|x|\}|\{|0\)|0|@|\)"
_HEAD_R = r"\|\|>|\|>|\|\||o\||\|\{|o\{|>>|>|\^|\*|o|\+|#|x|\{|\}|\(0|0|@|\("
_BODY = (r"[-.=](?:[-.=]|\[[^\]]*\]|\(0\)|\(0(?=[-.])|0\)(?=[-.])"
         r"|(?i:up|down|left|right|le|ri|do|u|d|l|r)(?=[-.=\[>|*o<+#x\"\s{}^0@(]|$))*")
_LINK_TAIL = r"\s*(?:(?:#[\w#:;.\-]+|<<[^>]*>>)\s*)*(?::(?!:)\s*(?P<label>.*))?$"
_C_GROUP_EP = rf"\(\s*{_C_EP}(?:\s*,\s*{_C_EP})+\s*\)"
_C_LINK_RE = re.compile(
    rf"^(?P<a>{_C_EP}|{_C_GROUP_EP})\s*(?:(?P<ca>{_QSTR})\s*)?(?P<lh>{_HEAD_L})?(?P<body>{_BODY})"
    rf"(?P<rh>{_HEAD_R})?\s*(?:(?P<cb>{_QSTR})\s*)?(?P<b>{_C_EP}|{_C_GROUP_EP}){_LINK_TAIL}"
)
_CROW = {"||": "1", "|o": "0..1", "o|": "0..1", "}|": "1..*", "|{": "1..*", "}o": "0..*", "o{": "0..*"}
_C_MEMBER_RE = re.compile(rf"^(?P<a>{_C_EP})(?<!:)\s*:(?!:)\s*(?P<m>.*)$")
_C_LOLLIPOP_RE = re.compile(
    rf"^(?P<a>{_C_EP})\s*(?:\(\)\s*-+|-+\s*\(\))\s*(?P<b>{_C_EP})\s*(?::\s*(?P<label>.*))?$")
_C_STEREO_RE = re.compile(rf"^(?P<a>{_C_EP})\s*(?P<s>(?:<<[^>]*>>\s*)+)(?:#\S+\s*)*$")
_C_PKG_RE = re.compile(
    r"^(?P<kw>package|namespace|folder|frame|cloud|node|database|rectangle|card|component|together|file"
    r"|storage|artifact|stack|queue|agent|hexagon|usecase|actor|person|boundary|control|collections|label"
    r"|port|portin|portout)"
    r"\b\s*(?P<rest>.*)$",
    re.I,
)
_MODIFIERS_RE = re.compile(r"\{(?:static|abstract|classifier|field|method)\}\s*", re.I)
_SEPARATOR_RE = re.compile(r"^(?:--|\.\.|==|__)")


def _link_style(body: str) -> tuple[bool, Optional[str], bool]:
    """Arrow body -> (dashed, style, hidden)."""
    opts = " ".join(re.findall(r"\[([^\]]*)\]", body)).lower()
    plain = re.sub(r"\[[^\]]*\]", "", body)
    dashed = "." in plain or "dashed" in opts or "dotted" in opts
    if "dotted" in opts:
        style = "dotted"
    elif dashed:
        style = "dashed"
    elif "bold" in opts or "thickness" in opts or "=" in plain:
        style = "thick"
    else:
        style = None
    return dashed, style, "hidden" in opts


def _tail_dashed(m: re.Match) -> bool:
    """`A --> B #red;line.dashed : x` - a line style given after the target."""
    end = m.start("label") if m.group("label") is not None else len(m.string)
    return bool(re.search(r"line\.(?:dashed|dotted)", m.string[m.end("b"):end], re.I))


def _class_head(h: Optional[str]) -> str:
    if h in ("<|", "|>", "^", "<||", "||>"):
        return "tri"
    if h == "*":
        return "comp"
    if h == "o":
        return "agg"
    if h in ("<", ">", "<<", ">>"):
        return "arrow"
    return ""


def _class_edge_kind(lh: Optional[str], rh: Optional[str], dashed: bool) -> tuple[str, bool]:
    """Heads -> (IR kind, reversed). reversed = the right-hand entity is the edge source."""
    left, right = _class_head(lh), _class_head(rh)
    if "tri" in (left, right):
        return ("realize" if dashed else "inherit"), (left == "tri" and right != "tri")
    if left in ("comp", "agg") or right in ("comp", "agg"):
        head = left if left in ("comp", "agg") else right
        return ("compose" if head == "comp" else "aggregate"), left not in ("comp", "agg")
    if left == "arrow" and right == "arrow":
        return "assoc", False
    if right == "arrow" or left == "arrow":
        return ("depend" if dashed else "assoc_dir"), left == "arrow"
    return "assoc", False


class _ClassParser:
    def __init__(self, src: Source):
        self.src = src
        self.b = _Build("class")
        self.by_key: dict[str, Node] = {}
        self.pkgs: dict[str, Group] = {}  # full package path -> group
        self.stack: list[Optional[str]] = []  # open containers: package path, None = together
        self.notes: dict[str, Node] = {}
        self.sep: Optional[str] = "."
        self.hides: list[tuple[str, str]] = []  # (hide|remove, argument)
        self.home: dict[str, str] = {}  # node id -> package path ("" = top level)
        self.short: dict[str, list[Node]] = {}  # short name -> nodes

    # ---- containers -------------------------------------------------------
    def _cur_pkg(self) -> str:
        for p in reversed(self.stack):
            if p is not None:
                return p
        return ""

    def _ensure_pkg(self, segs: list[str], base: str, kind: str = "package",
                    text: Optional[str] = None) -> tuple[str, Group]:
        """Nested packages `segs` under package path `base` -> (path, innermost group)."""
        path, g = base, self.pkgs.get(base) if base else None
        for k, seg in enumerate(segs):
            parent = g.id if g is not None else None
            path = f"{path}.{seg}" if path else seg
            g = self.pkgs.get(path)
            if g is None:
                label = text if (text is not None and k == len(segs) - 1) else seg
                g = self.b.group(kind, label, parent=parent)
                self.pkgs[path] = g
        assert g is not None
        return path, g

    def _place(self, node_id: str) -> None:
        cur = self._cur_pkg()
        if cur:
            self.pkgs[cur].members.append(node_id)

    # ---- entities ---------------------------------------------------------
    def _segs(self, name: str, quoted: bool) -> list[str]:
        """Name -> path segments; a name containing the namespace separator is absolute."""
        if not quoted and self.sep and self.sep in name.strip(self.sep):
            return [x for x in name.split(self.sep) if x]
        return [name]

    def find(self, name: str, quoted: bool = False, declare: bool = False) -> Optional[Node]:
        """PlantUML's (1.2023+) name resolution, checked against PlantUML's own entity counts.
        Absolute (dotted) names as they are. A declaration (`class X`) only reuses X of the current
        package. A reference (link, `X : member`) looks in the current package, its ancestors, the
        top level, then takes the one class of that name living in another package (namespaces keep
        their names to themselves)."""
        root = not quoted and name.startswith(".")
        if not quoted:
            name = name.lstrip(".")
        segs = self._segs(name, quoted)
        if len(segs) > 1:
            return self.by_key.get(".".join(segs))
        path = "" if root else self._cur_pkg()
        if declare:
            return self.by_key.get(f"{path}.{name}" if path else name)
        while True:
            node = self.by_key.get(f"{path}.{name}" if path else name)
            if node is not None or not path:
                break
            path = path.rsplit(".", 1)[0] if "." in path else ""
        if node is None and not quoted and "::" in name and self.sep != "::":
            base = self.find(name.split("::")[0])
            if base is not None and not any(n.text == name for n in self.short.get(name, [])):
                return base  # Class::member reference
        if node is not None or root:
            return node
        cur = self._cur_pkg()
        if cur and self.pkgs[cur].kind == "namespace":
            return None
        cands = [n for n in self.short.get(name, [])
                 if not self.home[n.id] or self.pkgs[self.home[n.id]].kind != "namespace"]
        return cands[0] if len(cands) == 1 else None

    def cls(self, name: str, quoted: bool = False, display: Optional[str] = None,
            declare: bool = False) -> Node:
        node = self.find(name, quoted, declare)
        if node is not None:
            return node
        cur = "" if (not quoted and name.startswith(".")) else self._cur_pkg()
        segs = self._segs(name if quoted else name.lstrip("."), quoted)
        if len(segs) > 1:
            path, group = self._ensure_pkg(segs[:-1], "")
        else:
            path, group = cur, (self.pkgs[cur] if cur else None)
        text = display if display is not None else (_text(segs[-1]) if quoted else segs[-1])
        node = self.b.node("class", text)
        self.by_key[f"{path}.{segs[-1]}" if path else segs[-1]] = node
        self.home[node.id] = path
        self.short.setdefault(segs[-1], []).append(node)
        if group is not None:
            group.members.append(node.id)
        return node

    def _short(self, name: str) -> str:
        name = name.lstrip(".")
        return name.split(self.sep)[-1] if self.sep and self.sep in name else name

    def ref(self, raw: str) -> tuple[str, bool]:
        """Link endpoint -> (node or group id, is_note)."""
        quoted = raw.startswith('"')
        name = _text(raw) if quoted else raw
        if name in self.notes:
            return self.notes[name].id, True
        node = self.find(name, quoted)
        if node is not None:
            return node.id, False
        if not quoted:
            cur = self._cur_pkg()
            for path in (name, f"{cur}.{name}" if cur else None):
                if path and path in self.pkgs:
                    return self.pkgs[path].id, False
        return self.cls(name, quoted).id, False

    # ---- statements -------------------------------------------------------
    def parse(self) -> Diagram:
        lines = self.src.lines
        i = 0
        while i < len(lines):
            line = lines[i]
            i += 1
            i = self.statement(line, lines, i)
        self.apply_hides()
        if not self.b.nodes and not self.b.groups:
            raise PlantUMLError("empty class diagram")
        return self.b.diagram(self.src, "TB")

    def statement(self, line: str, lines: list[str], i: int) -> int:
        if "[[" in line:  # hyperlinks on declarations and links are not drawn as text
            line = _URL_TOKEN_RE.sub(" ", line).strip()
        low = line.lower()
        if line.startswith("}"):
            if self.stack:
                self.stack.pop()
            return i
        if re.match(r"^end\s*(?:package|namespace|folder|frame|rectangle|node|cloud|database)\b", low):
            if self.stack:
                self.stack.pop()
            return i
        note = _parse_note(line)
        if note is not None:
            self.note(note)
            return i
        hm = re.match(r"^(hide|remove|show|restore)\s+(.*)$", line, re.I)
        if hm:
            if hm.group(1).lower() in ("hide", "remove"):
                self.hides.append((hm.group(1).lower(), hm.group(2).strip()))
            return i
        sm = re.match(r"^set\s+(?:namespace)?separator\s+(\S+)", line, re.I)
        if sm:
            self.sep = None if sm.group(1).lower() == "none" else sm.group(1)
            return i
        dm = _C_DECL_RE.match(line)
        if dm:
            got = self.declare(dm.group("kw").lower(), dm.group("rest"), lines, i)
            if got is not None:
                return got
        pm = _C_PKG_RE.match(line)
        if pm and not _C_LINK_RE.match(line) and not pm.group("rest").startswith(":"):
            return self.container(pm.group("kw").lower(), pm.group("rest"), lines, i)
        mm = _C_MEMBER_RE.match(line)  # before links: 'Foo.java : x' is not the link 'Foo . java'
        lm = None if mm else _C_LINK_RE.match(line)
        if lm:
            self.link(lm)
            return i
        if mm:
            node = self.cls(_text(mm.group("a")) if mm.group("a").startswith('"') else mm.group("a"),
                            quoted=mm.group("a").startswith('"'))
            mem = _class_member(mm.group("m"), node.kind == "enum")
            if mem is not None:
                node.members.append(mem)
            return i
        lp = _C_LOLLIPOP_RE.match(line)
        if lp:  # lollipop interface link: drawn as a plain line between the two
            a, b = self.ref(lp.group("a"))[0], self.ref(lp.group("b"))[0]
            self.b.edge(a, b, "assoc", label=_direction_free_label(lp.group("label")))
            return i
        st = _C_STEREO_RE.match(line)
        if st:
            node = self.cls(_text(st.group("a")) if st.group("a").startswith('"') else st.group("a"),
                            quoted=st.group("a").startswith('"'))
            stereos = [_text(x) for x in re.findall(r"<<\s*(.*?)\s*>>", st.group("s"))]
            node.stereotype = ", ".join(x for x in stereos if x) or node.stereotype
            return i
        dm = re.match(rf"^(?:\(\)|<>|circle\b|diamond\b)\s*(?P<n>{_QSTR}|{_C_DNAME})"
                      rf"(?:\s+(?i:as)\s+(?P<a>{_C_NAME}))?(?:\s+#\S+)*$", line, re.I)
        if dm:  # () lollipop interface / <> n-ary association diamond
            raw = dm.group("n")
            text = _text(raw)
            node = self.cls(dm.group("a") or text, quoted=dm.group("a") is None, display=text)
            circle = line.startswith("()") or line.lower().startswith("circle")
            node.kind = "interface" if circle else "other"
            node.attrs["keyword"] = "circle" if circle else "diamond"
            return i
        raise PlantUMLError(f"class diagram: cannot parse line: {line[:80]!r}")

    def declare(self, kw: str, rest: str, lines: list[str], i: int) -> Optional[int]:
        kw = re.sub(r"\s+", " ", kw)
        if kw == "json":
            raise UnsupportedPlantUML("json objects in class diagrams")
        pre = re.match(rf"^({_QSTR}|\S+)\s*((?:<<[^>]*>>\s*)+)(\s+(?i:as)\s.*)$", rest.strip())
        if pre:  # stereotype written before the alias: class "X" <<S>> as Y
            rest = f"{pre.group(1)}{pre.group(3)} {pre.group(2)}"
        parsed = _class_decl_name(rest)
        if parsed is None:
            return None
        key, display, quoted, generic, tail = parsed
        stereos, extends, implements, brace, inline, ok = _class_tail(tail)
        if not ok:
            return None
        self.has_kw = True
        node = self.cls(key, quoted=quoted, display=display, declare=True)
        node.text = (display if display is not None else self._short(key)) + generic
        kind = "abstract" if kw.startswith("abstract") else _C_KIND.get(kw, "class")
        node.kind = kind
        if kw in ("circle", "diamond"):
            node.attrs["keyword"] = kw
        elif kw not in _C_KIND and kw != "abstract class":
            if kw == "entity":
                stereos = stereos or ["entity"]
            else:
                node.attrs["keyword"] = kw
        if stereos:
            node.stereotype = ", ".join(stereos)
        for parent in extends:
            pid, _ = self.ref(parent)
            self.b.edge(node.id, pid, "inherit")
        for iface in implements:
            pid, _ = self.ref(iface)
            self.b.edge(node.id, pid, "realize")
        is_enum = kind == "enum"
        if inline:
            mem = _class_member(inline, is_enum)
            if mem is not None:
                node.members.append(mem)
        if brace == "{":
            while i < len(lines):
                ln = lines[i]
                i += 1
                if ln.startswith("}"):
                    break
                if kw == "map":
                    mem = Member("attr", _text(ln, creole=False)) if ln.strip() else None
                else:
                    mem = _class_member(ln, is_enum)
                if mem is not None:
                    node.members.append(mem)
        return i

    def container(self, kw: str, rest: str, lines: list[str], i: int) -> int:
        rest = rest.strip()
        if kw == "together":
            if rest.endswith("{"):
                self.stack.append(None)
            return i
        brace = ""
        if rest.endswith("{"):
            brace, rest = "{", rest[:-1].strip()
        elif re.search(r"\{\s*\}$", rest):
            brace, rest = "{}", re.sub(r"\{\s*\}$", "", rest).strip()
        rest = re.sub(r"<<[^>]*>>|\[\[[^\]]*\]\]|(?:^|\s)#\S+", " ", rest).strip()
        m = re.match(rf"^(?:({_QSTR})(?:\s+(?i:as)\s+([\w.$@:/\-]+))?"
                     rf"|([^\s\"]+)(?:\s+(?i:as)\s+({_QSTR}|[\w.$@:/\-]+))?)\s*$", rest)
        if not rest and brace:
            m = None
            key, text, split = f"{kw} {len(self.pkgs) + 1}", "", False
        elif not m:
            raise PlantUMLError(f"class diagram: bad {kw}: {rest[:60]!r}")
        elif m.group(1):
            text, key, split = _text(m.group(1)), m.group(2) or _text(m.group(1)), False
        else:
            key = m.group(3)
            text = _text(m.group(4)) if m.group(4) else None
            split = m.group(4) is None
        is_pkg = kw in ("package", "namespace")
        if not brace and not is_pkg:  # an allowmixing element (node, database, ...) in a class diagram
            node = self.cls(key, quoted=True, display=text or key, declare=True)
            node.kind = "other"
            node.attrs["keyword"] = kw
            return i
        gkind = "namespace" if kw == "namespace" else "package"
        segs = key.split(self.sep) if (split and self.sep and self.sep in key) else [key]
        path, _g = self._ensure_pkg(segs, self._cur_pkg(), gkind, text)
        if brace == "{":
            self.stack.append(path)
        elif not brace and is_pkg:
            end = _find(lines, i, re.compile(rf"^end\s*{kw}\b", re.I))
            if end >= 0:
                self.stack.append(path)
        return i

    def note(self, spec: _NoteSpec) -> None:
        node = self.b.node("note", spec.text)
        self._place(node.id)
        if spec.alias:
            self.notes[spec.alias] = node
        for raw in spec.targets:
            tid, _ = self.ref(raw.strip())
            self.b.edge(node.id, tid, "anchor")

    def link(self, m: re.Match) -> None:
        dashed, style, hidden = _link_style(m.group("body"))
        if _tail_dashed(m):
            dashed, style = True, "dashed"
        ra, rb = m.group("a"), m.group("b")
        if ra.startswith("(") or rb.startswith("("):
            ga = _split_args(ra.strip()[1:-1]) if ra.startswith("(") else [ra]
            gb = _split_args(rb.strip()[1:-1]) if rb.startswith("(") else [rb]
            if not hidden and not m.group("lh") and not m.group("rh") and dashed and len(ga) + len(gb) == 3:
                # association class (A, B) .. C: the A-B association; C's dashed line ends on it
                pair = ga if len(ga) == 2 else gb
                for raw in ga + gb:
                    self.ref(raw)
                self.b.edge(self.ref(pair[0])[0], self.ref(pair[1])[0], "assoc")
                return
            for x in ga:
                for y in gb:
                    self.link_one(m, x, y, dashed, style, hidden)
            return
        self.link_one(m, ra, rb, dashed, style, hidden)

    def link_one(self, m: re.Match, ra: str, rb: str, dashed: bool, style: Optional[str],
                 hidden: bool) -> None:
        a, a_note = self.ref(ra)
        b, b_note = self.ref(rb)
        if hidden:
            return
        if a_note or b_note:
            src, tgt = (a, b) if a_note else (b, a)
            self.b.edge(src, tgt, "anchor")
            return
        kind, rev = _class_edge_kind(m.group("lh"), m.group("rh"), dashed)
        sl = _text(m.group("ca")) or None if m.group("ca") else _CROW.get(m.group("lh") or "")
        tl = _text(m.group("cb")) or None if m.group("cb") else _CROW.get(m.group("rh") or "")
        if rev:
            a, b, sl, tl = b, a, tl, sl
        if kind in ("realize", "depend") and style == "dashed":
            style = None  # dashed is part of the notation
        self.b.edge(a, b, kind, label=_direction_free_label(m.group("label")),
                    source_label=sl, target_label=tl, style=style)

    def apply_hides(self) -> None:
        drop: set[str] = set()
        linked = {e.source for e in self.b.edges if e.kind != "anchor"} | \
            {e.target for e in self.b.edges if e.kind != "anchor"}
        classes = [n for n in self.b.nodes if n.kind != "note"]
        compartments = {"members": ("attr", "op", "lit"), "member": ("attr", "op", "lit"),
                        "methods": ("op",), "method": ("op",), "fields": ("attr", "lit"),
                        "field": ("attr", "lit"), "attributes": ("attr", "lit"), "attribute": ("attr", "lit")}
        for _verb, arg in self.hides:
            words = arg.split()
            low = arg.lower()
            if low.startswith("empty") or not words:
                continue
            if low in ("stereotype", "stereotypes"):
                for n in classes:
                    n.stereotype = None
                continue
            if low in ("@unlinked", "unlinked"):
                drop |= {n.id for n in classes if n.id not in linked}
                continue
            hidden = compartments.get(words[-1].lower())
            if hidden is not None:  # hide [target] members|methods|fields
                target = arg[: len(arg) - len(words[-1])].strip()
                for n in self._select(target) if target else classes:
                    n.members = [mm for mm in n.members if mm.kind not in hidden]
                continue
            if len(words) == 1 or (low.startswith("<<") and low.endswith(">>")):
                drop |= {n.id for n in self._select(arg)}
        self.b.drop_nodes(drop)

    def _select(self, target: str) -> list[Node]:
        target = target.strip()
        sm = re.match(r"^<<\s*(.*?)\s*>>$", target)
        if sm:
            return [n for n in self.b.nodes if n.stereotype and sm.group(1) in n.stereotype.split(", ")]
        if target.lower() in ("class", "interface", "enum", "abstract"):
            return [n for n in self.b.nodes if n.kind == target.lower()]
        node = self.find(_text(target) if target.startswith('"') else target, target.startswith('"'))
        return [node] if node is not None else []


def _generic(tail: str) -> tuple[str, str]:
    """Leading `<...>` type parameters (not a <<stereotype>>) -> (generic, rest)."""
    m = re.match(r"^\s*<(?!<)", tail)
    if not m:
        return "", tail
    s = tail[m.end() - 1:]
    depth = 0
    for k, c in enumerate(s):
        depth += (c == "<") - (c == ">")
        if depth == 0:
            return re.sub(r"\s+", "", s[:k + 1]), s[k + 1:]
    return "", tail


def _class_decl_name(rest: str) -> Optional[tuple[str, Optional[str], bool, str, str]]:
    """Declaration name part -> (key, display or None, quoted key, generic, tail)."""
    rest = re.sub(r'""((?:[^"\\]|\\.)*)""', r'"\1"', rest.strip())  # ""mono"" creole inside quotes
    m = re.match(rf"^({_QSTR})(.*)$", rest)
    if m:
        display = _text(m.group(1))
        gen, tail = _generic(m.group(2))
        a = re.match(rf"^\s*as\s+({_C_DNAME})(.*)$", tail)
        if a:
            gen2, tail = _generic(a.group(2))
            return a.group(1), display, False, gen or gen2, tail
        return display, display, True, gen, tail
    m = re.match(rf"^({_C_DNAME})(.*)$", rest)
    if not m:
        return None
    name = m.group(1)
    gen, tail = _generic(m.group(2))
    a = re.match(rf"^\s+as\s+(?:({_QSTR})|({_C_DNAME}))(.*)$", tail)
    if a:
        gen2, tail = _generic(a.group(3))
        if a.group(1):
            return name, _text(a.group(1)), False, gen or gen2, tail
        return a.group(2), _text(name, creole=False), False, gen or gen2, tail
    display = _text(name, creole=False) if "\\" in name else None
    return name, display, False, gen, tail


def _class_tail(tail: str) -> tuple[list[str], list[str], list[str], str, str, bool]:
    """-> (stereotypes, extends, implements, brace "" | "{" | "{}", inline body, ok)."""
    t = tail
    stereos = []
    for s in re.findall(r"<<(.*?)>>", t):
        s = re.sub(r"^\s*\(\s*\S\s*,\s*[^)]*\)", "", s).strip()  # spot: << (S,#FF7700) Name >>
        if s:
            stereos.append(_text(s))
    t = re.sub(r"<<.*?>>", " ", t)
    t = _URL_RE.sub(" ", t)
    brace, inline = "", ""
    s = t.rstrip()
    if s.endswith("{"):
        brace, t = "{", s[:-1]
    else:
        im = re.search(r"\{(.*)\}\s*$", s)
        if im and not re.match(r"^\s*(?:static|abstract)\s*$", im.group(1)):
            brace, inline, t = "{}", im.group(1).strip(), s[:im.start()]
    extends, implements = [], []
    for kw, dest in (("extends", extends), ("implements", implements)):
        km = re.search(rf"\b{kw}\s+(.+?)(?=\s+(?:extends|implements)\b|$)", t, re.I)
        if km:
            dest.extend(re.sub(r"\s*<.*>", "", x).strip() for x in _split_generic_list(km.group(1)))
            t = t[:km.start()] + t[km.end():]
    t = re.sub(r"(?:^|\s)##?\S*|\$\w+", " ", t)
    return stereos, extends, implements, brace, inline, not re.search(r"\w", t)


def _split_generic_list(s: str) -> list[str]:
    out, buf, depth = [], [], 0
    for c in s:
        depth += (c == "<") - (c == ">")
        if c == "," and depth == 0:
            out.append("".join(buf))
            buf = []
        else:
            buf.append(c)
    out.append("".join(buf))
    return [x.strip() for x in out if x.strip()]


def _class_member(line: str, enum: bool) -> Optional[Member]:
    text = line.strip()
    if not text or _SEPARATOR_RE.match(text):
        return None
    kind = None
    low = text.lower()
    if "{field}" in low:
        kind = "attr"
    elif "{method}" in low:
        kind = "op"
    text = _MODIFIERS_RE.sub("", text).strip()
    text = _text(text, creole=False) if ("<" in text or "\\n" in text or "&" in text) else text
    if not text:
        return None
    if kind is None:
        if "(" in text:
            kind = "op"
        elif enum and not re.match(r"^[-+#~]", text) and ":" not in text:
            kind = "lit"
        else:
            kind = "attr"
    return Member(kind, text)


# --------------------------------------------------------------------------- #
# Sequence diagrams
# --------------------------------------------------------------------------- #
_S_KINDS = ("participant", "actor", "boundary", "control", "entity", "database", "collections", "queue")
_S_DECL_RE = re.compile(rf"^(?P<create>create\s+)?(?P<kw>{'|'.join(_S_KINDS)})\b\s*(?P<rest>.*)$", re.I)
_S_NAME = r"[\w$@.]+"
_S_EP = rf"(?:{_QSTR}|{_S_NAME})"
_S_DNAME = r"[^\s\"<>#{}\[\]:,]*\w[^\s\"<>#{}\[\]:,]*"  # participant names in declarations
_S_MSG_RE = re.compile(
    rf"^(?:&\s*)?(?P<a>{_S_EP}|[\[\]?])?\s*"
    r"(?P<lh>[oxX]?(?:<<|<|\\\\|\\|//|/)|[oxX](?=[-\[]))?"
    r"(?P<body>(?:\[[^\]]*\])?-+(?:\[[^\]]*\]-*)?)"
    r"(?P<rh>(?:>>|>|\\\\|\\|//|/)[oxX]?|[oxX](?![\w$@.]))?"
    r"\s*(?:\(\d+\)\s*)?"
    rf"(?P<b>{_S_EP}|[\[\]?])?(?:\s+(?i:as)\s+(?P<balias>{_S_EP}))?\s*"
    r"(?P<mods>(?:(?:\+\+|--|\*\*|!!)\s*)*)(?:#\w+\s*)?"
    r"(?::\s*(?P<label>.*))?$"
)
_S_FRAGMENTS = {"alt": "alt", "opt": "opt", "loop": "loop", "par": "par", "par2": "par", "break": "break",
                "critical": "critical", "group": "group", "neg": "neg", "strict": "group",
                "seq": "group", "ignore": "group", "consider": "group", "assert": "group"}
_S_FRAG_RE = re.compile(rf"^(?P<kw>{'|'.join(sorted(_S_FRAGMENTS, key=len, reverse=True))})\b(?P<rest>.*)$",
                        re.I)
_S_IGNORE_RE = re.compile(
    r"^(?:autonumber\b|==.*==$|\.\.\..*|\|\|\|$|\|\|\d+\|\|$|delay\b|space\b|ignore\s+newpage$"
    r"|autoactivate\b|hide|show\b|skinparam\b|\{[^}]*\}\s*<->\s*\{[^}]*\}|@+(?:start|end)\w*\s*$)",
    re.I,
)


@dataclass
class _Frame:
    kind: str
    guards: list[str] = field(default_factory=lambda: [""])
    items: list[list] = field(default_factory=lambda: [[]])  # per operand: edge indexes / frames
    nodes: list[str] = field(default_factory=list)  # ref: covered participants


class _SequenceParser:
    def __init__(self, src: Source):
        self.src = src
        self.b = _Build("sequence")
        self.parts: dict[str, Node] = {}
        self.notes: list[Node] = []
        self.root = _Frame("root")
        self.frames = [self.root]
        self.box: Optional[Group] = None
        self.activations: list[tuple[Optional[str], str]] = []  # (caller, callee) participant ids
        self.autoactivate = False
        self.pending_create: set[str] = set()
        self.last_msg: Optional[Edge] = None
        self.hide_unlinked = False
        self.skipped_boundary = 0

    def part(self, raw: str, kind: str = "participant", display: Optional[str] = None,
             declared: bool = False) -> Node:
        key = _text(raw) if raw.startswith('"') else raw
        node = self.parts.get(key)
        if node is None:
            node = self.b.node(kind, display if display is not None else
                               (_text(raw) if raw.startswith('"') else _text(raw, creole=False)))
            self.parts[key] = node
            if declared and self.box is not None:
                self.box.members.append(node.id)
        elif declared:
            node.kind = kind
            if display is not None:
                node.text = display
        return node

    def parse(self) -> Diagram:
        lines = self.src.lines
        i = 0
        while i < len(lines):
            line = lines[i]
            i += 1
            i = self.statement(line, lines, i)
        while len(self.frames) > 1:  # unclosed fragments end with the diagram
            self.frames.pop()
        self.emit(self.root, None)
        if self.hide_unlinked:
            linked = {x for e in self.b.edges if e.kind != "anchor" for x in (e.source, e.target)}
            self.b.drop_nodes({n.id for n in self.parts.values() if n.id not in linked})
        # participants first (they head the diagram), then notes in order
        notes = [n for n in self.b.nodes if n.kind == "note"]
        self.b.nodes = [n for n in self.b.nodes if n.kind != "note"] + notes
        if not self.parts:
            raise PlantUMLError("empty sequence diagram")
        return self.b.diagram(self.src)

    def statement(self, line: str, lines: list[str], i: int) -> int:
        if line.startswith("{") and not _S_IGNORE_RE.match(line):  # teoz anchor: {name} A -> B
            line = re.sub(r"^\{[^}]*\}\s*", "", line)
        line = re.sub(r"^/\s+(?=[hr]?note\b)", "", line)  # teoz: note on the same line as the message
        low = line.lower()
        if _S_IGNORE_RE.match(line):
            if low.startswith("autoactivate"):
                self.autoactivate = "on" in low
            elif low == "hide unlinked":
                self.hide_unlinked = True
            return i
        note = _parse_note(line)
        if note is not None:
            self.note(note)
            return i
        am = re.match(rf"^(activate|deactivate)\b\s*({_S_EP})?\s*(?:#\S+\s*)*$", line, re.I) or \
            re.match(rf"^()({_S_EP})\s*(\+\+|--)\s*(?:#\S+\s*)*$", line)
        if am:
            verb = am.group(1).lower() or ("activate" if am.group(3) == "++" else "deactivate")
            if am.group(2) is None:  # bare deactivate: the latest activation
                if verb == "deactivate" and self.activations:
                    self.activations.pop()
                return i
            node = self.part(am.group(2))
            if verb == "activate":
                caller = None
                if self.last_msg is not None and self.last_msg.target == node.id:
                    caller = self.last_msg.source
                self.activations.append((caller, node.id))
            else:
                self.deactivate(node.id)
            return i
        dm = _S_DECL_RE.match(line)
        if dm and not _S_MSG_RE.match(line):
            return self.declare(dm.group("kw"), dm.group("rest"), lines, i, create=bool(dm.group("create")))
        rm = re.match(r"^return\b\s*(.*)$", line, re.I)
        if rm:  # back to the caller of the latest activation, else of the latest message (as PlantUML)
            caller: Optional[str] = None
            if self.activations:
                caller, callee = self.activations.pop()
            if caller is None and self.last_msg is not None:
                caller, callee = self.last_msg.source, self.last_msg.target
            if caller is not None:
                self.message(callee, caller, "reply", _text(rm.group(1)))
            return i
        cm = re.match(rf"^create\s+(?:(?P<kw>{'|'.join(_S_KINDS)})\s+)?(?P<rest>.+)$", line, re.I)
        if cm:
            return self.declare(cm.group("kw") or "participant", cm.group("rest"), lines, i, create=True)
        xm = re.match(rf"^destroy\s+({_S_EP})\s*$", line, re.I)
        if xm:
            node = self.part(xm.group(1))
            if self.last_msg is not None and self.last_msg.target == node.id and \
                    self.last_msg.kind in ("sync", "async", "create"):
                self.last_msg.kind = "destroy"
            return i
        bm = re.match(r"^box\b\s*(.*)$", line, re.I)
        if bm:
            title = re.sub(r"(?:^|\s)#\S+", " ", bm.group(1)).strip()
            self.box = self.b.group("box", _text(title))
            return i
        if re.match(r"^end\s*box$", low):
            self.box = None
            return i
        rf = re.match(r"^ref\s*(?:#\w+\s*)?over\s+(.*)$", line, re.I)
        if rf:
            sp = _split_colon(rf.group(1))
            who, text = (sp[0], sp[1]) if sp else (rf.group(1), "")
            who = re.sub(r"(?:^|\s)#\S+", " ", who)
            frame = _Frame("ref", guards=[_text(text)])
            frame.nodes = [self.part(w).id for w in _split_args(who) if w]
            self.frames[-1].items[-1].append(frame)
            return i
        fm = _S_FRAG_RE.match(line)
        if fm and not _S_MSG_RE.match(line):
            kind = _S_FRAGMENTS[fm.group("kw").lower()]
            text = re.sub(r"^\s*(?:#\w+\s*)+", "", fm.group("rest")).strip()
            guard = _text(text) if kind == "group" else _guard(text)
            frame = _Frame(kind, guards=[guard])
            self.frames[-1].items[-1].append(frame)
            self.frames.append(frame)
            return i
        em = re.match(r"^else\b\s*(.*)$", line, re.I)
        if em:
            if len(self.frames) == 1:
                raise PlantUMLError("sequence diagram: else outside a fragment")
            frame = self.frames[-1]
            text = re.sub(r"^\s*(?:#\w+\s*)+", "", em.group(1)).strip()
            frame.guards.append(_text(text) if frame.kind == "group" else _guard(text))
            frame.items.append([])
            return i
        if re.match(r"^end\b(?!\s*(?:box|note|ref)\b)", low):
            if len(self.frames) == 1:
                raise PlantUMLError("sequence diagram: end outside a fragment")
            self.frames.pop()
            return i
        mm = _S_MSG_RE.match(_URL_RE.sub("", line))
        if mm and (mm.group("a") or mm.group("b")) and (mm.group("lh") or mm.group("rh")):
            self.arrow(mm)
            return i
        raise PlantUMLError(f"sequence diagram: cannot parse line: {line[:80]!r}")

    def declare(self, kind: str, rest: str, lines: list[str], i: int, create: bool = False) -> int:
        kind = kind.lower()
        rest = rest.strip()
        multiline = rest.endswith("[")
        if multiline:
            rest = rest[:-1].strip()
        na = _name_alias(rest, _S_DNAME)
        if na is None:
            raise PlantUMLError(f"sequence diagram: bad participant: {rest[:60]!r}")
        display, code, tail = na
        stereos = [_text(s) for s in re.findall(r"<<\s*(.*?)\s*>>", tail)]
        tail = re.sub(r"<<.*?>>|\[\[[^\]]*\]\]|(?:^|\s)#\S+|(?i:\border\s+-?\d+)", " ", tail)
        if re.search(r"\w", tail):
            raise PlantUMLError(f"sequence diagram: bad participant: {rest[:60]!r}")
        if multiline:
            j = _find(lines, i, re.compile(r"^\]"))
            if j >= 0:
                body = [re.sub(r"^=+\s*", "", ln) for ln in lines[i:j] if not re.match(r"^-{3,}$", ln)]
                display = _text("\\n".join(body)) or display
                i = j + 1
        node = self.part(code, kind, display=display, declared=True)
        if stereos:
            node.stereotype = ", ".join(s for s in stereos if s) or None
        if create:
            self.pending_create.add(node.id)
        return i

    def deactivate(self, pid: str) -> None:
        for k in range(len(self.activations) - 1, -1, -1):
            if self.activations[k][1] == pid:
                del self.activations[k]
                return

    def message(self, src: str, tgt: str, kind: str, label: str = "") -> Edge:
        e = self.b.edge(src, tgt, kind, label=label)
        self.frames[-1].items[-1].append(len(self.b.edges) - 1)
        self.last_msg = e
        return e

    def arrow(self, m: re.Match) -> None:
        body = m.group("body")
        lh, rh = m.group("lh") or "", m.group("rh") or ""
        a, b = m.group("a"), m.group("b")
        opts = " ".join(re.findall(r"\[([^\]]*)\]", body)).lower()
        if "hidden" in opts:
            return
        if not a or not b or a in "[]?" or b in "[]?":  # boundary message: no gate node in the IR
            for raw in (a, b):
                if raw and raw not in "[]?":
                    self.part(raw)
            self.skipped_boundary += 1
            return
        alias = m.group("balias")
        if alias:  # A -> "Long name" as L  declares L on the fly
            if alias.startswith('"'):
                b, alias = alias, b
            tgt_node = self.part(alias, display=_text(b) if b.startswith('"') else b)
            src, tgt = self.part(a).id, tgt_node.id
        else:
            src, tgt = self.part(a).id, self.part(b).id
        if lh and not rh:
            src, tgt = tgt, src
        heads = (lh + rh).lower()
        plain = re.sub(r"\[[^\]]*\]", "", body)
        mods = m.group("mods") or ""
        if "x" in heads or "!!" in mods:
            kind = "destroy"
        elif plain.count("-") >= 2 or "dashed" in opts or "dotted" in opts:
            kind = "reply"
        elif any(h in heads for h in (">>", "<<", "\\", "/")):
            kind = "async"
        else:
            kind = "sync"
        if kind in ("sync", "async") and ("**" in mods or tgt in self.pending_create):
            kind = "create"
        self.pending_create.discard(tgt)
        self.message(src, tgt, kind, _text(m.group("label")))
        if "++" in mods or (self.autoactivate and kind in ("sync", "async", "create") and src != tgt):
            self.activations.append((src, tgt))
        if re.search(r"--(?!\+)", mods):
            self.deactivate(src)
        if self.autoactivate and kind == "reply" and self.activations and \
                self.activations[-1] == (tgt, src):
            self.activations.pop()

    def note(self, spec: _NoteSpec) -> None:
        node = self.b.node("note", spec.text)
        self.notes.append(node)
        targets = [self.part(t).id for t in spec.targets]
        if not targets and spec.pos in ("left", "right") and self.last_msg is not None:
            ends = [self.last_msg.source, self.last_msg.target]
            order = [n.id for n in self.b.nodes]
            ends.sort(key=order.index)
            targets = [ends[0] if spec.pos == "left" else ends[-1]]
        for t in targets:
            self.b.edge(node.id, t, "anchor")

    def emit(self, frame: _Frame, parent: Optional[str]) -> None:
        if frame.kind == "ref":
            g = self.b.group("ref", frame.guards[0], parent=parent)
            g.members.extend(frame.nodes)
            return
        if frame.kind == "root":
            for item in frame.items[0]:
                if isinstance(item, _Frame):
                    self.emit(item, None)
            return
        if len(frame.items) == 1:
            holders = [self.b.group(frame.kind, frame.guards[0], parent=parent)]
        else:
            top = self.b.group(frame.kind, "", parent=parent)
            holders = [self.b.group("operand", g, parent=top.id) for g in frame.guards]
        for holder, items in zip(holders, frame.items, strict=True):
            for item in items:
                if isinstance(item, _Frame):
                    self.emit(item, holder.id)
                else:
                    holder.members.append(f"e{item + 1}")


def _name_alias(rest: str, name_re: str) -> Optional[tuple[str, str, str]]:
    """`"Display" as code` / `code as "Display"` / `Display as code` / `code` -> (display, code, tail)."""
    m = re.match(rf"^({_QSTR})(?:\s+(?i:as)\s+({name_re}))?(.*)$", rest)
    if m:
        return _text(m.group(1)), m.group(2) or _text(m.group(1)), m.group(3)
    m = re.match(rf"^({name_re})(?:\s+(?i:as)\s+(?:({_QSTR})|({name_re})))?(?![\w$@.])(.*)$", rest)
    if not m:
        return None
    if m.group(2):
        return _text(m.group(2)), m.group(1), m.group(4)
    if m.group(3):
        return _text(m.group(1), creole=False), m.group(3), m.group(4)
    return _text(m.group(1), creole=False), m.group(1), m.group(4)


# --------------------------------------------------------------------------- #
# Activity diagrams (new syntax)
# --------------------------------------------------------------------------- #
_A_IF_RE = re.compile(
    r"^if\s*\((?P<c>.*?)\)\s*(?:(?:is|equals?)\s*\((?P<v>.*?)\)\s*)?(?:then\b\s*(?:\((?P<l>.*)\))?)?\s*;?$",
    re.I)
_A_ELSEIF_RE = re.compile(
    r"^(?:\((?P<p>[^)]*)\)\s*)?else\s*if\s*\((?P<c>.*?)\)\s*(?:(?:is|equals?)\s*\((?P<v>.*?)\)\s*)?"
    r"(?:then\b\s*(?:\((?P<l>.*)\))?)?\s*;?$",
    re.I)
_A_ELSE_RE = re.compile(r"^else\b\s*(?:\((?P<l>.*)\))?\s*;?$", re.I)
_A_WHILE_RE = re.compile(
    r"^while\s*\((?P<c>.*?)\)\s*(?:is\s*\((?P<l>.*?)\))?\s*(?:not\s*\((?P<n>.*?)\))?\s*;?$", re.I)
_A_ENDWHILE_RE = re.compile(r"^end\s*while\b\s*(?:\((?P<l>.*)\))?\s*;?$", re.I)
_A_REPEAT_WHILE_RE = re.compile(
    r"^repeat\s*while\b\s*(?:\((?P<c>.*?)\))?\s*(?:is\s*\((?P<l>.*?)\))?\s*(?:not\s*\((?P<n>.*?)\))?"
    r"\s*(?:->\s*(?P<a>[^;]*))?;?$",
    re.I)
_A_LANE_RE = re.compile(r"^\|(?P<a>[^|]+)\|(?:(?P<b>[^|]+)\|)?\s*(?P<rest>.*)$")
_A_ARROW_RE = re.compile(r"^-(?P<body>[^>]*)>\s*(?P<label>.*?)\s*;?\s*$")
_A_REGION_RE = re.compile(r"^(?P<kw>partition|group|package|rectangle|card)\b\s*(?P<rest>.*?)\s*$", re.I)
_A_STEREO = {"acceptevent": "receive_signal", "sendsignal": "send_signal", "timeevent": "time_event",
             "object": "object", "objectsignal": "send_signal", "input": "action", "output": "action",
             "procedure": "action", "load": "action", "save": "action", "continuous": "action",
             "task": "action", "trigger": "action"}
_A_ENDINGS = (";", "\\\\", "|", "<", ">", "/", "]", "}")


def _action_end(text: str) -> Optional[tuple[str, str, Optional[str]]]:
    """If this line closes an action, -> (body, ending, stereotype). PlantUML's rules for the SDL
    endings: `>` not after `>` or a short html tag; `/ < } ]` not after one of `/|<>}]`; `|` only
    when it is the only `|` on the line."""
    st = None
    sm = re.search(r"\s*<<\s*(\w+)\s*>>\s*$", text)
    if sm and sm.start() > 0 and text[:sm.start()].rstrip().endswith(_A_ENDINGS):
        st, text = sm.group(1), text[:sm.start()].rstrip()
    t = text.rstrip()
    if not t:
        return None
    end, body = t[-1], t[:-1]
    if end == ";":
        return body, ";", st
    if t.endswith("\\\\"):
        return t[:-2], "\\\\", st
    if end == ">":
        if body.endswith(">") or re.search(r"</?\w{1,5}$|<img[^>]*$|<[&$][\w-]+$", body):
            return None
        return body, ">", st
    if end in "/<}]":
        return None if body[-1:] in ("/", "|", "<", ">", "}", "]") else (body, end, st)
    if end == "|":
        return None if "|" in body else (body, "|", st)
    return None


_A_COND_START_RE = re.compile(
    r"^(?:\([^)]*\)\s*)?(?:if|else\s*if|elseif|while|repeat\s*while|switch|case)\s*\(", re.I)
_A_STMT_START_RE = re.compile(
    r"^(?::|\||-|#\w+:|\(|\}|(?:start|stop|end|if|else|elseif|endif|while|endwhile|repeat|fork|split"
    r"|switch|case|note|floating|partition|group|detach|kill|break|backward)\b)", re.I)


def _paren_depth(s: str) -> int:
    depth, quoted = 0, False
    for c in s:
        if c == '"':
            quoted = not quoted
        elif not quoted:
            depth += (c == "(") - (c == ")")
    return depth


class _ActivityParser:
    def __init__(self, src: Source):
        self.src = src
        self.b = _Build("activity")
        self.tails: list[tuple[str, str]] = []  # open flows: (node id, label of the next edge)
        self.arrow_label: Optional[str] = None
        self.hidden = False
        self.lane: Optional[Group] = None
        self.lanes: dict[str, Group] = {}
        self.regions: list[Group] = []
        self.ctx: list[dict] = []
        self.last: Optional[str] = None
        self.labels: dict[str, str] = {}
        self.pending_labels: list[str] = []
        self.gotos: list[tuple[list[tuple[str, str]], str]] = []

    def new(self, kind: str, text: str = "") -> str:
        nid = self.b.node(kind, text).id
        if self.lane is not None:
            self.lane.members.append(nid)
        if self.regions:
            self.regions[-1].members.append(nid)
        return nid

    def enter(self, nid: str) -> None:
        """Connect the open flows into `nid`, which becomes the only open flow."""
        for t, lbl in self.tails:
            if not self.hidden:
                self.b.edge(t, nid, "flow", label=lbl or self.arrow_label or "")
        self.arrow_label, self.hidden = None, False
        self.tails = [(nid, "")]
        self.last = nid
        for name in self.pending_labels:
            self.labels[name] = nid
        self.pending_labels = []

    def join(self, ends: list[tuple[str, str]]) -> None:
        """Flows meeting at endif / endswitch: a merge node only when two or more rejoin."""
        if len(ends) >= 2:
            m = self.new("merge")
            for t, lbl in ends:
                self.b.edge(t, m, "flow", label=lbl)
            self.tails = [(m, "")]
            self.last = m
        else:
            self.tails = list(ends)

    def top(self, typ: str, what: str) -> dict:
        if not self.ctx or self.ctx[-1]["type"] != typ:
            raise PlantUMLError(f"activity diagram: misplaced {what}")
        return self.ctx[-1]

    def parse(self) -> Diagram:
        lines = self.src.lines
        i = 0
        while i < len(lines):
            line = lines[i]
            i += 1
            i = self.statement(line, lines, i)
        while self.ctx:  # PlantUML closes the blocks still open at the end of the diagram
            self.close(self.ctx[-1]["type"])
        for tails, name in self.gotos:
            target = self.labels.get(name)
            if target is None:
                raise PlantUMLError(f"activity diagram: goto to unknown label {name!r}")
            for t, lbl in tails:
                self.b.edge(t, target, "flow", label=lbl)
        if not self.b.nodes:
            raise PlantUMLError("empty activity diagram")
        return self.b.diagram(self.src, "TB")

    # ---- block ends -------------------------------------------------------
    def close(self, typ: str, label: str = "", cond: str = "", no: str = "") -> None:
        c = self.ctx.pop()
        if typ != c["type"]:
            raise PlantUMLError(f"activity diagram: end of {typ} inside {c['type']}")
        if typ == "if":
            ends = c["ends"] + self.tails
            if not c["else"]:
                ends.append((c["decisions"][-1], ""))
            self.join(ends)
        elif typ == "while":
            for t, lbl in self.tails:
                self.b.edge(t, c["d"], "flow", label=lbl or self.arrow_label or "")
            self.arrow_label = None
            self.tails = [(c["d"], label or c["exit"])] + c["breaks"]
            self.last = c["d"]
        elif typ == "repeat":
            d = self.new("decision", cond)
            self.enter(d)
            if c["backward"] is not None:
                back = self.new(*c["backward"])
                self.b.edge(d, back, "flow", label=label)
                self.b.edge(back, c["top"], "flow")
            else:
                self.b.edge(d, c["top"], "flow", label=label)
            self.tails = [(d, no)] + c["breaks"]
        elif typ == "fork":
            ends = c["ends"] + self.tails
            if ends:
                j = self.new("merge" if label == "merge" else "join")
                for t, lbl in ends:
                    self.b.edge(t, j, "flow", label=lbl)
                self.tails = [(j, "")]
                self.last = j
        elif typ == "split":
            self.tails = c["ends"] + self.tails
        elif typ == "switch":
            self.join(c["ends"] + self.tails)

    # ---- statements -------------------------------------------------------
    def action(self, first: str, lines: list[str], i: int) -> tuple[tuple[str, str, Optional[str]], int]:
        """`first` is the text after ':'. -> ((text, ending, stereotype), next index)."""
        buf = [first]
        end = _action_end(first)
        while end is None and i < len(lines):
            buf.append(lines[i])
            i += 1
            end = _action_end(lines[i - 1])
        if end is None:
            raise PlantUMLError("activity diagram: unterminated action")
        return ("\\n".join(buf[:-1] + [end[0]]), end[1], end[2]), i

    def statement(self, line: str, lines: list[str], i: int) -> int:
        line = re.sub(r"^#[\w#]+(?:[/\\|-]#?\w+)?\s*:(?=\s*(?:if|while|repeat|switch)\b)", "", line)
        if _A_COND_START_RE.match(line):  # a condition may continue on the next lines
            while _paren_depth(line) > 0 and i < len(lines):
                line += "\\n" + lines[i]
                i += 1
        low = line.lower().rstrip(";").strip()
        if line.startswith(":") or re.match(r"^#[\w]+(?:[/\\|-]#?\w+)?\s*:", line):
            (text, ending, st), i = self.action(line[line.index(":") + 1:], lines, i)
            kind = {">": "send_signal", "<": "receive_signal"}.get(ending, "action")
            if st:
                kind = _A_STEREO.get(st.lower(), kind)
            self.enter(self.new(kind, _text(text)))
            return i
        if low == "start":
            self.enter(self.new("initial"))
            return i
        if low in ("stop", "end"):
            self.enter(self.new("final" if low == "stop" else "flow_final"))
            self.tails = []
            return i
        if low in ("kill", "detach"):
            self.tails = []
            return i
        note = _parse_note(line)
        if note is not None:
            node = self.new("note", note.text)
            if self.last is not None and not low.startswith("floating"):
                self.b.edge(node, self.last, "anchor")
            return i
        m = _A_ARROW_RE.match(line)
        if m:
            label = m.group("label")
            if label and not line.rstrip().endswith(";"):  # the label may continue on the next lines
                for j in range(i, min(i + 6, len(lines))):
                    if _A_STMT_START_RE.match(lines[j]):
                        break
                    if lines[j].endswith(";"):
                        label = "\\n".join([label] + lines[i:j] + [lines[j][:-1]])
                        i = j + 1
                        break
            if "hidden" in m.group("body").lower():
                self.hidden = True
            self.arrow_label = _text(label) or self.arrow_label
            return i
        m = _A_LANE_RE.match(line)
        if m:
            a, b = m.group("a").strip(), (m.group("b") or "").strip()
            name = b if (a.startswith(("#", "[#")) and b) else a
            rest = m.group("rest").strip()
            if name not in self.lanes:
                title = rest if rest and not rest.startswith(":") else name
                self.lanes[name] = self.b.group("lane", _text(title))
            self.lane = self.lanes[name]
            return self.statement(rest, lines, i) if rest.startswith(":") else i
        return self.control(line, low, lines, i)

    def control(self, line: str, low: str, lines: list[str], i: int) -> int:
        m = _A_IF_RE.match(line)
        if m:
            d = self.new("decision", _text(m.group("c")))
            self.enter(d)
            self.ctx.append({"type": "if", "decisions": [d], "ends": [], "else": False})
            self.tails = [(d, _text(m.group("l") or m.group("v") or ""))]
            return i
        m = _A_ELSEIF_RE.match(line)
        if m:
            c = self.top("if", "elseif")
            c["ends"].extend(self.tails)
            d = self.new("decision", _text(m.group("c")))
            self.b.edge(c["decisions"][-1], d, "flow", label=_text(m.group("p") or ""))
            c["decisions"].append(d)
            self.tails = [(d, _text(m.group("l") or m.group("v") or ""))]
            self.last = d
            return i
        m = _A_ELSE_RE.match(line)
        if m:
            c = self.top("if", "else")
            if c["else"]:
                raise PlantUMLError("activity diagram: second else")
            c["ends"].extend(self.tails)
            c["else"] = True
            self.tails = [(c["decisions"][-1], _text(m.group("l") or ""))]
            return i
        if re.match(r"^end\s*if$", low):
            self.top("if", "endif")
            self.close("if")
            return i
        m = _A_WHILE_RE.match(line)
        if m:
            d = self.new("decision", _text(m.group("c")))
            self.enter(d)
            self.ctx.append({"type": "while", "d": d, "breaks": [], "exit": _text(m.group("n") or "")})
            self.tails = [(d, _text(m.group("l") or ""))]
            return i
        m = _A_ENDWHILE_RE.match(line)
        if m:
            self.top("while", "endwhile")
            self.close("while", label=_text(m.group("l") or ""))
            return i
        m = _A_REPEAT_WHILE_RE.match(line)
        if m:
            self.top("repeat", "repeat while")
            self.close("repeat", label=_text(m.group("l") or ""), cond=_text(m.group("c") or ""),
                       no=_text(m.group("n") or m.group("a") or ""))
            return i
        m = re.match(r"^repeat\b\s*(?P<rest>.*)$", line, re.I)
        if m:
            rest = m.group("rest").strip()
            if rest.startswith(":"):  # `repeat :label;` draws the label box instead of the entry diamond
                (text, ending, _st), i = self.action(rest[1:], lines, i)
                top = self.new({">": "send_signal", "<": "receive_signal"}.get(ending, "action"), _text(text))
            else:
                top = self.new("merge")
            self.enter(top)
            self.ctx.append({"type": "repeat", "top": top, "breaks": [], "backward": None})
            return self.statement(rest, lines, i) if rest and not rest.startswith(":") else i
        m = re.match(r"^backward\s*:(?P<rest>.*)$", line, re.I)
        if m:
            c = next((x for x in reversed(self.ctx) if x["type"] == "repeat"), None)
            if c is None:
                raise PlantUMLError("activity diagram: backward outside repeat")
            (text, ending, _st), i = self.action(m.group("rest"), lines, i)
            c["backward"] = ({">": "send_signal", "<": "receive_signal"}.get(ending, "action"), _text(text))
            return i
        if low == "break":
            c = next((x for x in reversed(self.ctx) if x["type"] in ("while", "repeat")), None)
            if c is not None:
                c["breaks"].extend(self.tails)
            self.tails = []
            return i
        if low == "fork":
            f = self.new("fork")
            self.enter(f)
            self.ctx.append({"type": "fork", "f": f, "ends": []})
            return i
        if re.match(r"^fork\s*again$", low):
            c = self.top("fork", "fork again")
            c["ends"].extend(self.tails)
            self.tails = [(c["f"], "")]
            return i
        m = re.match(r"^(?:end\s*(fork|merge)|fork\s+end)\b.*$", low)
        if m:
            self.top("fork", "end fork")
            self.close("fork", label=m.group(1) or "fork")
            return i
        if low == "split":
            self.ctx.append({"type": "split", "start": list(self.tails), "ends": []})
            return i
        if re.match(r"^split\s*again$", low):
            c = self.top("split", "split again")
            c["ends"].extend(self.tails)
            self.tails = list(c["start"])
            return i
        if re.match(r"^(?:end\s*split|split\s+end)$", low):
            self.top("split", "end split")
            self.close("split")
            return i
        m = re.match(r"^switch\s*\((?P<c>.*)\)\s*;?$", line, re.I | re.S)
        if m:
            d = self.new("decision", _text(m.group("c")))
            self.enter(d)
            self.ctx.append({"type": "switch", "d": d, "ends": []})
            self.tails = []
            return i
        m = re.match(r"^case\s*\((?P<l>.*)\)\s*;?$", line, re.I | re.S)
        if m:
            c = self.top("switch", "case")
            c["ends"].extend(self.tails)
            self.tails = [(c["d"], _text(m.group("l")))]
            return i
        if re.match(r"^end\s*switch$", low):
            self.top("switch", "endswitch")
            self.close("switch")
            return i
        m = re.match(r"^(goto|label)\s+([\w.]+)\s*;?$", line, re.I)
        if m:
            if m.group(1).lower() == "goto":
                self.gotos.append((self.tails, m.group(2)))
                self.tails = []
            else:
                self.pending_labels.append(m.group(2))
            return i
        m = _A_REGION_RE.match(line)
        if m:
            rest = m.group("rest")
            brace = rest.endswith("{")
            rest = rest[:-1].strip() if brace else rest
            rest = re.sub(r"<<[^>]*>>|(?:^|\s)#\S+", " ", rest).strip()
            if not brace and m.group("kw").lower() not in ("partition", "group"):
                raise PlantUMLError(f"activity diagram: cannot parse line: {line[:80]!r}")
            parent = self.regions[-1].id if self.regions else None
            self.regions.append(self.b.group("region", _text(rest), parent=parent))
            return i
        if line.startswith("}") or re.match(r"^end\s*(?:partition|group)$", low):
            if self.regions:
                self.regions.pop()
            return i
        m = re.match(r"^\((?P<c>[^()]+)\)$", line)
        if m:  # connector
            self.enter(self.new("other", _text(m.group("c"))))
            return i
        raise PlantUMLError(f"activity diagram: cannot parse line: {line[:80]!r}")


# --------------------------------------------------------------------------- #
# Activity diagrams (legacy syntax)
# --------------------------------------------------------------------------- #
_L_NODE = rf"(?:\(\*(?:top|\d+)?\)|={{2,}}\s*[^=]+?\s*={{2,}}|{_QSTR}(?:\s+as\s+[\w.$@]+)?|[\w.$@]+)"
_L_ARROW = r"[-.=](?:[-.=]|\[[^\]]*\]|(?i:up|down|left|right|le|ri|do|u|d|l|r|\*))*>"
_L_LINK_RE = re.compile(
    rf"^(?:(?P<a>{_L_NODE})\s*(?:<<[^>]*>>\s*)?(?:#\w+\s*)?)?(?P<arrow>{_L_ARROW})\s*"
    rf"(?:\[(?P<label>[^\]]*)\]\s*)?(?P<b>.+?)\s*$"
)
_L_IF_RE = re.compile(rf"^if\s+(?P<c>{_QSTR}|.*?)(?:\s+as\s+(?P<alias>[\w.$@]+))?(?:\s+then)?\s*$", re.I)
_L_SRC_IF_RE = re.compile(rf"^(?P<a>{_L_NODE})\s*(?P<rest>if\s.*)$", re.I)


class _LegacyActivityParser:
    def __init__(self, src: Source):
        self.src = src
        self.b = _Build("activity")
        self.keys: dict[str, str] = {}
        self.bars: set[str] = set()
        self.start: Optional[str] = None
        self.final: Optional[str] = None
        self.last: list[str] = []  # implicit sources of the next sourceless arrow
        self.ctx: list[dict] = []
        self.regions: list[Group] = []

    def new(self, kind: str, text: str = "") -> str:
        nid = self.b.node(kind, text).id
        if self.regions:
            self.regions[-1].members.append(nid)
        return nid

    def ep(self, raw: str, as_target: bool) -> str:
        raw = re.sub(r"\s*(?:<<[^>]*>>|#\w+)\s*$", "", raw.strip())
        if raw.startswith("(*"):
            if as_target:
                if self.final is None:
                    self.final = self.new("final")
                return self.final
            if self.start is None:
                self.start = self.new("initial")
            return self.start
        if raw.startswith("=="):
            key = "===" + raw.strip("= ").strip()
            if key not in self.keys:
                self.keys[key] = self.new("fork")
                self.bars.add(self.keys[key])
            return self.keys[key]
        am = re.match(rf"^({_QSTR})\s+as\s+([\w.$@]+)$", raw)
        if am:
            key, text = am.group(2), _text(am.group(1))
        else:
            key = text = _text(raw)
        if key not in self.keys:
            self.keys[key] = self.new("action", text)
        return self.keys[key]

    def sources(self) -> list[str]:
        if self.ctx and self.ctx[-1]["fresh"]:
            self.ctx[-1]["fresh"] = False
            return [self.ctx[-1]["d"]]
        return self.last

    def decision(self, sources: list[str], label: str, rest: str) -> bool:
        m = _L_IF_RE.match(rest)
        if not m:
            return False
        d = self.new("decision", _text(m.group("c")))
        if m.group("alias"):
            self.keys[m.group("alias")] = d
        for s in sources:
            self.b.edge(s, d, "flow", label=label)
        self.ctx.append({"d": d, "ends": [], "fresh": True})
        self.last = [d]
        return True

    def parse(self) -> Diagram:
        lines = self.src.lines
        i = 0
        while i < len(lines):
            line = lines[i]
            i += 1
            while line.count('"') % 2 and i < len(lines):  # a quoted text may span lines
                line += "\\n" + lines[i]
                i += 1
            self.statement(line)
        for bar in self.bars:  # one way in: fork; several: join
            if sum(1 for e in self.b.edges if e.target == bar) > 1:
                next(n for n in self.b.nodes if n.id == bar).kind = "join"
        if not self.b.nodes:
            raise PlantUMLError("empty activity diagram")
        return self.b.diagram(self.src, "TB")

    def statement(self, line: str) -> None:
        low = line.lower()
        m = _L_LINK_RE.match(line)
        if m:
            sources = [self.ep(m.group("a"), as_target=False)] if m.group("a") else self.sources()
            label = _text(m.group("label") or "")
            if self.decision(sources, label, m.group("b")):
                return
            tgt = self.ep(m.group("b"), as_target=True)
            dashed, _style, hidden = _link_style(m.group("arrow"))
            if not hidden:
                for s in sources:
                    self.b.edge(s, tgt, "flow", label=label)
            self.last = [tgt]
            return
        m = _L_SRC_IF_RE.match(line)
        if m and self.decision([self.ep(m.group("a"), False)], "", m.group("rest")):
            return
        if self.decision(self.sources(), "", line):
            return
        if re.match(r"^else\b", low):
            if not self.ctx:
                raise PlantUMLError("legacy activity diagram: else outside if")
            c = self.ctx[-1]
            c["ends"].extend(x for x in self.last if x not in (c["d"], self.final))
            c["fresh"], c["else"] = True, True
            self.last = [c["d"]]
            return
        if re.match(r"^end\s*if$", low):
            if not self.ctx:
                raise PlantUMLError("legacy activity diagram: endif outside if")
            c = self.ctx.pop()
            ends = c["ends"] + [x for x in self.last if x != self.final]
            if not c.get("else"):  # no else branch: the false path continues from the decision
                ends.append(c["d"])
            self.last = list(dict.fromkeys(ends))
            return
        note = _parse_note(line)
        if note is not None:
            node = self.new("note", note.text)
            if self.last and not note.alias:
                self.b.edge(node, self.last[-1], "anchor")
            return
        m = re.match(r"^partition\s+(.*?)\s*\{?\s*$", line, re.I)
        if m:
            self.regions.append(self.b.group("region", _text(re.sub(r"(?:^|\s)#\S+", " ", m.group(1)))))
            return
        if line.startswith("}") or re.match(r"^end\s*partition$", low):
            if self.regions:
                self.regions.pop()
            return
        raise PlantUMLError(f"legacy activity diagram: cannot parse line: {line[:80]!r}")


# --------------------------------------------------------------------------- #
# State diagrams
# --------------------------------------------------------------------------- #
_ST_NAME = r"[\w$@.]+"
_ST_EP = rf"(?:\[\*\]|\[H\*?\]|{_QSTR}|{_ST_NAME}(?:\[H\*?\])?)"
_ST_LINK_RE = re.compile(
    rf"^(?P<a>{_ST_EP})\s*(?:(?:<<[^>]*>>|#[\w#]+)\s*)*(?P<lh><)?(?P<body>{_BODY})(?P<rh>>)?\s*"
    rf"(?P<b>{_ST_EP}){_LINK_TAIL}"
)
_ST_DESC_RE = re.compile(rf"^(?P<a>{_QSTR}|{_ST_NAME})(?<!:)\s*:(?!:)\s*(?P<d>.*)$")
_ST_STEREO = {"choice": "choice", "fork": "fork", "join": "join", "start": "initial", "end": "final",
              "history": "history", "history*": "deep_history", "entrypoint": "other",
              "exitpoint": "other", "inputpin": "other", "outputpin": "other",
              "expansioninput": "other", "expansionoutput": "other"}


@dataclass
class _Scope:
    group: Optional[Group] = None
    initial: Optional[str] = None
    final: Optional[str] = None
    history: dict[str, str] = field(default_factory=dict)


class _StateParser:
    def __init__(self, src: Source):
        self.src = src
        self.b = _Build("state")
        self.root = _Scope()
        self.scopes = [self.root]
        self.states: dict[str, Node] = {}
        self.scope_of: dict[str, _Scope] = {}  # node id -> scope
        self.declared: set[str] = set()
        self.comps: dict[str, tuple[Group, _Scope]] = {}  # name -> (group, parent scope)
        self.comp_scope: dict[str, _Scope] = {}  # name -> its inner (first) scope
        self.group_parent: dict[str, _Scope] = {}  # group id -> parent scope
        self.notes: dict[str, Node] = {}

    def cur(self) -> _Scope:
        return self.scopes[-1]

    def state(self, raw: str, declare: bool = False) -> Node:
        key = _text(raw) if raw.startswith('"') else raw
        node = self.states.get(key)
        if node is None:
            node = self.b.node("state", _text(raw) if raw.startswith('"') else key)
            self.states[key] = node
            self.scope_of[node.id] = self.cur()
        elif declare and node.id not in self.declared and self.cur() is not self.root:
            self.scope_of[node.id] = self.cur()
        if declare:
            self.declared.add(node.id)
        return node

    def pseudo(self, raw: str, as_target: bool) -> str:
        """[*], [H], [H*], X[H], X[H*] -> node id."""
        if raw == "[*]":
            sc = self.cur()
            attr = "final" if as_target else "initial"
            if getattr(sc, attr) is None:
                setattr(sc, attr, self.b.node(attr).id)
                self.scope_of[getattr(sc, attr)] = sc
            return getattr(sc, attr)
        m = re.match(rf"^({_ST_NAME})?\[(H\*?)\]$", raw)
        assert m
        owner, deep = m.group(1), m.group(2) == "H*"
        sc = self.comp_scope.get(owner) if owner else self.cur()
        key = f"{owner or ''}[{m.group(2)}]"
        hist_key = key if sc is None else m.group(2)
        holder = sc if sc is not None else self.root
        if hist_key not in holder.history:
            nid = self.b.node("deep_history" if deep else "history").id
            holder.history[hist_key] = nid
            self.scope_of[nid] = holder
            if owner and sc is None:
                self.state(owner)  # composite declared later or a plain state
        return holder.history[hist_key]

    def ref(self, raw: str, as_target: bool) -> str:
        if raw == "[*]" or raw.endswith("]"):
            return self.pseudo(raw, as_target)
        key = _text(raw) if raw.startswith('"') else raw
        if key in self.notes:
            return self.notes[key].id
        return self.state(raw).id

    def parse(self) -> Diagram:
        lines = self.src.lines
        i = 0
        while i < len(lines):
            line = lines[i]
            i += 1
            self.statement(line)
        # states that turned out to be composites become their group
        redirect: dict[str, str] = {}
        for name, (g, _parent) in self.comps.items():
            node = self.states.get(name)
            if node is not None:
                redirect[node.id] = g.id
                g.text = "\n".join([g.text] + [m.text for m in node.members])
        for e in self.b.edges:
            e.source = redirect.get(e.source, e.source)
            e.target = redirect.get(e.target, e.target)
        self.b.nodes = [n for n in self.b.nodes if n.id not in redirect]
        for gid, parent in self.group_parent.items():
            g = next(x for x in self.b.groups if x.id == gid)
            if parent.group is not None and parent.group.id != gid:
                g.parent = parent.group.id
        for n in self.b.nodes:
            sc = self.scope_of.get(n.id)
            if sc is not None and sc.group is not None:
                sc.group.members.append(n.id)
            if n.kind not in ("state", "note"):  # pseudo-states have no compartment
                n.members = []
        if not self.b.nodes and not self.b.groups:
            raise PlantUMLError("empty state diagram")
        return self.b.diagram(self.src, "TB")

    def statement(self, line: str) -> None:
        low = line.lower()
        if line.startswith("}"):
            if len(self.scopes) > 1:
                self.scopes.pop()
            return
        if re.match(r"^(?:--+|\|\|+)$", line):
            self.region()
            return
        if re.match(r"^(?:hide|show|remove|restore)\b", low):
            return
        note = _parse_note(line)
        if note is not None:
            node = self.b.node("note", note.text)
            self.scope_of[node.id] = self.cur()
            if note.alias:
                self.notes[note.alias] = node
            for t in note.targets:
                self.b.edge(node.id, self.ref(t.strip(), True), "anchor")
            return
        m = re.match(r"^state\s+(?P<rest>.+)$", line, re.I)
        if m and not _ST_LINK_RE.match(line):
            self.declare(m.group("rest"))
            return
        m = None if _ST_DESC_RE.match(line) else _ST_LINK_RE.match(line)  # 'A.b : x' is a description
        if m:
            dashed, style, hidden = _link_style(m.group("body"))
            a = self.ref(m.group("a"), as_target=bool(m.group("lh")) and not m.group("rh"))
            b = self.ref(m.group("b"), as_target=not (m.group("lh") and not m.group("rh")))
            if hidden:
                return
            if m.group("lh") and not m.group("rh"):
                a, b = b, a
            if a in self.notes_ids() or b in self.notes_ids():
                src, tgt = (a, b) if a in self.notes_ids() else (b, a)
                self.b.edge(src, tgt, "anchor")
                return
            self.b.edge(a, b, "transition", label=_text(m.group("label")), style=style)
            return
        m = _ST_DESC_RE.match(line)
        if m:
            key = _text(m.group("a")) if m.group("a").startswith('"') else m.group("a")
            if key in self.comps:  # described composite: the text is drawn under its name
                desc = _text(m.group("d"))
                if desc:
                    self.comps[key][0].text += "\n" + desc
                return
            node = self.state(m.group("a"))
            desc = _text(m.group("d"))
            if desc:
                node.members.append(Member("attr", desc))
            return
        raise PlantUMLError(f"state diagram: cannot parse line: {line[:80]!r}")

    def notes_ids(self) -> set[str]:
        return {n.id for n in self.notes.values()}

    def declare(self, rest: str) -> None:
        na = _name_alias(rest.strip(), _ST_NAME)
        if na is None:
            raise PlantUMLError(f"state diagram: bad state: {rest[:60]!r}")
        display, code, tail = na
        desc = None
        sp = _split_colon(tail)
        if sp:
            tail, desc = sp
        tail = tail.strip()
        brace = tail.endswith("{")
        if brace:
            tail = tail[:-1]
        elif re.search(r"\{\s*\}$", tail):
            tail = re.sub(r"\{\s*\}$", "", tail)
        stereo = None
        sm = re.search(r"<<\s*(.*?)\s*>>", tail)
        if sm:
            stereo = sm.group(1)
            tail = tail[:sm.start()] + tail[sm.end():]
        tail = re.sub(r"(?:^|\s)##?\S*|\[\[[^\]]*\]\]|\$\w+", " ", tail)
        if re.search(r"\w", tail):
            raise PlantUMLError(f"state diagram: bad state: {rest[:60]!r}")
        if brace:
            parent = self.cur()
            g = self.b.group("composite", display)
            self.comps[code] = (g, parent)
            self.group_parent[g.id] = parent
            sc = _Scope(group=g)
            self.comp_scope.setdefault(code, sc)
            self.scopes.append(sc)
            return
        node = self.state(code, declare=True)
        if display != code or node.text == code:
            node.text = display
        if stereo:
            kind = _ST_STEREO.get(stereo.lower())
            if kind:
                node.kind = kind
                if kind != "state":
                    node.text = "" if display == code else display
            else:
                node.stereotype = _text(stereo)
        if desc and _text(desc):
            node.members.append(Member("attr", _text(desc)))

    def region(self) -> None:
        sc = self.cur()
        if sc is self.root or sc.group is None:
            return
        comp = sc.group if sc.group.kind == "composite" else \
            next(x for x in self.b.groups if x.id == sc.group.parent)
        if sc.group.kind == "composite":  # first separator: what came so far is region 1
            r1 = self.b.group("region", "", parent=comp.id)
            for gid, parent in self.group_parent.items():
                if parent is sc:
                    self.group_parent[gid] = _Scope(group=r1)
            sc.group = r1
        r = self.b.group("region", "", parent=comp.id)
        self.scopes[-1] = _Scope(group=r)


# --------------------------------------------------------------------------- #
# Use case (and other element-style) diagrams -> graph
# --------------------------------------------------------------------------- #
_U_KINDS = {**ELEMENTS, "usecase": "ellipse", "actor": "actor", "port": "point", "portin": "point",
            "portout": "point"}
_U_KW = "|".join(sorted(_U_KINDS, key=len, reverse=True))
_U_NAME = rf'(?:{_QSTR}|\([^()]+\)|:[^:]+:|\[[^\]]+\]|[^\s"<>#{{}}\[\]():]+)'
_U_DECL_RE = re.compile(rf"^(?P<kw>{_U_KW})(?P<slash>/)?\s+(?P<rest>.+)$", re.I)
_U_EP = rf'(?:{_QSTR}|\([^()]+\)|:[^:]+:|\[[^\]]+\]|\(\)\s*(?:{_QSTR}|[\w.$@]+)|[\w.$@]+/?)'
_U_LINK_RE = re.compile(
    rf"^(?P<a>{_U_EP})\s*(?:(?P<ca>{_QSTR})\s*)?(?P<lh>{_HEAD_L})?(?P<body>{_BODY})(?P<rh>{_HEAD_R})?"
    rf"\s*(?:(?P<cb>{_QSTR})\s*)?(?P<b>{_U_EP}){_LINK_TAIL}"
)
_ARROW_HEADS = {"<", "<<", "<|", "^", ">", ">>", "|>", "<||", "||>"}


def _shape_of(raw: str) -> tuple[str, Optional[str]]:
    """Endpoint / name spelling -> (code, kind implied by the spelling or None)."""
    raw = raw.strip()
    if raw.startswith("()"):
        inner = raw[2:].strip()
        return (_text(inner) if inner.startswith('"') else inner), "circle"
    if len(raw) >= 2 and raw[0] == "(" and raw[-1] == ")":
        return _text(raw[1:-1]), "ellipse"
    if len(raw) >= 2 and raw[0] == ":" and raw[-1] == ":":
        return _text(raw[1:-1]), "actor"
    if len(raw) >= 2 and raw[0] == "[" and raw[-1] == "]":
        return _text(raw[1:-1]), "component"
    if raw.startswith('"'):
        return _text(raw), None
    return raw.rstrip("/"), None


class _UseCaseParser:
    def __init__(self, src: Source):
        self.src = src
        self.b = _Build("graph")
        self.by_code: dict[str, Node] = {}
        self.clusters: dict[str, Group] = {}
        self.cluster_kind: dict[str, str] = {}
        self.cluster_stereo: dict[str, Optional[str]] = {}
        self.stack: list[Optional[Group]] = []
        self.notes: dict[str, Node] = {}

    def place(self, nid: str) -> None:
        for g in reversed(self.stack):
            if g is not None:
                g.members.append(nid)
                return

    def element(self, code: str, kind: str, text: Optional[str] = None, declared: bool = False) -> Node:
        node = self.by_code.get(code)
        if node is None:
            node = self.b.node(kind, text if text is not None else code)
            self.by_code[code] = node
            self.place(node.id)
        elif declared:
            node.kind = kind
            if text is not None:
                node.text = text
        return node

    def close(self) -> None:
        """An empty container is drawn as a plain element."""
        g = self.stack.pop()
        if g is None or g.members or any(x.parent == g.id for x in self.b.groups):
            return
        code = next((c for c, x in self.clusters.items() if x is g), None)
        if code is None or code in self.by_code or any(g.id in (e.source, e.target) for e in self.b.edges):
            return
        self.b.groups.remove(g)
        del self.clusters[code]
        node = self.element(code, self.cluster_kind.get(g.id, "box"), g.text)
        node.stereotype = self.cluster_stereo.get(g.id)

    def ep(self, raw: str) -> str:
        code, kind = _shape_of(raw)
        if code in self.notes:
            return self.notes[code].id
        if kind is None and code in self.clusters and code not in self.by_code:
            return self.clusters[code].id
        return self.element(code, kind or "actor").id

    def parse(self) -> Diagram:
        lines = self.src.lines
        i = 0
        while i < len(lines):
            line = lines[i]
            i += 1
            while line.count('"') % 2 and i < len(lines):  # a quoted name may span lines
                line += "\\n" + lines[i]
                i += 1
            self.statement(line)
        if not self.b.nodes:
            raise PlantUMLError("empty use case diagram")
        return self.b.diagram(self.src, "TB")

    def statement(self, line: str) -> None:
        if line.startswith("}"):
            if self.stack:
                self.close()
            return
        low = line.lower()
        if re.match(r"^(?:hide|show|remove|restore)\b", low):
            return
        if re.match(r"^together\b.*\{$", low):
            self.stack.append(None)
            return
        note = _parse_note(line)
        if note is not None:
            node = self.b.node("note", note.text)
            self.place(node.id)
            if note.alias:
                self.notes[note.alias] = node
            for t in note.targets:
                self.b.edge(node.id, self.ep(t), "anchor")
            return
        m = _U_LINK_RE.match(line)
        if m:
            self.link(m)
            return
        m = _U_DECL_RE.match(line)
        if m:
            self.declare(m.group("kw").lower(), m.group("rest"))
            return
        if line.startswith("()"):
            self.declare("circle", line[2:])
            return
        self.declare(None, line)  # (Use) / :Actor: / [Comp] / "Text" as (Use) / Name <<S>> / Name as N

    def declare(self, kw: Optional[str], rest: str) -> None:
        rest = rest.strip()
        brace = rest.endswith("{")
        if brace:
            rest = rest[:-1].strip()
        elif re.search(r"\{\s*\}$", rest):
            rest = re.sub(r"\{\s*\}$", "", rest).strip()
        stereos = [_text(x) for x in re.findall(r"<<\s*(.*?)\s*>>", rest)]
        rest = re.sub(r"<<.*?>>|\[\[[^\]]*\]\]|(?:^|\s)##?[\w#:;.\-]*|\$\w+", " ", rest).strip()
        m = re.match(rf"^(?P<x>{_U_NAME})(?:\s+(?i:as)\s+(?P<y>{_U_NAME}))?\s*$", rest)
        if not m and brace and not rest:  # anonymous boundary
            parent = next((g for g in reversed(self.stack) if g is not None), None)
            self.stack.append(self.b.group("cluster", "", parent=parent.id if parent else None))
            return
        if not m or (kw is None and m.group("y") is None and not stereos and not brace
                     and not rest.startswith(("(", ":", "["))):
            what = "bad element" if kw else "cannot parse line"
            raise PlantUMLError(f"use case diagram: {what}: {rest[:60]!r}")
        x, y = m.group("x"), m.group("y")
        xc, xk = _shape_of(x)
        if y is None:
            code, text, shape = xc, xc, xk
        else:
            yc, yk = _shape_of(y)
            if (x.startswith(('"', "(", ":", "[")) or yk is None and not y.startswith('"')) and \
                    not y.startswith('"'):
                code, text, shape = yc, xc, xk or yk
            else:
                code, text, shape = xc, yc, yk or xk
        kind = (_U_KINDS.get(kw, "box") if kw != "circle" else "circle") if kw else (shape or "actor")
        if brace:
            parent = next((g for g in reversed(self.stack) if g is not None), None)
            g = self.b.group("cluster", text, parent=parent.id if parent else None)
            self.clusters[code] = g
            self.cluster_kind[g.id] = kind
            self.cluster_stereo[g.id] = ", ".join(x for x in stereos if x) or None
            self.stack.append(g)
            return
        node = self.element(code, kind, text, declared=True)
        if stereos:
            node.stereotype = ", ".join(s for s in stereos if s) or None

    def link(self, m: re.Match) -> None:
        dashed, _style, hidden = _link_style(m.group("body"))
        dashed = dashed or _tail_dashed(m)
        a, b = self.ep(m.group("a")), self.ep(m.group("b"))
        if hidden:
            return
        notes = {n.id for n in self.notes.values()}
        if a in notes or b in notes:
            src, tgt = (a, b) if a in notes else (b, a)
            self.b.edge(src, tgt, "anchor")
            return
        lh, rh = m.group("lh") in _ARROW_HEADS, m.group("rh") in _ARROW_HEADS
        sl = _text(m.group("ca")) or None if m.group("ca") else None
        tl = _text(m.group("cb")) or None if m.group("cb") else None
        kind = "bidir" if lh and rh else "arrow" if (lh or rh) else "line"
        if lh and not rh:
            a, b, sl, tl = b, a, tl, sl
        self.b.edge(a, b, kind, label=_direction_free_label(m.group("label")), source_label=sl,
                    target_label=tl, style="dashed" if dashed else None)


# --------------------------------------------------------------------------- #
# Kind detection and dispatch
# --------------------------------------------------------------------------- #
_UNSUPPORTED_RE = re.compile(
    r"^(?:(?:salt|ditaa|nwdiag|mindmap|wbs|gantt)\s*(?:\{.*)?$|(?:di)?graph\s+\w*\s*\{|archimate\s"
    r"|(?:robust|concise|clock|binary)\s+(?:\"|\w+\s+as\s)|@\d+\s*$|@[A-Za-z]\w*\s*$(?<!@enduml)"
    r"|project\s+starts|printscale\b"
    r"|(?-i:(?:Person|System|Container|Component|Rel|BiRel|Boundary|Deployment_Node|Enterprise_Boundary"
    r"|System_Boundary|Container_Boundary|LAYOUT|SHOW|HIDE|Lay|AddElementTag|AddRelTag|UpdateElementStyle"
    r"|UpdateRelStyle)(?:_\w+)?)\(.*\)\s*\{?$"  # C4-PlantUML macros from an !include
    r"|\[[^\]]+\]\s+(?:lasts|requires|starts|ends|happens|is\s+colored|displays)\b)",
    re.I,
)
_LEGACY_RE = re.compile(r"\(\*(?:top)?\)\s*(?:[-.]|if\b)|[-.>]\s*(?:\[[^\]]*\]\s*)?\(\*(?:top)?\)")
_ACT3_RE = re.compile(
    r"^(?:start|stop|detach|kill|endif|end\s+if|endwhile|end\s+while|fork|fork\s+again|end\s*fork|end\s*merge"
    r"|split|split\s+again|end\s*split|endswitch|end\s*switch)\s*;?$"
    r"|^(?:if|while|switch|case)\s*\(|^repeat\b|^backward\s*:|^\|[^|]+\||^-+(?:\[[^\]]*\])?-*>.*;$",
    re.I,
)
_USE_ACTOR_RE = re.compile(r"^:[^:;]+:(?:\s+as\s|\s*[-.<]|\s*$)")
_STATE_RE = re.compile(r"\[\*\]\s*[-.<]|[-.>]\s*\[\*\]|\[H\*?\]\s*[-.<]|[-.>]\s*\w*\[H\*?\]", re.I)
_STATE_DECL_RE = re.compile(r"^state\s+\S", re.I)
_CLASS_KW_RE = re.compile(
    r"^[-+#~]?(?:abstract|class|enum|annotation|protocol|struct|exception|metaclass|dataclass"
    r"|record|object|map)\s+(?![-.<=]|[*o+#x}|][-.])\S|^[-+#~]?interface\s.*\{$", re.I)
_DESC_HINT_RE = re.compile(
    r"^(?:rectangle|actor|usecase|database|node|cloud|frame|folder|component|artifact|storage|card|agent"
    r"|queue|collections|boundary|control|person|hexagon|file|stack)\b"
    r"(?!\s*(?:\"[^\"]*\"\s*)?(?:[-.<=]|[*o][-.]))", re.I)
_INTERFACE_RE = re.compile(r"^[-+#~]?interface\s+(?![-.<=]|[*o+#x}|][-.])\S", re.I)
_SEQ_RE = re.compile(
    r"^(?:participant|boundary|control|collections|queue|activate|deactivate|autonumber|autoactivate"
    r"|return|destroy|create|alt|opt|loop|par|par2|critical|ref\s+over|hnote|rnote|box"
    r"|end\s*box)\b|^note\s+over\b|^\.\.\.|^\|\|\d*\|\|?|^==.*==$",
    re.I,
)
_CLASS_RE = re.compile(
    r"^(?:abstract|class|interface|enum|annotation|protocol|struct|exception|metaclass|dataclass|record"
    r"|object|map|namespace|stereotype)\s|<\|[-.]|[-.]\|>|^entity\s.*\{$|^set\s+(?:namespace)?separator"
    r"|^\S+\s+(?:\"[^\"]*\"\s*)?[*o][-.]|[-.][*o]\s+(?:\"[^\"]*\"\s*)?\S+(?:\s*:.*)?$",
    re.I,
)
_USE_RE = re.compile(
    r"^usecase/?\s+(?![-.<=])\S|^actor/|^\([^()*][^()]*\)(?:\s+as\s|\s*[-.<]|\s*$)"
    r"|^(?::[^:;]+:)?[^:]*?[-.>]\s*(?:\([^()*][^()]*\)|:[^:;]+:)\s*(?::.*)?$|^:[^:;]+:(?:\s+as\s|\s*[-.<]|\s*$)",
    re.I,
)
_COMP_RE = re.compile(
    r"^(?:component|node|cloud|artifact|folder|frame|file|storage|stack|agent|card|person|hexagon"
    r"|port|portin|portout)\b(?!\s*(?:\"[^\"]*\"\s*)?(?:[-.<=]|[*o][-.]))"
    r"|^\[[^\]\-<>?*][^\]]*\]|^\(\)\s*[\w\"]|^[^:]*?[-.>]\s*\[[^\]\-<>?*][^\]]*\]\s*(?::.*)?$",
    re.I,
)
_ELEMENT_RE = re.compile(
    r"^(?:rectangle|actor|database|usecase|interface)\b|^(?:package|node|cloud|frame|folder|queue|storage"
    r"|together)\b.*\{$", re.I)

KINDS = ("class", "sequence", "activity", "activity_legacy", "state", "usecase", "component")


def _statements(src: Source) -> list[str]:
    """Statement lines without class bodies and notes (for kind detection)."""
    out, skip = [], False
    for ln in src.lines:
        if skip:
            skip = not ln.startswith("}")
            continue
        if _BODY_DECL_RE.match(ln):
            skip = True
        if _NOTE_HEAD_RE.match(ln) and not re.match(r"^note\s+over\b", ln, re.I):
            continue
        out.append(re.sub(r'"(?:[^"\\]|\\.)*"', '"Q"', ln))  # quoted text says nothing about the kind
    return out


def _route(src: Source) -> tuple[str, Optional[Diagram]]:
    """Diagram kind (and the sequence result when that grammar was tried). Unambiguous markers
    first (activity, class keywords, state); then, like PlantUML, the sequence grammar is tried
    unless the text has use case / component / class markers or a layout direction (invalid in
    sequence diagrams); description elements beat class-style arrows. On UML-in-the-Wild this
    agrees with the grammar PlantUML itself used (its element keys) on 99.8% of parsed files."""
    stmts = _statements(src)
    if not stmts:
        raise PlantUMLError("empty diagram")

    def has(rx: re.Pattern) -> bool:
        return any(rx.search(s) for s in stmts)

    if has(_UNSUPPORTED_RE):
        raise UnsupportedPlantUML("not a UML diagram (gantt, timing, salt, ditaa, mindmap, ...)")
    if has(_LEGACY_RE):
        return "activity_legacy", None
    if any(_ACT3_RE.search(s) or (s.startswith(":") and not _USE_ACTOR_RE.match(s)) for s in stmts):
        return "activity", None
    if has(_CLASS_KW_RE) or (has(_INTERFACE_RE) and not has(_COMP_RE) and not has(_USE_RE)
                             and not has(_DESC_HINT_RE)):
        return "class", None
    if has(_STATE_RE) or any(_STATE_DECL_RE.match(s) and not _C_LINK_RE.match(s) for s in stmts):
        return "state", None
    is_use, is_class, is_comp, is_seq = has(_USE_RE), has(_CLASS_RE), has(_COMP_RE), has(_SEQ_RE)
    seq_error: Optional[PlantUMLError] = None
    if not src.direction and (is_seq or not (is_use or is_class or is_comp)):
        try:  # PlantUML itself tries the sequence grammar first
            return "sequence", _SequenceParser(src).parse()
        except PlantUMLError as e:
            seq_error = e
    if is_use:
        return "usecase", None
    if is_comp:
        return "component", None
    if is_class and not has(_DESC_HINT_RE):  # class arrows between description elements: description
        return "class", None
    if has(_ELEMENT_RE) or has(_DESC_HINT_RE):
        return "usecase", None
    if is_seq and seq_error is not None:
        raise seq_error
    if has(_C_LINK_RE) or has(_C_DECL_RE) or (has(_C_MEMBER_RE) and not is_seq):
        return "class", None
    if seq_error is not None:
        raise seq_error
    raise PlantUMLError("cannot determine the diagram kind")


def detect_kind(code: str) -> str:
    """Diagram kind of the first @startuml block: one of KINDS (raises UnsupportedPlantUML)."""
    return _route(preprocess(code))[0]


def _checked(d: Diagram) -> Diagram:
    issues = validate(d, strict=True)
    if issues:
        raise PlantUMLError(f"result does not validate: {issues[0]}")
    return d


def _component(src: Source) -> Diagram:
    lines = [ln for ln in src.lines if not _NOTE_HEAD_RE.match(ln)]
    try:
        d = plantuml_graph_to_diagram("\n".join(lines))
        if src.direction:
            d.direction = src.direction
        d.title = src.title
        return _checked(d)
    except PlantUMLError:
        return _checked(_UseCaseParser(src).parse())


_PARSERS = {
    "class": lambda s: _ClassParser(s).parse(),
    "sequence": lambda s: _SequenceParser(s).parse(),
    "activity": lambda s: _ActivityParser(s).parse(),
    "activity_legacy": lambda s: _LegacyActivityParser(s).parse(),
    "state": lambda s: _StateParser(s).parse(),
    "usecase": lambda s: _UseCaseParser(s).parse(),
    "component": _component,
}


def parse_plantuml(code: str) -> Diagram:
    """Detect the kind of the first @startuml block of `code` and parse it."""
    src = preprocess(code)
    kind, d = _route(src)
    return _checked(d if d is not None else _PARSERS[kind](src))


def plantuml_class_to_diagram(code: str) -> Diagram:
    return _checked(_ClassParser(preprocess(code)).parse())


def plantuml_sequence_to_diagram(code: str) -> Diagram:
    return _checked(_SequenceParser(preprocess(code)).parse())


def plantuml_activity_to_diagram(code: str) -> Diagram:
    """New (`:action;`) or legacy (`(*) --> "action"`) activity syntax."""
    src = preprocess(code)
    legacy = any(_LEGACY_RE.search(s) for s in src.lines)
    return _checked((_LegacyActivityParser(src) if legacy else _ActivityParser(src)).parse())


def plantuml_state_to_diagram(code: str) -> Diagram:
    return _checked(_StateParser(preprocess(code)).parse())


def plantuml_usecase_to_diagram(code: str) -> Diagram:
    return _checked(_UseCaseParser(preprocess(code)).parse())
