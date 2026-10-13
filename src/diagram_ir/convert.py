"""JSON / YAML / XML converters for the Diagram IR, plus format detection.

All three are lossless views of the same `Diagram`, so any format converts to any
other through it:  DGM <-> Diagram <-> JSON | YAML | XML.

The JSON object shape is the reference (schema.json); YAML is the same object, and
XML maps it element-for-element (SPEC.md, "XML").
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any, Optional
from xml.etree import ElementTree as ET

import yaml

from . import dsl
from .model import (
    FORMAT_NAME,
    ID_RE,
    VERSION,
    Diagram,
    Edge,
    Group,
    Member,
    Node,
    default_edge_kind,
    edge_id,
)

try:  # hardened parser when available; stdlib expat already refuses external entities
    from defusedxml import ElementTree as _SafeET  # type: ignore
except ImportError:  # pragma: no cover
    _SafeET = ET

FORMATS = ("dgm", "json", "yaml", "xml")
EXTENSIONS = {".dgm": "dgm", ".json": "json", ".yaml": "yaml", ".yml": "yaml", ".xml": "xml"}

# Optional scalar fields, in output order. Names are identical in JSON, YAML and XML.
_EDGE_OPTIONAL = ("source_label", "target_label", "source_port", "target_port", "style")


# --------------------------------------------------------------------------- #
# Diagram <-> plain dict (the JSON / YAML object)
# --------------------------------------------------------------------------- #
def to_dict(d: Diagram) -> dict:
    out: dict[str, Any] = {"format": FORMAT_NAME, "version": VERSION, "type": d.type}
    if d.direction:
        out["direction"] = d.direction
    if d.title:
        out["title"] = d.title

    nodes = []
    for n in d.nodes:
        nd: dict[str, Any] = {"id": n.id, "kind": n.kind, "text": n.text}
        if n.bbox is not None:
            nd["bbox"] = list(n.bbox)
        if n.stereotype:
            nd["stereotype"] = n.stereotype
        if n.members:
            nd["members"] = []
            for m in n.members:
                md: dict[str, Any] = {"kind": m.kind, "text": m.text}
                if m.bbox is not None:
                    md["bbox"] = list(m.bbox)
                nd["members"].append(md)
        if n.attrs:
            nd["attrs"] = dict(n.attrs)
        nodes.append(nd)

    edges = []
    for i, e in enumerate(d.edges):
        ed: dict[str, Any] = {
            "id": edge_id(i), "source": e.source, "target": e.target,
            "kind": e.kind, "directed": e.directed,
        }
        if e.label:
            ed["label"] = e.label
        for key in _EDGE_OPTIONAL:
            if getattr(e, key) is not None:
                ed[key] = getattr(e, key)
        if e.path is not None:
            ed["path"] = [list(p) for p in e.path]
        if e.attrs:
            ed["attrs"] = dict(e.attrs)
        edges.append(ed)

    groups = []
    for g in d.groups:
        gd: dict[str, Any] = {"id": g.id, "kind": g.kind, "text": g.text, "members": list(g.members)}
        if g.parent:
            gd["parent"] = g.parent
        if g.bbox is not None:
            gd["bbox"] = list(g.bbox)
        if g.attrs:
            gd["attrs"] = dict(g.attrs)
        groups.append(gd)

    out["nodes"], out["edges"], out["groups"] = nodes, edges, groups
    if d.attrs:
        out["attrs"] = dict(d.attrs)
    return out


def _req(obj: dict, key: str, typ: type, where: str) -> Any:
    if key not in obj:
        raise ValueError(f"{where}: missing required field {key!r}")
    return _typed(obj[key], typ, f"{where}.{key}")


def _opt(obj: dict, key: str, typ: type, where: str, default: Any = None) -> Any:
    if obj.get(key) is None:
        return default
    return _typed(obj[key], typ, f"{where}.{key}")


def _typed(value: Any, typ: type, where: str) -> Any:
    if typ is int and isinstance(value, bool):
        raise ValueError(f"{where}: expected int, got bool")
    if not isinstance(value, typ):
        hint = " (quote it in YAML)" if typ is str and isinstance(value, (bool, int, float)) else ""
        raise ValueError(f"{where}: expected {typ.__name__}, got {type(value).__name__}{hint}")
    return value


def _bbox_from(value: Any, where: str) -> Optional[tuple[int, int, int, int]]:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"{where}: bbox must be a list of 4 integers")
    return tuple(_typed(v, int, where) for v in value)  # type: ignore[return-value]


def _attrs_from(value: Any, where: str) -> dict[str, str]:
    if value is None:
        return {}
    _typed(value, dict, where)
    out = {}
    for k, v in value.items():
        if isinstance(v, bool) or not isinstance(v, (str, int, float)):
            raise ValueError(f"{where}.{k}: attribute values must be strings")
        out[str(k)] = str(v)
    return out


def from_dict(obj: Any) -> Diagram:
    """Build a Diagram from a JSON/YAML object. Raises ValueError with a field path.
    `directed` on edges is informational and ignored (it follows from `kind`)."""
    _typed(obj, dict, "$")
    fmt = obj.get("format", FORMAT_NAME)
    if fmt != FORMAT_NAME:
        raise ValueError(f"$.format: expected {FORMAT_NAME!r}, got {fmt!r}")
    version = str(obj.get("version", VERSION))
    if version.split(".")[0] != VERSION.split(".")[0]:
        raise ValueError(f"$.version: unsupported major version {version!r}")

    d = Diagram(
        type=_req(obj, "type", str, "$"),
        direction=_opt(obj, "direction", str, "$"),
        title=_opt(obj, "title", str, "$"),
        attrs=_attrs_from(obj.get("attrs"), "$.attrs"),
    )

    for i, nd in enumerate(_opt(obj, "nodes", list, "$", [])):
        w = f"$.nodes[{i}]"
        _typed(nd, dict, w)
        members = []
        for j, md in enumerate(_opt(nd, "members", list, w, [])):
            mw = f"{w}.members[{j}]"
            _typed(md, dict, mw)
            members.append(Member(kind=_req(md, "kind", str, mw), text=_opt(md, "text", str, mw, ""),
                                  bbox=_bbox_from(md.get("bbox"), f"{mw}.bbox")))
        d.nodes.append(Node(
            id=_req(nd, "id", str, w), kind=_req(nd, "kind", str, w),
            text=_opt(nd, "text", str, w, ""), bbox=_bbox_from(nd.get("bbox"), f"{w}.bbox"),
            stereotype=_opt(nd, "stereotype", str, w), members=members,
            attrs=_attrs_from(nd.get("attrs"), f"{w}.attrs"),
        ))

    raw_edges = _opt(obj, "edges", list, "$", [])
    node_or_group = {n.id for n in d.nodes} | {
        g["id"] for g in obj.get("groups") or [] if isinstance(g, dict) and isinstance(g.get("id"), str)
    }
    edge_ids: dict[str, str] = {}  # id as written -> positional id
    for i, ed in enumerate(raw_edges):
        w = f"$.edges[{i}]"
        _typed(ed, dict, w)
        given = _opt(ed, "id", str, w, edge_id(i))
        if given in node_or_group:
            raise ValueError(f"{w}.id: {given!r} collides with a node/group id")
        if given in edge_ids:
            raise ValueError(f"{w}.id: duplicate edge id {given!r}")
        edge_ids[given] = edge_id(i)
        path = None
        if ed.get("path") is not None:
            path = []
            for k, p in enumerate(_typed(ed["path"], list, f"{w}.path")):
                if not isinstance(p, (list, tuple)) or len(p) != 2:
                    raise ValueError(f"{w}.path[{k}]: expected [x, y]")
                path.append((_typed(p[0], int, f"{w}.path[{k}]"), _typed(p[1], int, f"{w}.path[{k}]")))
        edge = Edge(
            source=_req(ed, "source", str, w), target=_req(ed, "target", str, w),
            kind=_opt(ed, "kind", str, w, default_edge_kind(d.type)),
            label=_opt(ed, "label", str, w, ""), path=path,
            attrs=_attrs_from(ed.get("attrs"), f"{w}.attrs"),
        )
        for key in _EDGE_OPTIONAL:
            setattr(edge, key, _opt(ed, key, str, w))
        if edge.style == "solid":
            edge.style = None
        d.edges.append(edge)

    for i, gd in enumerate(_opt(obj, "groups", list, "$", [])):
        w = f"$.groups[{i}]"
        _typed(gd, dict, w)
        members = []
        for j, m in enumerate(_opt(gd, "members", list, w, [])):
            ref = _typed(m, str, f"{w}.members[{j}]")
            members.append(edge_ids.get(ref, ref) if ref not in node_or_group else ref)
        d.groups.append(Group(
            id=_req(gd, "id", str, w), kind=_req(gd, "kind", str, w),
            text=_opt(gd, "text", str, w, ""), members=members,
            parent=_opt(gd, "parent", str, w), bbox=_bbox_from(gd.get("bbox"), f"{w}.bbox"),
            attrs=_attrs_from(gd.get("attrs"), f"{w}.attrs"),
        ))
    return d


# --------------------------------------------------------------------------- #
# JSON
# --------------------------------------------------------------------------- #
# A pretty-printed list of numbers / plain ids. It must span real newlines, which JSON
# strings cannot contain, so text inside labels is never touched.
_JSON_SHORT_LIST = re.compile(
    r'\[\n\s+((?:-?\d+|"[A-Za-z0-9_]*")(?:,\n\s+(?:-?\d+|"[A-Za-z0-9_]*"))*)\n\s*\]'
)


def to_json(d: Diagram, indent: Optional[int] = 2) -> str:
    text = json.dumps(to_dict(d), ensure_ascii=False, indent=indent)
    if indent:
        text = _JSON_SHORT_LIST.sub(lambda m: "[" + re.sub(r",\n\s+", ", ", m.group(1)) + "]", text)
    return text + "\n"


def from_json(text: str) -> Diagram:
    return from_dict(json.loads(text))


# --------------------------------------------------------------------------- #
# YAML
# --------------------------------------------------------------------------- #
class _YamlDumper(yaml.SafeDumper):
    pass


def _is_flow_list(data: list) -> bool:
    """Short homogeneous lists (bbox, path points, member refs) read best inline."""
    if not data:
        return True
    if all(isinstance(x, int) and not isinstance(x, bool) for x in data):
        return True
    if all(isinstance(x, str) and ID_RE.match(x) for x in data):
        return True
    return all(isinstance(x, list) and len(x) == 2 and all(isinstance(v, int) for v in x) for x in data)


def _represent_list(dumper: yaml.SafeDumper, data: list) -> yaml.Node:
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=_is_flow_list(data))


def _represent_str(dumper: yaml.SafeDumper, data: str) -> yaml.Node:
    style = "|" if "\n" in data else None  # multi-line labels as literal blocks
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)


_YamlDumper.add_representer(list, _represent_list)
_YamlDumper.add_representer(str, _represent_str)


def to_yaml(d: Diagram) -> str:
    return yaml.dump(to_dict(d), Dumper=_YamlDumper, sort_keys=False, allow_unicode=True, width=100)


def from_yaml(text: str) -> Diagram:
    return from_dict(yaml.safe_load(text))


# --------------------------------------------------------------------------- #
# XML
# --------------------------------------------------------------------------- #
def _xml_text(s: str) -> str:
    # XML 1.0 cannot carry most C0 control characters at all.
    return "".join(ch if ch in "\n\t" or ord(ch) >= 0x20 else " " for ch in s)


def _set(el: ET.Element, key: str, value: Optional[str]) -> None:
    if value is not None:
        el.set(key, _xml_text(value))


def _child_text(parent: ET.Element, tag: str, text: str) -> None:
    if text:
        ET.SubElement(parent, tag).text = _xml_text(text)


def _xml_attrs(parent: ET.Element, attrs: dict[str, str]) -> None:
    for k, v in attrs.items():
        ET.SubElement(parent, "attr", {"name": k}).text = _xml_text(v)


def _bbox_str(b: Optional[tuple[int, ...]]) -> Optional[str]:
    return None if b is None else ",".join(str(v) for v in b)


def to_xml(d: Diagram) -> str:
    root = ET.Element("diagram", {"format": FORMAT_NAME, "version": VERSION, "type": d.type})
    _set(root, "direction", d.direction)
    _child_text(root, "title", d.title or "")
    _xml_attrs(root, d.attrs)

    nodes_el = ET.SubElement(root, "nodes")
    for n in d.nodes:
        el = ET.SubElement(nodes_el, "node", {"id": n.id, "kind": n.kind})
        _set(el, "bbox", _bbox_str(n.bbox))
        _set(el, "stereotype", n.stereotype)
        _child_text(el, "text", n.text)
        for m in n.members:
            mel = ET.SubElement(el, "member", {"kind": m.kind})
            _set(mel, "bbox", _bbox_str(m.bbox))
            mel.text = _xml_text(m.text)
        _xml_attrs(el, n.attrs)

    edges_el = ET.SubElement(root, "edges")
    for i, e in enumerate(d.edges):
        el = ET.SubElement(edges_el, "edge", {
            "id": edge_id(i), "source": e.source, "target": e.target,
            "kind": e.kind, "directed": "true" if e.directed else "false",
        })
        for key in _EDGE_OPTIONAL:
            _set(el, key, getattr(e, key))
        if e.path is not None:
            el.set("path", " ".join(f"{x},{y}" for x, y in e.path))
        _child_text(el, "label", e.label)
        _xml_attrs(el, e.attrs)

    groups_el = ET.SubElement(root, "groups")
    for g in d.groups:
        el = ET.SubElement(groups_el, "group", {"id": g.id, "kind": g.kind})
        _set(el, "parent", g.parent)
        _set(el, "bbox", _bbox_str(g.bbox))
        _child_text(el, "text", g.text)
        for m in g.members:
            ET.SubElement(el, "ref", {"id": m})
        _xml_attrs(el, g.attrs)

    ET.indent(root, space="  ")
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(root, encoding="unicode") + "\n"


def _ints(s: str, where: str) -> list[int]:
    try:
        return [int(v) for v in s.replace(" ", ",").split(",") if v != ""]
    except ValueError:
        raise ValueError(f"{where}: expected comma-separated integers, got {s!r}") from None


def _el_attrs(el: ET.Element) -> dict[str, str]:
    return {a.get("name", ""): a.text or "" for a in el.findall("attr")}


def _el_text(el: ET.Element, tag: str) -> str:
    child = el.find(tag)
    return (child.text or "") if child is not None else ""


def from_xml(text: str) -> Diagram:
    root = _SafeET.fromstring(text.encode("utf-8") if isinstance(text, str) else text)
    if root.tag != "diagram":
        raise ValueError(f"root element must be <diagram>, got <{root.tag}>")
    header_keys = ("format", "version", "type", "direction")
    obj: dict[str, Any] = {k: v for k, v in root.attrib.items() if k in header_keys}
    if root.find("title") is not None:
        obj["title"] = _el_text(root, "title")
    obj["attrs"] = _el_attrs(root)

    def opt_bbox(el: ET.Element, where: str) -> Optional[list[int]]:
        return _ints(el.get("bbox"), f"{where}.bbox") if el.get("bbox") is not None else None

    obj["nodes"] = []
    for i, el in enumerate(root.findall("./nodes/node")):
        w = f"nodes[{i}]"
        obj["nodes"].append({
            "id": el.get("id"), "kind": el.get("kind"), "text": _el_text(el, "text"),
            "bbox": opt_bbox(el, w), "stereotype": el.get("stereotype"),
            "members": [{"kind": m.get("kind"), "text": m.text or "", "bbox": opt_bbox(m, w)}
                        for m in el.findall("member")],
            "attrs": _el_attrs(el),
        })

    obj["edges"] = []
    for i, el in enumerate(root.findall("./edges/edge")):
        ed: dict[str, Any] = {k: el.get(k) for k in ("id", "source", "target", "kind") + _EDGE_OPTIONAL}
        ed["label"] = _el_text(el, "label")
        if el.get("path") is not None:
            nums = _ints(el.get("path"), f"edges[{i}].path")
            ed["path"] = [nums[k:k + 2] for k in range(0, len(nums), 2)]
        ed["attrs"] = _el_attrs(el)
        obj["edges"].append(ed)

    obj["groups"] = []
    for i, el in enumerate(root.findall("./groups/group")):
        obj["groups"].append({
            "id": el.get("id"), "kind": el.get("kind"), "text": _el_text(el, "text"),
            "parent": el.get("parent"), "bbox": opt_bbox(el, f"groups[{i}]"),
            "members": [r.get("id") for r in el.findall("ref")], "attrs": _el_attrs(el),
        })
    return from_dict(obj)


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #
LOADERS: dict[str, Callable[[str], Diagram]] = {
    "dgm": dsl.loads, "json": from_json, "yaml": from_yaml, "xml": from_xml,
}
DUMPERS: dict[str, Callable[[Diagram], str]] = {
    "dgm": dsl.dumps, "json": to_json, "yaml": to_yaml, "xml": to_xml,
}


def detect_format(text: str) -> str:
    head = text.lstrip("﻿ \t\r\n")
    if head.startswith("<"):
        return "xml"
    if head.startswith("{"):
        return "json"
    for line in head.splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            return "dgm" if s == dsl.HEADER or s.startswith(dsl.HEADER + " ") else "yaml"
    return "yaml"


def load(text: str, fmt: Optional[str] = None) -> Diagram:
    fmt = fmt or detect_format(text)
    if fmt not in LOADERS:
        raise ValueError(f"unknown format {fmt!r}; expected one of {FORMATS}")
    return LOADERS[fmt](text)


def dump(d: Diagram, fmt: str) -> str:
    if fmt not in DUMPERS:
        raise ValueError(f"unknown format {fmt!r}; expected one of {FORMATS}")
    return DUMPERS[fmt](d)


def convert(text: str, to: str, src: Optional[str] = None, canonical: bool = False) -> str:
    from .canonical import canonicalize

    d = load(text, src)
    return dump(canonicalize(d) if canonical else d, to)


__all__ = [
    "EXTENSIONS", "FORMATS", "convert", "detect_format", "dump", "from_dict", "from_json",
    "from_xml", "from_yaml", "load", "to_dict", "to_json", "to_xml", "to_yaml",
]
