"""hdBPMN (hand-drawn BPMN, CC-BY-4.0) -> activity samples WITH boxes.

Each photo has a BPMN XML model with diagram-interchange bounds. The bounds live in the photo
rescaled so that its longer side is `backgroundSize` (stored in a leading XML comment), which
lets us normalize them to the 0..1000 grid. Mapping BPMN -> IR (type activity):

  task / subProcess / callActivity          -> action
  startEvent / endEvent                     -> initial / final
  intermediate events                       -> receive_signal / send_signal (message),
                                               time_event (timer), else other
  exclusive / inclusive / event gateway     -> decision (fan-out) or merge (fan-in)
  parallel gateway                          -> fork (fan-out) or join (fan-in)
  dataObject / dataStore reference          -> object (data stores get stereotype "datastore")
  textAnnotation                            -> note
  sequenceFlow -> flow; messageFlow -> flow, dashed; data association -> flow, dotted;
  association -> anchor. Pools and lanes -> lane groups (lanes nested in their pool); message
  flows to a collapsed (black-box) pool point at the pool group.

Split by writer (data/writer_split.csv), as in the papers.
"""

from __future__ import annotations

import csv
import re
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from pathlib import Path
from typing import Optional

from PIL import Image

from diagram_ir import Diagram, Edge, Group, Node
from diagram_ir.model import COORD_MAX

from ..common import DATA_ROOT, Sample, SkipLog, make_sample

SOURCE = "hdbpmn"
ROOT = DATA_ROOT / "handwritten" / "dwslab__hdBPMN" / "data"
LICENSE = "CC-BY-4.0"

M = "{http://www.omg.org/spec/BPMN/20100524/MODEL}"
DI = "{http://www.omg.org/spec/BPMN/20100524/DI}"
DC = "{http://www.omg.org/spec/DD/20100524/DC}"

ACTIONS = {"task", "userTask", "serviceTask", "sendTask", "receiveTask", "manualTask", "scriptTask",
           "businessRuleTask", "subProcess", "callActivity", "transaction"}
GATEWAYS = {"exclusiveGateway": "xor", "inclusiveGateway": "xor", "eventBasedGateway": "xor",
            "complexGateway": "xor", "parallelGateway": "and"}
_BG = re.compile(r'"backgroundSize"\s*:\s*(\d+)')


def _local(tag: str) -> str:
    return tag.split("}", 1)[-1]


def _name(el: ET.Element) -> str:
    if _local(el.tag) == "textAnnotation":
        t = el.find(f"{M}text")
        return (t.text or "") if t is not None else ""
    return el.get("name") or ""


def _event_kind(el: ET.Element, tag: str) -> str:
    defs = {_local(c.tag) for c in el}
    if tag == "startEvent":
        return "initial"
    if tag == "endEvent":
        return "final"
    if "timerEventDefinition" in defs:
        return "time_event"
    if "messageEventDefinition" in defs:
        return "send_signal" if tag == "intermediateThrowEvent" else "receive_signal"
    return "other"


def parse_bpmn(xml_text: str, frame: tuple[float, float]) -> Diagram:
    """frame = (width, height) of the coordinate system the bounds are expressed in."""
    root = ET.fromstring(xml_text.encode("utf-8"))
    fw, fh = frame

    bounds: dict[str, tuple[int, int, int, int]] = {}
    for shape in root.iter(f"{DI}BPMNShape"):
        b = shape.find(f"{DC}Bounds")
        if b is None:
            continue
        x, y, w, h = (float(b.get(k, 0)) for k in ("x", "y", "width", "height"))

        def nx(v: float) -> int:
            return max(0, min(COORD_MAX, round(v / fw * COORD_MAX)))

        def ny(v: float) -> int:
            return max(0, min(COORD_MAX, round(v / fh * COORD_MAX)))

        bounds[shape.get("bpmnElement", "")] = (nx(x), ny(y), nx(x + w), ny(y + h))

    nodes: dict[str, Node] = {}
    edges: list[Edge] = []
    fan_in: dict[str, int] = {}
    fan_out: dict[str, int] = {}
    gateway_type: dict[str, str] = {}
    process_of: dict[str, str] = {}  # node id -> process id

    for process in root.iter(f"{M}process"):
        pid = process.get("id", "")
        for el in process.iter():
            tag, eid = _local(el.tag), el.get("id")
            if not eid:
                continue
            kind: Optional[str] = None
            stereotype = None
            if tag in ACTIONS:
                kind = "action"
            elif tag.endswith("Event"):
                kind = _event_kind(el, tag)
            elif tag in GATEWAYS:
                kind, gateway_type[eid] = "decision", GATEWAYS[tag]
            elif tag in ("dataObjectReference", "dataStoreReference"):
                kind = "object"
                stereotype = "datastore" if tag == "dataStoreReference" else None
            elif tag == "textAnnotation":
                kind = "note"
            if kind is not None:
                nodes[eid] = Node(id=eid, kind=kind, text=_name(el), bbox=bounds.get(eid),
                                  stereotype=stereotype)
                process_of[eid] = pid
            if tag == "sequenceFlow":
                edges.append(Edge(source=el.get("sourceRef", ""), target=el.get("targetRef", ""),
                                  kind="flow", label=el.get("name") or ""))
            elif tag == "association":
                edges.append(Edge(source=el.get("sourceRef", ""), target=el.get("targetRef", ""),
                                  kind="anchor"))
            elif tag in ("dataInputAssociation", "dataOutputAssociation"):
                src = el.find(f"{M}sourceRef")
                tgt = el.find(f"{M}targetRef")
                owner = _owner_of(process, el)
                if tag == "dataInputAssociation" and src is not None and owner:
                    edges.append(Edge(source=src.text or "", target=owner, kind="flow", style="dotted"))
                elif tag == "dataOutputAssociation" and tgt is not None and owner:
                    edges.append(Edge(source=owner, target=tgt.text or "", kind="flow", style="dotted"))

    for mf in root.iter(f"{M}messageFlow"):
        edges.append(Edge(source=mf.get("sourceRef", ""), target=mf.get("targetRef", ""), kind="flow",
                          label=mf.get("name") or "", style="dashed"))

    # gateways: decision/merge, fork/join by fan-out vs fan-in of sequence flows
    for e in edges:
        if e.kind == "flow" and e.style is None:
            fan_out[e.source] = fan_out.get(e.source, 0) + 1
            fan_in[e.target] = fan_in.get(e.target, 0) + 1
    for gid, gtype in gateway_type.items():
        splits = fan_out.get(gid, 0) > 1 or fan_in.get(gid, 0) <= 1
        if gtype == "xor":
            nodes[gid].kind = "decision" if splits else "merge"
        else:
            nodes[gid].kind = "fork" if splits else "join"

    # pools and lanes
    groups: list[Group] = []
    in_lane: set[str] = set()
    pool_of_process: dict[str, str] = {}
    for part in root.iter(f"{M}participant"):
        gid = part.get("id", "")
        groups.append(Group(id=gid, kind="lane", text=part.get("name") or "", bbox=bounds.get(gid)))
        if part.get("processRef"):
            pool_of_process[part.get("processRef")] = gid
    for process in root.iter(f"{M}process"):
        pool = pool_of_process.get(process.get("id", ""))
        for lane in process.iter(f"{M}lane"):
            members = [r.text for r in lane.findall(f"{M}flowNodeRef") if r.text in nodes]
            in_lane.update(members)
            groups.append(Group(id=lane.get("id", ""), kind="lane", text=lane.get("name") or "",
                                members=members, parent=pool, bbox=bounds.get(lane.get("id", ""))))
    by_id = {g.id: g for g in groups}
    for nid, pid in process_of.items():
        pool = pool_of_process.get(pid)
        if pool and nid not in in_lane:
            by_id[pool].members.append(nid)

    known = set(nodes) | set(by_id)
    edges = [e for e in edges if e.source in known and e.target in known]
    return Diagram(type="activity", nodes=list(nodes.values()), edges=edges, groups=groups)


def _owner_of(process: ET.Element, assoc: ET.Element) -> Optional[str]:
    for el in process.iter():
        if assoc in list(el):
            return el.get("id")
    return None


def convert() -> Iterator[Sample]:
    skips = SkipLog(SOURCE)
    split_of = {}
    with open(ROOT / "writer_split.csv", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            split_of[row["writer"]] = row["split"]
    for bpmn_path in sorted((ROOT / "annotations").glob("*/*.bpmn")):
        stem = bpmn_path.stem
        image = _find_image(bpmn_path)
        if image is None:
            skips.skip("image missing", stem)
            continue
        text = bpmn_path.read_text(encoding="utf-8")
        bg = _BG.search(text)
        width, height = Image.open(image).size
        scale = int(bg.group(1)) / max(width, height) if bg else 1.0
        try:
            d = parse_bpmn(text, (width * scale, height * scale))
        except ET.ParseError as exc:
            skips.skip("xml error", f"{stem}: {exc}")
            continue
        writer = stem.split("_")[-1]
        split = {"train": "train", "val": "val", "test": "test"}.get(split_of.get(writer, "train"), "train")
        try:
            yield make_sample(source=SOURCE, local_id=stem, split=split, image=image, diagram=d,
                              use="train", license=LICENSE, meta={"exercise": stem.split("_")[0]})
        except ValueError as exc:
            skips.skip("invalid diagram", f"{stem}: {exc}")
    skips.report()


def _find_image(bpmn_path: Path) -> Optional[Path]:
    base = ROOT / "images" / bpmn_path.parent.name / bpmn_path.stem
    for ext in (".jpg", ".png", ".jpeg", ".JPG", ".PNG"):
        if base.with_suffix(ext).exists():
            return base.with_suffix(ext)
    return None
