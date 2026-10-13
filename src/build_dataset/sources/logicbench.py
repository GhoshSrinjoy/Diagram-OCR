"""LogicBench-1K -> circuit diagrams.

Annotations are gate-level directed graphs. Primary inputs / outputs are drawn as bare wires and
are not annotated, so the target has gates only. Images are 512x512 scans in varying
orientations; there are no boxes, so no direction is asserted.

Edge direction is not documented upstream. Read as driver -> receiver, 87 of 1,389 NOT gates
have fan-in > 1; read the other way, 269 would have fan-out > 1. So edges are kept as given
(driver -> receiver, the IR `wire` convention). ~6% of gates exceed their type's fan-in, i.e.
the labels are slightly noisy.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

from diagram_ir import Diagram, Edge, Node

from ..common import DATA_ROOT, Sample, SkipLog, make_sample

SOURCE = "logicbench_1k"
ROOT = DATA_ROOT / "circuits" / "TKCL-HF__LogicBench-1K"
LICENSE = "CC-BY-4.0"
GATES = {"AND", "OR", "NOT", "NAND", "NOR", "XOR", "XNOR"}


def convert() -> Iterator[Sample]:
    skips = SkipLog(SOURCE)
    split_of = {}
    for split in ("train", "val", "test"):
        for line in (ROOT / "annotations" / "splits" / f"{split}.txt").read_text().split():
            split_of[line.strip()] = split

    for ann_path in sorted((ROOT / "annotations").glob("lb1k_*.json")):
        sid = ann_path.stem
        ann = json.loads(ann_path.read_text(encoding="utf-8"))
        image = ROOT / ann.get("image_path", f"images/{sid}.jpg")
        if not image.exists():
            skips.skip("image missing", sid)
            continue
        if any(n["type"] not in GATES for n in ann["nodes"]):
            skips.skip("unknown gate type", sid)
            continue
        d = Diagram(
            type="circuit",
            nodes=[Node(id=n["id"], kind=n["type"].lower()) for n in ann["nodes"]],
            edges=[Edge(source=e["source"], target=e["target"], kind="wire") for e in ann["edges"]],
        )
        try:
            yield make_sample(source=SOURCE, local_id=sid, split=split_of.get(sid, "train"),
                              image=image, diagram=d, use="train", license=LICENSE)
        except ValueError as exc:
            skips.skip(f"invalid: {exc}", sid)
    skips.report()
