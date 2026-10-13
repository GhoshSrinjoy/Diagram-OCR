"""Handwritten diagram datasets (COCO; FA, FC_A, FC_Bscan) -> hand-drawn test samples WITH boxes.

Licenses (FC_A CC-BY-NC-SA, FA / FC_B / FC_Bscan none) rule out training a public model, so
every image becomes an evaluation sample (use=eval_only, split=test).

COCO annotations: shapes (terminator / process / decision / data / connection, or state /
final state), arrows with explicit `arrow_prev` / `arrow_next` shape ids, and text boxes.
  * text linked to a shape (`text_ann`, FC_A) or lying inside a shape -> the shape's text
  * other text -> label of the nearest arrow (FA transition labels, yes/no on flowcharts)
  * FA start arrows have no source -> an `initial` marker node at the arrow tail
FC_B is skipped: it has no transcriptions and shows the same diagrams as FC_Bscan.
FC_Bscan transcriptions use LaTeX ($M=1$\\\\$F=1$); they are reduced to plain text.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterator

from diagram_ir import Diagram, Edge, Node
from diagram_ir.model import COORD_MAX

from ..common import DATA_ROOT, Sample, SkipLog, make_sample

SOURCE = "handwritten_fc_fa"
ROOT = DATA_ROOT / "handwritten" / "bernhardschaefer__handwritten-diagram-datasets"
LICENSE = "FC_A CC-BY-NC-SA-3.0; FA, FC_Bscan no license (evaluation only)"
SUBSETS = {"fca": "flowchart", "fcb_scan": "flowchart", "fa": "state"}
KINDS = {
    "terminator": "terminal", "process": "process", "decision": "decision", "data": "io",
    "connection": "connector", "state": "state", "final state": "final",
}
_LATEX = [(r"\\cdot", "*"), (r"\\times", "x"), (r"\\leq?", "<="), (r"\\geq?", ">="), (r"\\neq", "!="),
          (r"\\leftarrow|\\gets", "<-"), (r"\\rightarrow|\\to", "->"), (r"\\ldots|\\dots", "...")]


def clean_latex(text: str) -> str:
    t = text.replace("\\\\", "\n").replace("$", "")
    for pat, rep in _LATEX:
        t = re.sub(pat, rep, t)
    t = re.sub(r"\\([A-Za-z]+)", r"\1", t)  # remaining commands: drop the backslash
    return re.sub(r"[{}]", "", t).strip()


def _inside(inner: list[float], outer: list[float], frac: float = 0.6) -> bool:
    ix1, iy1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix2 = min(inner[0] + inner[2], outer[0] + outer[2])
    iy2 = min(inner[1] + inner[3], outer[1] + outer[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    return inter >= frac * max(1.0, inner[2] * inner[3])


def _arrow_distance(point: tuple[float, float], arrow: dict) -> float:
    pts = arrow.get("waypoints") or []
    if not pts:
        kp = arrow.get("keypoints") or []
        pts = [kp[i:i + 2] for i in range(0, len(kp), 3) if kp[i + 2] > 0]
    if not pts:
        x, y, w, h = arrow["bbox"]
        pts = [[x + w / 2, y + h / 2]]
    return min(math.dist(point, p) for p in pts)


def convert_image(img: dict, anns: list[dict], dtype: str) -> Diagram:
    w, h = img["width"], img["height"]

    def clamp(v: float, size: float) -> int:
        return max(0, min(COORD_MAX, round(v / size * COORD_MAX)))

    def box(b: list[float]) -> tuple[int, int, int, int]:
        x, y, bw, bh = b
        return (clamp(x, w), clamp(y, h), clamp(x + bw, w), clamp(y + bh, h))

    shapes = {a["id"]: a for a in anns if a["category"] in KINDS}
    arrows = [a for a in anns if a["category"] == "arrow"]
    texts = [a for a in anns if a["category"] == "text"]
    node_text: dict[int, list[tuple[float, float, str]]] = {sid: [] for sid in shapes}
    arrow_text: dict[int, list[str]] = {a["id"]: [] for a in arrows}
    for t in texts:
        content = clean_latex(t.get("text") or "")
        if not content:
            continue
        x, y, bw, bh = t["bbox"]
        if t.get("text_ann") in shapes:
            node_text[t["text_ann"]].append((y, x, content))
            continue
        owner = next((sid for sid, s in shapes.items() if _inside(t["bbox"], s["bbox"])), None)
        if owner is not None:
            node_text[owner].append((y, x, content))
        elif arrows:
            centre = (x + bw / 2, y + bh / 2)
            near = min(arrows, key=lambda a: _arrow_distance(centre, a))
            arrow_text[near["id"]].append(content)

    nodes = []
    for sid, s in shapes.items():
        parts = [p[2] for p in sorted(node_text[sid])]
        text = " ".join(parts) if parts else clean_latex(s.get("text") or "")
        nodes.append(Node(id=f"s{sid}", kind=KINDS[s["category"]], text=text, bbox=box(s["bbox"])))
    edge_kind = "transition" if dtype == "state" else "flow"
    edges = []
    for a in arrows:
        src, tgt = a.get("arrow_prev"), a.get("arrow_next")
        if tgt not in shapes:
            continue
        if src not in shapes:  # start arrow (FA): a marker at the tail
            kp = a.get("keypoints") or [0, 0, 0, 0, 0, 0]
            tx, ty = kp[3], kp[4]
            nid = f"start{a['id']}"
            nodes.append(Node(id=nid, kind="initial", bbox=box([tx - 4, ty - 4, 8, 8])))
            edges.append(Edge(source=nid, target=f"s{tgt}", kind=edge_kind))
            continue
        edges.append(Edge(source=f"s{src}", target=f"s{tgt}", kind=edge_kind,
                          label=", ".join(arrow_text[a["id"]])))
    return Diagram(type=dtype, nodes=nodes, edges=edges)


def convert() -> Iterator[Sample]:
    skips = SkipLog(SOURCE)
    for subset, dtype in SUBSETS.items():
        for split_file in sorted((ROOT / "datasets" / subset).glob("*.json")):
            j = json.loads(split_file.read_text(encoding="utf-8"))
            by_image: dict[int, list[dict]] = {}
            for a in j["annotations"]:
                by_image.setdefault(a["image_id"], []).append(a)
            for img in j["images"]:
                image = _image_path(subset, split_file.stem, img["file_name"])
                if image is None:
                    skips.skip("image missing", f"{subset}/{img['file_name']}")
                    continue
                d = convert_image(img, by_image.get(img["id"], []), dtype)
                try:
                    local_id = f"{subset}/{split_file.stem}/{img['file_name']}"
                    yield make_sample(source=SOURCE, local_id=local_id, split="test", image=image,
                                      diagram=d, use="eval_only", license=LICENSE,
                                      meta={"subset": subset, "native_split": split_file.stem})
                except ValueError as exc:
                    skips.skip("invalid diagram", f"{subset}/{img['file_name']}: {exc}")
    skips.report()


def _image_path(subset: str, split: str, name: str):
    for cand in (ROOT / "datasets" / subset / split / name, ROOT / "datasets" / subset / name):
        if cand.exists():
            return cand
    hits = list((ROOT / "datasets" / subset).rglob(name))
    return hits[0] if hits else None
