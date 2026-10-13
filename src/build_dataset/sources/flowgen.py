"""FlowGen (ICLR 2026) -> flowchart / graph samples, verified against the gold triplets.

Layout on disk:  {train,test}/{mermaid,graphviz,plantuml,diagrams}/<level>/<renderer>_<i>.{png,txt,code}
  train levels: easy / medium / hard; test levels: graph_* (clean) and scanned_* (scanned look).
Each sample has the render code (.mmd / .dot / .puml) and gold triplets (.txt):
  <Source, relation, Target>   relation = edge label, "connectedTo" when unlabelled,
                               "partOf" for membership in a container.

The code is parsed into the IR, split/merge junction dots are folded into direct edges (as the
triplets do), and the sample is kept only if its triplets match the gold set exactly.

Known upstream defects, handled here:
  * "diagrams" DOT files name clusters without quotes (`subgraph cluster_Fund Disbursement {`),
    which Graphviz rejects; the names are quoted before parsing.
  * Gold triplets of graphviz-renderer containers say `X, partOf, X` instead of naming the
    container, so for those only the member set is compared.
  * Scanned test images are named `<name>_scanned.png`.

Shapes: the "diagrams" renderer does not draw what the DOT shape names say (box is drawn as a
diamond, diamond as a trapezoid, ...). DIAGRAMS_SHAPES is the decoded mapping, checked by eye
on rendered samples. Images were rendered by FlowGen with its own styling, so targets carry no
boxes (a fresh Graphviz layout does not match their PNGs).
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator
from pathlib import Path

from diagram_ir import Diagram
from diagram_ir.importers.dot import DotError, dot_to_diagram
from diagram_ir.importers.mermaid import MermaidError, mermaid_flowchart_to_diagram
from diagram_ir.importers.plantuml import PlantUMLError, plantuml_graph_to_diagram
from diagram_ir.transform import contract_nodes

from ..common import DATA_ROOT, Sample, SkipLog, make_sample

SOURCE = "flowgen"
ROOT = DATA_ROOT / "diagram_code" / "Anonymous112233__FlowGen"
LICENSE = "Apache-2.0"
VAL_FRACTION = 0.05

# Graphviz renderer: standard shapes plus the extra ones FlowGen uses.
GRAPHVIZ_SHAPES = {"mdiamond": "decision", "octagon": "other", "folder": "other",
                   "point": "connector"}
# "diagrams" renderer: DOT shape name -> what is actually drawn.
DIAGRAMS_SHAPES = {
    "action": "process", "box": "decision", "rect": "display", "rectangle": "subroutine",
    "predefinedprocess": "subroutine", "ellipse": "loop_limit", "diamond": "manual_op",
    "parallelogram": "terminal", "inputoutput": "io", "preparation": "preparation",
    "database": "database", "document": "document", "multipledocuments": "document",
    "vnode": "connector", "vnode_shape": "connector", "hexagon": "manual_input",
    "circle": "connector",
}
CODE_EXT = {"mermaid": ".mmd", "graphviz": ".dot", "diagrams": ".dot", "plantuml": ".puml"}
_TRIPLET = re.compile(r"^<(.*)>$")
_UNQUOTED_CLUSTER = re.compile(r'^(\s*subgraph\s+)(cluster_[^"{]*?)\s*\{', re.MULTILINE)


def parse_code(renderer: str, code: str) -> Diagram:
    if renderer == "mermaid":
        d = mermaid_flowchart_to_diagram(code)
    elif renderer == "graphviz":
        d = dot_to_diagram(code, "flowchart", shapes=GRAPHVIZ_SHAPES)
    elif renderer == "diagrams":
        code = _UNQUOTED_CLUSTER.sub(lambda m: f'{m.group(1)}"{m.group(2)}" {{', code)
        d = dot_to_diagram(code, "flowchart", shapes=DIAGRAMS_SHAPES)
    else:
        d = plantuml_graph_to_diagram(code)
    # Every real FlowGen node is labelled; unlabelled ones are split/merge junctions (dots in
    # mermaid / plantuml / diagrams, tiny 0.1in boxes in graphviz).
    return contract_nodes(d, lambda n: not n.text.strip())


def triplets_of(d: Diagram) -> set[str]:
    text = {n.id: " ".join(n.text.split()) for n in d.nodes}
    text.update({g.id: " ".join(g.text.split()) for g in d.groups})
    out = set()
    for e in d.edges:
        rel = " ".join(e.label.split()) or "connectedTo"
        out.add(f"{text[e.source]}, {rel}, {text[e.target]}")
    for g in d.groups:
        for m in g.members:
            if m in text:
                out.add(f"{text[m]}, partOf, {text[g.id]}")
    return out


def gold_triplets(path: Path) -> set[str]:
    out = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        m = _TRIPLET.match(line.strip())
        if m:
            out.add(" ".join(m.group(1).split()))
    return out


def relax_self_part_of(got: set[str], gold: set[str]) -> tuple[set[str], set[str]]:
    """Gold `X, partOf, X` (upstream bug) -> compare only which nodes are in some container."""
    if not any(_self_part_of(t) for t in gold):
        return got, gold

    def membership_only(ts: set[str]) -> set[str]:
        out = {t for t in ts if ", partOf, " not in t}
        out |= {"member:" + t.split(", partOf, ")[0] for t in ts if ", partOf, " in t}
        return out

    return membership_only(got), membership_only(gold)


def _self_part_of(t: str) -> bool:
    if ", partOf, " not in t:
        return False
    a, b = t.split(", partOf, ", 1)
    return a == b


def _split(split_dir: str, key: str) -> str:
    if split_dir == "test":
        return "test"
    h = int(hashlib.md5(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "val" if h < VAL_FRACTION else "train"


def convert() -> Iterator[Sample]:
    skips = SkipLog(SOURCE)
    for split_dir in ("train", "test"):
        for renderer, ext in CODE_EXT.items():
            for level_dir in sorted((ROOT / split_dir / renderer).iterdir()):
                for code_path in sorted(level_dir.glob(f"*{ext}")):
                    key = f"{split_dir}/{renderer}/{level_dir.name}/{code_path.stem}"
                    scanned = level_dir.name.startswith("scanned")
                    image = code_path.with_name(code_path.stem + ("_scanned.png" if scanned else ".png"))
                    triplet_path = code_path.with_suffix(".txt")
                    if not image.exists() or image.stat().st_size < 1024:
                        skips.skip("image missing or LFS pointer", key)
                        continue
                    try:
                        d = parse_code(renderer, code_path.read_text(encoding="utf-8"))
                    except (MermaidError, DotError, PlantUMLError) as exc:
                        skips.skip(f"{renderer} parse error", f"{key}: {exc}")
                        continue
                    got, gold = relax_self_part_of(triplets_of(d), gold_triplets(triplet_path))
                    if got != gold:
                        missing, extra = gold - got, got - gold
                        skips.skip(f"{renderer} triplet mismatch",
                                   f"{key}: missing {sorted(missing)[:2]} extra {sorted(extra)[:2]}")
                        continue
                    variant = "scanned" if scanned else "clean"
                    try:
                        yield make_sample(
                            source=SOURCE, local_id=key, split=_split(split_dir, key), image=image,
                            diagram=d, use="train", license=LICENSE,
                            meta={"renderer": renderer, "level": level_dir.name.split("_")[-1],
                                  "variant": variant},
                        )
                    except ValueError as exc:
                        skips.skip("invalid diagram", f"{key}: {exc}")
    skips.report()
