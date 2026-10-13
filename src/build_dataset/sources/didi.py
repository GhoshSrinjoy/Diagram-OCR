"""DiDi - Digital Ink Diagram data (Google, CC-BY-4.0) -> hand-drawn flowchart samples.

Each record is a tablet drawing (strokes of x/y/t points on a `writing_guide` canvas) of one of
6,555 GraphViz prompts (prompts/dot/<label_id>.dot, extracted from the prompts archive).
  * diagrams_20200131.ndjson        22,287 drawings whose prompts carry text the writer copied
  * diagrams_wo_text_20200131.ndjson 36,368 drawings of text-free prompts -> targets have no text
Strokes are rasterized (black ink on white, longest side 1024 px). The drawing follows the prompt
only loosely, so targets carry no boxes. Native train / valid / test splits are kept.
Prompt shapes: oval, box, diamond, parallelogram, octagon (-> "other": not a flowchart symbol).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from PIL import Image, ImageDraw

from diagram_ir import Diagram
from diagram_ir.importers.dot import DotError, dot_to_diagram

from ..common import DATA_ROOT, Sample, SkipLog, extracted_image_path, make_sample

SOURCE = "didi"
ROOT = DATA_ROOT / "handwritten" / "google__didi"
LICENSE = "CC-BY-4.0"
FILES = {"diagrams_20200131.ndjson": True, "diagrams_wo_text_20200131.ndjson": False}  # -> has text
SPLITS = {"train": "train", "valid": "val", "test": "test"}
MAX_SIDE = 1024
SHAPES = {"octagon": "other"}


def rasterize(drawing: list, guide: dict, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    w, h = guide.get("width", 2400), guide.get("height", 1432)
    scale = MAX_SIDE / max(w, h)
    img = Image.new("L", (max(1, round(w * scale)), max(1, round(h * scale))), 255)
    draw = ImageDraw.Draw(img)
    for stroke in drawing:
        pts = [(x * scale, y * scale) for x, y in zip(stroke[0], stroke[1], strict=False)]
        if len(pts) == 1:
            x, y = pts[0]
            draw.ellipse([x - 1.5, y - 1.5, x + 1.5, y + 1.5], fill=0)
        elif pts:
            draw.line(pts, fill=0, width=3, joint="curve")
    img.save(dest, optimize=False)


def convert() -> Iterator[Sample]:
    skips = SkipLog(SOURCE)
    prompts: dict[str, Diagram] = {}
    for fname, has_text in FILES.items():
        with open(ROOT / fname, encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                label_id, key = rec["label_id"], rec["key"]
                if label_id not in prompts:
                    dot_path = ROOT / "prompts" / "dot" / f"{label_id}.dot"
                    if not dot_path.exists():
                        skips.skip("prompt missing", label_id)
                        continue
                    try:
                        prompts[label_id] = dot_to_diagram(dot_path.read_text(encoding="utf-8"),
                                                           "flowchart", shapes=SHAPES)
                    except DotError as exc:
                        skips.skip("prompt parse error", f"{label_id}: {exc}")
                        continue
                d = prompts[label_id]
                if not has_text:
                    d = d.strip_layout()  # copy, then blank all text the writer did not draw
                    for n in d.nodes:
                        n.text = ""
                    for e in d.edges:
                        e.label = ""
                split = SPLITS.get(rec.get("split"), "train")
                image = extracted_image_path(SOURCE, f"{split}/{key}.png")
                rasterize(rec["drawing"], rec.get("writing_guide") or {}, image)
                try:
                    yield make_sample(source=SOURCE, local_id=key, split=split, image=image, diagram=d,
                                      use="train", license=LICENSE,
                                      meta={"text": has_text, "prompt": label_id})
                except ValueError as exc:
                    skips.skip("invalid diagram", f"{key}: {exc}")
    skips.report()
