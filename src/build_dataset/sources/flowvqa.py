"""FlowVQA -> flowchart samples.

Data/{train,test}_full.json map a key (code00340, wiki..., inst...) to a record whose
`mermaid` field is the flowchart. Images: "A. Main Set Flowchart Images/<key>.png" and the same
chart rendered bottom-to-top in "B. Directional Bias Bottom Top Set Images/<key>.png" - the
latter becomes a second sample with direction BT. The ~260 images without a JSON record are
skipped. Native train/test kept; 5% of train (by key, both variants together) becomes val.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

from diagram_ir.importers.mermaid import MermaidError, mermaid_flowchart_to_diagram

from ..common import DATA_ROOT, Sample, SkipLog, make_sample
from ._shared import hash_split, strip_fences

SOURCE = "flowvqa"
ROOT = DATA_ROOT / "diagram_code" / "flowvqa__flowvqa" / "Data"
LICENSE = "MIT"
VARIANTS = {"main": "A. Main Set Flowchart Images", "bottom_top": "B. Directional Bias Bottom Top Set Images"}


def convert() -> Iterator[Sample]:
    skips = SkipLog(SOURCE)
    for native in ("train", "test"):
        records = json.loads((ROOT / f"{native}_full.json").read_text(encoding="utf-8"))
        for key, rec in sorted(records.items()):
            try:
                base = mermaid_flowchart_to_diagram(strip_fences(rec.get("mermaid") or ""))
            except MermaidError as exc:
                skips.skip("mermaid parse error", f"{key}: {exc}")
                continue
            split = "test" if native == "test" else hash_split(key)
            for variant, folder in VARIANTS.items():
                image = ROOT / folder / f"{key}.png"
                if not image.exists():
                    skips.skip(f"{variant} image missing", key)
                    continue
                d = base
                if variant == "bottom_top":
                    d = base.strip_layout()  # copy
                    d.direction = "BT"
                try:
                    yield make_sample(source=SOURCE, local_id=f"{key}/{variant}", split=split, image=image,
                                      diagram=d, use="train", license=LICENSE,
                                      meta={"variant": variant, "origin": key[:4]})
                except ValueError as exc:
                    skips.skip("invalid diagram", f"{key}: {exc}")
    skips.report()
