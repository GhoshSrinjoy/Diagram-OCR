"""CAS2UML (CC-BY-4.0) -> hand-drawn UML class / activity samples.

Parquet rows: image (hand-drawn photo/scan), type, code (PlantUML ground truth), xmi. The
PlantUML importer matches the XMI exactly on all 271 class diagrams (classes, relations,
direction, multiplicities). Native train / test kept; 5% of train becomes val.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator

import pyarrow.parquet as pq

from diagram_ir.importers.plantuml import plantuml_to_diagram

from ..common import DATA_ROOT, Sample, SkipLog, make_sample
from ._shared import hash_split, image_ext, save_image_bytes

SOURCE = "cas2uml"
ROOT = DATA_ROOT / "uml" / "Seym0n__cas2uml" / "data"
LICENSE = "CC-BY-4.0"


def convert() -> Iterator[Sample]:
    skips = SkipLog(SOURCE)
    for native in ("train", "test"):
        for i, row in enumerate(pq.read_table(ROOT / f"{native}-00000-of-00001.parquet").to_pylist()):
            data = row["image"]["bytes"]
            sid = f"{native}_{i:04d}_{hashlib.md5(data).hexdigest()[:8]}"
            try:
                d = plantuml_to_diagram(row["code"])
            except Exception as exc:
                skips.skip(f"parse: {type(exc).__name__}", sid)
                continue
            image = save_image_bytes(SOURCE, f"{native}/{sid}{image_ext(data)}", data)
            split = "test" if native == "test" else hash_split(sid)
            try:
                yield make_sample(source=SOURCE, local_id=sid, split=split, image=image, diagram=d,
                                  use="train", license=LICENSE, meta={"uml_type": row["type"]})
            except ValueError as exc:
                skips.skip("invalid diagram", f"{sid}: {exc}")
    skips.report()
