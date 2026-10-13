"""DangIT02/flowchart-to-mermaid -> flowchart samples.

Parquet rows: id, image {bytes, path}, conversations [user prompt, assistant Mermaid].
Native train / validation / test splits are kept. Images are extracted once.
"""

from __future__ import annotations

from collections.abc import Iterator

import pyarrow.parquet as pq

from diagram_ir.importers.mermaid import MermaidError, mermaid_flowchart_to_diagram

from ..common import DATA_ROOT, Sample, SkipLog, make_sample
from ._shared import image_ext, save_image_bytes, strip_fences

SOURCE = "flowchart_to_mermaid"
ROOT = DATA_ROOT / "diagram_code" / "DangIT02__flowchart-to-mermaid" / "data"
LICENSE = "MIT"
SPLITS = {"train": "train", "validation": "val", "test": "test"}


def convert() -> Iterator[Sample]:
    skips = SkipLog(SOURCE)
    for file_split, split in SPLITS.items():
        table = pq.read_table(ROOT / f"{file_split}-00000-of-00001.parquet")
        for row in table.to_pylist():
            sid = row["id"]
            answer = next((c["value"] for c in reversed(row["conversations"])
                           if c["from"] in ("assistant", "gpt")), "")
            try:
                d = mermaid_flowchart_to_diagram(strip_fences(answer))
            except MermaidError as exc:
                skips.skip("mermaid parse error", f"{sid}: {exc}")
                continue
            data = row["image"]["bytes"]
            image = save_image_bytes(SOURCE, f"{split}/{sid}{image_ext(data)}", data)
            try:
                yield make_sample(source=SOURCE, local_id=sid, split=split, image=image, diagram=d,
                                  use="train", license=LICENSE,
                                  meta={"level": sid.split("_")[0]})
            except ValueError as exc:
                skips.skip("invalid diagram", f"{sid}: {exc}")
    skips.report()
