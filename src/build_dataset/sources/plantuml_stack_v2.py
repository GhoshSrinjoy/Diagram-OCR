"""The Stack v2 PlantUML (filtered) -> rendered UML samples (class, sequence, activity, state, ...).

Only files from permissively licensed repositories (license_type == "permissive": MIT, Apache,
BSD, ...) are used. Sources are parsed with diagram_ir.importers.plantuml, deduplicated by
content, picked per diagram type (caps below), and rendered with PlantUML.
"""

from __future__ import annotations

from collections.abc import Iterator

import pyarrow.parquet as pq

from ..common import DATA_ROOT, Sample, SkipLog
from ._plantuml_sets import balanced_pick, parse_pool, render_samples
from ._shared import hash_split

SOURCE = "plantuml_stack_v2"
PARQUET = (DATA_ROOT / "uml" / "devgpt-aimotion__the-stack-v2_PlantUML" / "filter_uml_no_nonlatin"
           / "train-00000-of-00001.parquet")
CAPS = {"class": 9000, "sequence": 7000, "graph": 5000, "*": 100000}  # activity / state: all


def convert() -> Iterator[Sample]:
    skips = SkipLog(SOURCE)
    columns = ["blob_id", "code", "license_type", "detected_licenses"]
    rows = pq.read_table(PARQUET, columns=columns).to_pylist()
    licenses = {}
    items = []
    for r in rows:
        if r["license_type"] != "permissive":
            skips.skip("not permissively licensed", r["blob_id"])
            continue
        licenses[r["blob_id"]] = ", ".join(r["detected_licenses"] or []) or "permissive"
        items.append((r["blob_id"], r["code"]))
    pool = parse_pool(items, skips)
    print("  parsed:", {k: len(v) for k, v in pool.items()}, flush=True)
    picked = balanced_pick(pool, CAPS, seed=7)

    def split_of(key: str) -> str:
        s = hash_split(key, 0.05)
        return "test" if s == "val" and hash_split(key + "#", 0.4) == "val" else s

    yield from render_samples(SOURCE, picked, skips, "train", licenses.get, split_of)
    skips.report()
