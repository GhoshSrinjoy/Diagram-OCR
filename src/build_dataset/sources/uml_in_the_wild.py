"""UML-in-the-Wild (Zenodo) -> rendered real-world UML TEST samples.

The compilation is CC-BY-4.0, but each file keeps its source repository's license and the
dataset does not record it. For a publicly released model these are therefore used for
evaluation only (use=eval_only, split=test): a type-balanced set of 3,000 diagrams. To train on
them instead, change USE to "train" (and give them a split) - a licensing decision, not a
technical one.
"""

from __future__ import annotations

import zipfile
from collections.abc import Iterator

from ..common import DATA_ROOT, Sample, SkipLog
from ._plantuml_sets import balanced_pick, parse_pool, render_samples

SOURCE = "uml_in_the_wild"
ZIP = DATA_ROOT / "uml" / "zenodo__uml-in-the-wild" / "puml_files.zip"
USE = "eval_only"
LICENSE = "CC-BY-4.0 compilation; per-file license of the source repository (unrecorded)"
CAPS = {"class": 900, "sequence": 700, "activity": 450, "state": 350, "graph": 600}


def _items(z: zipfile.ZipFile) -> Iterator[tuple[str, str]]:
    for name in z.namelist():
        if name.endswith(".puml"):
            yield name.rsplit("/", 1)[-1][:-5], z.read(name).decode("utf-8", "replace")


def convert() -> Iterator[Sample]:
    skips = SkipLog(SOURCE)
    with zipfile.ZipFile(ZIP) as z:
        pool = parse_pool(_items(z), skips)
    print("  parsed:", {k: len(v) for k, v in pool.items()}, flush=True)
    picked = balanced_pick(pool, CAPS, seed=11)
    yield from render_samples(SOURCE, picked, skips, USE, lambda _k: LICENSE, lambda _k: "test")
    skips.report()
