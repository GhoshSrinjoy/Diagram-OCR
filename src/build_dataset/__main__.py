"""Build the v2 (Diagram IR) training data from the raw downloads.

Usage (from src/):
    python -m build_dataset --list
    python -m build_dataset --source logicbench_1k
    python -m build_dataset --source flowgen --limit 200      # smoke test
    python -m build_dataset --all
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
from typing import Optional

from .common import OUT_ROOT, write_samples

# source key -> module under build_dataset.sources (each exposes SOURCE and convert())
SOURCES = {
    "cas2uml": "cas2uml",
    "didi": "didi",
    "flowchart_to_mermaid": "flowchart_to_mermaid",
    "flowgen": "flowgen",
    "flowvqa": "flowvqa",
    "handwritten_fc_fa": "handwritten",
    "hdbpmn": "hdbpmn",
    "image_to_mermaid_v3": "image_to_mermaid",
    "logicbench_1k": "logicbench",
    "plantuml_stack_v2": "plantuml_stack_v2",
    "rakitha": "rakitha",
    "synth_bpmn": "synth_bpmn",            # synthetic, generated in-house
    "synth_circuits": "synth_circuits",
    "synth_plantuml": "synth_plantuml",
    "uml_in_the_wild": "uml_in_the_wild",
}


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", action="append", default=[], choices=sorted(SOURCES))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="max samples per source (smoke test)")
    args = ap.parse_args(argv)

    if args.list:
        for key, mod in SOURCES.items():
            print(f"{key:24} build_dataset.sources.{mod}")
        return 0
    keys = sorted(SOURCES) if args.all else args.source
    if not keys:
        ap.error("give --source, --all or --list")

    for key in keys:
        module = importlib.import_module(f"build_dataset.sources.{SOURCES[key]}")
        t0 = time.time()
        print(f"[{key}]", flush=True)
        stats = write_samples(key, module.convert(), limit=args.limit)
        print(f"  -> {OUT_ROOT / (key + '.jsonl')}: {stats['samples']} samples {stats['by_split']}, "
              f"{stats['with_layout']} with layout ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
