"""Synthetic BPMN diagrams (generated in-house) -> activity samples WITH boxes.

Made to close the gap on hdBPMN (hand-drawn BPMN photos): random block-structured processes
(sequences, XOR / AND / OR / event-based gateway blocks, loops, early exits, timer and message
events), 1-3 lanes, optional partner pools (collapsed or expanded) with message flows, data
objects / stores and text annotations, laid out like BPMN (horizontal lanes, orthogonal flows)
and drawn in two styles:
  * clean   (~40%): tool style (bpmn.io / Camunda / Signavio-like themes), PNG or high-quality JPEG
  * sketch  (~60%): hand-drawn pen strokes and handwriting fonts on squared / lined / plain paper,
                    photographed (rotation, perspective, light, blur, noise, JPEG) or scanned

Target mapping is identical to sources/hdbpmn.py (type activity):
  task -> action; startEvent / endEvent -> initial / final; intermediate timer -> time_event,
  message catch / throw -> receive_signal / send_signal, other intermediate -> other;
  XOR / OR / event-based gateway -> decision (fan-out) or merge (fan-in); parallel -> fork / join;
  data object -> object, data store -> object stereo=datastore; text annotation -> note + anchor.
  Sequence flow -> flow (label = condition as drawn); message flow -> flow dashed (to the pool
  group for collapsed pools); data association -> flow dotted. Pools and lanes -> lane groups
  (lanes nested in their pool); lanes hold the flow nodes, the pool the data objects / notes.
Every visual line break is a "\\n" in the target; boxes enclose shapes only (not outside labels).

Samples are deterministic per index; ids are "<style>_<index>"; split by hash of the id
(3% test, 3% val). Rendering runs in a spawn process pool (top-level worker in synth/bpmn_worker).
"""

from __future__ import annotations

import hashlib
import multiprocessing as mp
import os
from collections.abc import Iterator
from pathlib import Path

from ..common import Sample, SkipLog, make_sample
from ..synth.bpmn_worker import make_one, worker_init

SOURCE = "synth_bpmn"
LICENSE = "synthetic (generated in-house)"
N_DEFAULT = 8000
MAX_PROCESSES = 7  # + the main process = 8; other generators run at the same time


def split_of(local_id: str) -> str:
    h = int(hashlib.md5(local_id.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "test" if h < 0.03 else "val" if h < 0.06 else "train"


def convert() -> Iterator[Sample]:
    n = int(os.environ.get("SYNTH_BPMN_N", N_DEFAULT))
    procs = max(1, min(MAX_PROCESSES, (os.cpu_count() or 2) - 1))
    skips = SkipLog(SOURCE)
    ctx = mp.get_context("spawn")
    with ctx.Pool(procs, initializer=worker_init) as pool:
        for res in pool.imap(make_one, range(n), chunksize=2):
            if "error" in res:
                skips.skip("generation failed", f"{res['i']}: {res['error'][-300:]}")
                continue
            try:
                yield make_sample(source=SOURCE, local_id=res["local_id"], split=split_of(res["local_id"]),
                                  image=Path(res["image"]), diagram=res["diagram"], use="train",
                                  license=LICENSE, meta=res["meta"])
            except ValueError as exc:
                skips.skip("invalid diagram", f"{res['local_id']}: {exc}")
    skips.report()
