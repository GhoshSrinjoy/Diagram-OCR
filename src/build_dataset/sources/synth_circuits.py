"""Synthetic logic circuits (generated in-house) -> circuit samples WITH boxes.

Fills the gap LogicBench-1K leaves (1K images, no boxes, weak edge recall): N_SAMPLES random
combinational circuits (2-6 inputs, 2-16 gates, 1-3 outputs; ~15% textbook circuits such as
adders, multiplexers, decoders), laid out in columns by logic depth with orthogonal wires, drawn
with schemdraw and put on white / scanned-cream / grid paper (build_dataset.synth.circuit_*).

Target conventions (type circuit):
  gates                and, or, not, nand, nor, xor, xnor, buf; text "" (nothing is written on them)
  primary inputs       "labeled" style (~50%): nodes of kind input / output whose text is the
  and outputs          drawn label (A, Cin, x1, S, Cout, ...); "bare" style: unlabeled wire stubs,
                       which are NOT nodes - the target holds gates only, as in LogicBench-1K
  wires                edge kind wire, driver -> sink; fan-out is one edge per sink; junction dots
                       are resolved (never nodes); a driver feeds at most one pin of a gate
  boxes                every node; gates tightly around the symbol (body and bubble, leads
                       excluded), terminals around label + stub end; 0..1000 on the saved image
  direction            visual signal flow after orientation: LR, RL, TB or BT

Each sample is deterministic in its index. Rendering runs in a spawn-context process pool;
samples are yielded in index order. Split by hash of the sample id: 3% test, 3% val, rest train.
"""

from __future__ import annotations

import hashlib
import multiprocessing as mp
import os
from collections import deque
from collections.abc import Iterator
from pathlib import Path

from diagram_ir import Diagram, Edge, Node, validate

from ..common import Sample, SkipLog, extracted_image_path, make_sample
from ..synth.circuit_render import render_sample

SOURCE = "synth_circuits"
LICENSE = "synthetic (generated in-house)"
N_SAMPLES = 8000
N_PROCS = 8  # other generators run concurrently
WINDOW = 64  # tasks in flight


def split_of(sample_id: str) -> str:
    h = int(hashlib.md5(sample_id.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "test" if h < 0.03 else ("val" if h < 0.06 else "train")


def to_diagram(res: dict) -> Diagram:
    return Diagram(
        type="circuit",
        direction=res["direction"],
        nodes=[Node(id=nid, kind=kind, text=text, bbox=tuple(bbox))
               for nid, kind, text, bbox in res["nodes"]],
        edges=[Edge(source=s, target=t, kind="wire") for s, t in res["edges"]],
    )


def convert() -> Iterator[Sample]:
    skips = SkipLog(SOURCE)
    out_dir = extracted_image_path(SOURCE, "_").parent
    os.environ.setdefault("MPLBACKEND", "Agg")  # inherited by the spawned workers
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=min(N_PROCS, os.cpu_count() or 1)) as pool:
        todo = iter(range(N_SAMPLES))
        pending: deque = deque()
        for i in todo:
            pending.append(pool.apply_async(render_sample, (i, str(out_dir))))
            if len(pending) >= WINDOW:
                break
        while pending:
            res = pending.popleft().get()
            nxt = next(todo, None)
            if nxt is not None:
                pending.append(pool.apply_async(render_sample, (nxt, str(out_dir))))
            if "error" in res:
                skips.skip(f"render: {res['error'][:60]}", str(res["index"]))
                continue
            d = to_diagram(res)
            issues = validate(d, strict=True)
            if issues:
                skips.skip(f"invalid: {issues[0]}", res["local_id"])
                continue
            try:
                yield make_sample(source=SOURCE, local_id=res["local_id"],
                                  split=split_of(f"{SOURCE}/{res['local_id']}"), image=Path(res["image"]),
                                  diagram=d, use="train", license=LICENSE, meta=res["meta"])
            except ValueError as exc:
                skips.skip(f"invalid: {exc}", res["local_id"])
    skips.report()
