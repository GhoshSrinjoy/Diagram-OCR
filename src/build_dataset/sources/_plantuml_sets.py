"""Shared flow for code-only PlantUML corpora: parse -> dedupe -> balanced pick -> render -> samples."""

from __future__ import annotations

import hashlib
import random
import re
from collections import defaultdict
from collections.abc import Iterable, Iterator

from diagram_ir import Diagram, canonicalize, dumps
from diagram_ir.importers.plantuml import plantuml_to_diagram

from ..common import OUT_ROOT, Sample, SkipLog, make_sample
from ..render_plantuml import render_batch

MAX_TARGET_CHARS = 4500
BATCH = 1500


def _norm_hash(code: str) -> str:
    return hashlib.md5(re.sub(r"\s+", " ", code).strip().encode("utf-8", "ignore")).hexdigest()


def parse_pool(items: Iterable[tuple[str, str]], skips: SkipLog) -> dict[str, list[tuple[str, str, Diagram]]]:
    """(key, code) -> {diagram type: [(key, code, diagram)]}, deduplicated by normalized content."""
    seen: set[str] = set()
    pool: dict[str, list[tuple[str, str, Diagram]]] = defaultdict(list)
    for key, code in items:
        h = _norm_hash(code)
        if h in seen:
            skips.skip("duplicate content", key)
            continue
        seen.add(h)
        try:
            d = plantuml_to_diagram(code)
        except Exception as exc:  # importer raises PlantUMLError subclasses; be robust to the rest
            skips.skip(f"parse: {type(exc).__name__}", key)
            continue
        if len(d.nodes) < 2:
            skips.skip("fewer than 2 nodes", key)
            continue
        if _duplicate_class_names(d):
            # Known importer issue: a namespaced class referenced by its short name becomes a
            # phantom duplicate. The target would list a class that is not drawn - skip it.
            skips.skip("duplicate class names (namespace short-name reference)", key)
            continue
        if len(dumps(canonicalize(d))) > MAX_TARGET_CHARS:
            skips.skip("target too long", key)
            continue
        pool[d.type].append((key, code, d))
    return pool


def _duplicate_class_names(d: Diagram) -> bool:
    if d.type != "class":
        return False
    names = [n.text for n in d.nodes if n.kind in ("class", "abstract", "interface", "enum")]
    return len(names) != len(set(names))


def balanced_pick(pool: dict[str, list], caps: dict[str, int], seed: int) -> list[tuple[str, str, Diagram]]:
    rng = random.Random(seed)
    picked = []
    for dtype, items in sorted(pool.items()):
        items = items[:]
        rng.shuffle(items)
        picked.extend(items[: caps.get(dtype, caps.get("*", len(items)))])
    rng.shuffle(picked)
    return picked


def render_samples(source: str, picked: list[tuple[str, str, Diagram]], skips: SkipLog, use: str,
                   license_of, split_of) -> Iterator[Sample]:
    out_dir = OUT_ROOT / "images" / source
    for start in range(0, len(picked), BATCH):
        chunk = picked[start:start + BATCH]
        safe = {key: re.sub(r"[^\w.-]", "_", key) for key, _, _ in chunk}
        rendered = render_batch([(safe[k], code) for k, code, _ in chunk], out_dir)
        print(f"  {source}: rendered {start + len(chunk)}/{len(picked)}", flush=True)
        for key, _, d in chunk:
            png = rendered.get(safe[key])
            if png is None:
                skips.skip("render failed / cropped", key)
                continue
            try:
                yield make_sample(source=source, local_id=key, split=split_of(key), image=png, diagram=d,
                                  use=use, license=license_of(key), meta={"diagram_type": d.type})
            except ValueError as exc:
                skips.skip("invalid diagram", f"{key}: {exc}")
