"""Shared plumbing for the v2 dataset builders: the sample record and the writer.

Every source converter yields `Sample`s whose target is canonical DGM (diagram_ir). Output:

    D:/data/diagram_extraction/processed_v2/
        <source>.jsonl            one Sample per line
        images/<source>/...       only for images that had to be extracted (parquet bytes,
                                  renders); files that already exist on disk are referenced
        stats.json                per-source counts, refreshed by each run

Image paths are stored relative to DATA_ROOT so the folder can move as a whole.
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from diagram_ir import Diagram, canonicalize, dumps, errors, validate

DATA_ROOT = Path("D:/data/diagram_extraction")
OUT_ROOT = DATA_ROOT / "processed_v2"

SPLITS = ("train", "val", "test")
USES = ("train", "eval_only")


@dataclass
class Sample:
    id: str  # globally unique, prefixed with the source key
    source: str
    split: str  # train | val | test
    image: str  # relative to DATA_ROOT
    type: str  # diagram type (flowchart, class, ...)
    dgm: str  # canonical DGM target
    has_layout: bool  # target carries boxes that match `image`
    use: str  # train | eval_only (license-driven)
    license: str
    meta: dict = field(default_factory=dict)


def make_sample(
    *,
    source: str,
    local_id: str,
    split: str,
    image: Path,
    diagram: Diagram,
    use: str,
    license: str,
    meta: Optional[dict] = None,
) -> Sample:
    """Canonicalize + validate `diagram` and wrap it. Raises ValueError on structural errors."""
    assert split in SPLITS, split
    assert use in USES, use
    issues = errors(validate(diagram))
    if issues:
        raise ValueError("; ".join(str(i) for i in issues[:3]))
    d = canonicalize(diagram)
    return Sample(
        id=f"{source}/{local_id}",
        source=source,
        split=split,
        image=image.resolve().relative_to(DATA_ROOT.resolve()).as_posix(),
        type=d.type,
        dgm=dumps(d),
        has_layout=d.has_layout(),
        use=use,
        license=license,
        meta=meta or {},
    )


def extracted_image_path(source: str, name: str) -> Path:
    path = OUT_ROOT / "images" / source / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_samples(source: str, samples: Iterable[Sample], limit: Optional[int] = None) -> dict:
    """Write `<source>.jsonl` (rewritten each run) and merge counts into stats.json."""
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = OUT_ROOT / f"{source}.jsonl"
    by_split: Counter = Counter()
    by_type: Counter = Counter()
    layout = 0
    t0 = time.time()
    with open(out_path, "w", encoding="utf-8") as fh:
        for i, s in enumerate(_take(samples, limit)):
            fh.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")
            by_split[s.split] += 1
            by_type[s.type] += 1
            layout += s.has_layout
            if (i + 1) % 1000 == 0:
                print(f"  {source}: {i + 1} samples ({time.time() - t0:.0f}s)", flush=True)
    stats = {
        "samples": sum(by_split.values()),
        "by_split": dict(by_split),
        "by_type": dict(by_type),
        "with_layout": layout,
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    # Several builders can finish at the same time: tolerate a half-written file and replace it
    # atomically (stats.json is informational; the .jsonl files are the data).
    stats_path = OUT_ROOT / "stats.json"
    try:
        all_stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}
    except (OSError, json.JSONDecodeError):
        all_stats = {}
    all_stats[source] = stats
    tmp = stats_path.with_name(f"stats.{source}.tmp")
    tmp.write_text(json.dumps(all_stats, indent=2), encoding="utf-8")
    os.replace(tmp, stats_path)
    return stats


def _take(items: Iterable[Sample], limit: Optional[int]) -> Iterator[Sample]:
    for i, item in enumerate(items):
        if limit is not None and i >= limit:
            return
        yield item


class SkipLog:
    """Count and sample the reasons rows were skipped, so converters never fail silently."""

    def __init__(self, source: str):
        self.source = source
        self.reasons: Counter = Counter()
        self.examples: dict[str, str] = {}

    def skip(self, reason: str, example: str = "") -> None:
        self.reasons[reason] += 1
        self.examples.setdefault(reason, example)

    def report(self) -> None:
        if not self.reasons:
            return
        print(f"  {self.source}: skipped {sum(self.reasons.values())} rows")
        for reason, n in self.reasons.most_common():
            print(f"    {n:6}  {reason}  e.g. {self.examples[reason][:100]}")
