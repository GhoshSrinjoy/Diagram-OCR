"""DangIT02/image-to-mermaid-v3 (MIT) -> flowchart / sequence / class / state / er / graph samples.

{train,val,test}.json: [{id, conversations: [user with <img>path</img>, assistant Mermaid]}].
26 Mermaid diagram types; those that are charts or timelines (pie, gantt, xychart, sankey,
journey, gitGraph, ...) have no graph structure and are skipped (UnsupportedMermaid).
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator

from diagram_ir.importers.mermaid import MermaidError, UnsupportedMermaid, mermaid_to_diagram

from ..common import DATA_ROOT, Sample, SkipLog, make_sample
from ._shared import strip_fences

SOURCE = "image_to_mermaid_v3"
ROOT = DATA_ROOT / "diagram_code" / "DangIT02__image-to-mermaid-v3"
LICENSE = "MIT"
_IMG = re.compile(r"<img>(.*?)</img>")


def convert() -> Iterator[Sample]:
    skips = SkipLog(SOURCE)
    for split in ("train", "val", "test"):
        for rec in json.loads((ROOT / f"{split}.json").read_text(encoding="utf-8")):
            sid = rec["id"]
            user = next((c["value"] for c in rec["conversations"] if c["from"] == "user"), "")
            code = next((c["value"] for c in reversed(rec["conversations"]) if c["from"] == "assistant"), "")
            m = _IMG.search(user)
            image = ROOT / m.group(1) if m else None
            if image is None or not image.exists():
                skips.skip("image missing", sid)
                continue
            try:
                d = mermaid_to_diagram(strip_fences(code))
            except UnsupportedMermaid:
                skips.skip(f"unsupported type ({sid.rsplit('_', 1)[0]})", sid)
                continue
            except MermaidError as exc:
                skips.skip("mermaid parse error", f"{sid}: {exc}")
                continue
            level = image.parent.name if image.parent.name in ("easy", "medium", "hard") else ""
            try:
                yield make_sample(source=SOURCE, local_id=sid, split=split, image=image, diagram=d,
                                  use="train", license=LICENSE,
                                  meta={"mermaid_type": sid.rsplit("_", 1)[0], "level": level})
            except ValueError as exc:
                skips.skip("invalid diagram", f"{sid}: {exc}")
    skips.report()
