"""rakitha/mermaid-flowchart-transformer -> flowchart samples.

{train,test}/metadata.jsonl holds file_name + chat messages whose assistant turn is the Mermaid.
The dataset card states no license, so everything is used as extra flowchart test data
(use=eval_only, split=test), including the augmented copies (`_gaussian`, `_rotate`, ...).
"""

from __future__ import annotations

import json
from collections.abc import Iterator

from diagram_ir.importers.mermaid import MermaidError, mermaid_flowchart_to_diagram

from ..common import DATA_ROOT, Sample, SkipLog, make_sample
from ._shared import strip_fences

SOURCE = "rakitha"
ROOT = DATA_ROOT / "diagram_code" / "rakitha__mermaid-flowchart-transformer"
LICENSE = "unspecified (HF dataset card has no license)"


def convert() -> Iterator[Sample]:
    skips = SkipLog(SOURCE)
    for native in ("train", "test"):
        for line in (ROOT / native / "metadata.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            fname = rec.get("file_name")
            code = ""
            for msg in rec.get("messages", []):
                if msg.get("role") == "assistant":
                    code = next((p["text"] for p in msg.get("content", [])
                                 if p.get("type") == "text" and p.get("text")), code)
            image = ROOT / native / (fname or "")
            if not fname or not image.exists():
                skips.skip("image missing", str(fname))
                continue
            try:
                d = mermaid_flowchart_to_diagram(strip_fences(code))
            except MermaidError as exc:
                skips.skip("mermaid parse error", f"{fname}: {exc}")
                continue
            base = fname.rsplit(".", 1)[0]
            try:
                yield make_sample(source=SOURCE, local_id=f"{native}/{base}", split="test", image=image,
                                  diagram=d, use="eval_only", license=LICENSE,
                                  meta={"augmented": any(t in base for t in ("_gaussian", "_rotate",
                                                                             "_blur", "_noise"))})
            except ValueError as exc:
                skips.skip("invalid diagram", f"{fname}: {exc}")
    skips.report()
