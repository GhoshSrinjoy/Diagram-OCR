"""Convert the raw downloaded datasets into one unified training schema.

The raw datasets stay untouched under their category folders. This script reads
each one, normalizes it into a single canonical row shape, and writes:

    <output-dir>/
        train.jsonl
        val.jsonl
        test.jsonl
        images/<source_dataset>/...        # extracted / copied / rendered images
        format-stats.json                  # per-dataset / per-split / per-task counts

Canonical row schema (one JSON object per JSONL line):

    {
      "id":             stable unique id, dataset-prefixed
      "source_dataset": datasets.yaml key (identity preserved)
      "source_split":   original split name when known, else null
      "category":       diagram type when known (flowchart|directed_graph|chart|...)
      "task":           diagram_to_graph | diagram_to_code | diagram_qa | diagram_caption
      "target_format":  mermaid | graphviz_dot | tikz_latex | graph_json | svg | natural_language
      "image_path":     path relative to output-dir, or null if no image
      "input_text":     the instruction / question shown to the model
      "target_text":    the supervised target string
      "graph":          {directed, nodes[], edges[]} canonical IR, or null
      "metadata":       free-form provenance (never required to train)
    }

`task` is the high-level skill used for sampling/balancing; `target_format` is the
concrete serialization of `target_text`. Tiering (primary vs auxiliary) is NOT done
here - every dataset is converted and labeled, and the mix is chosen later via
sampling weights keyed on `task` / `source_dataset`.

Usage:
    python format_dataset.py                       # convert everything
    python format_dataset.py --dataset rakitha     # one dataset (key or short name)
    python format_dataset.py --limit 50            # cap rows per dataset (smoke test)
    python format_dataset.py --no-render-vgqa      # skip VGQA code->image rendering
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

DATA_ROOT = Path("D:/data/diagram_extraction")

# Closed vocabularies. Keep these in sync with the schema docstring.
TASKS = {"diagram_to_graph", "diagram_to_code", "diagram_qa", "diagram_caption"}
TARGET_FORMATS = {
    "mermaid",
    "graphviz_dot",
    "tikz_latex",
    "graph_json",
    "svg",
    "natural_language",
}

# Default instruction text per (task, target_format).
PROMPTS = {
    ("diagram_to_graph", "mermaid"): (
        "Convert this diagram into a Mermaid graph. "
        "Preserve every node, edge, direction, and label."
    ),
    ("diagram_to_graph", "graphviz_dot"): (
        "Convert this diagram into Graphviz DOT, "
        "preserving all nodes, edges, directions, and labels."
    ),
    ("diagram_to_graph", "graph_json"): (
        "Extract this diagram as a structured graph of nodes and edges (JSON), "
        "preserving labels, directions, and hierarchy."
    ),
    ("diagram_to_code", "tikz_latex"): "Reproduce this diagram as LaTeX/TikZ code.",
    ("diagram_to_code", "svg"): "Reproduce this figure as SVG code.",
    ("diagram_caption", "natural_language"): (
        "Describe this diagram, including its components and how they relate."
    ),
}


# --------------------------------------------------------------------------- #
# Conversion context
# --------------------------------------------------------------------------- #
@dataclass
class Options:
    data_root: Path
    output_dir: Path
    images_dir: Path
    limit: Optional[int]
    seed: int
    val_ratio: float
    test_ratio: float
    skip_images: bool
    overwrite: bool
    render_vgqa: bool
    vgqa_formats: tuple[str, ...]
    render_timeout: int


@dataclass
class Ctx:
    opts: Options
    renderers: dict[str, bool] = field(default_factory=dict)

    # ---- image helpers (all return a path relative to output_dir, or None) ----
    def _dest(self, subdir: str, name: str) -> Path:
        dest = self.opts.images_dir / subdir / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        return dest

    def _rel(self, dest: Path) -> str:
        return dest.relative_to(self.opts.output_dir).as_posix()

    def save_bytes(self, subdir: str, stem: str, data: bytes) -> Optional[str]:
        if self.opts.skip_images or not data:
            return None
        ext = _ext_from_bytes(data)
        dest = self._dest(subdir, f"{stem}{ext}")
        if self.opts.overwrite or not dest.exists() or dest.stat().st_size == 0:
            dest.write_bytes(data)
        return self._rel(dest)

    def copy_file(self, subdir: str, name: str, src: Path) -> Optional[str]:
        if self.opts.skip_images:
            return None
        if not src.exists():
            return None
        dest = self._dest(subdir, name)
        if self.opts.overwrite or not dest.exists() or dest.stat().st_size != src.stat().st_size:
            shutil.copy2(src, dest)
        return self._rel(dest)

    def render(self, subdir: str, stem: str, code: str, fmt: str) -> Optional[str]:
        """Render diagram source code to a PNG. Returns rel path or None."""
        if self.opts.skip_images:
            return None
        dest = self._dest(subdir, f"{stem}.png")
        if dest.exists() and dest.stat().st_size > 0 and not self.opts.overwrite:
            return self._rel(dest)
        ok = False
        timeout = self.opts.render_timeout
        try:
            if fmt == "graphviz_dot":
                ok = _render_dot(code, dest, timeout)
            elif fmt == "tikz_latex":
                ok = _render_latex(code, dest, timeout)
            elif fmt == "svg":
                ok = _render_svg(code, dest)
        except Exception:
            ok = False
        if ok and dest.exists() and dest.stat().st_size > 0:
            return self._rel(dest)
        return None


def make_row(
    *,
    id: str,
    source_dataset: str,
    task: str,
    target_format: str,
    target_text: str,
    input_text: Optional[str] = None,
    image_path: Optional[str] = None,
    source_split: Optional[str] = None,
    category: Optional[str] = None,
    graph: Optional[dict] = None,
    metadata: Optional[dict] = None,
    split: Optional[str] = None,
) -> dict:
    assert task in TASKS, f"bad task {task!r}"
    assert target_format in TARGET_FORMATS, f"bad target_format {target_format!r}"
    if input_text is None:
        input_text = PROMPTS.get((task, target_format), "Describe this diagram.")
    return {
        "id": id,
        "source_dataset": source_dataset,
        "source_split": source_split,
        "category": category,
        "task": task,
        "target_format": target_format,
        "image_path": image_path,
        "input_text": input_text,
        "target_text": target_text,
        "graph": graph,
        "metadata": metadata or {},
        "_split": split,  # internal hint; stripped before writing
    }


# --------------------------------------------------------------------------- #
# Small utilities
# --------------------------------------------------------------------------- #
def _ext_from_bytes(data: bytes) -> str:
    if data[:8].startswith(b"\x89PNG"):
        return ".png"
    if data[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return ".gif"
    if data[:2] == b"BM":
        return ".bmp"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    return ".png"


def _hash8(text: str) -> str:
    return hashlib.md5(text.encode("utf-8", "ignore")).hexdigest()[:8]


def _strip_mermaid_fences(text: str) -> str:
    t = text.strip()
    t = re.sub(r"^```[A-Za-z]*\s*\n?", "", t)  # leading ```mermaid / ```
    t = re.sub(r"\n?```\s*$", "", t)           # trailing ``` (may share a line)
    return t.strip()


# --------------------------------------------------------------------------- #
# Graph IR parsers (best-effort; never raise)
# --------------------------------------------------------------------------- #
_MERMAID_NODE = re.compile(
    r"([A-Za-z0-9_]+)\s*(\[\[|\[\(|\(\(|\(\[|\[|\(|\{\{|\{|>)\s*(.*?)\s*(\]\]|\)\]|\)\)|\]\)|\]|\)|\}\}|\})"
)
_MERMAID_EDGE = re.compile(
    r"([A-Za-z0-9_]+)\s*(-{2,3}>|-{2,3}|-\.->|-\.-|={2,3}>|--[ox])\s*(?:\|([^|]*)\|\s*)?([A-Za-z0-9_]+)"
)
_MERMAID_EDGE_MIDLABEL = re.compile(
    r"([A-Za-z0-9_]+)\s*--\s*([^>\-][^>]*?)\s*-->\s*([A-Za-z0-9_]+)"
)


def mermaid_to_graph(code: str) -> dict:
    code = _strip_mermaid_fences(code)
    first = code.splitlines()[0].strip().lower() if code.splitlines() else ""
    undirected_orient = first.startswith("graph") and " --- " in code and "-->" not in code
    nodes: dict[str, dict] = {}

    def add_node(nid: str, label: Optional[str] = None) -> None:
        n = nodes.setdefault(nid, {"id": nid, "label": nid})
        if label:
            n["label"] = label.strip().strip('"')

    for nid, _open, label, _close in _MERMAID_NODE.findall(code):
        add_node(nid, label)

    edges: list[dict] = []
    for line in code.splitlines():
        for m in _MERMAID_EDGE_MIDLABEL.finditer(line):
            a, lbl, b = m.group(1), m.group(2), m.group(3)
            add_node(a); add_node(b)
            edges.append({"source": a, "target": b, "label": lbl.strip() or None, "directed": True})
        for m in _MERMAID_EDGE.finditer(line):
            a, conn, lbl, b = m.group(1), m.group(2), m.group(3), m.group(4)
            add_node(a); add_node(b)
            directed = ">" in conn
            edges.append({"source": a, "target": b, "label": (lbl or "").strip() or None, "directed": directed})

    return {"directed": not undirected_orient, "nodes": list(nodes.values()), "edges": edges}


_DOT_NAME = r'(?:"[^"]*"|[A-Za-z_][\w.]*)'
_DOT_NODE = re.compile(rf'({_DOT_NAME})\s*\[[^\]]*\blabel\s*=\s*"([^"]*)"', re.DOTALL)
_DOT_EDGE = re.compile(rf'({_DOT_NAME})\s*(->|--)\s*({_DOT_NAME})\s*(?:\[[^\]]*\blabel\s*=\s*"([^"]*)")?')


def dot_to_graph(code: str) -> dict:
    directed = bool(re.search(r"\bdigraph\b", code))
    nodes: dict[str, dict] = {}

    def unquote(s: str) -> str:
        return s.strip().strip('"').strip()

    def add_node(nid: str, label: Optional[str] = None) -> None:
        nid = unquote(nid)
        if not nid or nid in {"graph", "node", "edge"}:
            return
        n = nodes.setdefault(nid, {"id": nid, "label": nid})
        if label:
            n["label"] = label

    for nid, label in _DOT_NODE.findall(code):
        add_node(nid, label)

    edges: list[dict] = []
    for a, conn, b, lbl in _DOT_EDGE.findall(code):
        a, b = unquote(a), unquote(b)
        if not a or not b:
            continue
        add_node(a); add_node(b)
        edges.append({"source": a, "target": b, "label": lbl or None, "directed": conn == "->"})

    return {"directed": directed, "nodes": list(nodes.values()), "edges": edges}


def ai2d_to_graph(entities: dict, relationships: Optional[list]) -> dict:
    nodes = []
    for eid, e in entities.items():
        nodes.append(
            {
                "id": eid,
                "label": e.get("label", eid),
                "type": e.get("type"),
                "bounds": e.get("bounds"),
            }
        )
    edges: list[dict] = []
    for rel in relationships or []:
        rel = rel.strip()
        m = re.match(r"(\S+)\s+connects to\s+(\S+)", rel)
        if m:
            edges.append({"source": m.group(1), "target": m.group(2), "label": None, "directed": True})
            continue
        m = re.match(r"(\S+)\s+labels\s+(\S+)", rel)
        if m:
            edges.append({"source": m.group(1), "target": m.group(2), "label": "labels", "directed": True})
            continue
        m = re.match(r"(\S+)\s+(.+?)\s+(\S+)$", rel)
        if m:
            edges.append({"source": m.group(1), "target": m.group(3), "label": m.group(2), "directed": True})
    return {"directed": True, "nodes": nodes, "edges": edges}


def detect_code_format(code: str) -> str:
    head = code[:400].lower()
    if "documentclass" in head or "begin{tikzpicture}" in head or "usetikzlibrary" in head:
        return "tikz_latex"
    if "digraph" in head or "graph {" in head or "graph{" in head or "->" in head:
        return "graphviz_dot"
    return "tikz_latex"


# --------------------------------------------------------------------------- #
# Renderers
# --------------------------------------------------------------------------- #
def _render_dot(code: str, dest: Path, timeout: int) -> bool:
    proc = subprocess.run(
        ["dot", "-Tpng", "-o", str(dest)],
        input=code.encode("utf-8", "ignore"),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=timeout,
    )
    return proc.returncode == 0


def _render_latex(code: str, dest: Path, timeout: int) -> bool:
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        (tdp / "main.tex").write_text(code, encoding="utf-8")
        run = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
            cwd=td,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
        )
        pdf = tdp / "main.pdf"
        if run.returncode != 0 or not pdf.exists():
            return False
        prefix = tdp / "page"
        conv = subprocess.run(
            ["pdftoppm", "-png", "-r", "150", "-singlefile", str(pdf), str(prefix)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
        )
        out = prefix.with_suffix(".png")
        if conv.returncode == 0 and out.exists():
            shutil.move(str(out), str(dest))
            return True
        return False


def _render_svg(code: str, dest: Path) -> bool:
    try:
        import cairosvg  # type: ignore
    except ImportError:
        return False
    cairosvg.svg2png(bytestring=code.encode("utf-8", "ignore"), write_to=str(dest))
    return True


def detect_renderers() -> dict[str, bool]:
    have = {
        "dot": shutil.which("dot") is not None,
        "pdflatex": shutil.which("pdflatex") is not None,
        "pdftoppm": shutil.which("pdftoppm") is not None,
    }
    try:
        import cairosvg  # type: ignore  # noqa: F401

        have["cairosvg"] = True
    except ImportError:
        have["cairosvg"] = False
    return have


# --------------------------------------------------------------------------- #
# Parquet helper
# --------------------------------------------------------------------------- #
def iter_parquet(paths: list[Path], columns: list[str], batch_size: int = 64) -> Iterator[dict]:
    import pyarrow.parquet as pq

    for path in paths:
        pf = pq.ParquetFile(str(path))
        for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
            for rec in batch.to_pylist():
                yield rec


def _image_bytes(rec_image: Any) -> bytes:
    if isinstance(rec_image, dict):
        return rec_image.get("bytes") or b""
    return b""


# --------------------------------------------------------------------------- #
# Converters (one per source_dataset). Each yields canonical rows.
# --------------------------------------------------------------------------- #
def convert_diagramgen(ctx: Ctx) -> Iterator[dict]:
    base = ctx.opts.data_root / "diagram_code" / "OpenRaiser__DiagramGen"
    cat_map = {
        "directed graph": "directed_graph",
        "undirected graph": "undirected_graph",
        "flowchart": "flowchart",
        "mind map": "mind_map",
        "model architecture diagram": "model_architecture",
        "line chart": "chart",
        "bar chart": "chart",
        "table": "table",
    }
    # (file, field holding the code that matches the shown image)
    sources = [("DiagramCoding.json", "reference_answer"), ("DiagramEditing.json", "original_code")]
    for fname, code_field in sources:
        fpath = base / fname
        if not fpath.exists():
            continue
        subset = "coding" if "Coding" in fname else "editing"
        records = json.loads(fpath.read_text(encoding="utf-8"))
        for rec in records:
            code = (rec.get(code_field) or "").strip()
            if not code:
                continue
            images = rec.get("images") or []
            img_rel = None
            if images:
                src = base / Path(str(images[0]).lstrip("./"))
                img_rel = ctx.copy_file("diagramgen", src.name, src)
            fmt = detect_code_format(code)
            if fmt == "graphviz_dot":
                task, graph = "diagram_to_graph", dot_to_graph(code)
            else:
                task, graph = "diagram_to_code", None
            yield make_row(
                id=f"diagramgen-{subset}-{rec.get('id')}",
                source_dataset="diagramgen",
                source_split="test",
                category=cat_map.get(rec.get("category", ""), rec.get("category")),
                task=task,
                target_format=fmt,
                target_text=code,
                image_path=img_rel,
                graph=graph,
                metadata={"orig_id": rec.get("id"), "subset": subset, "orig_category": rec.get("category")},
            )


def convert_rakitha(ctx: Ctx) -> Iterator[dict]:
    base = ctx.opts.data_root / "diagram_code" / "rakitha__mermaid-flowchart-transformer"
    for split in ("train", "test"):
        meta = base / split / "metadata.jsonl"
        if not meta.exists():
            continue
        with meta.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                fname = rec.get("file_name")
                mermaid = ""
                for msg in rec.get("messages", []):
                    if msg.get("role") == "assistant":
                        for part in msg.get("content", []):
                            if part.get("type") == "text" and part.get("text"):
                                mermaid = _strip_mermaid_fences(part["text"])
                if not mermaid or not fname:
                    continue
                img_rel = ctx.copy_file(f"rakitha/{split}", fname, base / split / fname)
                yield make_row(
                    id=f"rakitha-{split}-{Path(fname).stem}",
                    source_dataset="mermaid_flowchart_transformer",
                    source_split=split,
                    category="flowchart",
                    task="diagram_to_graph",
                    target_format="mermaid",
                    target_text=mermaid,
                    image_path=img_rel,
                    graph=mermaid_to_graph(mermaid),
                    metadata={"file_name": fname},
                    split=split,  # honor native split
                )


def convert_kamizuru(ctx: Ctx) -> Iterator[dict]:
    base = ctx.opts.data_root / "diagram_code" / "Kamizuru00__diagram_image_to_text" / "data"
    files = sorted(base.glob("*.parquet"))
    for i, rec in enumerate(iter_parquet(files, ["image", "text"])):
        text = (rec.get("text") or "").strip()
        if not text:
            continue
        img_rel = ctx.save_bytes("kamizuru", f"{i:06d}", _image_bytes(rec.get("image")))
        # Target is natural-language prose, not structured code -> caption task.
        yield make_row(
            id=f"kamizuru-{i:06d}",
            source_dataset="diagram_image_to_text",
            source_split="train",
            category="flowchart",
            task="diagram_caption",
            target_format="natural_language",
            target_text=text,
            image_path=img_rel,
        )


def _convert_qa_parquet(
    ctx: Ctx,
    *,
    source_dataset: str,
    subdir: str,
    files_by_split: dict[str, list[Path]],
    columns: list[str],
    to_qa: Callable[[dict], tuple[str, str]],
) -> Iterator[dict]:
    """Shared driver for the chart/figure QA parquet datasets."""
    idx = 0
    for split, files in files_by_split.items():
        for rec in iter_parquet(files, columns):
            question, answer = to_qa(rec)
            if not question:
                continue
            img_rel = ctx.save_bytes(subdir, f"{idx:06d}", _image_bytes(rec.get("image")))
            yield make_row(
                id=f"{subdir}-{idx:06d}",
                source_dataset=source_dataset,
                source_split=split,
                category="chart",
                task="diagram_qa",
                target_format="natural_language",
                input_text=question,
                target_text=answer,
                image_path=img_rel,
                metadata={"qa": {"question": question, "answer": answer}},
                split=split if split in {"train", "val", "test"} else None,
            )
            idx += 1


def convert_chartqa(ctx: Ctx) -> Iterator[dict]:
    base = ctx.opts.data_root / "chart_qa" / "HuggingFaceM4__ChartQA" / "data"
    files_by_split = {
        "train": sorted(base.glob("train-*.parquet")),
        "val": sorted(base.glob("val-*.parquet")),
        "test": sorted(base.glob("test-*.parquet")),
    }

    def to_qa(rec: dict) -> tuple[str, str]:
        label = rec.get("label") or []
        answer = label[0] if isinstance(label, list) and label else (label or "")
        return (rec.get("query") or "").strip(), str(answer).strip()

    yield from _convert_qa_parquet(
        ctx,
        source_dataset="chartqa_hfm4",
        subdir="chartqa",
        files_by_split=files_by_split,
        columns=["image", "query", "label", "human_or_machine"],
        to_qa=to_qa,
    )


def convert_plotqa(ctx: Ctx) -> Iterator[dict]:
    base = ctx.opts.data_root / "chart_qa" / "jrc__cleaned-plotqa-v2" / "data"
    files_by_split = {"train": sorted(base.glob("*.parquet"))}

    def to_qa(rec: dict) -> tuple[str, str]:
        return (rec.get("user") or "").strip(), (rec.get("assistant") or "").strip()

    yield from _convert_qa_parquet(
        ctx,
        source_dataset="cleaned_plotqa_v2",
        subdir="plotqa",
        files_by_split=files_by_split,
        columns=["image", "user", "assistant"],
        to_qa=to_qa,
    )


def convert_figureqa(ctx: Ctx) -> Iterator[dict]:
    base = ctx.opts.data_root / "chart_qa" / "jrc__cleaned-figureqa-v2" / "data"
    files_by_split = {"train": sorted(base.glob("*.parquet"))}

    def to_qa(rec: dict) -> tuple[str, str]:
        return (rec.get("user") or "").strip(), (rec.get("assistant") or "").strip()

    yield from _convert_qa_parquet(
        ctx,
        source_dataset="cleaned_figureqa_v2",
        subdir="figureqa",
        files_by_split=files_by_split,
        columns=["image", "user", "assistant"],
        to_qa=to_qa,
    )


def convert_ai2d(ctx: Ctx) -> Iterator[dict]:
    base = ctx.opts.data_root / "science_diagrams" / "lmms-lab__ai2d" / "data"
    files = sorted(base.glob("*.parquet"))
    for i, rec in enumerate(iter_parquet(files, ["question", "options", "answer", "image"])):
        question = (rec.get("question") or "").strip()
        options = list(rec.get("options") or [])
        ans_raw = rec.get("answer")
        answer = ""
        try:
            answer = options[int(ans_raw)]
        except (ValueError, TypeError, IndexError):
            answer = str(ans_raw or "")
        if not question:
            continue
        img_rel = ctx.save_bytes("ai2d", f"{i:06d}", _image_bytes(rec.get("image")))
        prompt = question
        if options:
            prompt += "\nOptions: " + "; ".join(f"{j}) {o}" for j, o in enumerate(options))
        yield make_row(
            id=f"ai2d-{i:06d}",
            source_dataset="ai2d",
            source_split="test",
            category="science_diagram",
            task="diagram_qa",
            target_format="natural_language",
            input_text=prompt,
            target_text=str(answer),
            image_path=img_rel,
            metadata={"qa": {"question": question, "options": options, "answer": str(answer)}},
        )


def convert_ai2d_caption(ctx: Ctx) -> Iterator[dict]:
    base = ctx.opts.data_root / "science_diagrams" / "abhayzala__AI2D-Caption"
    images_dir = base / "ai2d_images"
    # gpt4v: nodes + caption (no edges). planner/test: full graph with relationships.
    sources = [
        ("ai2d_caption_gpt4v.json", "gpt4v", None),
        ("ai2d_caption_planner_llm_training_examples.json", "planner", "train"),
        ("ai2d_caption_test.json", "test", "test"),
    ]
    seen: set[str] = set()
    for fname, tag, split in sources:
        fpath = base / fname
        if not fpath.exists():
            continue
        for rec in json.loads(fpath.read_text(encoding="utf-8")):
            image = rec.get("image")
            if not image:
                continue
            uid = f"ai2dcap-{tag}-{Path(image).stem}"
            if uid in seen:
                continue
            seen.add(uid)
            entities = rec.get("entities") or {}
            rels = rec.get("relationships") or rec.get("relations")
            graph = ai2d_to_graph(entities, rels)
            img_rel = ctx.copy_file("ai2d_caption", Path(image).name, images_dir / Path(image).name)
            if graph["edges"]:
                yield make_row(
                    id=uid,
                    source_dataset="ai2d_caption",
                    source_split=split or tag,
                    category="science_diagram",
                    task="diagram_to_graph",
                    target_format="graph_json",
                    target_text=json.dumps(graph, ensure_ascii=False),
                    image_path=img_rel,
                    graph=graph,
                    metadata={"topic": rec.get("topic"), "layout": rec.get("layout")},
                    split=split,
                )
            else:
                caption = (rec.get("caption") or "").strip()
                if not caption:
                    continue
                yield make_row(
                    id=uid,
                    source_dataset="ai2d_caption",
                    source_split=split or tag,
                    category="science_diagram",
                    task="diagram_caption",
                    target_format="natural_language",
                    target_text=caption,
                    image_path=img_rel,
                    graph=graph,  # nodes only, edges empty
                    metadata={"entity_count": len(entities)},
                    split=split,
                )


def convert_vgqa(ctx: Ctx) -> Iterator[dict]:
    base = ctx.opts.data_root / "diagram_code" / "vgbench__VGQA"
    csv.field_size_limit(10 ** 7)
    family_fmt = {"graphviz": "graphviz_dot", "tikz": "tikz_latex", "svg": "svg"}
    rendered = failed = 0
    seen_ids: set[str] = set()  # global per family: same diagram recurs across CSVs
    for csv_path in sorted(base.glob("*.csv")):
        family = csv_path.stem.split("_")[0]
        if family not in ctx.opts.vgqa_formats:
            continue
        fmt = family_fmt.get(family)
        if fmt is None:
            continue
        n_done = 0
        t0 = time.time()
        print(f"  [vgqa] {csv_path.stem} ({fmt})...", flush=True)
        with csv_path.open(encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for rec in reader:
                code = (rec.get("Code") or "").strip()
                if not code:
                    continue
                h = _hash8(code)
                uid = f"vgqa-{family}-{h}"
                if uid in seen_ids:
                    continue  # dedup: emit each unique diagram once (across all CSVs)
                seen_ids.add(uid)
                img_rel = None
                if ctx.opts.render_vgqa:
                    img_rel = ctx.render("vgqa", f"{family}_{h}", code, fmt)
                    if img_rel:
                        rendered += 1
                    else:
                        failed += 1
                n_done += 1
                if ctx.opts.render_vgqa and n_done % 25 == 0:
                    rate = n_done / max(time.time() - t0, 1e-6)
                    print(f"    {csv_path.stem}: {n_done} unique | {rendered} ok {failed} fail "
                          f"| {rate:.1f}/s", flush=True)
                if fmt == "graphviz_dot":
                    task, graph = "diagram_to_graph", dot_to_graph(code)
                else:
                    task, graph = "diagram_to_code", None
                yield make_row(
                    id=uid,
                    source_dataset="vgqa",
                    source_split=None,
                    category="directed_graph" if family == "graphviz" else family,
                    task=task,
                    target_format=fmt,
                    target_text=code,
                    image_path=img_rel,
                    graph=graph,
                    metadata={"family": csv_path.stem, "rendered_image": img_rel is not None},
                )
    if ctx.opts.render_vgqa:
        print(f"  [vgqa] rendered {rendered} images, {failed} failed/skipped")


def convert_mermaid_50k(ctx: Ctx) -> Iterator[dict]:
    """Stub: enabled once gated access is granted and the jsonl is downloaded."""
    base = ctx.opts.data_root / "diagram_code" / "CoreFidelity__Mermaid_50K"
    jsonl = base / "mermaid_50k.jsonl"
    if not jsonl.exists():
        print("  [mermaid_50k] no data yet (gated) - skipping. Request access, then re-download.")
        return
    with jsonl.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            # Best-effort field discovery; tighten once the real schema is known.
            code = rec.get("mermaid") or rec.get("code") or rec.get("target") or ""
            code = _strip_mermaid_fences(str(code))
            if not code:
                continue
            img_field = rec.get("image") or rec.get("image_path")
            img_rel = None
            if isinstance(img_field, dict) and img_field.get("bytes"):
                img_rel = ctx.save_bytes("mermaid_50k", f"{i:06d}", img_field["bytes"])
            yield make_row(
                id=f"mermaid50k-{i:06d}",
                source_dataset="mermaid_50k",
                source_split="train",
                category="flowchart",
                task="diagram_to_graph",
                target_format="mermaid",
                target_text=code,
                image_path=img_rel,
                graph=mermaid_to_graph(code),
            )


CONVERTERS: dict[str, Callable[[Ctx], Iterator[dict]]] = {
    "diagramgen": convert_diagramgen,
    "mermaid_flowchart_transformer": convert_rakitha,
    "diagram_image_to_text": convert_kamizuru,
    "chartqa_hfm4": convert_chartqa,
    "cleaned_plotqa_v2": convert_plotqa,
    "cleaned_figureqa_v2": convert_figureqa,
    "ai2d": convert_ai2d,
    "ai2d_caption": convert_ai2d_caption,
    "vgqa": convert_vgqa,
    "mermaid_50k": convert_mermaid_50k,
}

# Convenient short aliases for --dataset.
ALIASES = {
    "rakitha": "mermaid_flowchart_transformer",
    "kamizuru": "diagram_image_to_text",
    "chartqa": "chartqa_hfm4",
    "plotqa": "cleaned_plotqa_v2",
    "figureqa": "cleaned_figureqa_v2",
    "ai2dcaption": "ai2d_caption",
    "mermaid50k": "mermaid_50k",
}


# --------------------------------------------------------------------------- #
# Split assignment + writing
# --------------------------------------------------------------------------- #
def decide_split(row: dict, opts: Options) -> str:
    hint = row.get("_split")
    if hint in {"train", "val", "test"}:
        return hint
    h = int(hashlib.md5(f"{row['id']}|{opts.seed}".encode()).hexdigest(), 16)
    frac = (h % 10_000) / 10_000
    if frac < opts.test_ratio:
        return "test"
    if frac < opts.test_ratio + opts.val_ratio:
        return "val"
    return "train"


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the unified processed training format.")
    p.add_argument("--data-root", type=Path, default=DATA_ROOT)
    p.add_argument("--output-dir", type=Path, default=None, help="Default: <data-root>/processed")
    p.add_argument("--dataset", action="append", dest="datasets", default=[],
                   help="Dataset key or alias. Repeat for several. Default: all.")
    p.add_argument("--limit", type=int, default=None, help="Cap rows per dataset (smoke test).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-ratio", type=float, default=0.05)
    p.add_argument("--test-ratio", type=float, default=0.05)
    p.add_argument("--skip-images", action="store_true", help="Do not copy/extract/render images.")
    p.add_argument("--overwrite", action="store_true", help="Rewrite images that already exist.")
    p.add_argument("--append", action="store_true",
                   help="Append to existing {train,val,test}.jsonl instead of overwriting "
                        "(stage a fast bulk run, then a separate slow VGQA run).")
    p.add_argument("--no-render-vgqa", action="store_true", help="Skip VGQA code->image rendering.")
    p.add_argument("--vgqa-formats", default="graphviz,tikz,svg",
                   help="Comma list of VGQA families to convert: graphviz,tikz,svg")
    p.add_argument("--render-timeout", type=int, default=60, help="Per-render timeout (seconds).")
    return p.parse_args(argv)


def resolve_selection(names: list[str]) -> list[str]:
    if not names:
        return list(CONVERTERS.keys())
    resolved: list[str] = []
    for raw in names:
        key = ALIASES.get(raw.lower(), raw)
        if key not in CONVERTERS:
            raise SystemExit(f"Unknown dataset {raw!r}. Choices: {', '.join(CONVERTERS)}")
        resolved.append(key)
    return resolved


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    output_dir = (args.output_dir or (args.data_root / "processed")).resolve()
    opts = Options(
        data_root=args.data_root.resolve(),
        output_dir=output_dir,
        images_dir=output_dir / "images",
        limit=args.limit,
        seed=args.seed,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        skip_images=args.skip_images,
        overwrite=args.overwrite,
        render_vgqa=not args.no_render_vgqa,
        vgqa_formats=tuple(f.strip() for f in args.vgqa_formats.split(",") if f.strip()),
        render_timeout=args.render_timeout,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    ctx = Ctx(opts=opts, renderers=detect_renderers())

    selected = resolve_selection(args.datasets)
    print(f"Output -> {output_dir}")
    print(f"Renderers: {ctx.renderers}")
    if opts.render_vgqa and "svg" in opts.vgqa_formats and not ctx.renderers.get("cairosvg"):
        print("  note: cairosvg not installed - VGQA svg rows will have no image. `pip install cairosvg`")
    print(f"Datasets: {', '.join(selected)}\n")

    mode = "a" if args.append else "w"
    writers = {s: (output_dir / f"{s}.jsonl").open(mode, encoding="utf-8") for s in ("train", "val", "test")}
    stats: dict[str, Counter] = {
        "by_dataset": Counter(),
        "by_split": Counter(),
        "by_task": Counter(),
        "by_format": Counter(),
        "with_image": Counter(),
        "with_graph": Counter(),
    }
    per_dataset_split: dict[str, Counter] = defaultdict(Counter)

    try:
        for name in selected:
            print(f"[{name}] converting...")
            count = 0
            for row in CONVERTERS[name](ctx):
                if opts.limit is not None and count >= opts.limit:
                    break
                split = decide_split(row, opts)
                row.pop("_split", None)
                writers[split].write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
                stats["by_dataset"][name] += 1
                stats["by_split"][split] += 1
                stats["by_task"][row["task"]] += 1
                stats["by_format"][row["target_format"]] += 1
                if row.get("image_path"):
                    stats["with_image"][name] += 1
                if row.get("graph") and row["graph"].get("edges"):
                    stats["with_graph"][name] += 1
                per_dataset_split[name][split] += 1
            print(f"[{name}] {count} rows")
    finally:
        for w in writers.values():
            w.close()

    summary = {
        "by_dataset": dict(stats["by_dataset"]),
        "by_split": dict(stats["by_split"]),
        "by_task": dict(stats["by_task"]),
        "by_target_format": dict(stats["by_format"]),
        "rows_with_image": dict(stats["with_image"]),
        "rows_with_edges": dict(stats["with_graph"]),
        "per_dataset_split": {k: dict(v) for k, v in per_dataset_split.items()},
    }
    (output_dir / "format-stats.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    total = sum(stats["by_split"].values())
    print(f"\nDone. {total} rows -> {output_dir}")
    print(f"  splits: {dict(stats['by_split'])}")
    print(f"  tasks:  {dict(stats['by_task'])}")
    print(f"  stats written to {output_dir / 'format-stats.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
