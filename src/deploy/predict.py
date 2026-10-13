"""Read diagram images and write their structure as DGM, JSON, YAML or XML (PyTorch / transformers).

Usage (from src/, D:/envs/diagram Python):
    python -m deploy.predict diagram.png                              # DGM to stdout
    python -m deploy.predict diagram.png --to json                    # JSON to stdout
    python -m deploy.predict a.png b.jpg --to json,yaml,xml -o out/   # out/a.json, out/a.yaml, ...
    python -m deploy.predict photo.jpg --boxes --to json              # also ask for 0..1000 boxes

Uses the GPU when there is one, otherwise the CPU (slow; on CPU prefer the GGUF build with
llama.cpp, see the README). The output is parsed leniently, so a cut-off answer still yields
every complete element.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from diagram_ir import dump, parse
from train_v2.data import Collator, Example
from train_v2.trainer import decode_answers, stop_token_ids

DEFAULT_CHECKPOINT = Path("D:/data/diagram_extraction/runs/merges/rl-m2-wise")
EXTENSIONS = {"dgm": ".dgm", "json": ".json", "yaml": ".yaml", "xml": ".xml"}
MIN_PIXELS, MAX_PIXELS = 65536, 1254400  # as in training: ~64 to ~1225 visual tokens


def predict(images: list[Path], checkpoint: Path, boxes: bool = False, batch_size: int = 4,
            max_new_tokens: int = 1536) -> list[str]:
    """DGM text for each image."""
    cuda = torch.cuda.is_available()
    processor = AutoProcessor.from_pretrained(checkpoint)
    model = AutoModelForImageTextToText.from_pretrained(
        checkpoint, dtype=torch.bfloat16 if cuda else torch.float32).to("cuda" if cuda else "cpu").eval()
    collator = Collator(processor, MIN_PIXELS, MAX_PIXELS, layout_prob=0.0)
    stops = stop_token_ids(processor.tokenizer)
    out = []
    for i in range(0, len(images), batch_size):
        batch = [Example(id=str(p), source="input", image=str(p), dgm="", has_layout=False, est_tokens=0)
                 for p in images[i:i + batch_size]]
        inputs = collator.generation_inputs(batch, with_layout=boxes)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, eos_token_id=stops,
                                 pad_token_id=collator.pad_id)
        out.extend(decode_answers(processor.tokenizer, gen[:, inputs["input_ids"].shape[1]:]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", type=Path, nargs="+")
    ap.add_argument("--to", default="dgm", help="comma-separated: dgm, json, yaml, xml")
    ap.add_argument("-o", "--out", type=Path, help="output folder (default: print to stdout)")
    ap.add_argument("--boxes", action="store_true", help="ask for element boxes (0..1000 coordinates)")
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    formats = [f.strip() for f in args.to.split(",") if f.strip()]
    unknown = [f for f in formats if f not in EXTENSIONS]
    if unknown:
        ap.error(f"unknown format(s): {', '.join(unknown)}")
    texts = predict(args.images, args.checkpoint, args.boxes, args.batch_size)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
    for image, text in zip(args.images, texts, strict=True):
        diagram, problems = parse(text)
        if problems:
            print(f"{image}: {len(problems)} malformed line(s) skipped", file=sys.stderr)
        for fmt in formats:
            rendered = dump(diagram, fmt)
            if args.out:
                target = args.out / (image.stem + EXTENSIONS[fmt])
                target.write_text(rendered, encoding="utf-8")
                print(f"wrote {target}", file=sys.stderr)
            else:
                sys.stdout.write(rendered if rendered.endswith("\n") else rendered + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
