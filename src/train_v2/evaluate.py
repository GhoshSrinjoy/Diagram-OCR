"""Evaluate a checkpoint: generate DGM for images and score against the gold graphs.

Usage (from src/, with the D:/envs/diagram Python):
    python -m train_v2.evaluate D:/data/diagram_extraction/runs/v2-qwen35-0.8b/final
    python -m train_v2.evaluate <ckpt> --split test --per-source 50 --sources hdbpmn,handwritten_fc_fa
    python -m train_v2.evaluate <ckpt> --split val --per-source 10       # the trainer's eval set size

Writes <ckpt>/eval_<split>.json (per-source + overall metrics) and eval_<split>_samples.jsonl.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional

import torch
import yaml
from transformers import AutoModelForImageTextToText, AutoProcessor

from diagram_ir.metrics import VERSION as METRIC_VERSION
from diagram_ir.metrics import aggregate, score

from .data import Collator, Example, ImageSizeCache, eval_subset, load_rows, to_examples
from .trainer import _strip, decode_answers, stop_token_ids

ALL_EVAL_USES = ("train", "eval_only")
KEYS = ("node_f1", "edge_f1", "edge_label_f1", "kind_acc", "text_exact_rate", "valid_rate", "exact_rate")
HEADER = ("source", "n", "node F1", "edge F1", "+label", "kind", "valid", "exact")


def summary(rs: list[dict]) -> dict:
    agg = aggregate(rs)
    return {k: round(agg[k], 4) for k in KEYS} | {"n": len(rs)}


def gold_of(ex: Example) -> str:
    """The structure-only target used for scoring (generation uses the plain prompt)."""
    return _strip(ex.dgm) if ex.has_layout else ex.dgm


@torch.no_grad()
def run_eval(model: Any, processor: Any, collator: Collator, examples: list[Example], batch_size: int = 8,
             max_new_tokens: int = 1536, progress: Optional[Callable[[int, int], None]] = None
             ) -> tuple[dict, list[dict]]:
    """Greedy-decode every example, score it, and return (report, per-sample records)."""
    stops = stop_token_ids(processor.tokenizer)
    results: dict[str, list[dict]] = {}
    samples = []
    for i in range(0, len(examples), batch_size):
        exs = examples[i:i + batch_size]
        inputs = {k: v.cuda() for k, v in collator.generation_inputs(exs).items()}
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                             eos_token_id=stops, pad_token_id=collator.pad_id)
        preds = decode_answers(processor.tokenizer, out[:, inputs["input_ids"].shape[1]:])
        for ex, pred in zip(exs, preds, strict=True):
            gold = gold_of(ex)
            r = score(pred, gold)
            results.setdefault(ex.source, []).append(r)
            samples.append({"id": ex.id, "source": ex.source, "edge_f1": r["edge_f1"],
                            "node_f1": r["node_f1"], "valid": r["valid"], "pred": pred, "gold": gold})
        if progress is not None:
            progress(i + len(exs), len(examples))
    return report_of(results), samples


def report_of(results: dict[str, list[dict]]) -> dict:
    all_results = [r for rs in results.values() for r in rs]
    return {"overall": summary(all_results) if all_results else {},
            "per_source": {src: summary(rs) for src, rs in sorted(results.items())}}


def rescore(samples: list[dict]) -> tuple[dict, list[dict]]:
    """Score stored predictions again with the current metric (no generation needed)."""
    results: dict[str, list[dict]] = {}
    out = []
    for s in samples:
        r = score(s["pred"], s["gold"])
        results.setdefault(s["source"], []).append(r)
        out.append(s | {"edge_f1": r["edge_f1"], "node_f1": r["node_f1"], "valid": r["valid"]})
    return report_of(results), out


def print_table(report: dict) -> None:
    print("\n{:24} {:>4} {:>8} {:>8} {:>7} {:>6} {:>6} {:>6}".format(*HEADER))
    rows = list(report["per_source"].items()) + [("OVERALL", report["overall"])]
    for name, m in rows:
        print(f"{name:24} {m['n']:>4} {m['node_f1']:8.3f} {m['edge_f1']:8.3f} {m['edge_label_f1']:7.3f} "
              f"{m['kind_acc']:6.3f} {m['valid_rate']:6.1%} {m['exact_rate']:6.1%}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    ap.add_argument("--split", default="test", choices=("val", "test"))
    ap.add_argument("--sources", default="", help="comma-separated; default: every built source")
    ap.add_argument("--per-source", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=1536)
    ap.add_argument("--seed", type=int, default=2)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    d = cfg["data"]
    root = Path(d["root"])
    names = args.sources.split(",") if args.sources else sorted(p.stem for p in root.glob("*.jsonl"))
    rows = load_rows(root, dict.fromkeys(names, {}), args.split, ALL_EVAL_USES)
    sizes = ImageSizeCache(root / "_image_sizes.json")
    by_source = to_examples(rows, Path(d["images_root"]), sizes, d["min_pixels"], d["max_pixels"], 10**9)
    sizes.save()
    examples = eval_subset(by_source, args.per_source, seed=args.seed)
    print(f"{len(examples)} examples from {len(by_source)} sources ({args.split})", flush=True)

    processor = AutoProcessor.from_pretrained(args.checkpoint)
    model = AutoModelForImageTextToText.from_pretrained(args.checkpoint, dtype=torch.bfloat16).cuda().eval()
    collator = Collator(processor, d["min_pixels"], d["max_pixels"], layout_prob=0.0)
    t0 = time.time()

    def progress(done: int, total: int) -> None:
        print(f"  {done}/{total}  ({(time.time() - t0) / done:.1f}s/example)", flush=True)

    report, samples = run_eval(model, processor, collator, examples, args.batch_size, args.max_new_tokens,
                               progress)
    report = {"checkpoint": str(args.checkpoint), "split": args.split, "per_source_n": args.per_source,
              "metric_version": METRIC_VERSION,
              **report}
    out_json = args.checkpoint / f"eval_{args.split}.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with open(args.checkpoint / f"eval_{args.split}_samples.jsonl", "w", encoding="utf-8") as fh:
        for sample in samples:
            fh.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print_table(report)
    print(f"\nwrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
