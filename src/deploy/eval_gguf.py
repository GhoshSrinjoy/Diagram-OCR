"""Check a GGUF export with llama.cpp: accuracy on the train_v2 validation images, and speed.

The images are the first --per-source of train_v2's model-selection set (config_compare.yaml,
`select`), so they are a subset of the images the PyTorch candidates were scored on and the two
can be compared image by image (--torch-samples). llama-mtmd-cli runs once per image, so each
wall time includes loading the model; `load_s` (a 1-token run) estimates that part.

Usage (from src/, D:/envs/diagram Python):
    python -m deploy.eval_gguf --llama-bin D:/data/tools/llama-b11577-cpu --threads 16 \\
        --model <dir>/tadpole-0.8b-Q8_0.gguf --mmproj <dir>/mmproj-tadpole-0.8b-f16.gguf \\
        --torch-samples D:/data/diagram_extraction/runs/compare-post-rl/<name>.select_samples.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path

import yaml

from diagram_ir.metrics import VERSION as METRIC_VERSION
from diagram_ir.metrics import score
from train_v2.compare import load_examples
from train_v2.data import PROMPT_PLAIN
from train_v2.evaluate import gold_of, print_table, report_of

CONFIG = Path(__file__).resolve().parents[1] / "train_v2" / "config_compare.yaml"


def command(args: argparse.Namespace, image: str, n_predict: int) -> list[str]:
    cmd = [str(args.llama_bin / "llama-mtmd-cli.exe"), "-m", str(args.model), "--mmproj", str(args.mmproj),
           "--image", image, "-p", PROMPT_PLAIN, "--jinja", "--temp", "0", "-n", str(n_predict), "-c", "8192"]
    if not args.native_image_size:  # Tadpole 0.8B: the image token range it was trained with
        cmd += ["--image-min-tokens", "64", "--image-max-tokens", "1225"]
    if args.gpu_layers:
        cmd += ["-ngl", str(args.gpu_layers)]
    if args.threads:
        cmd += ["-t", str(args.threads)]
    return cmd


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llama-bin", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--mmproj", type=Path, required=True)
    ap.add_argument("--per-source", type=int, default=5)
    ap.add_argument("--gpu-layers", type=int, default=0, help="99 = whole model on the GPU (CUDA build)")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--native-image-size", action="store_true",
                    help="tiled models (Nano): keep the image size stored in the model file")
    ap.add_argument("--torch-samples", type=Path, help="PyTorch predictions for the same images")
    ap.add_argument("--out", type=Path, help="default: next to the model, eval_<model>_<device>.json")
    args = ap.parse_args()

    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    examples = load_examples(cfg["data"], cfg["select"], args.per_source)
    t0 = time.time()
    subprocess.run(command(args, examples[0].image, 1), capture_output=True, check=True)
    load_s = time.time() - t0

    results: dict[str, list[dict]] = {}
    samples = []
    for i, ex in enumerate(examples, 1):
        t0 = time.time()
        proc = subprocess.run(command(args, ex.image, 1536), capture_output=True, text=True, encoding="utf-8",
                              errors="replace")
        wall = time.time() - t0
        pred = proc.stdout.strip()
        r = score(pred, gold_of(ex))
        results.setdefault(ex.source, []).append(r)
        samples.append({"id": ex.id, "source": ex.source, "edge_f1": r["edge_f1"], "node_f1": r["node_f1"],
                        "valid": r["valid"], "seconds": round(wall, 2), "pred": pred, "gold": gold_of(ex)})
        if i % 10 == 0 or i == len(examples):
            print(f"  {i}/{len(examples)}  ({statistics.mean(s['seconds'] for s in samples):.1f}s/image)",
                  flush=True)

    report = report_of(results)
    secs = [s["seconds"] for s in samples]
    device = f"gpu{args.gpu_layers}" if args.gpu_layers else "cpu"
    report = {"model": str(args.model), "mmproj": str(args.mmproj), "device": device,
              "metric_version": METRIC_VERSION, "load_s": round(load_s, 2),
              "seconds_per_image": {"mean": round(statistics.mean(secs), 2),
                                    "median": round(statistics.median(secs), 2), "max": round(max(secs), 2)},
              **report}
    if args.torch_samples:
        torch_by_id = {}
        with open(args.torch_samples, encoding="utf-8") as fh:
            for line in fh:
                s = json.loads(line)
                torch_by_id[s["id"]] = s
        paired = [(s, torch_by_id[s["id"]]) for s in samples if s["id"] in torch_by_id]
        torch_report = report_of({src: [score(t["pred"], t["gold"]) for s, t in paired if s["source"] == src]
                                  for src in sorted({s["source"] for s, _ in paired})})
        same = sum(s["pred"] == t["pred"].strip() for s, t in paired)
        report["torch_same_images"] = {"n": len(paired), "overall": torch_report["overall"],
                                       "identical_text": same}
    out = args.out or args.model.with_name(f"eval_{args.model.stem}_{device}.json")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with open(out.with_name(out.stem + "_samples.jsonl"), "w", encoding="utf-8") as fh:
        for s in samples:
            fh.write(json.dumps(s, ensure_ascii=False) + "\n")
    print_table(report)
    sp = report["seconds_per_image"]
    print(f"\n{device}: {sp['mean']}s/image mean, {sp['median']}s median, {sp['max']}s max "
          f"(includes ~{load_s:.1f}s model load per run)")
    if "torch_same_images" in report:
        t = report["torch_same_images"]
        print(f"PyTorch on the same {t['n']} images: edge F1 {t['overall']['edge_f1']:.3f}, "
              f"node F1 {t['overall']['node_f1']:.3f}; identical output text {t['identical_text']}/{t['n']}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
