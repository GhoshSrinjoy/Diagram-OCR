"""Pick the final model: evaluate candidate checkpoints on the same examples, then test the winner.

Reads a YAML recipe (config_compare.yaml):
  candidates  name -> checkpoint dir
  merges      name -> {checkpoint: weight}, built with train_v2.merge; each merge is a candidate too
  select      split / sources / per_source / seed of the ranking set
  final       the same for the report set; the winner and the `baseline` are evaluated on it
Candidates are ranked by the mean of per-source edge F1, so every source counts the same however
many edges its diagrams have. Each finished evaluation is written to <output>/ straight away and
reused on a rerun (unless its checkpoint or example set changed), so an interrupted comparison
picks up where it stopped.

Usage (from src/, D:/envs/diagram Python):
    python -m train_v2.compare --config train_v2/config_compare.yaml --dry-run   # plan only
    python -m train_v2.compare --config train_v2/config_compare.yaml
    .\\train_v2\\launch.ps1 -Module train_v2.compare -Config train_v2\\config_compare.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
import warnings
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForImageTextToText, AutoProcessor

from diagram_ir.metrics import VERSION as METRIC_VERSION

from .data import Collator, Example, ImageSizeCache, eval_subset, load_rows, to_examples
from .evaluate import ALL_EVAL_USES, print_table, rescore, run_eval
from .merge import WEIGHTS, merge

Recipe = list[tuple[Path, float]]


def load_examples(data: dict, spec: dict, per_source: int) -> list[Example]:
    root = Path(data["root"])
    rows = load_rows(root, dict.fromkeys(spec["sources"], {}), spec["split"], ALL_EVAL_USES)
    sizes = ImageSizeCache(root / "_image_sizes.json")
    by_source = to_examples(rows, Path(data["images_root"]), sizes, data["min_pixels"], data["max_pixels"],
                            10**9)
    sizes.save()
    missing = [s for s in spec["sources"] if s not in by_source]
    if missing:
        raise SystemExit(f"no {spec['split']} rows for: {', '.join(missing)}")
    return eval_subset(by_source, per_source, seed=spec["seed"])


def examples_key(examples: list[Example]) -> str:
    return hashlib.sha1("\n".join(sorted(e.id for e in examples)).encode("utf-8")).hexdigest()


def macro_edge_f1(report: dict) -> float:
    return statistics.mean(m["edge_f1"] for m in report["per_source"].values())


def resolve(cfg: dict) -> tuple[dict[str, Path], dict[str, Recipe]]:
    paths = {name: Path(p) for name, p in (cfg.get("candidates") or {}).items()}
    recipes: dict[str, Recipe] = {}
    for name, parts in (cfg.get("merges") or {}).items():
        if name in paths:
            raise SystemExit(f"'{name}' is both a candidate and a merge")
        recipes[name] = [(Path(p), float(w)) for p, w in parts.items()]
        paths[name] = Path(cfg["merge_root"]) / name
    return paths, recipes


def merge_is_current(out: Path, parts: Recipe) -> bool:
    """True when `out` was built from exactly these parts and none of them changed since."""
    recipe_file = out / "merge.json"
    if not recipe_file.exists() or not (out / WEIGHTS).exists():
        return False
    built = json.loads(recipe_file.read_text(encoding="utf-8"))["parts"]
    wanted = [{"checkpoint": str(p), "weight": w} for p, w in parts]
    newest_part = max((p / WEIGHTS).stat().st_mtime for p, _ in parts)
    return built == wanted and (out / WEIGHTS).stat().st_mtime > newest_part


def save_result(out_dir: Path, name: str, tag: str, path: Path, key: str, seconds: float, report: dict,
                samples: list[dict]) -> dict:
    report = {"name": name, "checkpoint": str(path), "set": tag, "examples_key": key,
              "metric_version": METRIC_VERSION, "macro_edge_f1": round(macro_edge_f1(report), 4),
              "seconds": round(seconds), **report}
    with open(out_dir / f"{name}.{tag}_samples.jsonl", "w", encoding="utf-8") as fh:
        for sample in samples:
            fh.write(json.dumps(sample, ensure_ascii=False) + "\n")
    (out_dir / f"{name}.{tag}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")  # last
    return report


def evaluate(name: str, path: Path, examples: list[Example], tag: str, cfg: dict, out_dir: Path) -> dict:
    """Score one checkpoint on one example set, reusing an earlier run's predictions when it can."""
    out_json = out_dir / f"{name}.{tag}.json"
    samples_file = out_dir / f"{name}.{tag}_samples.jsonl"
    key = examples_key(examples)
    if out_json.exists():
        old = json.loads(out_json.read_text(encoding="utf-8"))
        if old.get("examples_key") == key and out_json.stat().st_mtime > (path / WEIGHTS).stat().st_mtime:
            if old.get("metric_version") == METRIC_VERSION:
                print(f"{name}: {tag} result reused from {out_json.name}", flush=True)
                return old
            if samples_file.exists():  # same predictions, newer metric: score them again
                with open(samples_file, encoding="utf-8") as fh:
                    report, samples = rescore([json.loads(line) for line in fh])
                print(f"{name}: {tag} predictions re-scored with metric v{METRIC_VERSION}", flush=True)
                return save_result(out_dir, name, tag, path, key, old.get("seconds", 0), report, samples)
    print(f"\n== {name} on the {tag} set ({len(examples)} examples): {path}", flush=True)
    data = cfg["data"]
    processor = AutoProcessor.from_pretrained(path)
    model = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16).cuda().eval()
    collator = Collator(processor, data["min_pixels"], data["max_pixels"], layout_prob=0.0)
    batch_size = cfg["batch_size"]
    t0 = time.time()

    def progress(done: int, total: int) -> None:
        if done % 40 < batch_size or done == total:
            print(f"  {name} {tag}: {done}/{total}  ({(time.time() - t0) / done:.1f}s/example)", flush=True)

    report, samples = run_eval(model, processor, collator, examples, batch_size, cfg["max_new_tokens"],
                               progress)
    del model
    torch.cuda.empty_cache()
    return save_result(out_dir, name, tag, path, key, time.time() - t0, report, samples)


def print_ranking(reports: dict[str, dict]) -> list[str]:
    order = sorted(reports, key=lambda n: (reports[n]["macro_edge_f1"], reports[n]["overall"]["edge_f1"]),
                   reverse=True)
    print(f"\n{'rank':>4}  {'candidate':20} {'mean src edge':>13} {'edge F1':>8} {'node F1':>8} "
          f"{'+label':>7} {'valid':>7}")
    for i, n in enumerate(order, 1):
        o = reports[n]["overall"]
        print(f"{i:>4}  {n:20} {reports[n]['macro_edge_f1']:13.3f} {o['edge_f1']:8.3f} {o['node_f1']:8.3f} "
              f"{o['edge_label_f1']:7.3f} {o['valid_rate']:7.1%}")
    print("\nedge F1 per source")
    print(f"{'source':24}" + "".join(f"{n[:14]:>15}" for n in order))
    for s in sorted(reports[order[0]]["per_source"]):
        print(f"{s:24}" + "".join(f"{reports[n]['per_source'][s]['edge_f1']:15.3f}" for n in order))
    return order


def print_paired(new: dict, base: dict) -> None:
    print(f"\n{new['name']} vs {base['name']} (same images): edge F1 per source")
    print(f"{'source':24} {base['name'][:12]:>12} {new['name'][:12]:>12} {'change':>8}")
    rows = [(s, base["per_source"][s], new["per_source"][s]) for s in sorted(new["per_source"])]
    for s, b, n in rows + [("OVERALL", base["overall"], new["overall"])]:
        print(f"{s:24} {b['edge_f1']:12.3f} {n['edge_f1']:12.3f} {n['edge_f1'] - b['edge_f1']:+8.3f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path(__file__).with_name("config_compare.yaml"))
    ap.add_argument("--dry-run", action="store_true", help="print the plan and set sizes; no merges, no GPU")
    ap.add_argument("--smoke", action="store_true", help="2 examples per source, separate output dir")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    paths, recipes = resolve(cfg)
    baseline = cfg["baseline"]
    if baseline not in paths:
        raise SystemExit(f"baseline '{baseline}' is not a candidate")
    out_dir = Path(cfg["output"] + ("-smoke" if args.smoke else ""))
    select_spec, final_spec = cfg["select"], cfg["final"]
    select_set = load_examples(cfg["data"], select_spec, 2 if args.smoke else select_spec["per_source"])
    final_set = load_examples(cfg["data"], final_spec, 2 if args.smoke else final_spec["per_source"])

    missing = sorted({str(p) for n, p in paths.items() if n not in recipes and not (p / WEIGHTS).exists()}
                     | {str(p) for parts in recipes.values() for p, _ in parts if not (p / WEIGHTS).exists()})
    print(f"select set: {len(select_set)} {select_spec['split']} examples from {len(select_spec['sources'])} "
          f"sources; final set: {len(final_set)} {final_spec['split']} examples from "
          f"{len(final_spec['sources'])} sources")
    for name, path in paths.items():
        what = ("merge of " + ", ".join(f"{w:+g} x {p.parent.name}/{p.name}" for p, w in recipes[name])
                if name in recipes else str(path))
        print(f"  {name:20} {what}")
    if missing:
        print("not there yet:\n  " + "\n  ".join(missing))
    if args.dry_run:
        return 0
    if missing:
        raise SystemExit("missing checkpoints - nothing evaluated")

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, parts in recipes.items():
        if merge_is_current(paths[name], parts):
            print(f"merge {name}: up to date")
            continue
        print(f"building merge {name} ...", flush=True)
        merge(parts, paths[name])

    t0 = time.time()
    reports = {name: evaluate(name, path, select_set, "select", cfg, out_dir) for name, path in paths.items()}
    order = print_ranking(reports)
    winner = order[0]
    print(f"\nwinner: {winner}  ({paths[winner]})")

    final = {name: evaluate(name, paths[name], final_set, "final", cfg, out_dir)
             for name in dict.fromkeys([winner, baseline])}
    for name, report in final.items():
        print(f"\n{name} on the {final_spec['split']} set")
        print_table(report)
    if winner != baseline:
        print_paired(final[winner], final[baseline])

    summary = {
        "winner": winner, "checkpoint": str(paths[winner]),
        "ranked_by": f"mean per-source edge F1 on {len(select_set)} {select_spec['split']} examples",
        "ranking": [{"name": n, "checkpoint": str(paths[n]),
                     "mean_source_edge_f1": reports[n]["macro_edge_f1"],
                     "edge_f1": reports[n]["overall"]["edge_f1"], "node_f1": reports[n]["overall"]["node_f1"]}
                    for n in order],
        "final": {n: {"mean_source_edge_f1": r["macro_edge_f1"], "overall": r["overall"],
                      "per_source_edge_f1": {s: m["edge_f1"] for s, m in r["per_source"].items()}}
                  for n, r in final.items()},
        "minutes": round((time.time() - t0) / 60, 1),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
