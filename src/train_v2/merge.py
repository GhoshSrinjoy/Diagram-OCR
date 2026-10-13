"""Combine checkpoints of the same model as a weighted sum of their weights.

Every run in the chain (stage 2 -> stage 3 -> RL) starts from the one before it, so the weights
stay close enough to combine linearly:
  * interpolation (WiSE-FT, Wortsman et al. 2022): 0.5 * stage2 + 0.5 * rl often keeps most of the
    later run's gains and recovers robustness the earlier model had on image styles it never saw;
  * task arithmetic: move one run's update onto another starting point, e.g.
    stage2 + (rl - stage3) = the RL update without the stage-3 synthetic round.
The weights must sum to 1 so the result stays on the scale of a real checkpoint.

Usage (from src/, D:/envs/diagram Python):
    python -m train_v2.merge --out <dir> -m <stage2 ckpt> 1 -m <rl ckpt> 1 -m <stage3 ckpt> -1
"""

from __future__ import annotations

import argparse
import json
import shutil
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

WEIGHTS = "model.safetensors"
COPIED_SUFFIXES = (".json", ".jinja")  # model/generation/processor/tokenizer configs, chat template


def merge(parts: list[tuple[Path, float]], out: Path) -> None:
    """Write sum(weight * checkpoint) to `out`. Configs and tokenizer come from the first part."""
    total = sum(w for _, w in parts)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"merge weights must sum to 1, got {total:g}")
    tmp = out.with_name(out.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    with ExitStack() as stack:
        handles = [stack.enter_context(safe_open(str(p / WEIGHTS), framework="pt")) for p, _ in parts]
        keys = list(handles[0].keys())
        for (path, _), h in zip(parts[1:], handles[1:], strict=True):
            if set(h.keys()) != set(keys):
                raise ValueError(f"{path} holds different tensors than {parts[0][0]}")
        merged = {}
        for key in keys:
            tensors = [h.get_tensor(key) for h in handles]
            if not tensors[0].is_floating_point():
                merged[key] = tensors[0]
                continue
            # float64 so that +1/-1 task-arithmetic terms cancel without rounding the small deltas away
            acc = torch.zeros(tensors[0].shape, dtype=torch.float64)
            for (_, w), t in zip(parts, tensors, strict=True):
                acc += w * t.double()
            merged[key] = acc.to(tensors[0].dtype)
        save_file(merged, str(tmp / WEIGHTS), metadata=handles[0].metadata())
    for f in parts[0][0].iterdir():
        if f.is_file() and f.suffix in COPIED_SUFFIXES and not f.name.startswith("eval_"):
            shutil.copy2(f, tmp / f.name)
    recipe = {"parts": [{"checkpoint": str(p), "weight": w} for p, w in parts]}
    (tmp / "merge.json").write_text(json.dumps(recipe, indent=2), encoding="utf-8")
    if out.exists():
        shutil.rmtree(out)
    tmp.rename(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("-m", "--model", nargs=2, action="append", required=True,
                    metavar=("CHECKPOINT", "WEIGHT"))
    args = ap.parse_args()
    merge([(Path(p), float(w)) for p, w in args.model], args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
