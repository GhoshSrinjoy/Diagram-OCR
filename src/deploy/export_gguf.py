"""Export a trained Tadpole checkpoint to GGUF for llama.cpp (CPU and GPU) in several sizes.

Writes into --out:
  tadpole-0.8b-bf16.gguf         language model, bf16 (reference quality)
  tadpole-0.8b-Q8_0.gguf         about half the size, near-lossless
  tadpole-0.8b-Q4_K_M.gguf       smallest
  mmproj-tadpole-0.8b-f16.gguf   vision encoder + projector, needed with every size
The DGM chat template (the empty think block the model was trained with) is stored in the model
files, so `llama-mtmd-cli --jinja` formats prompts exactly as in training.

Usage (from src/, D:/envs/diagram Python):
    python -m deploy.export_gguf <checkpoint> --out D:/data/diagram_extraction/gguf/<name> \\
        --llama-src D:/data/tools/llama.cpp-b11577 --llama-bin D:/data/tools/llama-b11577-cpu

Run a file:
    llama-mtmd-cli -m tadpole-0.8b-Q8_0.gguf --mmproj mmproj-tadpole-0.8b-f16.gguf --jinja \\
        --image diagram.png -p "Convert this diagram to DGM." --temp 0 -n 1536 \\
        --image-min-tokens 64 --image-max-tokens 1225
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

TEMPLATE = Path(__file__).with_name("chat_template.jinja")  # Qwen-style DGM template (Tadpole 0.8B)
STEM = "tadpole-0.8b"
QUANTS = ("Q8_0", "Q4_K_M")


def run(cmd: list[str]) -> None:
    print("$", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


def export(checkpoint: Path, out: Path, llama_src: Path, llama_bin: Path, name: str, stem: str = STEM,
           template: Path | None = TEMPLATE) -> list[Path]:
    """`template=None` keeps the checkpoint's own chat template (models trained with it, e.g. Nano)."""
    out.mkdir(parents=True, exist_ok=True)
    convert = llama_src / "convert_hf_to_gguf.py"
    raw = out / f"{stem}-bf16.raw.gguf"
    bf16 = out / f"{stem}-bf16.gguf"
    mmproj = out / f"mmproj-{stem}-f16.gguf"
    run([sys.executable, convert, checkpoint, "--outtype", "bf16", "--outfile", raw])
    template_args = ["--chat-template-file", template] if template else []
    run([sys.executable, llama_src / "gguf-py" / "gguf" / "scripts" / "gguf_new_metadata.py", raw, bf16,
         *template_args, "--general-name", name, "--force"])
    raw.unlink()
    run([sys.executable, convert, checkpoint, "--mmproj", "--outtype", "f16", "--outfile", mmproj])
    files = [bf16, mmproj]
    for q in QUANTS:
        target = out / f"{stem}-{q}.gguf"
        run([llama_bin / "llama-quantize.exe", bf16, target, q])
        files.append(target)
    return files


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--llama-src", type=Path, required=True, help="llama.cpp source checkout (converter)")
    ap.add_argument("--llama-bin", type=Path, required=True, help="llama.cpp binaries (llama-quantize)")
    ap.add_argument("--name", default="Tadpole 0.8B")
    ap.add_argument("--stem", default=STEM, help="file name stem, e.g. tadpole-nano-258m")
    ap.add_argument("--keep-template", action="store_true",
                    help="keep the checkpoint's own chat template instead of the DGM template")
    args = ap.parse_args()
    files = export(args.checkpoint, args.out, args.llama_src, args.llama_bin, args.name, args.stem,
                   None if args.keep_template else TEMPLATE)
    for f in files:
        print(f"{f.stat().st_size / 1e6:8.0f} MB  {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
