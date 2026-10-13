"""Training data for Tadpole: source mixing, prompts, and batch collation.

Rows come from processed_v2/<source>.jsonl (see build_dataset). Each epoch is a fresh mixture:
per source, take its train rows, apply `cap`, scale by `weight` (fractional weights subsample,
weights > 1 repeat), shuffle, then group by estimated length so batches pad little.

Samples whose target carries boxes are shown with the "with boxes" prompt `layout_prob` of
the time and otherwise with their boxes stripped, so the model learns both output modes.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from PIL import Image

from diagram_ir import canonicalize, dumps, loads

PROMPT_PLAIN = "Convert this diagram to DGM."
PROMPT_LAYOUT = "Convert this diagram to DGM with boxes."
TOKENS_PER_PATCH_AREA = 32 * 32  # 16 px patches, 2x2 merge -> one visual token per 32x32 px
# End-of-turn / end-of-text tokens of the supported model families (Qwen; Granite-Docling; SmolVLM).
END_TOKENS = ("<|im_end|>", "<|endoftext|>", "<|end_of_text|>", "<end_of_utterance>")
TILED_PLACEHOLDERS = ("<image>", "<fake_token_around_image>", "<global-img>")


def stop_token_ids(tokenizer: Any) -> list[int]:
    """Ids of the END_TOKENS this tokenizer has; generation stops at any of them."""
    vocab = tokenizer.get_vocab()
    return [vocab[t] for t in END_TOKENS if t in vocab]


@dataclass
class Example:
    id: str
    source: str
    image: str  # absolute path
    dgm: str
    has_layout: bool
    est_tokens: int


def load_rows(
    root: Path, sources: dict[str, dict], split: str, uses: tuple[str, ...]
) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for name in sources:
        path = root / f"{name}.jsonl"
        if not path.exists():
            continue
        rows = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                if r["split"] == split and r["use"] in uses:
                    rows.append(r)
        if rows:
            out[name] = rows
    return out


class ImageSizeCache:
    """Image dimensions are needed for length estimates; reading 100k headers takes a minute,
    so they are cached next to the processed data."""

    def __init__(self, path: Path):
        self.path = path
        self.sizes: dict[str, list[int]] = {}
        if path.exists():
            self.sizes = json.loads(path.read_text(encoding="utf-8"))
        self.dirty = False

    def get(self, image: str) -> tuple[int, int]:
        hit = self.sizes.get(image)
        if hit is None:
            try:
                with Image.open(image) as im:
                    hit = list(im.size)
            except OSError:
                hit = [0, 0]
            self.sizes[image] = hit
            self.dirty = True
        return hit[0], hit[1]

    def save(self) -> None:
        if self.dirty:
            self.path.write_text(json.dumps(self.sizes), encoding="utf-8")
            self.dirty = False


def visual_tokens(w: int, h: int, min_pixels: int, max_pixels: int) -> int:
    area = max(min(w * h, max_pixels), min_pixels)
    return max(1, int(area / TOKENS_PER_PATCH_AREA))


def to_examples(rows: dict[str, list[dict]], data_root: Path, sizes: ImageSizeCache,
                min_pixels: int, max_pixels: int, max_target_chars: int) -> dict[str, list[Example]]:
    out: dict[str, list[Example]] = {}
    for name, rs in rows.items():
        exs = []
        for r in rs:
            if len(r["dgm"]) > max_target_chars:
                continue
            image = str(data_root / r["image"])
            w, h = sizes.get(image)
            if w == 0:
                continue
            exs.append(Example(id=r["id"], source=name, image=image, dgm=r["dgm"],
                               has_layout=bool(r.get("has_layout")),
                               est_tokens=visual_tokens(w, h, min_pixels, max_pixels) + len(r["dgm"]) // 3))
        out[name] = exs
    return out


def build_epoch(by_source: dict[str, list[Example]], sources: dict[str, dict], rng: random.Random,
                batch_size: int, group_size: int = 64) -> list[list[Example]]:
    """Mixture for one epoch, returned as length-grouped batches in random order."""
    pool: list[Example] = []
    for name, exs in by_source.items():
        cfg = sources.get(name) or {}
        items = exs[:]
        rng.shuffle(items)
        if cfg.get("cap"):
            items = items[: int(cfg["cap"])]
        weight = float(cfg.get("weight", 1.0))
        whole, frac = int(weight), weight - int(weight)
        chosen = items * whole + items[: int(round(len(items) * frac))]
        pool.extend(chosen)
    rng.shuffle(pool)
    batches: list[list[Example]] = []
    chunk = batch_size * group_size
    for start in range(0, len(pool), chunk):
        block = sorted(pool[start:start + chunk], key=lambda e: e.est_tokens)
        batches.extend(block[i:i + batch_size] for i in range(0, len(block), batch_size))
    rng.shuffle(batches)
    return batches


def epoch_mix(batches: list[list[Example]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for b in batches:
        for e in b:
            counts[e.source] = counts.get(e.source, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def _coin(key: str) -> float:
    return int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF


def target_for(ex: Example, salt: int, layout_prob: float) -> tuple[str, str]:
    """(prompt, target DGM) for one example. The output mode is a deterministic coin flip on
    (example, salt), so DataLoader workers and resumed runs make the same choices."""
    if ex.has_layout and _coin(f"{ex.id}|{salt}") < layout_prob:
        return PROMPT_LAYOUT, ex.dgm.rstrip("\n")
    if ex.has_layout:
        return PROMPT_PLAIN, dumps(canonicalize(loads(ex.dgm).strip_layout())).rstrip("\n")
    return PROMPT_PLAIN, ex.dgm.rstrip("\n")


class Collator:
    """Builds model inputs; labels cover the answer (DGM + end-of-turn token) only.

    Works for Qwen-style processors (images sized by a pixel budget) and Idefics3-style ones
    (Granite-Docling, SmolVLM: 512-px tiles, one image list per sample, the checkpoint's own size)."""

    def __init__(self, processor: Any, min_pixels: int, max_pixels: int, layout_prob: float,
                 pad_multiple: int = 128):
        self.processor = processor
        self.tok = processor.tokenizer
        self.tiled = hasattr(processor, "image_seq_len")
        self.images_kwargs = {} if self.tiled else {"size": {"longest_edge": max_pixels,
                                                             "shortest_edge": min_pixels}}
        self.layout_prob = layout_prob
        # Padding lengths to a multiple keeps the set of sequence shapes small, so the Triton
        # (linear-attention) kernels compile a few dozen variants instead of one per length.
        self.pad_multiple = pad_multiple
        # The answer starts right after the assistant header the template adds for generation
        # (for Qwen that includes the empty think block).
        user = self._messages(PROMPT_PLAIN)
        with_header = processor.apply_chat_template(user, add_generation_prompt=True, enable_thinking=False)
        without = processor.apply_chat_template(user, add_generation_prompt=False, enable_thinking=False)
        assert with_header.startswith(without), "chat template: generation prompt is not a suffix"
        self.marker = self.tok(with_header[len(without):], add_special_tokens=False)["input_ids"]
        self.stop_ids = set(stop_token_ids(self.tok))
        self.pad_id = self.tok.pad_token_id

    @staticmethod
    def _messages(prompt: str) -> list[dict]:
        return [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]

    def _images(self, images: list[Image.Image]) -> list:
        return [[im] for im in images] if self.tiled else images

    def prompt_text(self, prompt: str) -> str:
        return self.processor.apply_chat_template(self._messages(prompt), add_generation_prompt=True,
                                                  enable_thinking=False)

    def full_text(self, prompt: str, answer: str) -> str:
        msgs = self._messages(prompt) + [{"role": "assistant", "content": [{"type": "text", "text": answer}]}]
        return self.processor.apply_chat_template(msgs, enable_thinking=False)

    def load_image(self, path: str) -> Image.Image:
        """RGB image; transparent areas become white. A plain convert("RGB") turns a transparent
        background black (some PlantUML themes render transparent PNGs), hiding black lines."""
        with Image.open(path) as im:
            if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
                rgba = im.convert("RGBA")
                page = Image.new("RGB", rgba.size, (255, 255, 255))
                page.paste(rgba, mask=rgba.getchannel("A"))
                return page
            return im.convert("RGB")

    def __call__(self, batch: list[Example], salt: int = 0) -> dict[str, torch.Tensor]:
        if self.tiled:
            # Idefics3-style processors count "<image>" anywhere in the text as an image slot, so a
            # target containing it (one PlantUML diagram has `list<image>`) cannot be encoded: skip it.
            kept = [ex for ex in batch if not any(p in ex.dgm for p in TILED_PLACEHOLDERS)]
            assert kept, "every sample in the batch contains an image placeholder string"
            batch = kept
        texts, images = [], []
        for ex in batch:
            prompt, answer = target_for(ex, salt, self.layout_prob)
            texts.append(self.full_text(prompt, answer))
            images.append(self.load_image(ex.image))
        self.tok.padding_side = "right"
        enc = self.processor(text=texts, images=self._images(images), return_tensors="pt", padding=True,
                             pad_to_multiple_of=self.pad_multiple, images_kwargs=self.images_kwargs)
        labels = torch.full_like(enc["input_ids"], -100)
        for i, ids in enumerate(enc["input_ids"].tolist()):
            start = _find_last(ids, self.marker)
            if start < 0:
                continue
            start += len(self.marker)
            # up to and including the end-of-turn token (the stop signal)
            end = next((j + 1 for j in range(start, len(ids)) if ids[j] in self.stop_ids), len(ids))
            labels[i, start:end] = enc["input_ids"][i, start:end]
        enc["labels"] = labels
        return dict(enc)

    def generation_inputs(self, batch: list[Example], with_layout: bool = False) -> dict[str, torch.Tensor]:
        texts = [self.prompt_text(PROMPT_LAYOUT if with_layout else PROMPT_PLAIN) for _ in batch]
        images = [self.load_image(ex.image) for ex in batch]
        self.tok.padding_side = "left"
        enc = self.processor(text=texts, images=self._images(images), return_tensors="pt", padding=True,
                             images_kwargs=self.images_kwargs)
        self.tok.padding_side = "right"
        return dict(enc)


class MicroBatches(torch.utils.data.Dataset):
    """Collates micro-batches in DataLoader workers (image decode + resize is CPU-heavy)."""

    def __init__(self, batches: list[list[Example]], collator: Collator, salt: int, start: int = 0):
        self.batches, self.collator, self.salt, self.start = batches, collator, salt, start

    def __len__(self) -> int:
        return len(self.batches) - self.start

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return self.collator(self.batches[self.start + i], self.salt)


def _find_last(seq: list[int], pattern: list[int]) -> int:
    n = len(pattern)
    for i in range(len(seq) - n, -1, -1):
        if seq[i:i + n] == pattern:
            return i
    return -1


def eval_subset(by_source: dict[str, list[Example]], per_source: int, seed: int,
                total: Optional[int] = None) -> list[Example]:
    """A fixed, source-balanced sample used for generation metrics at every eval."""
    rng = random.Random(seed)
    picked: list[Example] = []
    for name in sorted(by_source):
        exs = by_source[name][:]
        rng.shuffle(exs)
        picked.extend(exs[:per_source])
    if total is not None and len(picked) > total:
        rng.shuffle(picked)
        picked = picked[:total]
    return sorted(picked, key=lambda e: e.est_tokens)


def steps_per_epoch(n_batches: int, grad_accum: int) -> int:
    return math.ceil(n_batches / grad_accum)
