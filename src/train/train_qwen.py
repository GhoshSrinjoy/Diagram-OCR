"""LoRA SFT of Qwen2-VL-2B on the unified diagram dataset.

Reads everything from config.yaml. The raw processed data is ~96% chart-QA, so we
do NOT train on it directly: we materialize a *balanced* epoch weighted toward the
image->graph objective (see data.task_weights) and cap any single source.

    python train_qwen.py                 # real run, uses config.yaml
    python train_qwen.py --smoke         # tiny set + few steps, validates the loop
    python train_qwen.py --config x.yaml # alternate config

Design notes:
- bf16 LoRA on the language model only (vision tower frozen; its modules have
  different names so the target_modules list never matches them).
- Labels mask the prompt: we tokenize the prompt-with-generation-prompt to get its
  length, then set those positions to -100 so loss is only on the answer.
- pixel_values are cast to bf16 in the collator to match the model.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def apply_smoke(cfg: dict) -> dict:
    smoke = cfg.get("smoke", {})
    cfg["data"]["train_samples"] = smoke.get("train_samples", 16)
    cfg["data"]["eval_samples"] = smoke.get("eval_samples", 8)
    cfg["train"]["_max_steps"] = smoke.get("max_steps", 5)
    cfg["train"]["dataloader_workers"] = 0
    cfg["train"]["save_steps"] = 10**9
    cfg["train"]["eval_steps"] = 10**9
    return cfg


# --------------------------------------------------------------------------- #
# Balanced sampling
# --------------------------------------------------------------------------- #
def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _cap_sources(pool: list[dict], caps: dict, k: int, rng: random.Random) -> list[dict]:
    if not caps:
        return pool
    by_src: dict[str, list] = defaultdict(list)
    for r in pool:
        by_src[r["source_dataset"]].append(r)
    out: list[dict] = []
    for src, items in by_src.items():
        if src in caps and len(items) > max(1, int(caps[src] * k)):
            items = rng.sample(items, max(1, int(caps[src] * k)))
        out += items
    return out


def build_balanced(rows: list[dict], n: int, task_weights: dict, caps: dict,
                   seed: int, processed_dir: Path) -> list[dict]:
    """Materialize n rows allocated across tasks by weight, upsampling small pools."""
    by_task: dict[str, list] = defaultdict(list)
    for r in rows:
        ip = r.get("image_path")
        if ip and (processed_dir / ip).exists():
            by_task[r["task"]].append(r)
    rng = random.Random(seed)
    out: list[dict] = []
    for task, w in task_weights.items():
        pool = _cap_sources(by_task.get(task, []), caps, int(n * w), rng)
        if not pool:
            continue
        k = int(round(n * w))
        if len(pool) >= k:
            out += rng.sample(pool, k)
        else:  # upsample the tiny graph pool with replacement
            out += [rng.choice(pool) for _ in range(k)]
    rng.shuffle(out)
    return out


def describe(rows: list[dict], title: str) -> None:
    task = Counter(r["task"] for r in rows)
    src = Counter(r["source_dataset"] for r in rows)
    print(f"  {title}: {len(rows)} rows")
    print(f"    by task:   {dict(task)}")
    print(f"    by source: {dict(src)}")


# --------------------------------------------------------------------------- #
# Dataset + collator
# --------------------------------------------------------------------------- #
class DiagramDataset(Dataset):
    def __init__(self, rows: list[dict], processed_dir: Path):
        self.rows = rows
        self.root = processed_dir

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        r = self.rows[i]
        try:
            img = Image.open(self.root / r["image_path"]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (32, 32), (255, 255, 255))
        return {"image": img, "input_text": r["input_text"], "target_text": r["target_text"]}


class Collator:
    def __init__(self, processor: Any, max_seq_len: int):
        self.p = processor
        self.max_seq_len = max_seq_len

    def _messages(self, input_text: str, with_answer: str | None):
        user = {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": input_text}]}
        if with_answer is None:
            return [user]
        return [user, {"role": "assistant", "content": [{"type": "text", "text": with_answer}]}]

    def __call__(self, batch: list[dict]) -> dict:
        images = [b["image"] for b in batch]
        full_texts, prompt_texts = [], []
        for b in batch:
            full_texts.append(self.p.apply_chat_template(
                self._messages(b["input_text"], b["target_text"]),
                tokenize=False, add_generation_prompt=False))
            prompt_texts.append(self.p.apply_chat_template(
                self._messages(b["input_text"], None),
                tokenize=False, add_generation_prompt=True))

        enc = self.p(text=full_texts, images=images, padding=True, truncation=True,
                     max_length=self.max_seq_len, return_tensors="pt")
        # Per-example prompt length (same images -> same image-token count).
        prompt_enc = self.p(text=prompt_texts, images=images, padding=True, return_tensors="pt")
        prompt_lens = prompt_enc["attention_mask"].sum(dim=1)

        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        for i, pl in enumerate(prompt_lens):
            labels[i, : min(int(pl), labels.shape[1])] = -100
        enc["labels"] = labels
        if "pixel_values" in enc:
            enc["pixel_values"] = enc["pixel_values"].to(torch.bfloat16)
        return dict(enc)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    ap.add_argument("--smoke", action="store_true", help="Tiny set + few steps to validate the loop.")
    args = ap.parse_args()

    from transformers import (
        AutoProcessor,
        EarlyStoppingCallback,
        Qwen2VLForConditionalGeneration,
        Trainer,
        TrainingArguments,
    )
    from transformers.trainer_utils import get_last_checkpoint
    from peft import LoraConfig, get_peft_model

    cfg = load_config(args.config)
    if args.smoke:
        cfg = apply_smoke(cfg)
    random.seed(cfg["data"]["seed"])
    torch.manual_seed(cfg["data"]["seed"])

    mcfg, dcfg, lcfg, tcfg = cfg["model"], cfg["data"], cfg["lora"], cfg["train"]
    processed_dir = Path(dcfg["processed_dir"])

    print("Loading processor + model...")
    processor = AutoProcessor.from_pretrained(
        mcfg["path"], min_pixels=mcfg["min_pixels"], max_pixels=mcfg["max_pixels"])
    processor.tokenizer.padding_side = "right"

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        mcfg["path"], torch_dtype=torch.bfloat16,
        attn_implementation=mcfg.get("attn_implementation", "sdpa"))
    model.config.use_cache = False
    if tcfg.get("gradient_checkpointing", True):
        model.enable_input_require_grads()  # PEFT needs this; Trainer toggles checkpointing via args

    model = get_peft_model(model, LoraConfig(
        r=lcfg["r"], lora_alpha=lcfg["alpha"], lora_dropout=lcfg["dropout"],
        target_modules=lcfg["target_modules"], task_type="CAUSAL_LM", bias="none"))
    model.print_trainable_parameters()

    print("Building balanced datasets...")
    train_rows = build_balanced(
        read_jsonl(processed_dir / "train.jsonl"), dcfg["train_samples"],
        dcfg["task_weights"], dcfg.get("source_caps", {}), dcfg["seed"], processed_dir)
    describe(train_rows, "train")

    val_all = read_jsonl(processed_dir / "val.jsonl")
    eval_tasks = set(dcfg.get("eval_tasks", []))
    val_pool = [r for r in val_all if r.get("image_path")
                and (processed_dir / r["image_path"]).exists()
                and (not eval_tasks or r["task"] in eval_tasks)]
    rng = random.Random(dcfg["seed"])
    eval_rows = rng.sample(val_pool, min(dcfg["eval_samples"], len(val_pool)))
    describe(eval_rows, "eval")

    train_ds = DiagramDataset(train_rows, processed_dir)
    eval_ds = DiagramDataset(eval_rows, processed_dir)
    collator = Collator(processor, tcfg["max_seq_len"])

    do_eval = len(eval_rows) > 0 and not args.smoke
    use_muon = str(tcfg.get("optimizer", "adamw_torch_fused")).lower() == "muon"
    load_best = bool(tcfg.get("load_best_model_at_end", True)) and do_eval

    targs = TrainingArguments(
        output_dir=tcfg["output_dir"],
        num_train_epochs=tcfg["epochs"],
        per_device_train_batch_size=tcfg["per_device_batch_size"],
        per_device_eval_batch_size=tcfg["per_device_batch_size"],
        gradient_accumulation_steps=tcfg["grad_accum"],
        learning_rate=float(tcfg["lr"]),
        warmup_ratio=tcfg["warmup_ratio"],
        weight_decay=tcfg["weight_decay"],
        max_grad_norm=tcfg["max_grad_norm"],
        lr_scheduler_type="cosine",
        optim="adamw_torch" if use_muon else tcfg.get("optimizer", "adamw_torch_fused"),
        bf16=tcfg["bf16"],
        gradient_checkpointing=tcfg.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=tcfg["logging_steps"],
        save_strategy="no" if args.smoke else "steps",
        save_steps=tcfg["save_steps"],
        save_total_limit=tcfg.get("save_total_limit", 3),
        eval_strategy="steps" if do_eval else "no",
        eval_steps=tcfg["eval_steps"],
        load_best_model_at_end=load_best,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        max_steps=tcfg.get("_max_steps", -1),
        dataloader_num_workers=tcfg.get("dataloader_workers", 0),
        remove_unused_columns=False,
        label_names=["labels"],
        report_to="none",
    )

    class DiagramTrainer(Trainer):
        """Trainer that swaps in the Muon(2D)+AdamW hybrid when configured."""

        def create_optimizer(self):
            if use_muon and self.optimizer is None:
                from pytorch_optimizer import Muon
                params = [p for p in self.model.parameters() if p.requires_grad]
                self.optimizer = Muon(
                    params, lr=float(tcfg.get("muon_lr", 0.02)),
                    weight_decay=self.args.weight_decay,
                    adamw_lr=self.args.learning_rate, adamw_betas=(0.9, 0.95),
                    adamw_wd=self.args.weight_decay)
                return self.optimizer
            return super().create_optimizer()

    callbacks = []
    if load_best and tcfg.get("early_stopping_patience"):
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=int(tcfg["early_stopping_patience"])))

    trainer = DiagramTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds if do_eval else None,
        data_collator=collator,
        processing_class=processor,
        callbacks=callbacks,
    )

    # Resume from the latest checkpoint unless disabled or in smoke mode.
    resume_cfg = tcfg.get("resume", "auto")
    ckpt = None
    if not args.smoke and resume_cfg not in (False, "false", "none", None):
        if isinstance(resume_cfg, str) and resume_cfg not in ("auto", "true", "True"):
            ckpt = resume_cfg
        elif os.path.isdir(tcfg["output_dir"]):
            ckpt = get_last_checkpoint(tcfg["output_dir"])

    opt_name = "muon (hybrid)" if use_muon else targs.optim
    print(f"Training -> {tcfg['output_dir']} | optimizer={opt_name} | resume={ckpt or False}")
    trainer.train(resume_from_checkpoint=ckpt)
    trainer.save_model(tcfg["output_dir"])
    processor.save_pretrained(tcfg["output_dir"])
    print(f"Done. Adapter saved to {tcfg['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
