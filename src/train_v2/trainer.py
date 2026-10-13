"""Training loop for Tadpole (full fine-tune, vision tower included).

fp32 master weights + bf16 autocast, AdamW (fused), linear warmup + cosine decay, gradient
accumulation, gradient checkpointing. Every `eval.every_steps`: validation loss on a fixed
subset plus generation metrics (node / edge F1, ...) from diagram_ir.metrics on a fixed,
source-balanced subset. Checkpoints keep the last N plus the best by edge F1; `resume: auto`
continues from the newest checkpoint (model, optimizer, scheduler, step, RNG).

Ctrl+C stops gracefully: the current step finishes, a checkpoint is written, then exit.
"""

from __future__ import annotations

import json
import logging
import math
import random
import re
import shutil
import signal
import time
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForImageTextToText, AutoProcessor

from diagram_ir import canonicalize, dumps, loads
from diagram_ir.metrics import aggregate, score

from .dashboard import Dashboard, State, TailHandler
from .data import (
    Collator,
    ImageSizeCache,
    MicroBatches,
    build_epoch,
    epoch_mix,
    eval_subset,
    load_rows,
    steps_per_epoch,
    to_examples,
)

log = logging.getLogger("train_v2")
LOSS_CHUNK = 1024  # answer tokens per lm_head + cross-entropy chunk


class Trainer:
    def __init__(self, cfg: dict, smoke: bool = False, dashboard: bool = True):
        self.cfg, self.smoke = cfg, smoke
        self.run_dir = Path(cfg["output_root"]) / (cfg["run_name"] + ("-smoke" if smoke else ""))
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.tail = TailHandler()
        self._setup_logging()
        self.state = State(run_name=self.run_dir.name, model_name=Path(cfg["model_path"]).name)
        self.dash = Dashboard(self.state, self.tail, enabled=dashboard)
        self.stop_requested = False
        self.skipped_oom = 0
        self.metrics_fh = open(self.run_dir / "metrics.jsonl", "a", encoding="utf-8")

    # ------------------------------------------------------------------ setup
    def _setup_logging(self) -> None:
        root = logging.getLogger()
        root.setLevel(logging.INFO)
        for h in list(root.handlers):
            root.removeHandler(h)
        fh = logging.FileHandler(self.run_dir / "train.log", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-5s %(message)s"))
        root.addHandler(fh)
        root.addHandler(self.tail)
        for noisy in ("httpx", "urllib3", "PIL", "transformers.modeling_utils"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    def _on_sigint(self, *_: Any) -> None:
        if self.stop_requested:
            raise KeyboardInterrupt
        self.stop_requested = True
        log.warning("stop requested - finishing the current step, then saving a checkpoint "
                    "(press Ctrl+C again to abort immediately)")

    def _load_data(self) -> None:
        d = self.cfg["data"]
        root = Path(d["root"])
        data_root = Path(d["images_root"])
        sizes = ImageSizeCache(root / "_image_sizes.json")
        sources = d["sources"]
        train_rows = load_rows(root, sources, "train", ("train",))
        val_rows = load_rows(root, sources, "val", ("train",))
        missing = [s for s in sources if s not in train_rows]
        if missing:
            log.warning("sources without train data (not built yet?): %s", ", ".join(missing))
        log.info("indexing images (cached in %s) ...", sizes.path.name)
        args = (data_root, sizes, d["min_pixels"], d["max_pixels"], d["max_target_chars"])
        self.train_by_source = to_examples(train_rows, *args)
        self.val_by_source = to_examples(val_rows, *args)
        sizes.save()
        for name, exs in self.train_by_source.items():
            log.info("  %-22s train %6d  val %5d", name, len(exs), len(self.val_by_source.get(name, [])))
        ev = self.cfg["eval"]
        n_loss = 16 if self.smoke else ev["val_loss_samples"]
        n_gen = 6 if self.smoke else ev["gen_samples"]
        n_src = max(1, len(self.val_by_source))
        self.val_loss_set = eval_subset(self.val_by_source, math.ceil(n_loss / n_src), seed=1, total=n_loss)
        self.val_gen_set = eval_subset(self.val_by_source, math.ceil(n_gen / n_src), seed=2, total=n_gen)

    def _load_model(self) -> None:
        path = self.cfg["model_path"]
        self.processor = AutoProcessor.from_pretrained(path)
        self.model = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.float32).cuda()
        t = self.cfg["train"]
        if t.get("grad_checkpointing", True):
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.model.config.use_cache = False
        # Chat answers end with <|im_end|>, but the base checkpoint's generation config only stops
        # at <|endoftext|>: generation ran past the answer into a new "assistant" turn. Stop at
        # both - this is saved with every checkpoint, so inference with the model stops too.
        self.stop_ids = stop_token_ids(self.processor.tokenizer)
        self.model.generation_config.eos_token_id = self.stop_ids
        d = self.cfg["data"]
        self.collator = Collator(self.processor, d["min_pixels"], d["max_pixels"], d["layout_prob"],
                                 pad_multiple=d.get("pad_multiple", 128))
        decay, no_decay, vision = [], [], []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if "visual" in name:
                vision.append(p)
            elif p.ndim < 2 or "norm" in name or "embed" in name:
                no_decay.append(p)
            else:
                decay.append(p)
        lr, vlr = t["lr"], t["lr"] * t.get("vision_lr_scale", 1.0)
        groups = [{"params": decay, "lr": lr, "weight_decay": t["weight_decay"]},
                  {"params": no_decay, "lr": lr, "weight_decay": 0.0},
                  {"params": vision, "lr": vlr, "weight_decay": t["weight_decay"]}]
        if t.get("optimizer", "adamw") == "adamw8bit":
            import bitsandbytes as bnb  # 8-bit states: ~5 GB less VRAM than fp32 AdamW for 0.85B

            self.optimizer = bnb.optim.AdamW8bit(groups, betas=(0.9, 0.95), eps=1e-8)
        else:
            self.optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.95), eps=1e-8, fused=True)
        self.base_lrs = [g["lr"] for g in self.optimizer.param_groups]
        n_params = sum(p.numel() for p in self.model.parameters())
        log.info("model %s: %.0fM params (vision %.0fM), lr %.1e (vision %.1e)",
                 Path(path).name, n_params / 1e6, sum(p.numel() for p in vision) / 1e6, lr, vlr)

    def _lr_at(self, step: int) -> float:
        t = self.cfg["train"]
        warm = max(1, int(t["warmup_ratio"] * self.total_steps))
        if step < warm:
            return (step + 1) / warm
        progress = (step - warm) / max(1, self.total_steps - warm)
        floor = t.get("min_lr_ratio", 0.1)
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    # ------------------------------------------------------------------ checkpoints
    def _checkpoints(self) -> list[Path]:
        """Complete, resumable checkpoints, oldest first. Skips `.tmp` leftovers from an
        interrupted save (e.g. a power cut) and folders missing their weights or state."""
        found = []
        for p in self.run_dir.glob("checkpoint-*"):
            if (re.fullmatch(r"checkpoint-\d+", p.name) and (p / "trainer_state.pt").exists()
                    and (p / "model.safetensors").exists()):
                found.append(p)
        return sorted(found, key=lambda p: int(p.name.split("-")[1]))

    def _save(self, step: int, tag: Optional[str] = None) -> Path:
        self.state.phase = "saving"
        out = self.run_dir / (tag or f"checkpoint-{step}")
        tmp = out.with_name(out.name + ".tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        self.model.save_pretrained(tmp, safe_serialization=True)
        self.processor.save_pretrained(tmp)
        if tag is None:  # full training state only for resumable checkpoints
            torch.save({"optimizer": self.optimizer.state_dict(), "step": step, "epoch": self.epoch,
                        "batch_in_epoch": self.batch_in_epoch, "best": self.state.best,
                        "rng": random.getstate(), "torch_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state()}, tmp / "trainer_state.pt")
        if out.exists():
            shutil.rmtree(out)
        tmp.rename(out)
        if tag is None:
            keep = self.cfg["save"]["keep_last"]
            for old in self._checkpoints()[:-keep]:
                shutil.rmtree(old, ignore_errors=True)
        log.info("saved %s", out.name)
        self.state.phase = "training"
        return out

    def _maybe_resume(self) -> None:
        self.step, self.epoch, self.batch_in_epoch = 0, 0, 0
        if self.smoke or self.cfg["train"].get("resume", "auto") != "auto":
            return
        ckpts = self._checkpoints()
        if not ckpts:
            return
        last = ckpts[-1]
        log.info("resuming from %s", last.name)
        state_dict = AutoModelForImageTextToText.from_pretrained(last, dtype=torch.float32).state_dict()
        self.model.load_state_dict(state_dict)
        ts = torch.load(last / "trainer_state.pt", weights_only=False)
        self.optimizer.load_state_dict(ts["optimizer"])
        self.step, self.epoch, self.batch_in_epoch = ts["step"], ts["epoch"], ts["batch_in_epoch"]
        self.state.best = ts.get("best")
        random.setstate(ts["rng"])
        torch.set_rng_state(ts["torch_rng"])
        torch.cuda.set_rng_state(ts["cuda_rng"])

    def _loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Next-token loss on the answer only, without ever materializing full logits.

        With a 248k vocabulary, logits for a 2k-token answer are ~2 GB (bf16), ~4 GB as fp32
        for the loss, plus the same again for their gradient. So: run the base model for hidden
        states, keep only positions that predict an answer token, and apply lm_head + CE in
        chunks under activation checkpointing (each chunk's logits are recomputed in backward)."""
        labels = batch.pop("labels")
        targets = labels[:, 1:]  # position t predicts token t + 1
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = self.model.model(**batch, use_cache=False)[0]
        mask = targets != -100
        h, y = hidden[:, :-1][mask], targets[mask]
        total = torch.zeros((), device=h.device, dtype=torch.float32)
        for i in range(0, h.shape[0], LOSS_CHUNK):
            total = total + checkpoint(self._chunk_ce, h[i:i + LOSS_CHUNK], y[i:i + LOSS_CHUNK],
                                       use_reentrant=False)
        return total / max(1, y.numel())

    def _chunk_ce(self, h: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = self.model.lm_head(h)
        return F.cross_entropy(logits.float(), y, reduction="sum")

    # ------------------------------------------------------------------ eval
    @torch.no_grad()
    def evaluate(self) -> dict:
        self.state.phase = "evaluating"
        self.model.eval()
        ev = self.cfg["eval"]
        bs = self.cfg["train"]["batch_size"]
        losses = []
        for i in range(0, len(self.val_loss_set), bs):
            batch = self.collator(self.val_loss_set[i:i + bs])
            batch = {k: v.cuda() for k, v in batch.items()}
            losses.append(self._loss(batch).item())
        val_loss = sum(losses) / max(1, len(losses))

        self.model.config.use_cache = True
        results, samples = [], []
        gbs = ev["gen_batch_size"]
        for i in range(0, len(self.val_gen_set), gbs):
            exs = self.val_gen_set[i:i + gbs]
            inputs = {k: v.cuda() for k, v in self.collator.generation_inputs(exs).items()}
            max_new = min(ev["gen_max_new_tokens"], max(64, int(max(len(e.dgm) for e in exs) / 2.5) + 64))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = self.model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                                          eos_token_id=self.stop_ids, pad_token_id=self.collator.pad_id,
                                          use_cache=True)
            new_tokens = out[:, inputs["input_ids"].shape[1]:]
            texts = decode_answers(self.processor.tokenizer, new_tokens)
            for ex, pred in zip(exs, texts, strict=True):
                gold = ex.dgm if not ex.has_layout else _strip(ex.dgm)
                r = score(pred, gold)
                results.append(r)
                samples.append({"id": ex.id, "source": ex.source, "edge_f1": r["edge_f1"],
                                "node_f1": r["node_f1"], "pred": pred, "gold": gold})
            if self.stop_requested:
                break
        self.model.config.use_cache = False
        agg = aggregate(results) if results else {}
        metrics = {"val_loss": val_loss, **{k: agg[k] for k in (
            "node_f1", "edge_f1", "edge_label_f1", "kind_acc", "valid_rate", "exact_rate", "text_exact_rate")
            if k in agg}}
        by_source: dict[str, list[dict]] = {}
        for s, r in zip(samples, results, strict=True):
            by_source.setdefault(s["source"], []).append(r)
        metrics["per_source"] = {k: round(aggregate(v)["edge_f1"], 4) for k, v in by_source.items()}
        with open(self.run_dir / "eval_samples.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"step": self.step, "samples": samples}, ensure_ascii=False) + "\n")
        self.model.train()
        self.state.phase = "training"
        return metrics

    # ------------------------------------------------------------------ train
    def run(self) -> None:
        signal.signal(signal.SIGINT, self._on_sigint)
        with self.dash:
            try:
                self._run()
            except Exception:
                log.exception("training crashed")
                self.state.phase = "stopped"
                time.sleep(2)
                raise

    def _run(self) -> None:
        t = self.cfg["train"]
        torch.manual_seed(self.cfg["seed"])
        random.seed(self.cfg["seed"])
        torch.backends.cuda.matmul.allow_tf32 = True
        # On Windows the driver silently spills past-VRAM allocations into system RAM (10-100x
        # slower) instead of failing. Cap PyTorch below the card size so we get a clean OOM.
        torch.cuda.set_per_process_memory_fraction(self.cfg["train"].get("vram_fraction", 0.94))
        self._load_data()
        self._load_model()
        rng = random.Random(self.cfg["seed"])
        bs, accum = t["batch_size"], t["grad_accum"]
        epochs = 1 if self.smoke else t["epochs"]
        first = build_epoch(self.train_by_source, self.cfg["data"]["sources"],
                            random.Random(rng.random()), bs)
        per_epoch = steps_per_epoch(len(first), accum)
        self.total_steps = t.get("smoke_steps", 8) if self.smoke else min(
            per_epoch * epochs, t.get("max_steps") or 10**12)
        self.state.total_steps, self.state.epochs = self.total_steps, epochs
        self._maybe_resume()
        self.state.step = self.state.start_step = self.step
        log.info("run dir %s | %d optimizer steps (%d per epoch, batch %d x accum %d)",
                 self.run_dir, self.total_steps, per_epoch, bs, accum)
        (self.run_dir / "config.used.json").write_text(json.dumps(self.cfg, indent=2), encoding="utf-8")

        ev_every = 4 if self.smoke else self.cfg["eval"]["every_steps"]
        save_every = 8 if self.smoke else self.cfg["save"]["every_steps"]
        self.model.train()
        self.state.phase = "training"
        while self.step < self.total_steps and not self.stop_requested:
            # seeded by epoch number, so a resumed run rebuilds the same batch order
            batches = build_epoch(self.train_by_source, self.cfg["data"]["sources"],
                                  random.Random(self.cfg["seed"] * 1000 + self.epoch), bs)
            self.state.epoch, self.state.mix = self.epoch + 1, epoch_mix(batches)
            log.info("epoch %d: %d samples", self.epoch + 1, sum(self.state.mix.values()))
            loader = torch.utils.data.DataLoader(
                MicroBatches(batches, self.collator, salt=self.epoch, start=self.batch_in_epoch),
                batch_size=None, shuffle=False, num_workers=self.cfg["train"].get("num_workers", 4),
                prefetch_factor=4 if self.cfg["train"].get("num_workers", 4) else None,
                pin_memory=True, persistent_workers=False)
            it = iter(loader)
            while self.batch_in_epoch + accum <= len(batches) and self.step < self.total_steps:
                self._train_step([next(it) for _ in range(accum)])
                self.batch_in_epoch += accum
                if self.step % ev_every == 0 or self.step == self.total_steps:
                    self._run_eval()
                if self.stop_requested:
                    break
                if self.step % save_every == 0:
                    self._save(self.step)
            if self.stop_requested:
                break
            self.epoch += 1
            self.batch_in_epoch = 0
        self._save(self.step)
        self._save(self.step, tag="final")
        self.state.phase = "stopped" if self.stop_requested else "done"
        log.info("finished at step %d. Run dir: %s", self.step, self.run_dir)
        time.sleep(3 if self.smoke else 10)

    def _train_step(self, micro_batches: list[dict]) -> None:
        t0 = time.time()
        scale = self._lr_at(self.step)
        for g, base in zip(self.optimizer.param_groups, self.base_lrs, strict=True):
            g["lr"] = base * scale
        total_loss, n_tokens, n_samples = 0.0, 0, 0
        try:
            for mb in micro_batches:
                batch = {k: v.cuda(non_blocking=True) for k, v in mb.items()}
                n_tokens += int(batch["attention_mask"].sum())
                n_samples += int(mb["input_ids"].shape[0])
                loss = self._loss(batch) / len(micro_batches)
                loss.backward()
                total_loss += loss.item()
        except torch.cuda.OutOfMemoryError:
            self.optimizer.zero_grad(set_to_none=True)
            batch = loss = None
            torch.cuda.empty_cache()
            self.skipped_oom += 1
            log.warning("out of memory on a step of %d tokens - skipped (%d so far)",
                        n_tokens, self.skipped_oom)
            return
        max_norm = self.cfg["train"]["max_grad_norm"]
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.step += 1
        dt = time.time() - t0
        s = self.state
        s.step, s.loss, s.lr = self.step, total_loss, self.optimizer.param_groups[0]["lr"]
        s.loss_ema = total_loss if s.loss_ema is None else 0.98 * s.loss_ema + 0.02 * total_loss
        s.loss_hist.append(total_loss)
        s.grad_norm = float(grad_norm)
        s.tokens_per_s, s.samples_per_s = n_tokens / dt, n_samples / dt
        s.samples_seen += n_samples
        rec = {"step": self.step, "loss": total_loss, "lr": s.lr, "grad_norm": s.grad_norm,
               "tokens_per_s": round(s.tokens_per_s), "time": time.time()}
        self.metrics_fh.write(json.dumps(rec) + "\n")
        self.metrics_fh.flush()
        if self.step % self.cfg["train"].get("log_every", 10) == 0 or self.smoke:
            log.info("step %d  loss %.4f  ema %.4f  lr %.2e  gnorm %.2f  %.0f tok/s",
                     self.step, total_loss, s.loss_ema, s.lr, s.grad_norm, s.tokens_per_s)
        if not math.isfinite(total_loss):
            log.error("non-finite loss at step %d - stopping", self.step)
            self.stop_requested = True

    def _run_eval(self) -> None:
        log.info("evaluating at step %d ...", self.step)
        t0 = time.time()
        m = self.evaluate()
        self.state.eval, self.state.eval_step = m, self.step
        log.info("eval step %d: val_loss %.4f  node_f1 %.3f  edge_f1 %.3f  label_f1 %.3f  valid %.0f%%  "
                 "(%.0fs)  per-source edge F1 %s", self.step, m["val_loss"], m.get("node_f1", 0),
                 m.get("edge_f1", 0), m.get("edge_label_f1", 0), 100 * m.get("valid_rate", 0),
                 time.time() - t0, m.get("per_source"))
        self.metrics_fh.write(json.dumps({"step": self.step, "eval": m}) + "\n")
        self.metrics_fh.flush()
        if not self.smoke and (self.state.best is None or m.get("edge_f1", 0) > self.state.best):
            self.state.best = m.get("edge_f1", 0)
            self._save(self.step, tag="best")


def stop_token_ids(tokenizer: Any) -> list[int]:
    ids = [tokenizer.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")]
    return [i for i in ids if isinstance(i, int) and i >= 0]


def decode_answers(tokenizer: Any, new_tokens: torch.Tensor) -> list[str]:
    """Decode generated answers, cutting each at its first <|im_end|> (end of the turn)."""
    texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=False)
    out = []
    for t in texts:
        t = t.split("<|im_end|>", 1)[0].replace("<|endoftext|>", "")
        out.append(re.sub(r"<\|[^|>]+\|>", "", t))
    return out


def _strip(dgm: str) -> str:
    return dumps(canonicalize(loads(dgm).strip_layout()))
