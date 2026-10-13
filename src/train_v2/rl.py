"""Reinforcement learning on graph metrics: DAPO-style GRPO with the Dr. GRPO fixes.

For each training image the model samples a group of K answers; each is scored against the gold
graph with diagram_ir.metrics (all order-invariant, so a correct graph written in another order
is not penalized). Then:
  * reward     = weighted edge / node / edge+label F1 + valid-output bonus, minus a penalty when
                 the answer is cut off at the length limit (DAPO overlong shaping)
  * advantage  = reward - group mean, NOT divided by the group std (Dr. GRPO: no difficulty bias)
  * groups whose answers all score the same carry no signal; they are dropped and replaced with
    fresh prompts (DAPO dynamic sampling)
  * loss       = token-level mean over the whole step of the clipped policy-gradient objective
                 with asymmetric clip (DAPO clip-higher; active when ppo_epochs > 1), no KL term
Before training, a greedy pre-pass scores candidate prompts and keeps mostly those the model
does not already get exactly right, so generation time goes where there is something to learn.

Usage (from src/, D:/envs/diagram Python):
    python -m train_v2.rl --config train_v2/config_rl.yaml          # live dashboard
    python -m train_v2.rl --config train_v2/config_rl.yaml --smoke
    .\\train_v2\\launch.ps1 -Module train_v2.rl -Config train_v2\\config_rl.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import re
import shutil
import signal
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F
import yaml
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForImageTextToText, AutoProcessor

from diagram_ir.metrics import score

from .dashboard import Dashboard, State, TailHandler
from .data import Collator, Example, ImageSizeCache, eval_subset, load_rows, to_examples
from .evaluate import gold_of, run_eval
from .trainer import decode_answers, stop_token_ids

log = logging.getLogger("train_v2.rl")
LOGP_CHUNK = 1024  # completion tokens per lm_head + log-softmax chunk


class RLTrainer:
    def __init__(self, cfg: dict, smoke: bool = False, dashboard: bool = True):
        self.cfg, self.smoke = cfg, smoke
        self.rl = cfg["rl"]
        self.run_dir = Path(cfg["output_root"]) / (cfg["run_name"] + ("-smoke" if smoke else ""))
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.tail = TailHandler()
        self._setup_logging()
        self.state = State(run_name=self.run_dir.name, model_name=Path(cfg["init_from"]).name)
        self.dash = Dashboard(self.state, self.tail, enabled=dashboard)
        self.stop_requested = False
        self.metrics_fh = open(self.run_dir / "metrics.jsonl", "a", encoding="utf-8")
        self.rng = random.Random(cfg["seed"])

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

    def _on_sigint(self, *_: Any) -> None:
        if self.stop_requested:
            raise KeyboardInterrupt
        self.stop_requested = True
        log.warning("stop requested - finishing the current step, then saving a checkpoint")

    def _load_model(self) -> None:
        path = self.cfg["init_from"]
        self.processor = AutoProcessor.from_pretrained(path)
        self.model = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.float32).cuda()
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.stop_ids = stop_token_ids(self.processor.tokenizer)
        self.model.generation_config.eos_token_id = self.stop_ids
        d = self.cfg["data"]
        self.collator = Collator(self.processor, d["min_pixels"], d["max_pixels"], layout_prob=0.0)
        import bitsandbytes as bnb

        self.optimizer = bnb.optim.AdamW8bit(self.model.parameters(), lr=self.rl["lr"],
                                             betas=(0.9, 0.99), eps=1e-8, weight_decay=0.0)
        log.info("policy %s loaded (%.0fM params), lr %.1e", Path(path).name,
                 sum(p.numel() for p in self.model.parameters()) / 1e6, self.rl["lr"])

    def _load_data(self) -> None:
        d = self.cfg["data"]
        root = Path(d["root"])
        sizes = ImageSizeCache(root / "_image_sizes.json")
        weights = self.rl["sources"]
        args = (Path(d["images_root"]), sizes, d["min_pixels"], d["max_pixels"], d["max_target_chars"])
        self.train_by_source = to_examples(load_rows(root, weights, "train", ("train",)), *args)
        val_by_source = to_examples(load_rows(root, weights, "val", ("train",)), *args)
        sizes.save()
        missing = [s for s in weights if s not in self.train_by_source]
        if missing:
            log.warning("RL sources without data (not built yet?): %s", ", ".join(missing))
        per_source = 2 if self.smoke else self.rl["eval_per_source"]
        self.eval_set = eval_subset(val_by_source, per_source, seed=3)

    # ------------------------------------------------------------------ prompt pool
    def _prompt_pool(self) -> None:
        """Greedy pre-pass: keep prompts the policy does not already solve exactly."""
        cache = self.run_dir / "prompt_pool.json"
        by_id = {e.id: e for exs in self.train_by_source.values() for e in exs}
        if cache.exists():
            ids = json.loads(cache.read_text(encoding="utf-8"))
            self.pool = [by_id[i] for i in ids if i in by_id]
            log.info("prompt pool: %d prompts (cached)", len(self.pool))
            return
        n = 4 if self.smoke else self.rl["candidates_per_source"]
        candidates = eval_subset(self.train_by_source, n, seed=self.cfg["seed"])
        log.info("prompt pre-pass: scoring %d candidate prompts with greedy decoding ...", len(candidates))
        self.model.eval()
        t0 = time.time()

        def progress(done: int, total: int) -> None:
            self.state.note = f"pre-pass {done}/{total}"
            if done % 80 < 8:
                log.info("  pre-pass %d/%d (%.1fs/prompt)", done, total, (time.time() - t0) / done)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, samples = run_eval(self.model, self.processor, self.collator, candidates,
                                  batch_size=8, max_new_tokens=self.rl["max_new_tokens"], progress=progress)
        keep_perfect = self.rl["keep_perfect_fraction"]
        prng = random.Random(self.cfg["seed"])
        kept, perfect = [], 0
        for ex, s in zip(candidates, samples, strict=True):
            solved = s["edge_f1"] >= 1.0 and s["node_f1"] >= 1.0
            perfect += solved
            if not solved or prng.random() < keep_perfect:
                kept.append(ex)
        self.pool = kept
        cache.write_text(json.dumps([e.id for e in kept]), encoding="utf-8")
        log.info("prompt pool: kept %d of %d (%d already perfect)", len(kept), len(candidates), perfect)

    def _sample_prompts(self, n: int) -> list[Example]:
        weights = [float(self.rl["sources"].get(e.source, 1.0)) for e in self.pool]
        return self.rng.choices(self.pool, weights=weights, k=n)

    # ------------------------------------------------------------------ rollouts
    def _reward(self, pred: str, gold: str, truncated: bool) -> tuple[float, dict]:
        r = score(pred, gold)
        w = self.rl["reward_weights"]
        value = (w["edge_f1"] * r["edge_f1"] + w["node_f1"] * r["node_f1"]
                 + w["edge_label_f1"] * r["edge_label_f1"] + w["valid"] * r["valid"])
        if truncated:
            value -= self.rl["overlong_penalty"]
        return value, r

    @torch.no_grad()
    def _rollout(self, exs: list[Example]) -> list[dict]:
        k = self.rl["group_size"]
        rows = [ex for ex in exs for _ in range(k)]
        caps = [min(self.rl["max_new_tokens"], int(len(gold_of(ex)) / 2.5 * self.rl["length_factor"]) + 64)
                for ex in rows]
        inputs = {key: v.cuda() for key, v in self.collator.generation_inputs(rows).items()}
        self.model.eval()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.model.generate(**inputs, do_sample=True, temperature=self.rl["temperature"],
                                      top_p=self.rl["top_p"], max_new_tokens=max(caps),
                                      eos_token_id=self.stop_ids, pad_token_id=self.collator.pad_id)
        new = out[:, inputs["input_ids"].shape[1]:].tolist()
        groups = []
        for gi, ex in enumerate(exs):
            gold = gold_of(ex)
            comps, rewards, metrics, truncs = [], [], [], []
            for j in range(k):
                idx = gi * k + j
                toks = new[idx]
                end = next((t for t, tok in enumerate(toks) if tok in self.stop_ids), None)
                truncated = end is None or end >= caps[idx]
                toks = toks[: caps[idx]] if truncated else toks[: end + 1]  # keep the stop token
                text = decode_answers(self.processor.tokenizer, torch.tensor([toks]))[0]
                value, m = self._reward(text, gold, truncated)
                comps.append(toks)
                rewards.append(value)
                metrics.append(m)
                truncs.append(truncated)
            groups.append({"ex": ex, "comps": comps, "rewards": rewards, "metrics": metrics,
                           "truncated": truncs})
        return groups

    # ------------------------------------------------------------------ policy update
    def _chunk_logp(self, h: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = self.model.lm_head(h)
        return torch.gather(F.log_softmax(logits.float(), dim=-1), 1, y.unsqueeze(1)).squeeze(1)

    def _group_logp(self, ex: Example, comps: list[list[int]]) -> list[torch.Tensor]:
        """Per-token log-probs (with grad) of each completion, given the image prompt."""
        enc = self.collator.generation_inputs([ex] * len(comps))
        prompt = enc["input_ids"][0].tolist()
        plen = len(prompt)
        tlen = plen + max(len(c) for c in comps)
        pad = self.collator.pad_id
        ids = torch.full((len(comps), tlen), pad, dtype=torch.long)
        attn = torch.zeros((len(comps), tlen), dtype=torch.long)
        for i, c in enumerate(comps):
            ids[i, : plen + len(c)] = torch.tensor(prompt + c)
            attn[i, : plen + len(c)] = 1
        batch = {"input_ids": ids, "attention_mask": attn}
        for key, v in enc.items():  # image tensors; per-token extras get padded to the new length
            if key in batch:
                continue
            if v.dim() == 2 and v.shape[1] == plen:
                ext = torch.zeros((v.shape[0], tlen), dtype=v.dtype)
                ext[:, :plen] = v
                v = ext
            batch[key] = v
        batch = {key: v.cuda() for key, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = self.model.model(**batch, use_cache=False)[0]
        out = []
        for i, c in enumerate(comps):
            h = hidden[i, plen - 1: plen - 1 + len(c)]  # position t predicts token t + 1
            y = batch["input_ids"][i, plen: plen + len(c)]
            parts = [checkpoint(self._chunk_logp, h[s:s + LOGP_CHUNK], y[s:s + LOGP_CHUNK],
                                use_reentrant=False)
                     for s in range(0, len(c), LOGP_CHUNK)]
            out.append(torch.cat(parts))
        return out

    def _update(self, groups: list[dict]) -> dict:
        """One optimizer step per PPO epoch over the kept groups (token-level normalization)."""
        total_tokens = sum(len(c) for g in groups for c in g["comps"])
        lo, hi = 1.0 - self.rl["clip_low"], 1.0 + self.rl["clip_high"]
        mb = self.rl["train_micro_batch"]
        stats = {"pg_loss": 0.0, "clip_frac": 0.0}
        self.model.train()
        for _epoch in range(self.rl["ppo_epochs"]):
            clipped = n_tok = 0
            for g in groups:
                for s in range(0, len(g["comps"]), mb):
                    comps = g["comps"][s:s + mb]
                    advs = g["adv"][s:s + mb]
                    logps = self._group_logp(g["ex"], comps)
                    loss = torch.zeros((), device="cuda")
                    for j, (lp, adv) in enumerate(zip(logps, advs, strict=True)):
                        key = s + j
                        old = g.setdefault("old_logp", {}).get(key)
                        if old is None:  # first epoch is on-policy: ratio == 1
                            old = g["old_logp"][key] = lp.detach()
                        ratio = torch.exp(lp - old)
                        surr = torch.minimum(ratio * adv, torch.clamp(ratio, lo, hi) * adv)
                        loss = loss - surr.sum() / total_tokens
                        clipped += int(((ratio < lo) | (ratio > hi)).sum())
                        n_tok += lp.numel()
                    loss.backward()
                    stats["pg_loss"] += float(loss) / self.rl["ppo_epochs"]
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.rl["max_grad_norm"])
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            stats["clip_frac"] = clipped / max(1, n_tok)
            stats["grad_norm"] = float(grad_norm)
        return stats

    # ------------------------------------------------------------------ checkpoints
    def _checkpoints(self) -> list[Path]:
        found = [p for p in self.run_dir.glob("checkpoint-*")
                 if re.fullmatch(r"checkpoint-\d+", p.name) and (p / "trainer_state.pt").exists()]
        return sorted(found, key=lambda p: int(p.name.split("-")[1]))

    def _save(self, tag: Optional[str] = None) -> None:
        self.state.phase = "saving"
        out = self.run_dir / (tag or f"checkpoint-{self.step}")
        tmp = out.with_name(out.name + ".tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        self.model.save_pretrained(tmp, safe_serialization=True)
        self.processor.save_pretrained(tmp)
        if tag is None:
            torch.save({"optimizer": self.optimizer.state_dict(), "step": self.step, "best": self.state.best,
                        "rng": self.rng.getstate(), "torch_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state()}, tmp / "trainer_state.pt")
        if out.exists():
            shutil.rmtree(out)
        tmp.rename(out)
        if tag is None:
            for old in self._checkpoints()[: -self.cfg["save"]["keep_last"]]:
                shutil.rmtree(old, ignore_errors=True)
        log.info("saved %s", out.name)
        self.state.phase = "training"

    def _maybe_resume(self) -> None:
        self.step = 0
        ckpts = [] if self.smoke else self._checkpoints()
        if not ckpts:
            return
        last = ckpts[-1]
        log.info("resuming from %s", last.name)
        self.model.load_state_dict(
            AutoModelForImageTextToText.from_pretrained(last, dtype=torch.float32).state_dict())
        ts = torch.load(last / "trainer_state.pt", weights_only=False)
        self.optimizer.load_state_dict(ts["optimizer"])
        self.step, self.state.best = ts["step"], ts.get("best")
        self.rng.setstate(ts["rng"])
        torch.set_rng_state(ts["torch_rng"])
        torch.cuda.set_rng_state(ts["cuda_rng"])

    # ------------------------------------------------------------------ main loop
    def _evaluate(self) -> None:
        self.state.phase = "evaluating"
        self.model.eval()
        t0 = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            report, samples = run_eval(self.model, self.processor, self.collator, self.eval_set,
                                       batch_size=8, max_new_tokens=self.rl["max_new_tokens"])
        o = report["overall"]
        shown = ("node_f1", "edge_f1", "edge_label_f1", "kind_acc", "valid_rate", "exact_rate")
        self.state.eval = {k: o[k] for k in shown}
        self.state.eval_step = self.step
        per = {s: m["edge_f1"] for s, m in report["per_source"].items()}
        log.info("eval step %d: node_f1 %.3f  edge_f1 %.3f  label_f1 %.3f  valid %.0f%%  exact %.0f%%  "
                 "(%.0fs)  per-source edge F1 %s", self.step, o["node_f1"], o["edge_f1"], o["edge_label_f1"],
                 100 * o["valid_rate"], 100 * o["exact_rate"], time.time() - t0, per)
        self.metrics_fh.write(json.dumps({"step": self.step, "eval": report}) + "\n")
        self.metrics_fh.flush()
        with open(self.run_dir / "eval_samples.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"step": self.step, "samples": samples}, ensure_ascii=False) + "\n")
        if not self.smoke and (self.state.best is None or o["edge_f1"] > self.state.best):
            self.state.best = o["edge_f1"]
            self._save(tag="best")
        self.state.phase = "training"

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
        torch.manual_seed(self.cfg["seed"])
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.cuda.set_per_process_memory_fraction(self.rl.get("vram_fraction", 0.94))
        self._load_data()
        self._load_model()
        self._maybe_resume()
        total = self.rl["smoke_steps"] if self.smoke else self.rl["steps"]
        self.state.total_steps, self.state.step = total, self.step
        self.state.start_step = self.step
        (self.run_dir / "config.used.json").write_text(json.dumps(self.cfg, indent=2), encoding="utf-8")
        self._prompt_pool()
        if not self.pool:
            log.error("empty prompt pool - nothing to learn from")
            return
        mix: dict[str, int] = {}
        for e in self.pool:
            mix[e.source] = mix.get(e.source, 0) + 1
        self.state.mix = dict(sorted(mix.items(), key=lambda kv: -kv[1]))
        if self.step == 0:
            self._evaluate()  # baseline before any RL update
        eval_every = 2 if self.smoke else self.rl["eval_every"]
        save_every = 4 if self.smoke else self.cfg["save"]["every_steps"]
        self.state.phase = "training"
        while self.step < total and not self.stop_requested:
            self._step()
            if self.step % eval_every == 0 or self.step == total:
                self._evaluate()
            if self.step % save_every == 0 and not self.smoke:
                self._save()
        if not self.smoke:
            self._save()
            self._save(tag="final")
        self.state.phase = "stopped" if self.stop_requested else "done"
        log.info("finished at step %d. Run dir: %s", self.step, self.run_dir)
        time.sleep(3 if self.smoke else 10)

    def _step(self) -> None:
        t0 = time.time()
        warm = self.rl.get("warmup_steps", 0)
        for g in self.optimizer.param_groups:
            g["lr"] = self.rl["lr"] * min(1.0, (self.step + 1) / max(1, warm))
        need = self.rl["prompts_per_step"]
        kept, sampled, all_rewards, all_metrics, truncs, lengths = [], 0, [], [], [], []
        for _ in range(1 + self.rl["max_resample"]):
            groups = self._rollout(self._sample_prompts(need - len(kept)))
            for g in groups:
                sampled += 1
                all_rewards += g["rewards"]
                all_metrics += g["metrics"]
                truncs += g["truncated"]
                lengths += [len(c) for c in g["comps"]]
                mean = sum(g["rewards"]) / len(g["rewards"])
                spread = max(g["rewards"]) - min(g["rewards"])
                if spread > 1e-6:  # dynamic sampling: identical rewards carry no signal
                    g["adv"] = [r - mean for r in g["rewards"]]
                    kept.append(g)
            if len(kept) >= need or self.stop_requested:
                break
        stats = self._update(kept) if kept else {"pg_loss": 0.0, "clip_frac": 0.0, "grad_norm": 0.0}
        self.step += 1
        dt = time.time() - t0
        n = max(1, len(all_rewards))
        reward = sum(all_rewards) / n
        edge = sum(m["edge_f1"] for m in all_metrics) / n
        s = self.state
        s.step, s.loss, s.grad_norm = self.step, stats["pg_loss"], stats["grad_norm"]
        s.lr = self.optimizer.param_groups[0]["lr"]
        s.loss_ema = reward if s.loss_ema is None else 0.95 * s.loss_ema + 0.05 * reward
        s.loss_hist.append(reward)
        s.tokens_per_s, s.samples_per_s = sum(lengths) / dt, n / dt
        s.samples_seen += n
        s.extra = {"reward": f"{reward:.3f}", "rollout edge F1": f"{edge:.3f}",
                   "groups kept": f"{len(kept)}/{sampled}", "truncated": f"{sum(truncs) / n:.0%}",
                   "answer tokens": f"{sum(lengths) / n:.0f}", "clip frac": f"{stats['clip_frac']:.3f}"}
        rec = {"step": self.step, "reward": reward, "rollout_edge_f1": edge, "kept": len(kept),
               "sampled": sampled, "truncated": sum(truncs) / n, "answer_tokens": sum(lengths) / n,
               "time": dt, **stats}
        self.metrics_fh.write(json.dumps(rec) + "\n")
        self.metrics_fh.flush()
        log.info("step %d  reward %.3f  edge_f1 %.3f  kept %d/%d  trunc %.0f%%  tok %.0f  pg %.4f  "
                 "gnorm %.2f  %.0fs", self.step, reward, edge, len(kept), sampled, 100 * sum(truncs) / n,
                 sum(lengths) / n, stats["pg_loss"], stats["grad_norm"], dt)
        if not math.isfinite(stats["pg_loss"]):
            log.error("non-finite loss - stopping")
            self.stop_requested = True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=Path(__file__).with_name("config_rl.yaml"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-dashboard", action="store_true")
    args = ap.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    warnings.filterwarnings("ignore")
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    trainer = RLTrainer(cfg, smoke=args.smoke, dashboard=not args.no_dashboard)
    if args.no_dashboard:
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    trainer.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
