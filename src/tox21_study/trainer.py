"""Centralized and C1 FedAvg training with optional CleaR routing (M1/M2).

Budget: C1 = rounds x local_epochs, centralized = centralized_epochs (equal by config assertion).
Scheduler: cosine with min_lr_ratio on CUMULATIVE optimizer-step progress, warm-up applied once over
the whole run (never restarted per round).
FedAvg: n_k / sum(n_k) over trainable tensors only; the frozen base never moves.
Checkpoint: best validation AP, earliest on ties.
"""
from __future__ import annotations
# --- portable roots (added for the public release) -------------------------
# FEDFORMULA_ROOT         : working tree holding data/, teacher/, experiments/
# FEDFORMULA_PAPER_ROOT   : tree holding the figure/table build outputs
# FEDFORMULA_ARCHIVE_ROOT : optional archive of moved-out run outputs
import os as _os
from pathlib import Path as _Path
_ROOT = _os.environ.get('FEDFORMULA_ROOT', str(_Path.cwd()))
_PAPER = _os.environ.get('FEDFORMULA_PAPER_ROOT', _ROOT)
_ARCHIVE = _os.environ.get('FEDFORMULA_ARCHIVE_ROOT', _ROOT)
_HOME = _os.path.expanduser('~')
# ---------------------------------------------------------------------------

import copy
import json
import math
import os
import time
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .batching import eval_batches
from .clear_routing import GMMFitError, consistency_loss, period_state, score_pass
from .config import seed_streams
from .metrics import detection_metrics, evaluate, threshold_by_validation_ba
from .routing import deterministic_mask, sample_training_mask, stable_molecule_hash

ROOT = f"{_ROOT}"
DERIVED = f"{ROOT}/data/derived/tox21_nr_er"


def cosine_lr(step: int, total: int, base_lr: float, warmup_ratio: float, min_ratio: float) -> float:
    w = max(1, int(round(warmup_ratio * total)))
    if step < w:
        return base_lr * (step + 1) / w
    t = (step - w) / max(1, total - w)
    return base_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, t))))


def load_data(cfg):
    coh = pd.read_csv(f"{DERIVED}/cohort.csv")
    sp = pd.read_csv(f"{DERIVED}/split_seed{cfg.split_seed}.csv")
    tr_noisy = pd.read_csv(f"{DERIVED}/train_{cfg.noise}_seed{cfg.repeat_seed}.csv")
    audit = pd.read_csv(f"{DERIVED}/noise_audit_{cfg.noise}_seed{cfg.repeat_seed}.csv")
    idx = {m: i for i, m in enumerate(coh.molecule_id)}
    tr = coh.iloc[[idx[m] for m in tr_noisy.molecule_id]].reset_index(drop=True)
    train = pd.DataFrame({"molecule_id": tr_noisy.molecule_id, "smiles": tr.canonical_smiles.values,
                          "client_id": tr_noisy.client_id, "y": tr_noisy.observed_label.values,
                          "is_flip": audit.is_injected_flip.values})
    out = {"train": train}
    for s in ("validation", "test"):
        sub = coh[sp.split.values == s]
        out[s] = pd.DataFrame({"molecule_id": sub.molecule_id.values, "smiles": sub.canonical_smiles.values,
                               "y": sub.label.values})
    return out


class Runner:
    def __init__(self, cfg, device: str, out_dir: str):
        self.cfg, self.device, self.out = cfg, device, out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.data = load_data(cfg)
        self.streams = seed_streams(cfg.repeat_seed)
        torch.manual_seed(cfg.repeat_seed)
        np.random.seed(cfg.repeat_seed)
        from .backbones.small import build_backbone
        self.bb = build_backbone(cfg, device)
        self.uses_routing = cfg.method in ("M1", "M2", "M2F")
        self.frozen_state: dict = {}       # M2F: client_id -> (p_map, target_map), estimated once
        self.ctx = self.bb.enable_routing() if (self.uses_routing and hasattr(self.bb, "enable_routing")) else None
        self.history, self.gmm_audit = [], []
        self.best = {"average_precision": -1.0, "period": None}
        self.t0 = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)

    # ---------- core ----------
    def _micro_batches(self, df, order):
        mb = self.cfg.micro_batch_size
        for s in range(0, len(order), mb):
            yield df.iloc[order[s:s + mb]]

    def _train_one_epoch(self, df, opt, step_ref, total_steps, p_map, target_map, rng):
        cfg = self.cfg
        self.bb.model.train()
        order = rng.permutation(len(df))
        acc_n, acc_loss = 0, 0.0
        eff, mb = cfg.effective_batch_size, cfg.micro_batch_size
        accum = max(1, eff // mb)
        opt.zero_grad()
        micro_i = 0
        for sub in self._micro_batches(df, order):
            lr = cosine_lr(step_ref[0], total_steps, cfg.learning_rate,
                           cfg.scheduler_warmup_ratio, cfg.min_lr_ratio)
            for g in opt.param_groups:
                g["lr"] = lr
            batch = self.bb.encode(list(sub.smiles),
                                   augment=(self.bb.kind in ("chemfm_3b", "qwen_3b")
                                            and cfg.smiles_augmentation_probability > 0))
            y = torch.tensor(sub.y.values, dtype=torch.float32, device=self.device)
            if self.ctx is not None:
                if p_map is None:                       # warm-up: p_i = 1
                    p_i = torch.ones(len(sub), device=self.device)
                else:
                    p_i = torch.tensor([p_map[m] for m in sub.molecule_id], dtype=torch.float32,
                                       device=self.device)
                self.ctx.enabled = True
                self.ctx.set_mask(sample_training_mask(p_i, self.ctx.n_layers, cfg.routing_gamma,
                                                       self.train_gen))
            logits = self.bb.forward(batch)
            loss = F.binary_cross_entropy_with_logits(logits, y, reduction="mean")   # unweighted
            if target_map is not None:
                t = torch.tensor([target_map[m] for m in sub.molecule_id], dtype=torch.float32,
                                 device=self.device)
                loss = loss + cfg.consistency_lambda * consistency_loss(logits, t)
            (loss / accum).backward()
            acc_loss += float(loss.item()) * len(sub)
            acc_n += len(sub)
            micro_i += 1
            if micro_i % accum == 0:
                opt.step()
                opt.zero_grad()
                step_ref[0] += 1
        if micro_i % accum != 0:
            opt.step()
            opt.zero_grad()
            step_ref[0] += 1
        return acc_loss / max(1, acc_n)

    @torch.no_grad()
    def predict(self, df, mc: bool):
        cfg = self.cfg
        self.bb.model.eval()
        n = len(df)
        if mc and self.ctx is not None:
            h = stable_molecule_hash(list(df.molecule_id))
            probs = np.zeros((cfg.clear_mc_passes, n))
            smi = list(df.smiles.values)
            for m in range(cfg.clear_mc_passes):
                mask = deterministic_mask(h, self.ctx.n_layers, mc_index=m,
                                          base_seed=cfg.repeat_seed, gamma=cfg.routing_gamma)
                for pos in eval_batches(self.bb, smi, cfg.eval_batch_size, cfg.sort_eval_by_length):
                    self.ctx.enabled = True
                    self.ctx.set_mask(torch.from_numpy(mask[pos]).to(self.device))
                    b = self.bb.encode([smi[i] for i in pos], augment=False)
                    probs[m, pos] = torch.sigmoid(self.bb.forward(b)).float().cpu().numpy()
            return probs.mean(axis=0)                      # probability average (spec §12)
        if self.ctx is not None:
            self.ctx.enabled = False
        out = np.zeros(n)
        smi = list(df.smiles.values)
        for pos in eval_batches(self.bb, smi, cfg.eval_batch_size, cfg.sort_eval_by_length):
            b = self.bb.encode([smi[i] for i in pos], augment=False)
            out[pos] = torch.sigmoid(self.bb.forward(b)).float().cpu().numpy()
        return out

    def _period_clear_state(self, df, period, client_id):
        """Returns (p_map, target_map) or (None, None) during warm-up."""
        if not self.uses_routing or period < self.cfg.warmup_periods:
            return None, None
        if self.cfg.method == "M2F" and client_id in self.frozen_state:
            return self.frozen_state[client_id]        # estimated once, reused unchanged
        h = stable_molecule_hash(list(df.molecule_id))
        b, bce = score_pass(self.bb, self.ctx, list(df.smiles), df.y.values.astype(float), h,
                            self.cfg, period, self.cfg.eval_batch_size, self.device)
        p, target = period_state(self.cfg.method, b, bce, df.y.values.astype(int), self.cfg,
                                 period, client_id, self.gmm_audit)
        det = detection_metrics(df.is_flip.values, p, self.cfg.diagnostic_clean_threshold)
        det.update({"period": period, "client_id": client_id})
        self.gmm_audit.append({"group": f"period{period}|client{client_id}|detection", **det})
        out = (dict(zip(df.molecule_id, p)), dict(zip(df.molecule_id, target)))
        if self.cfg.method == "M2F":
            self.frozen_state[client_id] = out
        return out

    def _evaluate_period(self, period, extra):
        val = self.data["validation"]
        p = self.predict(val, mc=self.uses_routing)
        m = evaluate(val.y.values, p)
        row = {"period": period, "val_average_precision": m["average_precision"],
               "val_roc_auc": m["roc_auc"], "elapsed_s": round(time.time() - self.t0, 1),
               "peak_vram_mb": round(torch.cuda.max_memory_allocated(self.device) / 2 ** 20, 1)
               if torch.cuda.is_available() else 0, **extra}
        self.history.append(row)
        improved = m["average_precision"] > self.best["average_precision"] + 1e-12   # earliest tie wins
        if improved:
            self.best = {"average_precision": m["average_precision"], "roc_auc": m["roc_auc"],
                         "period": period}
            self.best_state = copy.deepcopy(self.bb.trainable_state())
            self.best_val_probs = p
        print(f"  [{self.cfg.run_id}] period {period}: val AP {m['average_precision']:.4f} "
              f"AUROC {m['roc_auc']:.4f}{' *' if improved else ''}", flush=True)
        return improved

    # ---------- loops ----------
    def run(self):
        cfg = self.cfg
        self.train_gen = torch.Generator(device=self.device).manual_seed(
            int(np.random.SeedSequence(self.streams["routing"]).generate_state(1)[0]) % (2 ** 31))
        loader_rng = np.random.default_rng(np.random.SeedSequence(self.streams["loader"]))
        train = self.data["train"]
        if cfg.environment == "centralized":
            steps_per_epoch = math.ceil(len(train) / cfg.effective_batch_size)
            total = steps_per_epoch * cfg.centralized_epochs
            opt = torch.optim.AdamW([p for p in self.bb.model.parameters() if p.requires_grad],
                                    lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
            step = [0]
            for ep in range(cfg.centralized_epochs):
                pm, tm = self._period_clear_state(train, ep, "central")
                tl = self._train_one_epoch(train, opt, step, total, pm, tm, loader_rng)
                self._evaluate_period(ep, {"train_loss": round(tl, 6), "cum_steps": step[0]})
        else:
            clients = {c: train[train.client_id == c].reset_index(drop=True)
                       for c in range(cfg.num_clients)}
            spe = {c: math.ceil(len(d) / cfg.effective_batch_size) for c, d in clients.items()}
            total = sum(spe.values()) * cfg.rounds * cfg.local_epochs // max(1, cfg.num_clients)
            gstate = self.bb.trainable_state()
            step = [0]
            for rd in range(cfg.rounds):
                states, weights, losses = [], [], []
                for c in range(cfg.num_clients):          # participation_fraction = 1.0
                    self.bb.load_trainable(gstate)
                    pm, tm = self._period_clear_state(clients[c], rd, c)
                    opt = torch.optim.AdamW(                # reset_local_optimizer_each_round = True
                        [p for p in self.bb.model.parameters() if p.requires_grad],
                        lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
                    local_step = [step[0]]
                    for _ in range(cfg.local_epochs):
                        losses.append(self._train_one_epoch(clients[c], opt, local_step, total,
                                                            pm, tm, loader_rng))
                    states.append(self.bb.trainable_state())
                    weights.append(len(clients[c]))
                step[0] += max(1, int(np.mean([spe[c] for c in range(cfg.num_clients)]))) * cfg.local_epochs
                tot = float(sum(weights))
                gstate = OrderedDict((k, sum(s[k].float() * (w / tot) for s, w in zip(states, weights))
                                      .to(states[0][k].dtype)) for k in states[0])
                self.bb.load_trainable(gstate)
                self._evaluate_period(rd, {"train_loss": round(float(np.mean(losses)), 6),
                                           "cum_steps": step[0],
                                           "client_sizes": [len(clients[c]) for c in range(cfg.num_clients)]})
        return self.finalize()

    def finalize(self):
        cfg = self.cfg
        self.bb.load_trainable(self.best_state)
        val, test = self.data["validation"], self.data["test"]
        thr = threshold_by_validation_ba(val.y.values, self.best_val_probs)
        vm = evaluate(val.y.values, self.best_val_probs, thr)
        tp = self.predict(test, mc=self.uses_routing)
        tm = evaluate(test.y.values, tp, thr)
        pd.DataFrame({"molecule_id": np.concatenate([val.molecule_id, test.molecule_id]),
                      "split": ["validation"] * len(val) + ["test"] * len(test),
                      "label": np.concatenate([val.y.values, test.y.values]),
                      "probability": np.concatenate([self.best_val_probs, tp])}).to_csv(
            f"{self.out}/predictions.csv", index=False)
        pd.DataFrame(self.history).to_csv(f"{self.out}/history.csv", index=False)
        if self.gmm_audit:
            pd.DataFrame(self.gmm_audit).to_csv(f"{self.out}/gmm_audit.csv", index=False)
        torch.save(self.best_state, f"{self.out}/best_trainable_state.pt")
        summary = {"run_id": cfg.run_id, "config_hash": cfg.hash(), "status": "completed",
                   "best_period": self.best["period"], "decision_threshold": thr,
                   "validation": vm, "test": tm,
                   "wall_time_s": round(time.time() - self.t0, 1),
                   "peak_vram_mb": round(torch.cuda.max_memory_allocated(self.device) / 2 ** 20, 1)
                   if torch.cuda.is_available() else 0,
                   "n_trainable": self.bb.n_trainable(), "config": cfg.to_dict()}
        json.dump(summary, open(f"{self.out}/summary.json", "w"), indent=2)
        return summary
