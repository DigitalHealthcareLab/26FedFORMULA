#!/usr/bin/env python
"""mt_v1 — 12 엔드포인트 멀티태스크. run_main_report.py 의 멀티태스크 판.

단일 태스크 러너와 프로토콜은 동일하다: 사전 등록된 예산의 마지막 체크포인트가 주 결과이고,
원 validation split 은 어디서도 읽지 않는다. 다른 것은 셋뿐이다.

  1) 헤드가 12 로짓이고, 손실이 관측 마스크로 가려진다 (data_mt.masked_loss).
  2) T1 의 타깃 t 가 분자당 스칼라가 아니라 (N, 12) 행렬이다.
  3) lambda_miss 가 미관측 칸에 교사 타깃을 얹는다 — 단일 태스크에는 없던 축이다.

    python run_main_mt.py --backbone chemfm_1b --mode fl --noise PN20 --method T1 \
        --lam_miss 0.5 --arm L5
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
import argparse, hashlib, json, math, os, sys, time
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp

R = f"{_ROOT}"
sys.path.insert(0, f"{R}/src")
from tox21_study.config import RunConfig                                   # noqa: E402
from tox21_study.trainer import cosine_lr                                  # noqa: E402
from tox21_study.data_mt import load_mt, masked_loss, task_weights, N_TASKS, TASKS  # noqa: E402
from tox21_study.fastbatch import (budget_micro_batches, token_lengths,    # noqa: E402
                                   bucketed_batches, BUCKET_POOL, eval_groups)
from tox21_study.runutil import (require_env, require_free_vram, atomic_json,  # noqa: E402
                                 atomic_to_csv, atomic_torch_save)
from tox21_study.clear_routing import period_state, score_pass_mt          # noqa: E402
from tox21_study.routing import sample_training_mask, stable_molecule_hash  # noqa: E402

OUT = os.environ.get("JYL_OUT", f"{R}/experiments/sweep_v1/outputs")   # fedformula: results dir
CKPT = f"{OUT}/ckpt"
TEACHER = f"{R}/teacher"                                                 # teacher_q_*.csv, teacher_quality_*.csv
TEACHERS = ["chemdfm_8b", "qwen2_5_0_5b", "qwen2_5_1_5b", "qwen2_5_3b"]   # first = study teacher (dir teacher/)


def set_teacher(name):
    """Exp3: the study teacher's files live in teacher/, the others in teacher/<name>/ (score_teacher_mt.py)."""
    global TEACHER
    TEACHER = f"{R}/teacher" if name == TEACHERS[0] else f"{R}/teacher/{name}"
TRAIN_BUDGET = int(os.environ.get("JYL_TRAIN_BUDGET", 3072))   # tokens per micro-batch; lower = less VRAM, same gradient math
EVAL_BUDGET = int(os.environ.get("JYL_EVAL_BUDGET", 4096))     # tokens per scoring group (no grad)
SAVE_AT = (3, 10, 15, 20)
# 실측 (12 로짓 헤드, 워커 1, 부모는 CPU 초기화): 학습 워커 하나가 ~14 GB.
# 여유를 얹어 16 GB 를 요구한다. 워커를 늘리면 카드당 이만큼씩 더 든다.
# 실측 (12 로짓 헤드, 워커 1, 부모는 CPU 초기화): ChemFM-1B 학습 워커가 **14,814 MB**.
# 2026-09-21: GPU 1 만 쓰게 되고 another user 가 32.5 GB 를 점유하면서 내 창이 16.0 GB 로 좁아졌다.
# 기존 요구값 16000 은 실측보다 1.2 GB 과다해 간발의 차로 큐가 계속 대기했다.
# 실측 + 약 400MB 여유인 15200 으로 낮춘다. 남의 잡이 1.2 GB 이상 커지면 OOM 이 나지만,
# 그 경우는 실행기가 30 분 냉각 후 재시도한다 — 몇 시간씩 놀리는 것보다 낫다.
# 2026-09-22 실측 (12 로짓 헤드, 워커 1, 부모 CPU 초기화):
#   ChemFM-1B 14.8 GB / Qwen2.5-3B 37.9 GB / ChemFM-3B 36.8 GB
# ChemFM-3B 는 가중치가 Qwen-3B 의 두 배(12 vs 6.2 GB)인데도 총 사용량은 오히려 조금 적다 —
# 어휘 크기가 작아 임베딩·로짓 텐서가 작기 때문으로 보인다. 42000 은 5 GB 과대추정이지만
# 처음엔 42000 으로 두었으나(과대추정은 대기만 시키니 안전하다고 봤다), 09:24 에 다른
# 사용자가 7.5 GB 를 점유하자 여유가 40988 MB 로 떨어져 **그 과대추정 자체가 큐를 막았다.**
# 실측 36.8 GB + 1.7 GB 여유인 38500 으로 내린다. 과대추정도 공짜가 아니다.
# 2026-09-22 실측: Qwen2.5-3B 학습이 **37.9 GB**. 처음 잡은 18000 은 2 배 이상 과소추정이었다
# (카드가 비어 있어 우연히 통과했다). ChemFM-3B 는 가중치가 12GB 로 Qwen-3B(6.2GB)의 두 배라
# 더 클 것으로 보고 42000 으로 잡는다. 과소추정은 OOM 을 반복시키지만 과대추정은 대기만
# 시키므로, 공유 카드에서는 후자가 안전하다.
# 3B 급은 가중치만 2~3 배라 그만큼 더 든다. 여유를 얹은 값이고, 모자라면 OOM 으로 죽고
# 실행기가 재시도한다 (측정되면 이 값을 실측으로 갱신할 것).
NEED_MB = {"chemfm_1b": 15200, "chemfm_3b": 38500, "qwen25_3b": 38000,
           "qwen25_0_5b": 7000, "qwen25_1_5b": 17000,
           "ecfp_mlp": 2000, "d_mpnn": 2000}
if os.environ.get("PN_NEED_MB"):
    NEED_MB = {k: int(os.environ["PN_NEED_MB"]) for k in NEED_MB}

# 셀 이름. 단일 태스크 표와 행이 1:1 대응하도록 접두를 맞춘다.
#   MC/MF = Centralized / FL 평범한 LoRA,  MR = CleaR,  MT = 우리 방법(T1)
# **백본 코드가 맨 앞에 붙는다.** 이것이 없으면 ChemFM-1B 의 MF20 과 ChemFM-3B 의 MF20 이
# 같은 cell_id 를 갖고 체크포인트·지표가 서로를 덮어쓴다. 기본 백본(chemfm_1b)만 빈 접두를
# 유지해 기존 37 셀의 이름이 바뀌지 않게 한다.
BB_CODE = {"chemfm_1b": "", "chemfm_3b": "Q", "qwen25_3b": "Y",
           "qwen25_0_5b": "Z", "qwen25_1_5b": "V"}
NOISE_SUFFIX = {"N0_v13": "00", "PN10": "10", "PN20": "20", "PN30": "30"}


def cell_of(mode, noise, method, arm=None, backbone="chemfm_1b"):
    b = BB_CODE[backbone]
    s = NOISE_SUFFIX[noise]
    if method == "T1":
        return f"{b}MT{s}{arm}"
    if method == "M0":
        return b + ("MC" if mode == "cen" else "MF") + s
    return b + "MR" + s


def canon(k):
    return k.replace(".lora_layer.", ".")


def substream(*parts):
    return int(hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:8], 16)


def state_hash(state) -> str:
    h = hashlib.sha256()
    for k in sorted(state):
        h.update(k.encode()); h.update(state[k].contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()[:16]


def make_cfg(backbone, noise, method="M0", seed=11, lam_miss=0.0,
             task_weight="uniform", floor=0.0, **kw):
    return RunConfig(backbone=backbone, method=method, environment="C1", noise=noise,
                     repeat_seed=seed, eval_batch_size=32, n_tasks=N_TASKS,
                     lambda_miss=lam_miss, task_weight=task_weight,
                     teacher_auroc_floor=floor, **kw)


FAST_LOADER = os.environ.get("JYL_FAST_LOADER", "1") == "1"


class GpuRows:
    """fedformula loader speed-up (2026-10-13): one client's model inputs kept on the GPU for the run.

    The trainer rebuilt every micro-batch on the CPU (token ids, attention mask, labels, observed mask,
    soft target), copied each one to the GPU and read two losses back with float(). Every one of those
    waits for the GPU to finish the previous micro-batch, so the GPU sat idle while Python prepared the
    next one. Here the token table holds exactly what encode(augment=False) returns for these strings
    (same ids, right-padded with the pad id); a micro-batch is a GPU gather of its rows cut to its
    longest member, i.e. the batch encode() would have built. Checked bit-for-bit against the old path
    in scripts/verify_fast_loader.py; JYL_FAST_LOADER=0 restores the old path."""

    def __init__(self, bb, smiles, dev):
        assert bb.tokenizer.padding_side == "right", "row table assumes right padding"
        enc = bb.encode(list(smiles), augment=False)
        self.ids, self.att = enc["input_ids"], enc["attention_mask"]
        self.len = enc["attention_mask"].sum(1).cpu().numpy()
        self.dev = dev

    def index(self, groups):
        """All groups of a pass -> one GPU index tensor (a single copy) + (start, end, max_len) per group."""
        flat = (np.concatenate([np.asarray(g, dtype=np.int64) for g in groups]) if groups
                else np.zeros(0, dtype=np.int64))
        gi = torch.as_tensor(flat).to(self.dev)
        spans, s0 = [], 0
        for g in groups:
            spans.append((s0, s0 + len(g), int(self.len[np.asarray(g)].max())))
            s0 += len(g)
        return gi, spans

    def batch(self, gi, span):
        s0, s1, L = span
        r = gi[s0:s1]
        return r, {"input_ids": self.ids[r, :L].contiguous(), "attention_mask": self.att[r, :L].contiguous()}


def score_rows(bb, smi, groups, rows=None):
    """sigmoid probabilities (n, N_TASKS) for one client's molecules, eval mode, no grad."""
    if rows is None:                                  # the study's path
        out = np.zeros((len(smi), N_TASKS), dtype=np.float32)
        for g in groups:
            b = bb.encode([smi[i] for i in g], augment=False)
            out[g] = torch.sigmoid(bb.forward(b)).float().cpu().numpy()
        return out
    gi, spans = rows.index(groups)
    og = torch.empty((len(smi), N_TASKS), dtype=torch.float32, device=rows.dev)
    for sp in spans:
        r, b = rows.batch(gi, sp)
        og[r] = torch.sigmoid(bb.forward(b)).float()
    return og.cpu().numpy()


def train_one_pass(bb, d, Y, M, lens, cfg, opt, step0, total_steps, rng_key, dev,
                   ctx=None, p_map=None, target_map=None, mask_gen=None,
                   T=None, w_obs=1.0, tw=None, t_use=None, rows=None):
    """`d` 위를 한 epoch. (보고용 관측 CE, 실제 최적화한 목적값, 배치 수) 를 돌려준다.

    Y/M/T 는 d 와 행 순서가 같은 (len(d), 12) 배열이다."""
    rng = np.random.default_rng(substream("loader", *rng_key))
    batches = bucketed_batches(len(d), lens, cfg.effective_batch_size, rng, BUCKET_POOL)
    # fedformula: the study's random-SMILES augmentation is skipped. Its output was passed to
    # bb.encode(..., augment=False), which re-canonicalises every string, so the model always saw the
    # canonical SMILES (6,260/6,261 molecules identical after the round trip; measured 2026-10-13).
    # Feeding the canonical strings directly gives the same inputs without ~1.6 s/round of RDKit work,
    # and lets encode() reuse cached token ids.
    aug = list(d.smiles.values)
    bb.model.train()
    tot_ce, tot_mix, n = 0.0, 0.0, 0.0
    twt = None if tw is None else torch.tensor(tw, device=dev)
    tut = None if t_use is None else torch.tensor(t_use, dtype=torch.float32, device=dev)
    fast = rows is not None and ctx is None and target_map is None     # CleaR arms keep the old path
    if fast:
        plan = [budget_micro_batches(chunk, lens, TRAIN_BUDGET) if chunk else [] for chunk in batches]
        gi, spans = rows.index([g for groups in plan for g in groups])
        Yg, Mg = torch.as_tensor(Y).to(dev), torch.as_tensor(M).to(dev)
        Tg = None if T is None else torch.as_tensor(T, dtype=torch.float32).to(dev)
        acc = torch.zeros(3, dtype=torch.float64, device=dev)        # sums kept on the GPU, read once
        k = 0
    for bi, chunk in enumerate(batches):
        if not chunk:
            continue
        for g in opt.param_groups:
            g["lr"] = cosine_lr(step0 + bi, total_steps, cfg.learning_rate,
                                cfg.scheduler_warmup_ratio, cfg.min_lr_ratio)
        opt.zero_grad(set_to_none=True)
        # 배치 전체의 관측 칸 수로 한 번만 나눈다. 마이크로배치마다 관측 칸 수가 다르므로
        # 각자 자기 분모로 나누면 관측이 적은 마이크로배치가 과대 가중된다.
        denom = float(M[chunk].sum()) or 1.0
        for grp in (plan[bi] if fast else budget_micro_batches(chunk, lens, TRAIN_BUDGET)):
            if fast:
                r, b = rows.batch(gi, spans[k]); k += 1
                lg = bb.forward(b).float()
                rep, loss, n_obs = masked_loss(lg, Yg[r], Mg[r], t=None if Tg is None else Tg[r],
                                               w_obs=w_obs, lam_miss=cfg.lambda_miss, tw=twt, t_use=tut)
                acc += torch.stack([rep.detach().double(), loss.detach().double(), n_obs.double()])
                (loss / denom).backward()
                continue
            b = bb.encode([aug[j] for j in grp], augment=False)
            if ctx is not None:
                p_i = (torch.ones(len(grp), device=dev) if p_map is None else
                       torch.tensor([p_map[m] for m in d.molecule_id.values[grp]],
                                    dtype=torch.float32, device=dev))
                ctx.enabled = True
                ctx.set_mask(sample_training_mask(p_i, ctx.n_layers, cfg.routing_gamma, mask_gen))
            lg = bb.forward(b).float()
            y = torch.tensor(Y[grp], device=lg.device)
            m = torch.tensor(M[grp], device=lg.device)
            t = None if T is None else torch.tensor(T[grp], dtype=torch.float32, device=lg.device)
            rep, loss, n_obs = masked_loss(lg, y, m, t=t, w_obs=w_obs,
                                           lam_miss=cfg.lambda_miss, tw=twt, t_use=tut)
            if target_map is not None:                      # CleaR 자기일관성 항
                tg = torch.tensor(np.stack([target_map[mm] for mm in d.molecule_id.values[grp]]),
                                  dtype=torch.float32, device=lg.device)
                _, cons, _ = masked_loss(lg, tg, m, tw=twt)
                loss = rep + cfg.consistency_lambda * cons
            tot_ce += float(rep.detach()); tot_mix += float(loss.detach())
            n += float(n_obs)
            (loss / denom).backward()
        opt.step()
    if fast:
        tot_ce, tot_mix, n = acc.tolist()
    return tot_ce / max(1.0, n), tot_mix / max(1.0, n), len(batches)


def worker(rank, devices, backbone, noise, in_q, out_q, method, seed, cfgkw):
    torch.manual_seed(seed); np.random.seed(seed)
    dev = devices[rank % len(devices)]
    cfg = make_cfg(backbone, noise, method, seed, **cfgkw)
    D = load_mt(cfg)
    train, Y, M = D["train"], D["train_Y"], D["train_M"]
    tw = task_weights(M, cfg.task_weight)
    from tox21_study.backbones.small import build_backbone
    bb = build_backbone(cfg, dev)
    ctx = bb.enable_routing() if method in ("M2", "M2F") else None
    kmap = {canon(n): n for n, p in bb.model.named_parameters() if p.requires_grad}
    lens_all = token_lengths(bb, list(train.smiles.values))
    sel = {c: np.where(train.client_id.values == c)[0] for c in range(cfg.num_clients)}
    clients = {c: train.iloc[i].reset_index(drop=True) for c, i in sel.items()}
    cY = {c: Y[i] for c, i in sel.items()}
    cM = {c: M[i] for c, i in sel.items()}
    lens_of = {c: lens_all[i] for c, i in sel.items()}
    spe = {c: math.ceil(len(d) / cfg.effective_batch_size) for c, d in clients.items()}
    rows_of = {c: GpuRows(bb, d.smiles.values, dev) for c, d in clients.items()} if FAST_LOADER else {}
    groups_of = {}                     # eval groups depend only on the strings: tokenise once, not every round
    frozen: dict = {}
    total = sum(spe.values()) * cfg.rounds * cfg.local_epochs // max(1, cfg.num_clients)
    out_q.put(("ready", rank))
    while True:
        m = in_q.get()
        if m[0] == "stop":
            return
        if m[0] == "score":
            # 현재 전역 모델을 이 클라이언트 행 위에서 채점. 12 확률/분자.
            _, gstate, client = m
            bb.load_trainable({kmap[k]: v.to(dev) for k, v in gstate.items()})
            d = clients[client]
            bb.model.eval()
            with torch.no_grad():
                smi = list(d.smiles.values)
                if client not in groups_of:
                    groups_of[client] = eval_groups(bb, smi, EVAL_BUDGET)[0]
                out = score_rows(bb, smi, groups_of[client], rows_of.get(client))
            out_q.put(("scored", rank, client, list(d.molecule_id.values), out))
            continue
        _, rd, client, gstate, Tsub, w_obs, t_use = m
        bb.load_trainable({kmap[k]: v.to(dev) for k, v in gstate.items()})
        opt = torch.optim.AdamW([p for p in bb.model.parameters() if p.requires_grad],
                                lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        step0 = int(np.mean(list(spe.values()))) * rd * cfg.local_epochs
        loss = mix = 0.0
        p_map = target_map = mask_gen = None
        audit = []
        if ctx is not None:
            mask_gen = torch.Generator(device=dev)
            mask_gen.manual_seed(substream("clear_mask", seed, rd, client) % (2 ** 63))
            if rd >= cfg.warmup_periods and not (method == "M2F" and client in frozen):
                d, yy, mm = clients[client], cY[client], cM[client]
                h = stable_molecule_hash(list(d.molecule_id))
                # CleaR 의 GMM 은 태스크마다 따로 돈다. 태스크별 오염률이 다르므로
                # 12 태스크를 한 혼합모형에 넣으면 태스크 간 난이도 차이를 오염으로 읽는다.
                PI = np.ones((len(d), N_TASKS)); TG = yy.copy()
                BLG, BCE = score_pass_mt(bb, ctx, list(d.smiles.values), yy, mm, h,
                                       cfg, rd, cfg.eval_batch_size, dev)
                for ti in range(N_TASKS):
                    obs = np.where(mm[:, ti] > 0)[0]
                    # 태스크마다 GMM 을 따로 적합한다. 태스크별 오염률과 난이도가 다르므로
                    # 12 태스크를 한 혼합모형에 넣으면 난이도 차이를 오염으로 읽는다.
                    if len(obs) < 20 or len(set(yy[obs, ti])) < 2:
                        audit.append({"group": f"period{rd}|client{client}|task{TASKS[ti]}",
                                      "n_samples": int(len(obs)), "status": "skipped_task"})
                        continue
                    pi, tg = period_state(method, BLG[obs, ti], BCE[obs, ti],
                                          yy[obs, ti].astype(int),
                                          cfg, rd, f"{client}|{TASKS[ti]}", audit)
                    PI[obs, ti] = pi; TG[obs, ti] = tg
                # 라우팅 마스크는 분자 단위 스칼라라서 태스크 평균을 쓴다
                p_map = dict(zip(d.molecule_id, PI.mean(1)))
                target_map = dict(zip(d.molecule_id, TG))
                if method == "M2F":
                    frozen[client] = (p_map, target_map)
            elif method == "M2F" and client in frozen:
                p_map, target_map = frozen[client]
            ctx.enabled = False
        for e in range(cfg.local_epochs):
            loss, mix, _ = train_one_pass(bb, clients[client], cY[client], cM[client],
                                          lens_of[client], cfg, opt,
                                          step0 + e * spe[client], total,
                                          ("fl", seed, rd, client, e), dev,
                                          ctx=ctx, p_map=p_map, target_map=target_map,
                                          mask_gen=mask_gen, T=Tsub, w_obs=w_obs,
                                          tw=tw, t_use=t_use, rows=rows_of.get(client))
        if ctx is not None:
            ctx.enabled = False
        st = OrderedDict((canon(n), p.detach().cpu().clone())
                         for n, p in bb.model.named_parameters() if p.requires_grad)
        out_q.put(("done", rank, client, st, len(clients[client]), loss, audit, mix))


def teacher_gate(floor, verbose=True):
    """교사 AUROC 이 floor 미만인 태스크를 끈다. 길이 12 의 0/1 벡터를 돌려준다.

    품질은 **train split 관측 칸** 위에서 잰 것을 쓴다. test 위에서 고르면 테스트를 보고
    방법을 정하는 것이 되고, 원 validation 은 이 연구에서 읽지 않기로 되어 있다."""
    if floor <= 0:
        return np.ones(N_TASKS, dtype=np.float32), {}
    q = pd.read_csv(f"{TEACHER}/teacher_quality_train.csv").set_index("task")
    u = np.array([1.0 if (pd.notna(q.AUROC.get(t)) and q.AUROC[t] >= floor) else 0.0
                  for t in TASKS], dtype=np.float32)
    info = {t: (None if pd.isna(q.AUROC.get(t)) else float(q.AUROC[t])) for t in TASKS}
    if verbose:
        print(json.dumps({"teacher_gate_floor": floor, "n_on": int(u.sum()),
                          "off": [t for t, k in zip(TASKS, u) if k == 0]}), flush=True)
    return u, info


def _abandon(procs, in_qs, out_q):
    """워커가 죽은 뒤 부모가 깨끗하게 빠져나가게 한다.

    이게 없으면 부모가 **영원히 매달린다**. 죽은 소비자의 큐에 put 을 하면 데이터가
    피더 스레드 버퍼에 남고, 파이썬은 종료할 때 그 스레드를 join 하려 들기 때문이다.
    소비자가 없으니 버퍼는 영원히 비지 않는다.

    그 결과가 단순한 지연이 아니다. 좀비 부모는 명령줄이 살아 있는 채로 남아서,
    큐 실행기의 '이미 돌고 있는 잡은 다시 띄우지 않는다' 검사에 걸린다 — 즉
    OOM 으로 죽은 셀이 **재시도조차 되지 않고** 밤새 방치된다.

    그래서 순서가 중요하다: 먼저 자식을 죽이고, 그 다음 큐의 join 을 포기한다.
    """
    for p in procs:
        try:
            if p.is_alive():
                p.terminate()
        except Exception:
            pass
    for p in procs:
        try:
            p.join(timeout=10)
        except Exception:
            pass
    for q in list(in_qs) + [out_q]:
        try:
            q.cancel_join_thread()      # 배출되지 않는 버퍼를 기다리지 않는다
            q.close()
        except Exception:
            pass


def predict_final(state, cfg, dev, backbone, suffix=""):
    """fedformula: score the final global model on validation / test / external right after
    training, so a sweep keeps ~50 KB of predictions per run instead of a 300 MB checkpoint set.
    Validation is for model selection; test and external are reported only."""
    import evaluate_mt as E
    from tox21_study.backbones.small import build_backbone
    from tox21_study.data_mt import load_mt
    torch.cuda.empty_cache()
    bb = build_backbone(cfg, dev)
    kmap = {canon(n): n for n, _ in bb.model.named_parameters()}
    bb.load_trainable({kmap.get(k, k): v.to(dev) for k, v in state.items()})
    D = load_mt(cfg)
    ext, EY, EM = E.load_external()
    out, arrays = {}, {}
    for name, frame, Y, M in (("val", D["validation"], D["validation_Y"], D["validation_M"]),
                              ("test", D["test"], D["test_Y"], D["test_M"]),
                              ("ext", ext, EY, EM)):
        P = E.predict(bb, list(frame.smiles.values))
        m = E.macro(Y, M, P)
        out[f"{name}_AUROC"], out[f"{name}_AP"] = m["macro_AUROC"], m["macro_AP"]
        arrays[f"{name}_p"] = P.astype(np.float32)
        arrays[f"{name}_ids"] = frame.molecule_id.values.astype(str)
    np.savez_compressed(f"{OUT}/pred/{TAG_FOR_PRED[0]}{suffix}.npz", **arrays)
    del bb; torch.cuda.empty_cache()
    print(json.dumps({"final_metrics": {k: round(v, 4) for k, v in out.items()}}), flush=True)
    return out


TAG_FOR_PRED = [None]


def w_schedule(rd, R, ws, wf, shape="linear", ramp=None):
    """10-23 W-schedule experiments: observed-label weight for 0-based round rd. t = rd/(L-1) clipped to 1, L = ramp or R;
    linear 1-t, convex (1-t)^2 (fast early), concave 1-t^2 (slow early); w = wf + (ws-wf)*f(t), so w = wf from round L on.
    Not used for the default (linear, no ramp): main() keeps the original expression there (bitwise unchanged runs)."""
    L = R if ramp is None else ramp
    t = min(1.0, rd / max(1, L - 1))
    f = {"linear": 1.0 - t, "convex": (1.0 - t) ** 2, "concave": 1.0 - t * t}[shape]
    return wf + (ws - wf) * f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="chemfm_1b",
                    choices=["chemfm_1b", "chemfm_3b", "qwen25_3b",
                             "qwen25_0_5b", "qwen25_1_5b"],
                    help="백본 크기 스케일링용. cell_id 앞에 백본 코드가 붙는다")
    ap.add_argument("--mode", required=True, choices=["cen", "fl"])
    ap.add_argument("--noise", required=True, choices=["PN10", "PN20", "PN30", "N0_v13"])
    ap.add_argument("--devices", default="cuda:0")
    ap.add_argument("--device", default=None,
                    help="--devices 의 단수 별칭. 큐 실행기가 두 러너를 같은 문법으로 부른다")
    ap.add_argument("--workers", type=int, default=1,
                    help="한 카드에 워커 둘을 올리면 모델 사본이 둘이 된다 — 같은 SM 을 두고 "
                         "경쟁해서 속도 이득은 작고 메모리는 정확히 두 배다. 카드가 여러 장일 "
                         "때만 올려라")
    ap.add_argument("--seed", type=int, default=11, choices=[11, 12, 13])
    ap.add_argument("--method", default="M0", choices=["M0", "M2", "M2F", "T1"])
    ap.add_argument("--arm", default="01", help="T1 only: 2-char arm id")
    ap.add_argument("--beta", type=float, default=0.94)
    ap.add_argument("--w_obs_start", type=float, default=1.0)
    ap.add_argument("--w_obs_final", type=float, default=0.0)
    ap.add_argument("--t_init", default="teacher", choices=["teacher", "self", "shuffle"])
    ap.add_argument("--lam_miss", type=float, default=0.0,
                    help="미관측 칸에 실을 교사 타깃의 가중. 0 이면 단일 태스크와 같은 취급")
    ap.add_argument("--task_weight", default="uniform", choices=["uniform", "inverse_freq"])
    ap.add_argument("--lora_dropout", type=float, default=None,
                    help="LoRA dropout. 기본 0.4 는 단일 태스크용으로 고른 값이라 "
                         "멀티태스크에 맞는지 미확인 — 하이퍼파라미터 3 후보")
    ap.add_argument("--teacher_floor", type=float, default=0.0,
                    help="교사 train AUROC 이 이 값 미만인 태스크에는 교사를 쓰지 않는다")
    # ---- fedformula sweep axes ----
    ap.add_argument("--rounds", type=int, default=20, help="FL rounds (= centralized epochs)")
    ap.add_argument("--num_clients", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=0.5, help="scaffold Dirichlet alpha of the partition")
    ap.add_argument("--t_mode", default="ema", choices=["ema", "latest"],
                    help="ema: t_r = beta*t_(r-1) + (1-beta)*P_global (study default). latest ('EMA X'): "
                         "t_r = beta^r*P_teacher + (1-beta^r)*P_global(latest only) - same teacher share, no history")
    ap.add_argument("--eval_at", type=int, nargs="*", default=[],
                    help="also save weights and score val/test/ext at these rounds (baselines: 100 rounds, eval 15 30 50)")
    ap.add_argument("--save_ckpt", default="none", choices=["none", "final", "all"],
                    help="none: keep only predictions/metrics (sweep default, ~300MB saved per run)")
    ap.add_argument("--train_budget", type=int, default=None,
                    help="tokens per training micro-batch (default JYL_TRAIN_BUDGET or 3072); lower = less VRAM, "
                         "same gradient math. Exp6 uses it to fit ChemFM-3B next to other users' jobs")
    ap.add_argument("--need_mb", type=int, default=None, help="free-VRAM check for this run (default NEED_MB[backbone])")
    ap.add_argument("--teacher", default=TEACHERS[0], choices=TEACHERS,
                    help="T1 only: zero-shot teacher for the initial soft target (Exp3)")
    ap.add_argument("--only_client", type=int, default=None,
                    help="10-22 local-only baseline: only this client of the K-client partition trains and nothing is "
                         "aggregated (it continues from its own model every round). M0 only")
    ap.add_argument("--size_alpha", type=float, default=None,
                    help="10-22 quantity skew: partition with client sizes ~ Dir(size_alpha) (data files *_q<size_alpha>)")
    ap.add_argument("--w_shape", default="linear", choices=["linear", "convex", "concave"],
                    help="10-23 T1 only: shape of the w_obs decay (convex = fast early, concave = slow early)")
    ap.add_argument("--w_ramp", type=int, default=None,
                    help="10-23 T1 only: w_obs reaches w_obs_final at round L and stays there (default L = rounds)")
    ap.add_argument("--no_snap_ckpt", action="store_true",
                    help="10-23: keep --eval_at snapshots in memory only (predictions still saved), no __R*.pt files")
    ap.add_argument("--save_train_p", action="store_true",
                    help="10-23 T1 FL only: save the global model's train-set predictions of every round (the target's "
                         "P_global, rounds 0..R-1) plus the final model -> pred/{tag}__trainP.npz (analyze_run's P)")
    a = ap.parse_args()
    # load_mt reads the partition name from the environment, here and in the spawned workers (cfg.hash() unchanged)
    if a.size_alpha is None:
        os.environ.pop("JYL_SIZE_ALPHA", None)
    else:
        os.environ["JYL_SIZE_ALPHA"] = str(a.size_alpha)
    if a.only_client is not None and not (a.mode == "fl" and a.method == "M0" and 0 <= a.only_client < a.num_clients):
        raise SystemExit("--only_client needs --mode fl --method M0 and 0 <= c < num_clients")
    if a.teacher != TEACHERS[0] and a.method != "T1":
        raise SystemExit("--teacher 는 T1 에서만 의미가 있다")
    set_teacher(a.teacher)
    if a.train_budget:                 # module global here, environment for the spawned workers (they re-import)
        global TRAIN_BUDGET
        TRAIN_BUDGET = a.train_budget; os.environ["JYL_TRAIN_BUDGET"] = str(a.train_budget)
    if a.device:
        a.devices = a.device
    require_env("train", f"mt {a.backbone} {a.mode} {a.noise}")
    devices = a.devices.split(",")
    require_free_vram(devices, a.need_mb or NEED_MB[a.backbone], f"mt {a.backbone} {a.mode}")
    cell = cell_of(a.mode, a.noise, a.method, a.arm, a.backbone)
    if a.method != "M0" and a.mode != "fl":
        raise SystemExit(f"{a.method} cells are FL-only")
    if a.method == "T1" and not (0.0 <= a.w_obs_final <= 1.0 and 0.0 <= a.w_obs_start <= 1.0
                                 and 0.0 <= a.beta <= 1.0):   # 10-22: beta 0 allowed (t_r = P_global, q never enters; beta ablation)
        raise SystemExit("T1 needs w_obs in [0,1] and 0 <= beta <= 1")
    if a.method != "T1" and (a.lam_miss or a.teacher_floor):
        raise SystemExit("lam_miss/teacher_floor 는 T1 에서만 의미가 있다")
    if (a.w_shape != "linear" or a.w_ramp is not None or a.save_train_p) and (a.method != "T1" or a.mode != "fl"):
        raise SystemExit("--w_shape / --w_ramp / --save_train_p are T1 FL options")
    if a.w_ramp is not None and not 2 <= a.w_ramp <= a.rounds:
        raise SystemExit("--w_ramp must be in [2, rounds]")
    data_noise = "N0" if a.noise == "N0_v13" else a.noise
    cfgkw = dict(lam_miss=a.lam_miss, task_weight=a.task_weight, floor=a.teacher_floor,
                 rounds=a.rounds, centralized_epochs=a.rounds,
                 num_clients=a.num_clients, dirichlet_alpha=a.alpha)
    if a.lora_dropout is not None:
        cfgkw["lora_dropout"] = a.lora_dropout
    cfg = make_cfg(a.backbone, data_noise, a.method, a.seed, **cfgkw)
    _ = (cfg.learning_rate, cfg.periods, cfg.hash())
    sched = (f"__w{a.w_obs_start:g}-{a.w_obs_final:g}" + ("" if a.w_ramp is None else f"_L{a.w_ramp}")
             + {"linear": "", "convex": "_cvx", "concave": "_ccv"}[a.w_shape] + f"_b{a.beta:g}_{a.t_init}"
             + ("_lat" if a.t_mode == "latest" else "")
             + ("" if a.teacher == TEACHERS[0] else f"_T{a.teacher}") if a.method == "T1" else "")
    tag = (f"{cell}__{a.backbone}__{a.mode}__{a.noise}__K{a.num_clients}_a{a.alpha:g}"
           + ("" if a.size_alpha is None else f"_q{a.size_alpha:g}") + f"_R{a.rounds}"
           + ("" if a.only_client is None else f"_loc{a.only_client}")
           + f"{sched}__s{a.seed}__lm{a.lam_miss:g}__tw{a.task_weight[:3]}__fl{a.teacher_floor:g}")
    TAG_FOR_PRED[0] = tag
    print(f"[{time.strftime('%H:%M:%S')}] cell={cell} tag={tag} hash={cfg.hash()}", flush=True)
    os.makedirs(CKPT, exist_ok=True); os.makedirs(f"{OUT}/pred", exist_ok=True)
    if os.path.exists(f"{OUT}/run_meta__{tag}.json"):
        print(f"{tag} already complete, skipping"); return

    D = load_mt(cfg)
    train, Y, M = D["train"], D["train_Y"], D["train_M"]
    tw = task_weights(M, cfg.task_weight)
    t_use, gate_info = (teacher_gate(a.teacher_floor) if a.method == "T1"
                        else (None, {}))
    t0 = time.time(); rows = []
    snaps = {}                     # --eval_at snapshots (FL branch fills it)
    gmm_audit: list = []
    from tox21_study.backbones.small import build_backbone

    if a.mode == "cen":
        dev = devices[0]
        torch.manual_seed(a.seed); np.random.seed(a.seed)
        bb = build_backbone(cfg, dev)
        init_state = OrderedDict((canon(n), p.detach().cpu().clone())
                                 for n, p in bb.model.named_parameters() if p.requires_grad)
        init_hash = state_hash(init_state)
        if a.save_ckpt == "all": atomic_torch_save(init_state, f"{CKPT}/{tag}__init.pt")
        lens = token_lengths(bb, list(train.smiles.values))
        spe = math.ceil(len(train) / cfg.effective_batch_size)
        epochs = cfg.rounds * cfg.local_epochs
        total = spe * epochs
        opt = torch.optim.AdamW([p for p in bb.model.parameters() if p.requires_grad],
                                lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        for ep in range(epochs):
            te = time.time()
            ce, mix, _ = train_one_pass(bb, train, Y, M, lens, cfg, opt, ep * spe, total,
                                        ("cen", a.seed, ep), dev, tw=tw)
            rows.append({"cell": cell, "unit": "epoch", "index": ep + 1, "train_ce": ce,
                         "train_objective": mix, "seconds": round(time.time() - te, 1)})
            print(json.dumps(rows[-1]), flush=True)
            if a.save_ckpt == "all" and ep + 1 in SAVE_AT:
                atomic_torch_save(OrderedDict(
                    (canon(n), p.detach().cpu().clone())
                    for n, p in bb.model.named_parameters() if p.requires_grad),
                    f"{CKPT}/{tag}__E{ep+1}.pt")
        final = OrderedDict((canon(n), p.detach().cpu().clone())
                            for n, p in bb.model.named_parameters() if p.requires_grad)
        optimizer_steps = total
        del bb, opt; torch.cuda.empty_cache()      # free the card before predict_final rebuilds the model
    else:
        ctxmp = mp.get_context("spawn")
        in_qs = [ctxmp.Queue() for _ in range(a.workers)]
        out_q = ctxmp.Queue()
        procs = [ctxmp.Process(target=worker,
                               args=(r, devices, a.backbone, data_noise, in_qs[r], out_q,
                                     a.method, a.seed, cfgkw), daemon=True)
                 for r in range(a.workers)]
        for p in procs:
            p.start()
        for i in range(len(procs)):
            try:
                assert out_q.get(timeout=600)[0] == "ready"
            except Exception as e:
                _abandon(procs, in_qs, out_q)
                raise RuntimeError(f"a client worker never reported ready: {e}")
        # 초기 가중치는 **CPU 에서** 만든다. 부모는 forward 를 한 번도 돌리지 않고
        # named_parameters() 만 읽으므로 GPU 에 14 GB 짜리 사본을 하나 더 올릴 이유가 없다.
        # 워커가 이미 자기 사본을 들고 있는 상태라 그 둘이 겹치면 카드가 넘친다.
        # LoRA 초기화 추첨은 .to(device) 이전에 CPU 에서 일어나므로 가중치는 동일하다.
        torch.manual_seed(a.seed); np.random.seed(a.seed)
        bb = build_backbone(cfg, "cpu")
        gstate = OrderedDict((canon(n), p.detach().cpu().clone())
                             for n, p in bb.model.named_parameters() if p.requires_grad)
        init_hash = state_hash(gstate)
        if a.save_ckpt == "all": atomic_torch_save(gstate, f"{CKPT}/{tag}__init.pt")
        del bb; torch.cuda.empty_cache()
        spe = {c: math.ceil((train.client_id == c).sum() / cfg.effective_batch_size)
               for c in range(cfg.num_clients)}
        sel = {c: np.where(train.client_id.values == c)[0] for c in range(cfg.num_clients)}
        cli = {c: train.iloc[i].reset_index(drop=True) for c, i in sel.items()}
        run_c = list(range(cfg.num_clients)) if a.only_client is None else [a.only_client]   # clients that train

        # ---- T1 타깃: 분자 x 12 행렬 ----
        row_of = {m: i for i, m in enumerate(train.molecule_id.values)}
        Tm = None
        if a.method == "T1":
            if a.t_init == "self":
                Tm = None                      # 첫 EMA 갱신이 씨앗을 놓는다
            else:
                qd = pd.read_csv(f"{TEACHER}/teacher_q_train.csv").set_index("molecule_id")
                miss = set(train.molecule_id) - set(qd.index)
                if miss:
                    raise SystemExit(f"teacher q missing for {len(miss)} molecules; "
                                     f"run score_teacher_mt.py --split train first")
                Tm = qd.loc[train.molecule_id, TASKS].to_numpy(dtype=np.float32)
                if a.t_init == "shuffle":
                    # 대조군: 소프트 타깃의 주변분포는 그대로, 붙는 분자만 틀리게 한다
                    rg = np.random.default_rng(20260918)
                    Tm = Tm[rg.permutation(len(Tm))]
                print(json.dumps({"t_init": a.t_init, "t_mean": round(float(Tm.mean()), 4),
                                  "per_task_mean": {t: round(float(Tm[:, i].mean()), 3)
                                                    for i, t in enumerate(TASKS)}}), flush=True)

        T0 = None if Tm is None else Tm.copy()          # P_teacher, needed by --t_mode latest
        snaps = {}                                       # --eval_at: round -> global state (CPU)
        trainP = (np.zeros((cfg.rounds + 1, len(train), N_TASKS), dtype=np.float32)
                  if a.save_train_p else None)           # 10-23: P_global of every round (+ final), analyze_run's P
        for rd in range(cfg.rounds):
            te = time.time()
            w_obs = 1.0
            if a.method == "T1":
                if a.w_shape == "linear" and a.w_ramp is None:
                    w_obs = a.w_obs_start + (a.w_obs_final - a.w_obs_start) * (
                        rd / max(1, cfg.rounds - 1))
                else:
                    w_obs = w_schedule(rd, cfg.rounds, a.w_obs_start, a.w_obs_final, a.w_shape, a.w_ramp)
                for c in range(cfg.num_clients):
                    in_qs[c % a.workers].put(("score", gstate, c))
                PG = np.zeros((len(train), N_TASKS), dtype=np.float32)
                for _ in range(cfg.num_clients):
                    msg = out_q.get(timeout=3600)
                    assert msg[0] == "scored", msg[0]
                    PG[[row_of[m] for m in msg[3]]] = msg[4]
                if trainP is not None:
                    trainP[rd] = PG
                if a.t_mode == "latest" and T0 is not None:
                    Tm = (a.beta ** (rd + 1)) * T0 + (1.0 - a.beta ** (rd + 1)) * PG
                else:
                    Tm = PG.copy() if Tm is None else a.beta * Tm + (1.0 - a.beta) * PG
                print(json.dumps({"cell": cell, "unit": "target", "round": rd + 1,
                                  "w_obs": round(w_obs, 4),
                                  "t_mean": round(float(Tm.mean()), 4),
                                  "t_std": round(float(Tm.std()), 4)}), flush=True)
            for c in run_c:
                Tsub = None if Tm is None else Tm[sel[c]]
                in_qs[c % a.workers].put(("train", rd, c, gstate, Tsub, w_obs, t_use))
            got, losses, mixes = {}, [], []
            for _ in run_c:
                msg, waited = None, 0
                while msg is None:
                    try:
                        msg = out_q.get(timeout=20)
                    except Exception:
                        waited += 20
                        dead = [i for i, p in enumerate(procs) if not p.is_alive()]
                        if dead:
                            _abandon(procs, in_qs, out_q)
                            raise RuntimeError(f"client workers {dead} died at round {rd+1}")
                        if waited >= 3600:
                            _abandon(procs, in_qs, out_q)
                            raise RuntimeError(f"no worker reported for {waited}s at round {rd+1}")
                _, _, c, st, n, loss, audit, mix = msg
                if n != len(sel[c]):          # 10-22: worker and parent must read the same partition (JYL_SIZE_ALPHA)
                    _abandon(procs, in_qs, out_q)
                    raise RuntimeError(f"client {c}: worker trained on {n} molecules, parent partition has {len(sel[c])}")
                got[c] = (st, n); losses.append(loss); mixes.append(mix)
                gmm_audit.extend(audit)
            tot_w = float(sum(n for _, n in got.values()))
            g0 = got[run_c[0]][0]
            gstate = OrderedDict(
                (k, sum(got[c][0][k].float() * (got[c][1] / tot_w) for c in got)
                 .to(g0[k].dtype)) for k in g0)
            rows.append({"cell": cell, "unit": "round", "index": rd + 1,
                         "train_ce": float(np.mean(losses)),
                         "train_objective": float(np.mean(mixes)), "w_obs": round(w_obs, 4),
                         "seconds": round(time.time() - te, 1)})
            print(json.dumps(rows[-1]), flush=True)
            if a.save_ckpt == "all" and rd + 1 in SAVE_AT:
                atomic_torch_save(gstate, f"{CKPT}/{tag}__R{rd+1}.pt")
            if rd + 1 in a.eval_at and rd + 1 != cfg.rounds:
                snaps[rd + 1] = OrderedDict((k, v.clone()) for k, v in gstate.items())
                if not a.no_snap_ckpt:
                    atomic_torch_save(snaps[rd + 1], f"{CKPT}/{tag}__R{rd+1}.pt")
        if trainP is not None:                           # final model, same worker "score" path as every round
            for c in range(cfg.num_clients):
                in_qs[c % a.workers].put(("score", gstate, c))
            for _ in range(cfg.num_clients):
                msg = out_q.get(timeout=3600)
                assert msg[0] == "scored", msg[0]
                trainP[cfg.rounds][[row_of[m] for m in msg[3]]] = msg[4]
            np.savez_compressed(f"{OUT}/pred/{tag}__trainP.npz", P=trainP,
                                molecule_id=train.molecule_id.values.astype(str))
        for q in in_qs:
            q.put(("stop",))
        for p in procs:
            p.join(timeout=120)
            if p.is_alive():
                p.terminate()
        final = gstate
        optimizer_steps = (int(np.mean(list(spe.values()))) if a.only_client is None
                           else spe[a.only_client]) * cfg.rounds * cfg.local_epochs

    if a.save_ckpt in ("final", "all"):
        atomic_torch_save(final, f"{CKPT}/{tag}__final.pt")
    preds = predict_final(final, cfg, devices[0], a.backbone)
    preds_at = {f"R{r}": predict_final(st, cfg, devices[0], a.backbone, suffix=f"__R{r}")
                for r, st in sorted(snaps.items())}
    atomic_to_csv(pd.DataFrame(rows), f"{OUT}/curve__{tag}.csv")
    if gmm_audit:
        atomic_to_csv(pd.DataFrame(gmm_audit), f"{OUT}/gmm_audit__{tag}.csv")
    atomic_json({
        "cell_id": cell, "tag": tag, "study": "mt_v1", "n_tasks": N_TASKS, "tasks": list(TASKS),
        "backbone": a.backbone, "mode": a.mode, "noise": a.noise, "training_seed": a.seed,
        "method": a.method, "arm": a.arm if a.method == "T1" else None,
        "t_init": a.t_init if a.method == "T1" else None,
        "teacher": a.teacher if a.method == "T1" else None,
        "beta": a.beta if a.method == "T1" else None,
        "w_obs_start": a.w_obs_start if a.method == "T1" else None,
        "w_obs_final": a.w_obs_final if a.method == "T1" else None,
        "lambda_miss": a.lam_miss, "task_weight": a.task_weight,
        "teacher_auroc_floor": a.teacher_floor,
        "teacher_tasks_on": (None if t_use is None else
                             [t for t, k in zip(TASKS, t_use) if k > 0]),
        "teacher_train_auroc": gate_info or None,
        "observed_cells_train": int(M.sum()), "total_cells_train": int(M.size),
        "rounds_or_epochs": cfg.rounds * cfg.local_epochs, "local_epochs": cfg.local_epochs,
        "num_clients": cfg.num_clients if a.mode == "fl" else 1,
        "optimizer_steps": int(optimizer_steps), "lr": cfg.learning_rate,
        "lora_rank": cfg.lora_rank, "lora_dropout": cfg.lora_dropout,
        "n_trainable": int(sum(v.numel() for v in final.values())),
        "init_state_hash": init_hash,
        "primary_checkpoint": f"{tag}__final.pt",
        "primary_rule": "last checkpoint of the pre-registered budget; no validation selection",
        "validation_used": False, "test_used_during_training": False,
        "devices": devices, "wall_s": round(time.time() - t0, 1), "config_hash": cfg.hash(),
        "rounds": a.rounds, "num_clients": a.num_clients, "dirichlet_alpha": a.alpha,
        "save_ckpt": a.save_ckpt, "metrics": preds, "metrics_at": preds_at, "t_mode": a.t_mode,
        "only_client": a.only_client, "size_alpha": a.size_alpha,
        "w_shape": a.w_shape, "w_ramp": a.w_ramp, "no_snap_ckpt": a.no_snap_ckpt, "save_train_p": a.save_train_p,
        "n_train_only_client": None if a.only_client is None else int((train.client_id == a.only_client).sum()),
    }, f"{OUT}/run_meta__{tag}.json")
    print(f"DONE {tag} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
