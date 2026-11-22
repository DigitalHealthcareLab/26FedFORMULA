"""12-엔드포인트 멀티태스크 데이터 로더와 마스킹 손실.

단일 태스크 로더(trainer.load_data)와 형태가 다른 점이 하나 있다: 라벨이 벡터가 아니라
(N, 12) 행렬이고, 그 옆에 같은 모양의 **관측 마스크**가 따라다닌다. Tox21 은 모든 분자를
모든 assay 에 돌리지 않아서 전체 칸의 17.1% 가 비어 있다. 비어 있음은 음성이 아니므로
손실에서도 지표에서도 빠져야 한다 — 그것을 0 으로 채우면 12 태스크 전부에 가짜 음성을
17% 주입하는 셈이 된다.

y_ref / is_flip 은 평가자 전용이다. load_mt(..., with_ref=True) 는 평가 스크립트만 쓴다.
"""
from __future__ import annotations
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .cohort_mt import DERIVED, TASKS

N_TASKS = len(TASKS)


def _mat(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """DataFrame -> (Y, M). NaN 은 미관측이고, Y 의 해당 칸은 0 으로 채우되 M=0 이다."""
    A = df[TASKS].to_numpy(dtype=float)
    M = (~np.isnan(A)).astype(np.float32)
    return np.nan_to_num(A, nan=0.0).astype(np.float32), M


def load_mt(cfg, with_ref: bool = False) -> dict:
    """train (노이즈 주입된 관측 라벨) + validation/test (원 라벨)."""
    coh = pd.read_csv(f"{DERIVED}/cohort.csv")
    sp = pd.read_csv(f"{DERIVED}/split_seed{cfg.split_seed}.csv")
    part = f"K{cfg.num_clients}_a{cfg.dirichlet_alpha}"          # partition tag (fedformula)
    if os.environ.get("JYL_SIZE_ALPHA"):     # 10-22 size skew: set by the trainers' --size_alpha (env, so cfg.hash() is unchanged)
        part += f"_q{float(os.environ['JYL_SIZE_ALPHA'])}"
    trn = pd.read_csv(f"{DERIVED}/train_{cfg.noise}_{part}_seed{cfg.repeat_seed}.csv")
    Y, M = _mat(trn)
    train = pd.DataFrame({"molecule_id": trn.molecule_id.values,
                          "smiles": trn.canonical_smiles.values,
                          "client_id": trn.client_id.values})
    out = {"train": train, "train_Y": Y, "train_M": M}
    if with_ref:
        # 평가자 전용. 어떤 학습 워커도 이 키를 읽어서는 안 된다.
        aud = pd.read_csv(f"{DERIVED}/noise_audit_{cfg.noise}_{part}_seed{cfg.repeat_seed}.csv")
        assert list(aud.molecule_id) == list(trn.molecule_id)
        out["train_Yref"] = np.nan_to_num(
            aud[[f"{t}__pre" for t in TASKS]].to_numpy(dtype=float), nan=0.0).astype(np.float32)
        out["train_flip"] = aud[[f"{t}__flip" for t in TASKS]].to_numpy(dtype=int)
    for s in ("validation", "test"):
        sub = coh[sp.split.values == s].reset_index(drop=True)
        sY, sM = _mat(sub)
        out[s] = pd.DataFrame({"molecule_id": sub.molecule_id.values,
                               "smiles": sub.canonical_smiles.values})
        out[f"{s}_Y"], out[f"{s}_M"] = sY, sM
    return out


def task_weights(M: np.ndarray, how: str) -> np.ndarray:
    """태스크별 손실 가중 (길이 12, 평균 1 로 정규화).

    uniform  : 전부 1. 관측 칸 수로만 나누므로 관측이 많은 태스크가 그만큼 더 기여한다.
    inverse_freq : 관측 칸 수의 역수. 모든 태스크가 같은 총 가중을 갖는다.
    """
    if how == "uniform":
        return np.ones(M.shape[1], dtype=np.float32)
    n = M.sum(0)
    w = np.where(n > 0, 1.0 / np.maximum(n, 1.0), 0.0)
    return (w / w[w > 0].mean()).astype(np.float32)


def masked_loss(lg, y, m, t=None, w_obs=1.0, lam_miss=0.0, tw=None, t_use=None):
    """관측 칸 기준 멀티태스크 손실. **합계**를 돌려준다 (평균이 아니다).

        관측 칸  : w_obs * BCE(z, y_obs) + (1 - w_obs) * BCE(z, t)
        미관측 칸: lam_miss * BCE(z, t)

    미관측 칸은 assay 가 돌지 않았다는 뜻이므로 **라벨 오염이 있을 수 없다**. 교사 타깃만이
    쓸 수 있는 유일한 깨끗한 추가 신호이고, lam_miss 는 그것을 얼마나 믿을지의 손잡이다.

    평균이 아니라 합계를 돌려주는 이유: 하나의 유효 배치가 여러 마이크로배치로 쪼개지는데,
    마이크로배치마다 관측 칸 수가 다르다. 각자 자기 분모로 나눈 뒤 더하면 관측이 적은
    마이크로배치가 과대 가중된다. 호출부가 배치 전체의 관측 칸 수로 한 번만 나눈다.

    t_use: 길이 12 의 0/1. 교사 품질이 문턱 아래인 태스크를 0 으로 꺼서, 그 태스크는
    관측 CE 만 쓰게 한다 (해당 태스크의 미관측 칸도 손실에서 빠진다).

    반환 (보고용 관측 CE 합, 실제 최적화하는 손실 합, 관측 칸 수).
    """
    ce = F.binary_cross_entropy_with_logits(lg, y, reduction="none")
    if tw is not None:
        ce = ce * tw
    n_obs = m.sum()
    rep = (ce * m).sum()                    # 방법 간 비교용: 언제나 관측 CE 만
    if t is None:
        return rep, rep, n_obs
    u = 1.0 if t_use is None else t_use
    ce_t = F.binary_cross_entropy_with_logits(lg, t, reduction="none")
    if tw is not None:
        ce_t = ce_t * tw
    # 교사가 켜진 태스크는 혼합, 꺼진 태스크는 관측 CE 그대로
    obs = (u * (w_obs * ce + (1.0 - w_obs) * ce_t) + (1.0 - u) * ce) * m
    mis = lam_miss * ce_t * u * (1.0 - m)
    return rep, obs.sum() + mis.sum(), n_obs
