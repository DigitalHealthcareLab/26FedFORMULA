"""Explicit argument mapping from the design YAML to runtime values (instruction §0: do not pass the
YAML `config`/`parameter_registry` straight to a runner). Every field below is set from a named key."""
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

import hashlib
import json
from dataclasses import dataclass, asdict, field

ROOT = f"{_ROOT}"


@dataclass
class RunConfig:
    # identity
    backbone: str                      # ecfp_mlp | d_mpnn | chemfm_1b | chemfm_3b
                                       # | qwen_3b (Instruct) | qwen25_1_5b/3b/7b (Base)
    method: str                        # M0 | M2 | M2F  (routing methods need a LoRA backbone)
    environment: str                   # centralized | C1
    noise: str                         # N0 | S10 | S20 | PN10 | PN20
    repeat_seed: int                   # 11 | 12 | 13

    # data / split  (data.* in YAML)
    endpoint: str = "NR-ER"
    split_seed: int = 42

    # multitask (mt_v1). n_tasks=1 은 기존 단일 엔드포인트 동작 그대로다.
    n_tasks: int = 1
    # 관측되지 않은 (분자, 태스크) 칸에 교사 타깃을 얼마나 실을지. 0 이면 완전히 무시한다.
    # 이 칸들은 실험된 적이 없으므로 정의상 라벨 오염이 없다 — 교사가 쓸 수 있는 유일한
    # 깨끗한 추가 신호이고, 그래서 스윕 대상이다.
    lambda_miss: float = 0.0
    # 태스크별 손실 가중. uniform = 관측 칸 수로만 정규화(기본).
    # inverse_freq = 태스크별 관측 칸 수의 역수로 가중해 희소 태스크를 끌어올린다.
    task_weight: str = "uniform"
    # 교사 AUROC 이 이 값 미만인 태스크에는 교사를 쓰지 않는다 (해당 태스크는 순수 CE).
    # 단일 태스크에서 교사 AUROC 0.526 은 해로웠고 0.672 는 이로웠다 — 문턱이 존재한다.
    teacher_auroc_floor: float = 0.0

    # federated  (fl.*)
    num_clients: int = 4
    dirichlet_alpha: float = 0.5
    participation_fraction: float = 1.0
    rounds: int = 20
    local_epochs: int = 1
    reset_local_optimizer_each_round: bool = True
    centralized_epochs: int = 20

    # optimisation  (training.*, runtime.*)
    optimizer: str = "adamw"
    weight_decay: float = 0.01
    scheduler: str = "cosine_with_min_lr"
    min_lr_ratio: float = 0.1
    scheduler_warmup_ratio: float = 0.05
    effective_batch_size: int = 32
    micro_batch_size: int = 8            # training only; grad accumulation keeps the effective batch
    gradient_accumulation_steps: int = 4
    eval_batch_size: int = 32            # gradient-free passes: prediction, CleaR score pass, MC eval
    sort_eval_by_length: bool = True     # batch similar-length molecules together, see batching.py
    data_parallel_world_size: int = 1
    precision: str = "bf16"
    max_sequence_length: int = 512
    smiles_augmentation_probability: float = 1.0

    # models
    ecfp_radius: int = 2
    ecfp_bits: int = 2048
    mlp_hidden: tuple = (1024, 512)
    mlp_dropout: float = 0.2
    d_mpnn_depth: int = 3
    d_mpnn_hidden: int = 300
    lora_rank: int = 32
    lora_alpha_ratio: float = 1.0
    lora_dropout: float = 0.4
    lora_target_modules: tuple = ("q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj")

    # CleaR  (clear.*, gmm.*)
    routing_gamma: float = 0.85
    warmup_periods: int = 3
    consistency_lambda: float = 1.0
    consistency_mc_forwards: int = 5
    consistency_temperature: float = 1.0
    diagnostic_clean_threshold: float = 0.5
    gmm_n_components: int = 2
    gmm_random_state: int = 0
    gmm_covariance_type: str = "full"
    gmm_tol: float = 1e-3
    gmm_reg_covar: float = 1e-6
    gmm_max_iter: int = 100
    gmm_n_init: int = 1
    gmm_init_params: str = "kmeans"

    # evaluation
    clear_mc_passes: int = 5

    def __post_init__(self):
        assert self.environment in ("centralized", "C1")
        assert self.n_tasks >= 1
        assert self.task_weight in ("uniform", "inverse_freq")
        assert self.lambda_miss >= 0.0
        # PN10/PN20 budget the flips on the ORIGINAL POSITIVE count and use equal counts in
        # both directions, so the observed prevalence is unchanged (12.77% either way). The
        # S-series instead flips a fixed rate per stratum, which under a 13% prevalence pushes
        # far more negatives into the positive class and moves prevalence to 27.7%.
        assert self.noise in ("N0", "S10", "S20", "PN10", "PN20", "PN30")
        # T1 (v13, Main Table row 8): teacher-initialised EMA target. It changes the objective
        # only, not the architecture, but it is registered here so its config hash differs from M0.
        assert self.method in ("M0", "M1", "M2", "M2F", "T1")
        if self.method == "M1":
            raise ValueError("M1 is excluded by protocol v3 (2026-09-15); use M0 or M2")
        if self.method != "M0":
            # M2F: the per-sample clean probability is estimated once, at the first period after
            # warm-up, and reused unchanged for the rest of training. Same routing, same gamma,
            # same consistency term; only the re-estimation is dropped.
            # v3: class-wise CleaR routing applies to both LoRA backbones (was ChemFM-only in v2)
            assert self.backbone in ("chemfm_1b", "chemfm_3b", "qwen_3b", "qwen25_0_5b",
                                     "qwen25_1_5b", "qwen25_3b", "qwen25_7b"), \
                "routing methods require a LoRA backbone; MLP/GNN would need an architecture change"
        assert (self.micro_batch_size * self.gradient_accumulation_steps
                * self.data_parallel_world_size == self.effective_batch_size), "effective batch mismatch"
        assert self.rounds * self.local_epochs == self.centralized_epochs, "budget mismatch"

    @property
    def learning_rate(self) -> float:
        """training.peft_learning_rate / dnn_learning_rate / gnn_learning_rate."""
        # every LoRA backbone shares the peft learning rate; the v8.4 Qwen2.5 Base line is
        # added here explicitly so a missing key fails at construction, not inside a worker
        lr = {"ecfp_mlp": 1e-3, "d_mpnn": 1e-3, "chemfm_1b": 8e-5, "chemfm_3b": 8e-5,
              "qwen_3b": 8e-5, "qwen25_0_5b": 8e-5, "qwen25_1_5b": 8e-5,
              "qwen25_3b": 8e-5, "qwen25_7b": 8e-5}
        if self.backbone not in lr:
            raise KeyError(f"no learning rate registered for backbone {self.backbone!r}; "
                           f"known: {sorted(lr)}")
        return lr[self.backbone]

    @property
    def periods(self) -> int:
        return self.rounds if self.environment == "C1" else self.centralized_epochs

    @property
    def run_id(self) -> str:
        mt = f"__T{self.n_tasks}" if self.n_tasks > 1 else ""
        return (f"{self.backbone}__{self.method}__{self.environment}__{self.noise}"
                f"__s{self.repeat_seed}{mt}")

    def hash(self) -> str:
        d = asdict(self)
        d = {k: (list(v) if isinstance(v, tuple) else v) for k, v in d.items()}
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        d = asdict(self)
        d = {k: (list(v) if isinstance(v, tuple) else v) for k, v in d.items()}
        d.update({"run_id": self.run_id, "config_hash": self.hash(),
                  "learning_rate": self.learning_rate, "periods": self.periods})
        return d


def seed_streams(repeat_seed: int) -> dict:
    """Separate RNG streams (instruction §6). noise=[seed,2001] is fixed by the parameter spec."""
    return {"partition": [repeat_seed, 1001], "noise": [repeat_seed, 2001],
            "init": [repeat_seed, 3001], "loader": [repeat_seed, 4001], "routing": [repeat_seed, 5001]}
