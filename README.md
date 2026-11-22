# FedFORMULA — modelling code (foundation-model initialisation → federated training)

This is the **minimal code share**: the code that produced the reported FedFORMULA model,
from ChemFM foundation-model initialisation through the complete federated training loop.
**Inference, evaluation, data construction, baselines, analysis and figure code are not in
this tree** — see §6 for the exact boundary and what to add if you need to cross it.

Every file here is byte-identical to the corresponding file in the full study repository;
nothing was rewritten for this share. `SCOPE_JYL.md` carries the sha256 of all 20 files.

---

## 1. The model in four lines

FedFORMULA trains a **ChemFM-1B** student across **K = 4** clients while a **frozen
ChemDFM-v1.5-8B** prior `q` supplies a per-cell soft label. For an observed cell the loss
target is a convex mixture of the observed label and a running teacher target; for an
unobserved cell (the assay was never run, so it cannot carry label noise) only the teacher
target enters, scaled by `lambda_miss`.

```
observed cell   :  w_r * BCE(z, y_obs) + (1 - w_r) * BCE(z, t_r)
unobserved cell :  lambda_miss * BCE(z, t_r)

w_r  = 1 -> 0, linear over R rounds        (observed-label weight)
t_0  = q                                   (zero-shot ChemDFM prior)
t_r  = beta * t_(r-1) + (1 - beta) * P_global_r ,  beta = 0.95
```

Expanding the EMA gives the effective target used in the paper,
`t_r = beta^r * q + (1 - beta^r) * (history of global predictions)`.

**`beta` and the `w` schedule are algorithm constants, not per-dataset hyperparameters.**
They are never tuned against validation or test data.

Where this lives in the code:

| | |
|---|---|
| observed-label weight `w_r` | `train/run_main_mt.py:654-656` (linear) and `w_schedule()` at `:428` (shape/ramp ablations) |
| teacher target `t_r` (EMA) | `train/run_main_mt.py:671` — `Tm = beta * Tm + (1 - beta) * PG` |
| `t_r` (`--t_mode latest`, no history) | `train/run_main_mt.py:669` |
| `t_0 = q` from the prior file | `train/run_main_mt.py:629-642` |
| the mixture itself | `src/tox21_study/data_mt.py:73-105` (`masked_loss`) |
| per-round global aggregation | `train/run_main_mt.py:700-704` (sample-count weighted FedAvg over LoRA params) |

---

## 2. Foundation-model initialisation

`build_backbone(cfg, device)` (`src/tox21_study/backbones/small.py:142`) dispatches to
`LMBackbone` (`src/tox21_study/backbones/lm.py:42`). For `chemfm_1b` the initialisation is:

1. **Pinned weights.** The HF repo, revision and local snapshot path come from
   `configs/tox21_study/model_revisions.json` (`lm.py:38`). ChemFM-1B is pinned to
   `f99dc2e89726539bb9cf31b2e2b4360650bac6a8`. `HF_HUB_OFFLINE=1` is set by default
   (`lm.py:33`) so a run can never silently pick up a different revision.
2. **ChemFM's own tokenizer and prompt template**, loaded from the official ChemFM
   repository, not reimplemented (`lm.py:64-69`): `tokenizer/` and `string_template.json`
   give `MOLECULE_START_STRING{smiles}END_STRING`.
3. **A 12-logit sequence-classification head** — `num_labels = cfg.n_tasks`, attention
   dropout forced to 0, bf16 (`lm.py:75-81`).
4. **ChemFM's official `smart_tokenizer_and_embedding_resize`** adds `[PAD]` and resizes the
   embedding (`lm.py:84-87`). This is imported from ChemFM's `utils.py`, so the padding
   behaviour matches the upstream fine-tuning recipe exactly.
5. **LoRA** via `peft`: `r = 32`, `alpha = r`, `dropout = 0.4`, adapter name `nrer`, applied to
   `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj` (`lm.py:89-93`, constants in
   `src/tox21_study/config.py`). 25,255,936 trainable parameters; the base model stays frozen.

Only the LoRA tensors are exchanged between clients and server, so the federated state is
~25 M parameters rather than the full 1 B.

---

## 3. Layout

```
2026_FedFORMULA/
├── train/run_main_mt.py              the single training entry point (federated + centralised)
├── src/tox21_study/
│   ├── config.py                     RunConfig: every registered constant + config hash
│   ├── backbones/lm.py               ChemFM / Qwen LoRA backbone  <- FM initialisation
│   ├── backbones/small.py            build_backbone dispatcher (+ ECFP-MLP / D-MPNN builders)
│   ├── data_mt.py                    load_mt, masked_loss (the effective-target mixture)
│   ├── cohort_mt.py, cohort.py       12-endpoint cohort + scaffold helpers, TASKS order
│   ├── fastbatch.py, batching.py     token-budgeted micro-batching, length bucketing
│   ├── routing.py                    LoRA layer routing hooks (CleaR arms)
│   ├── clear_routing.py              GMM clean-probability scoring (CleaR arms)
│   ├── trainer.py                    cosine LR schedule + the centralised trainer
│   ├── metrics.py                    metric functions imported by trainer.py
│   └── runutil.py                    env/VRAM guards, atomic json/csv/torch writes
├── configs/tox21_study/model_revisions.json    pinned HF revisions
├── requirements-train.txt            exact versions of the reported runs
├── pyproject.toml, LICENSE, CITATION.cff
└── SCOPE_JYL.md                      provenance, sha256 manifest, verification log
```

These 13 modules (plus two empty `__init__.py`) are the **complete transitive import closure**
of `train/run_main_mt.py` — nothing more, nothing less. `SCOPE_JYL.md` §3 records the check.

---

## 4. Install and configure

Python 3.10.12, CUDA 12.1.

```bash
python3.10 -m venv .venv && . .venv/bin/activate
pip install -r requirements-train.txt

export FEDFORMULA_ROOT=/path/to/working/tree     # where data/, teacher/, third_party/ live
export JYL_OUT=$FEDFORMULA_ROOT/experiments/<name>/outputs    # results dir (optional)
```

Every path in the tree is derived from `FEDFORMULA_ROOT`; there are no absolute paths in the
code. `FEDFORMULA_ROOT` may be this directory, in which case `configs/` is already in place
and you only need to add `data/`, `teacher/` and `third_party/`.

### Inputs you must supply

| Path (under `$FEDFORMULA_ROOT`) | What | Read by |
|---|---|---|
| `third_party/ChemFM/finetuning/property_prediction/` | official ChemFM repo — `tokenizer/`, `string_template.json`, `utils.py` | `lm.py:31,64,84` |
| HF snapshot at the pinned revision | ChemFM-1B weights (3.61 GB) | `lm.py:80` |
| `data/derived/tox21_mt/cohort.csv` | 12-endpoint cohort | `data_mt.py:33` |
| `data/derived/tox21_mt/split_seed42.csv` | train / validation / test assignment | `data_mt.py:34` |
| `data/derived/tox21_mt/train_{noise}_{part}_seed{seed}.csv` | observed (noise-injected) training labels + `client_id` | `data_mt.py:38` |
| `data/derived/tox21_mt/noise_audit_{noise}_{part}_seed{seed}.csv` | flip indicators — **read only by a separate evaluator, never by training** | `data_mt.py:46` |
| `teacher/teacher_q_train.csv` | the frozen prior `q`, molecule_id × 12 endpoints | `run_main_mt.py:632` |
| `teacher/teacher_quality_train.csv` | per-endpoint teacher AUROC, for `--teacher_floor` | `run_main_mt.py:355` |

The builders that *produce* the `data/derived/` files and the scorer that produces
`teacher_q_train.csv` are **data preparation and prior inference**, both outside this share
(§6).

---

## 5. The reported run

This is the exact invocation behind the PN30 / seed 11 FedFORMULA main-reference run
(`run_meta` tag `MT30SW__chemfm_1b__fl__PN30__K4_a1_R50__w1-0_b0.95_teacher__s11__lm0__twuni__fl0`):

```bash
python train/run_main_mt.py \
  --backbone chemfm_1b --mode fl --method T1 --arm SW \
  --noise PN30 --seed 11 \
  --rounds 50 --num_clients 4 --alpha 1.0 \
  --w_obs_start 1.0 --w_obs_final 0.0 --beta 0.95 \
  --t_init teacher --t_mode ema --teacher chemdfm_8b \
  --lam_miss 0 --task_weight uniform --teacher_floor 0 \
  --devices cuda:0 --save_ckpt final
```

Swap `--noise` for `N0_v13 | PN10 | PN20` and `--seed` for `11 | 12 | 13`. The reported number
is the seed ensemble: probabilities are averaged over the three seeds **first**, then the
per-endpoint metric, then the 12-endpoint macro.

Resulting configuration — reproduced by this tree, `RunConfig.hash() == d1e94812d30e9f14`,
identical to the hash recorded in the original run's `run_meta` JSON:

| | |
|---|---|
| learning rate | 8e-5, AdamW, cosine with 5 % warmup and 0.1 min-LR ratio |
| effective batch | 32 (micro 8 × accumulation 4), token-budgeted micro-batching |
| precision | bf16, max sequence length 512 |
| rounds × local epochs | 50 × 1 = 2,450 optimiser steps |
| clients | K = 4, Dirichlet α = 1.0, full participation, local optimiser reset each round |
| trainable | 25,255,936 LoRA parameters |

### Protocol invariants enforced in the code

- **The final-round global model is the primary result.** There is no validation-based model
  selection and no early stopping (`run_meta` records `validation_used: false`,
  `primary_rule: "last checkpoint of the pre-registered budget; no validation selection"`).
- **Test data is never read during training** (`test_used_during_training: false`).
- **Noise flip indicators are never read by the training path.** `load_mt(cfg)` loads the audit
  file only when called with `with_ref=True`, which the training entry point never does.
- Each run writes a `run_meta__<tag>.json` with the full configuration, the config hash, the
  initial-state hash and the per-round curve, so a run can be audited without the checkpoints.

---

## 6. The boundary: what is not here

**Deliberately excluded as inference** — `evaluate_mt.py`, `score_teacher_mt.py` and the
explainability runners. Consequence, stated plainly:

> `train/run_main_mt.py` ends by calling `predict_final()` (defined at `:397`, called at
> `:737-738`), which scores the trained model and does `import evaluate_mt as E` at `:401`.
> That is the **only** inference touchpoint in this tree, and it is unconditional. Training
> itself — foundation-model initialisation, all 50 federated rounds, the target schedule, the
> aggregation and the final checkpoint write at `:736` — runs without it, but the script will
> raise `ModuleNotFoundError: evaluate_mt` at the very end unless you either add
> `pipeline/evaluate_mt.py` from the full repository to `PYTHONPATH`, or stop after the
> checkpoint. The code was not modified to hide this.

Also excluded:

- **Data construction** — the raw-to-cohort build, the 80/10/10 split, the scaffold-LDA client
  partition and the symmetric noise injection. Their library modules (`noise.py`,
  `partition.py`) are therefore absent too; they are not imported by the training path.
- **Baselines** — masked BCE, SPLC, ELR, ProSelfLC, FLR, CleaR, FedGR, FedClean, D-MPNN, and
  the other label-noise methods, with their ports and adaptation record.
- **Analysis, figures and tables**, the run registry and the per-seed prediction arrays.
- **Model weights and checkpoints** — fetch ChemFM-1B from the Hugging Face Hub at the pinned
  revision; per-round checkpoints are ~205 GB and are not distributable.
- **The ChemFM repository itself**, which carries its own licence. Clone it to
  `$FEDFORMULA_ROOT/third_party/ChemFM`.

`src/tox21_study/backbones/small.py` does ship the ECFP-MLP and D-MPNN builders and
`clear_routing.py` ships the CleaR routing machinery, because `build_backbone` and `trainer.py`
import them unconditionally. They are dependencies of the entry point, not part of the
FedFORMULA model path; the reported model never takes those branches.

---

## 7. Licence and citation

MIT (`LICENSE`), citation metadata in `CITATION.cff`. ChemFM and ChemDFM are third-party
models under their own licences and are not redistributed here.
