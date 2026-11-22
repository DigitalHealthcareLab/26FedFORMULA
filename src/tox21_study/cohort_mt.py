"""12 엔드포인트 멀티태스크 코호트. 단일 엔드포인트 cohort.py 와 같은 정책을 태스크별로 편다.

단일 버전과 달라지는 점은 하나다. 구조 중복을 병합할 때 라벨 충돌 판정을 **태스크마다
독립으로** 한다. 같은 구조의 두 행이 NR-ER 에서는 일치하고 SR-MMP 에서는 어긋날 수 있으므로,
분자를 통째로 버리면 멀쩡한 라벨까지 잃는다. 충돌한 태스크만 결측으로 만들고 나머지는 남긴다.

산출물은 분자 하나에 12 개 라벨과 12 개 관측 마스크를 갖는 wide 표다. 결측은 원래 결측과
충돌로 인한 결측을 구분해 기록한다 — 후자는 우리가 만든 것이라 감사 대상이다.
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

import hashlib
import json
import os

import numpy as np
import pandas as pd

from .cohort import canonical_smiles, murcko_scaffold, sha256_df

ROOT = f"{_ROOT}"
RAW_CSV = f"{ROOT}/data/tox21/tox21.csv.gz"
DERIVED = f"{ROOT}/data/derived/tox21_mt"
TASKS = ["NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
         "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"]


def build_cohort(write: bool = True):
    raw = pd.read_csv(RAW_CSV)
    for t in TASKS:
        assert t in raw.columns, f"{t} 열이 없다"
        assert set(raw[t].dropna().unique()) <= {0.0, 1.0}, f"{t} 인코딩이 0/1 이 아니다"
    n_raw = len(raw)

    # 1) 정준화. 단일 버전과 같은 규칙을 쓴다.
    can, err = zip(*[canonical_smiles(s) for s in raw["smiles"]])
    raw = raw.assign(canonical_smiles=can, parse_error=err)
    failed = raw[raw.canonical_smiles.isna()]
    ok = raw[raw.canonical_smiles.notna()].copy()

    # 2) 구조 중복 병합. 충돌은 **태스크별로** 판정한다.
    rows, conflicts = [], []
    for smi, g in ok.groupby("canonical_smiles", sort=False):
        rec = {"canonical_smiles": smi, "n_source_rows": len(g),
               "source_mol_ids": ";".join(g["mol_id"]), "source_smiles": g["smiles"].iloc[0]}
        for t in TASKS:
            v = g[t].dropna().unique()
            if len(v) == 0:
                rec[t] = np.nan                      # 원래 결측
                rec[f"{t}__conflict"] = 0
            elif len(v) == 1:
                rec[t] = float(v[0])
                rec[f"{t}__conflict"] = 0
            else:
                rec[t] = np.nan                      # 충돌 -> 결측 처리
                rec[f"{t}__conflict"] = 1
                conflicts.append({"canonical_smiles": smi, "task": t,
                                  "n_source_rows": len(g),
                                  "labels": ";".join(str(int(x)) for x in g[t].dropna()),
                                  "source_mol_ids": ";".join(g["mol_id"])})
        rows.append(rec)
    coh = pd.DataFrame(rows)

    # 3) 라벨이 하나도 없는 분자는 학습에 쓸 수 없다
    obs = coh[TASKS].notna()
    keep = obs.sum(axis=1) > 0
    dropped_all_missing = int((~keep).sum())
    coh = coh[keep].reset_index(drop=True)

    coh.insert(0, "molecule_id", [f"TOX{i:05d}" for i in range(len(coh))])
    coh["scaffold_smiles"] = [murcko_scaffold(s) for s in coh.canonical_smiles]
    uniq = {s: i for i, s in enumerate(sorted(coh.scaffold_smiles.unique()))}
    coh["scaffold_id"] = coh.scaffold_smiles.map(uniq)

    obs = coh[TASKS].notna()
    stats = {
        "n_raw_rows": n_raw, "n_parse_failed": int(len(failed)),
        "n_unique_structures": int(len(rows)),
        "n_dropped_all_labels_missing": dropped_all_missing,
        "n_molecules": int(len(coh)), "n_tasks": len(TASKS),
        "n_scaffolds": int(coh.scaffold_id.nunique()),
        "n_cells": int(len(coh) * len(TASKS)),
        "n_observed_cells": int(obs.values.sum()),
        "observed_fraction": round(float(obs.values.mean()), 4),
        "n_conflict_cells": int(len(conflicts)),
        "per_task": {t: {"observed": int(obs[t].sum()),
                         "positives": int(coh[t].sum()),
                         "prevalence": round(float(coh[t].mean()), 4),
                         "conflicts": int(coh[f"{t}__conflict"].sum())} for t in TASKS},
        "cohort_sha256": sha256_df(coh[["molecule_id", "canonical_smiles"] + TASKS]),
    }
    if write:
        os.makedirs(DERIVED, exist_ok=True)
        coh.to_csv(f"{DERIVED}/cohort.csv", index=False)
        pd.DataFrame(conflicts).to_csv(f"{DERIVED}/label_conflicts.csv", index=False)
        json.dump(stats, open(f"{DERIVED}/cohort_stats.json", "w"), indent=2)
    return coh, stats


def make_split(coh: pd.DataFrame, seed: int = 42, fractions=(0.8, 0.1, 0.1), write: bool = True):
    """80/10/10. 단일 버전은 관측 라벨로 층화했지만 12 태스크에는 층화 키가 하나가 아니다.

    관측된 태스크 수와 양성을 가진 태스크 수로 만든 결합 키로 층화한다. 이렇게 하면
    '라벨이 거의 없는 분자'와 '양성이 많은 분자'가 세 분할에 고르게 퍼진다.
    """
    rng = np.random.default_rng(seed)
    n_obs = coh[TASKS].notna().sum(axis=1)
    n_pos = (coh[TASKS] == 1).sum(axis=1)
    key = pd.Series([f"{min(o,12)//3}_{min(p,3)}" for o, p in zip(n_obs, n_pos)])
    split = pd.Series(index=coh.index, dtype=object)
    for _, idx in key.groupby(key):
        i = idx.index.to_numpy()[rng.permutation(len(idx))]
        a = int(round(fractions[0] * len(i)))
        b = a + int(round(fractions[1] * len(i)))
        split.iloc[i[:a]] = "train"
        split.iloc[i[a:b]] = "validation"
        split.iloc[i[b:]] = "test"
    out = pd.DataFrame({"molecule_id": coh.molecule_id, "split": split.values})
    meta = {"seed": seed, "fractions": list(fractions),
            "split_method": "stratified by (n observed tasks bucket, n positive tasks bucket)",
            "counts": out.split.value_counts().to_dict(),
            "per_split_task_prevalence": {
                s: {t: round(float(coh.loc[out.split.values == s, t].mean()), 4) for t in TASKS}
                for s in ("train", "validation", "test")}}
    if write:
        os.makedirs(DERIVED, exist_ok=True)
        out.to_csv(f"{DERIVED}/split_seed{seed}.csv", index=False)
        json.dump(meta, open(f"{DERIVED}/split_seed{seed}_meta.json", "w"), indent=2)
    return out, meta


if __name__ == "__main__":
    c, s = build_cohort()
    print(json.dumps({k: v for k, v in s.items() if k != "per_task"}, indent=2))
    print("\n태스크별:")
    print(f"{'task':16s} {'관측':>6s} {'양성':>6s} {'유병률':>7s} {'충돌':>5s}")
    for t, v in s["per_task"].items():
        print(f"{t:16s} {v['observed']:6d} {v['positives']:6d} {v['prevalence']:7.4f} {v['conflicts']:5d}")
    sp, m = make_split(c)
    print("\n분할:", m["counts"])
