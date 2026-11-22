"""NR-ER cohort construction: label extraction, minimal standardization, duplicate policy, 80/10/10 split.

Standardization scope (instruction §6): none of this study's references (FedChem, ChemFM, CleaR, chemprop, exmol)
define salt/tautomer rules, so we apply ONLY RDKit parsing + canonicalization and record the RDKit version.
ChemFM's own DataCollator canonicalises with `Chem.MolToSmiles(Chem.MolFromSmiles(s))`, i.e. the same operation.
We do NOT reuse the previous project's MELLODDY-TUNER pipeline (not a reference of this study).

Duplicate policy `merge_consistent_exclude_conflicting`, applied BEFORE the split.
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
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

ROOT = f"{_ROOT}"
RAW_CSV = f"{ROOT}/data/tox21/tox21.csv.gz"
RAW_SHA256 = "45d09792492ce049039dd24aa27b07fc79ce20c573187d4d90bcd178c0c0d360"
DERIVED = f"{ROOT}/data/derived/tox21_nr_er"
ENDPOINT = "NR-ER"


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_df(df: pd.DataFrame) -> str:
    return hashlib.sha256(df.to_csv(index=False).encode()).hexdigest()


@dataclass
class CohortStats:
    n_raw_rows: int
    n_endpoint_observed: int
    n_missing_endpoint: int
    n_parse_failed: int
    n_after_parse: int
    n_unique_structures: int
    n_merged_groups: int
    n_rows_in_merged_groups: int
    n_conflicting_groups: int
    n_rows_excluded_conflict: int
    n_cohort: int
    n_active: int
    n_inactive: int


def canonical_smiles(smi: str):
    """Minimal standardization: parse + canonical SMILES. Returns (canonical, error_or_None)."""
    try:
        mol = Chem.MolFromSmiles(smi)
    except Exception as e:  # pragma: no cover - RDKit raises rarely
        return None, f"parse_exception:{e}"
    if mol is None:
        return None, "rdkit_parse_failed"
    try:
        return Chem.MolToSmiles(mol), None
    except Exception as e:  # pragma: no cover
        return None, f"canonicalization_failed:{e}"


def murcko_scaffold(smi: str) -> str:
    """Bemis-Murcko scaffold SMILES; '' for acyclic molecules (no ring system)."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError(f"unparsable canonical smiles: {smi}")
    return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))


def build_cohort(write: bool = True) -> tuple[pd.DataFrame, CohortStats, dict]:
    assert sha256_file(RAW_CSV) == RAW_SHA256, "raw Tox21 file hash changed"
    raw = pd.read_csv(RAW_CSV)
    assert ENDPOINT in raw.columns, f"{ENDPOINT} column missing"
    n_raw = len(raw)

    # 1) keep rows with an observed NR-ER label (missing/inconclusive excluded)
    obs = raw[raw[ENDPOINT].notna()].copy()
    assert set(obs[ENDPOINT].unique()) <= {0.0, 1.0}, "unexpected NR-ER encoding"
    n_missing = n_raw - len(obs)

    # 2) minimal standardization
    can, err = zip(*[canonical_smiles(s) for s in obs["smiles"]])
    obs["canonical_smiles"] = can
    obs["parse_error"] = err
    failed = obs[obs.canonical_smiles.isna()].copy()
    ok = obs[obs.canonical_smiles.notna()].copy()

    # 3) duplicate policy on the canonical structure
    groups = ok.groupby("canonical_smiles", sort=False)
    keep_rows, conflict_rows, merge_rows = [], [], []
    for smi, g in groups:
        labels = set(g[ENDPOINT].astype(int))
        if len(labels) == 1:
            keep_rows.append({
                "canonical_smiles": smi,
                "label": int(labels.pop()),
                "n_source_rows": len(g),
                "source_mol_ids": ";".join(g["mol_id"]),
                "source_smiles": g["smiles"].iloc[0],
            })
            if len(g) > 1:
                merge_rows.append({"canonical_smiles": smi, "n_source_rows": len(g),
                                   "source_mol_ids": ";".join(g["mol_id"]),
                                   "label": int(g[ENDPOINT].iloc[0])})
        else:
            conflict_rows.append({"canonical_smiles": smi, "n_source_rows": len(g),
                                  "source_mol_ids": ";".join(g["mol_id"]),
                                  "labels": ";".join(str(int(v)) for v in g[ENDPOINT])})

    cohort = pd.DataFrame(keep_rows).sort_values("canonical_smiles", kind="mergesort").reset_index(drop=True)
    cohort.insert(0, "molecule_id", [f"NRER{i:05d}" for i in range(len(cohort))])
    cohort["scaffold_smiles"] = [murcko_scaffold(s) for s in cohort.canonical_smiles]
    uniq = {s: i for i, s in enumerate(sorted(cohort.scaffold_smiles.unique()))}
    cohort["scaffold_id"] = cohort.scaffold_smiles.map(uniq)

    stats = CohortStats(
        n_raw_rows=n_raw,
        n_endpoint_observed=len(obs),
        n_missing_endpoint=int(n_missing),
        n_parse_failed=len(failed),
        n_after_parse=len(ok),
        n_unique_structures=ok.canonical_smiles.nunique(),
        n_merged_groups=len(merge_rows),
        n_rows_in_merged_groups=int(sum(r["n_source_rows"] for r in merge_rows)),
        n_conflicting_groups=len(conflict_rows),
        n_rows_excluded_conflict=int(sum(r["n_source_rows"] for r in conflict_rows)),
        n_cohort=len(cohort),
        n_active=int((cohort.label == 1).sum()),
        n_inactive=int((cohort.label == 0).sum()),
    )
    meta = {
        "endpoint": ENDPOINT,
        "raw_csv": RAW_CSV,
        "raw_sha256": RAW_SHA256,
        "label_mapping": {"inactive": 0, "active": 1},
        "standardization": "rdkit_parse_then_canonical_smiles_only (no salt stripping, no tautomer canonicalisation)",
        "standardization_rationale": "no reference of this study defines salt/tautomer rules; ChemFM DataCollator uses the same MolToSmiles(MolFromSmiles(x))",
        "rdkit_version": Chem.rdBase.rdkitVersion,
        "duplicate_policy": "merge_consistent_exclude_conflicting",
        "scaffold": "rdkit.Chem.Scaffolds.MurckoScaffold.GetScaffoldForMol -> MolToSmiles ('' for acyclic)",
        "n_scaffolds": int(cohort.scaffold_id.nunique()),
        "stats": asdict(stats),
    }

    if write:
        os.makedirs(DERIVED, exist_ok=True)
        cohort.to_csv(f"{DERIVED}/cohort.csv", index=False)
        failed[["mol_id", "smiles", "parse_error"]].to_csv(f"{DERIVED}/excluded_parse_failed.csv", index=False)
        pd.DataFrame(conflict_rows, columns=["canonical_smiles", "n_source_rows", "source_mol_ids", "labels"]).to_csv(
            f"{DERIVED}/excluded_conflicting_labels.csv", index=False)
        pd.DataFrame(merge_rows, columns=["canonical_smiles", "n_source_rows", "source_mol_ids", "label"]).to_csv(
            f"{DERIVED}/merged_duplicates.csv", index=False)
        meta["cohort_sha256"] = sha256_file(f"{DERIVED}/cohort.csv")
        json.dump(meta, open(f"{DERIVED}/cohort_meta.json", "w"), indent=2)
    return cohort, stats, meta


def make_split(cohort: pd.DataFrame, seed: int = 42, fractions=(0.8, 0.1, 0.1), write: bool = True):
    """compound_stratified_random: stratify by label, shuffle within stratum, split 80/10/10.

    Duplicate structures were already merged into single rows, so no structure spans two splits.
    """
    assert abs(sum(fractions) - 1.0) < 1e-9
    rng = np.random.RandomState(seed)
    split = np.empty(len(cohort), dtype=object)
    for lab in sorted(cohort.label.unique()):
        idx = np.where(cohort.label.values == lab)[0]
        idx = idx[rng.permutation(len(idx))]
        n = len(idx)
        n_tr = int(np.floor(fractions[0] * n))
        n_va = int(np.floor(fractions[1] * n))
        split[idx[:n_tr]] = "train"
        split[idx[n_tr:n_tr + n_va]] = "validation"
        split[idx[n_tr + n_va:]] = "test"
    out = pd.DataFrame({"molecule_id": cohort.molecule_id, "split": split})
    assert out.split.notna().all()
    if write:
        out.to_csv(f"{DERIVED}/split_seed{seed}.csv", index=False)
        meta = {
            "split_method": "compound_stratified_random (stratified by observed NR-ER label)",
            "split_seed": seed,
            "fractions": list(fractions),
            "counts": {s: int((out.split == s).sum()) for s in ["train", "validation", "test"]},
            "positives": {s: int(cohort.label[(out.split == s).values].sum()) for s in ["train", "validation", "test"]},
            "sha256": sha256_file(f"{DERIVED}/split_seed{seed}.csv"),
            "note": "structure deduplication happens before splitting; this is NOT a scaffold-disjoint test set",
        }
        json.dump(meta, open(f"{DERIVED}/split_seed{seed}_meta.json", "w"), indent=2)
    return out
