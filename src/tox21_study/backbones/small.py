"""ECFP-MLP (new_baseline_glue) and Chemprop D-MPNN (unchanged_reference import) backbones.

ECFP-MLP: the spec architecture (ECFP4 r=2/2048 -> 1024 -> 512 -> 1 logit, dropout 0.2) matches no
upstream library, so this is a small RDKit + PyTorch baseline and is NOT called an official reproduction
(instruction §4.6).
D-MPNN: chemprop v1.7.1 `MoleculeModel` with depth=3, hidden=300 (= chemprop defaults), imported unmodified.
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

import os
import sys

import numpy as np
import torch
import torch.nn as nn

ROOT = f"{_ROOT}"


class EcfpMlp(nn.Module):
    def __init__(self, cfg, device="cuda:0"):
        super().__init__()
        from rdkit import Chem, RDLogger
        from rdkit.Chem import AllChem
        RDLogger.DisableLog("rdApp.*")
        self.cfg, self.device_str, self.kind = cfg, device, "ecfp_mlp"
        self._Chem, self._AllChem = Chem, AllChem
        h1, h2 = cfg.mlp_hidden
        self.net = nn.Sequential(
            nn.Linear(cfg.ecfp_bits, h1), nn.ReLU(), nn.Dropout(cfg.mlp_dropout),
            nn.Linear(h1, h2), nn.ReLU(), nn.Dropout(cfg.mlp_dropout),
            nn.Linear(h2, cfg.n_tasks),
        ).to(device)
        self.model = self.net
        self.ctx = None
        self._cache: dict[str, np.ndarray] = {}

    def fingerprint(self, smi: str) -> np.ndarray:
        v = self._cache.get(smi)
        if v is None:
            mol = self._Chem.MolFromSmiles(smi)
            fp = self._AllChem.GetMorganFingerprintAsBitVect(mol, self.cfg.ecfp_radius,
                                                             nBits=self.cfg.ecfp_bits)
            v = np.zeros(self.cfg.ecfp_bits, dtype=np.float32)
            from rdkit import DataStructs
            DataStructs.ConvertToNumpyArray(fp, v)
            self._cache[smi] = v
        return v

    def encode(self, smiles_list, augment: bool = False):
        x = np.stack([self.fingerprint(s) for s in smiles_list])
        return {"x": torch.from_numpy(x).to(self.device_str)}

    def forward(self, batch):
        """(B,) for a single endpoint, (B, n_tasks) for the multitask head."""
        lg = self.net(batch["x"])
        return lg.squeeze(-1) if lg.shape[-1] == 1 else lg

    def trainable_state(self):
        return {n: p.detach().cpu().clone() for n, p in self.net.named_parameters()}

    def load_trainable(self, state):
        self.net.load_state_dict({k: v.to(self.device_str) for k, v in state.items()}, strict=False)

    def n_trainable(self):
        return sum(p.numel() for p in self.net.parameters() if p.requires_grad)

    def token_length_report(self, smiles_list):
        return {"note": "fingerprint input, no tokenizer"}


class DMpnn(nn.Module):
    def __init__(self, cfg, device="cuda:0"):
        super().__init__()
        sys.path.insert(0, f"{ROOT}/third_party/chemprop")
        import tempfile
        from chemprop.args import TrainArgs
        from chemprop.models import MoleculeModel
        from chemprop.data import MoleculeDatapoint, MoleculeDataset, set_cache_graph, set_cache_mol
        self.cfg, self.device_str, self.kind = cfg, device, "d_mpnn"
        d = tempfile.mkdtemp()
        p = os.path.join(d, "d.csv")
        open(p, "w").write("smiles,t\nCCO,1\n")
        args = TrainArgs().parse_args(["--data_path", p, "--dataset_type", "classification",
                                       "--save_dir", d, "--quiet"])
        # chemprop sizes its readout from len(task_names); the multitask study passes 12.
        if cfg.n_tasks == 1:
            args.task_names = ["NR-ER"]
        else:
            from ..cohort_mt import TASKS
            assert cfg.n_tasks == len(TASKS), (cfg.n_tasks, len(TASKS))
            args.task_names = list(TASKS)
        args.hidden_size = cfg.d_mpnn_hidden
        args.depth = cfg.d_mpnn_depth
        args.device = torch.device(device)
        self.args = args
        self.model = MoleculeModel(args)
        self.model.classification = False        # keep logits in eval; sigmoid applied by the caller
        self.model.to(device)
        set_cache_graph(True); set_cache_mol(True)
        self._MoleculeDatapoint, self._MoleculeDataset = MoleculeDatapoint, MoleculeDataset
        self._dp: dict[str, object] = {}
        self.ctx = None

    def encode(self, smiles_list, augment: bool = False):
        dps = []
        for s in smiles_list:
            if s not in self._dp:
                self._dp[s] = self._MoleculeDatapoint(smiles=[s])
            dps.append(self._dp[s])
        return {"batch": self._MoleculeDataset(dps).batch_graph()}

    def forward(self, batch):
        """(B,) for a single endpoint, (B, n_tasks) for the multitask readout."""
        lg = self.model(batch["batch"])
        return lg.squeeze(-1) if lg.shape[-1] == 1 else lg

    def trainable_state(self):
        return {n: p.detach().cpu().clone() for n, p in self.model.named_parameters()}

    def load_trainable(self, state):
        self.model.load_state_dict({k: v.to(self.device_str) for k, v in state.items()}, strict=False)

    def n_trainable(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def token_length_report(self, smiles_list):
        return {"note": "graph input, no tokenizer"}


def build_backbone(cfg, device: str):
    if cfg.backbone == "ecfp_mlp":
        return EcfpMlp(cfg, device)
    if cfg.backbone == "d_mpnn":
        return DMpnn(cfg, device)
    from .lm import LMBackbone
    return LMBackbone(cfg.backbone, cfg, device)
