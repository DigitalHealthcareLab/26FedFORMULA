"""ChemFM-3B / Qwen2.5-3B-Instruct LoRA backbones with a 1-logit binary head.

ChemFM path reuses the official artefacts from third_party/ChemFM/finetuning/property_prediction
(tokenizer directory, string_template.json, DataCollator with SMILES augmentation,
smart_tokenizer_and_embedding_resize) and the LoraConfig from its main.py.
Qwen path uses the official AutoTokenizer with a fixed SMILES prompt recorded as metadata;
no answer, explanation or noise information is placed in the input.
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

import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

ROOT = f"{_ROOT}"
CHEMFM_PP = f"{ROOT}/third_party/ChemFM/finetuning/property_prediction"
os.environ.setdefault("HF_HOME", f"{ROOT}/cache/hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

QWEN_PROMPT = "SMILES: {smiles}"   # fixed format, recorded in run metadata


def _revisions() -> dict:
    return json.load(open(f"{ROOT}/configs/tox21_study/model_revisions.json"))


class LMBackbone(nn.Module):
    """Wraps a causal-LM-derived sequence classifier (num_labels=cfg.n_tasks) with peft LoRA."""

    def __init__(self, kind: str, cfg, device="cuda:0"):
        super().__init__()
        # v8.4 P0.1 adds the Qwen2.5 pretrained Base line (non-Instruct). qwen_3b stays bound to
        # the Instruct checkpoint the earlier study used, so the two variants never silently merge.
        assert kind in ("chemfm_1b", "chemfm_3b", "qwen_3b", "qwen25_0_5b",
                        "qwen25_1_5b", "qwen25_3b", "qwen25_7b")
        from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
        from peft import LoraConfig, get_peft_model

        self.kind, self.cfg, self.device_str = kind, cfg, device
        revs = _revisions()
        repo = {"chemfm_1b": "ChemFM/ChemFM-1B", "chemfm_3b": "ChemFM/ChemFM-3B",
                "qwen_3b": "Qwen/Qwen2.5-3B-Instruct",
                "qwen25_0_5b": "Qwen/Qwen2.5-0.5B",
                "qwen25_1_5b": "Qwen/Qwen2.5-1.5B", "qwen25_3b": "Qwen/Qwen2.5-3B",
                "qwen25_7b": "Qwen/Qwen2.5-7B"}[kind]
        self.repo, self.revision, path = repo, revs[repo]["revision"], revs[repo]["path"]

        if kind.startswith("chemfm"):          # every ChemFM size shares ChemFM's tokenizer and prompt
            sys.path.insert(0, CHEMFM_PP)
            self.tokenizer = AutoTokenizer.from_pretrained(f"{CHEMFM_PP}/tokenizer",
                                                           padding_side="right", use_fast=True)
            tmpl = json.load(open(f"{CHEMFM_PP}/string_template.json"))
            self.start, self.end = tmpl["MOLECULE_START_STRING"], tmpl["END_STRING"]
            self.prompt_format = f"{self.start}{{smiles}}{self.end}"
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(path, padding_side="right", use_fast=True)
            self.start = self.end = ""
            self.prompt_format = QWEN_PROMPT

        hf_cfg = AutoConfig.from_pretrained(path)
        hf_cfg.num_labels = cfg.n_tasks       # 1 logit per endpoint (12 in the multitask study)
        if hasattr(hf_cfg, "attention_dropout"):
            hf_cfg.attention_dropout = 0.0
        dtype = torch.bfloat16 if cfg.precision == "bf16" else torch.float32
        self.model = AutoModelForSequenceClassification.from_pretrained(
            path, config=hf_cfg, torch_dtype=dtype)

        if self.tokenizer.pad_token is None or kind.startswith("chemfm"):
            from utils import smart_tokenizer_and_embedding_resize    # official ChemFM helper
            smart_tokenizer_and_embedding_resize(special_tokens_dict=dict(pad_token="[PAD]"),
                                                 tokenizer=self.tokenizer, model=self.model)
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        lcfg = LoraConfig(task_type="SEQ_CLS", r=cfg.lora_rank,
                          lora_alpha=int(cfg.lora_rank * cfg.lora_alpha_ratio),
                          lora_dropout=cfg.lora_dropout, target_modules=list(cfg.lora_target_modules),
                          modules_to_save=[], inference_mode=False, use_rslora=False)
        self.model = get_peft_model(self.model, lcfg, adapter_name="nrer")
        self.adapter = "nrer"
        self.n_layers = hf_cfg.num_hidden_layers
        self.model.to(device)

        self.ctx = None      # set by enable_routing()
        self._aug = None

    # ---------- routing ----------
    def enable_routing(self):
        from ..routing import install_routing
        self.ctx = install_routing(self.model, self.adapter, self.n_layers)
        return self.ctx

    # ---------- input ----------
    def _augmenter(self):
        if self._aug is None:
            from pysmilesutils.augment import SMILESAugmenter
            self._aug = SMILESAugmenter()
        return self._aug

    def encode_raw(self, texts):
        """Tokenise SMILES strings exactly as given, with no canonicalisation or augmentation.

        `encode(augment=False)` runs Chem.MolToSmiles on its input, so feeding it strings that
        were already randomised collapses every view back to one canonical string. Any caller
        that produced its own views must come through here instead.
        """
        enc = self.tokenizer([self.prompt_format.format(smiles=s) for s in texts],
                             max_length=self.cfg.max_sequence_length, truncation=True,
                             add_special_tokens=(self.kind != "chemfm_3b"), padding=True,
                             return_tensors="pt")
        # same contract as encode(): the model rejects token_type_ids
        return {k: v.to(self.device_str) for k, v in enc.items()
                if k in ("input_ids", "attention_mask")}

    def _encode_cached(self, smiles_list):
        """encode(augment=False) without re-running RDKit and the tokenizer on every call.

        fedformula speed-up. Each distinct input string is canonicalised + tokenised ONCE (same
        calls, same arguments as the uncached path) and its ids are kept; a batch is then right-padded
        with the tokenizer's pad id exactly as tokenizer(..., padding=True) does. The equivalence is
        checked against the uncached path before use (see scripts/verify_speedups.py)."""
        if not hasattr(self, "_tok_cache"):
            self._tok_cache = {}
        from rdkit import Chem
        ids = []
        for s in smiles_list:
            t = self._tok_cache.get(s)
            if t is None:
                c = s
                try:
                    c = Chem.MolToSmiles(Chem.MolFromSmiles(s))
                except Exception:
                    pass
                t = self.tokenizer(self.prompt_format.format(smiles=c), max_length=self.cfg.max_sequence_length,
                                   truncation=True, add_special_tokens=(self.kind != "chemfm_3b"))["input_ids"]
                self._tok_cache[s] = t
            ids.append(t)
        L = max(len(t) for t in ids)
        pad = self.tokenizer.pad_token_id
        inp = torch.full((len(ids), L), pad, dtype=torch.long)
        att = torch.zeros((len(ids), L), dtype=torch.long)
        for i, t in enumerate(ids):
            inp[i, :len(t)] = torch.tensor(t, dtype=torch.long)
            att[i, :len(t)] = 1
        return {"input_ids": inp.to(self.device_str), "attention_mask": att.to(self.device_str)}

    def encode(self, smiles_list, augment: bool):
        from rdkit import Chem
        if not augment and self.tokenizer.padding_side == "right":
            return self._encode_cached(smiles_list)
        texts = []
        for s in smiles_list:
            if augment:
                try:
                    s = self._augmenter().augment([s])[0]
                except Exception:
                    pass
            else:
                try:
                    s = Chem.MolToSmiles(Chem.MolFromSmiles(s))
                except Exception:
                    pass
            texts.append(self.prompt_format.format(smiles=s))
        enc = self.tokenizer(texts, max_length=self.cfg.max_sequence_length, truncation=True,
                             # NOTE: chemfm_1b has always been tokenised WITH special tokens and
                             # chemfm_3b without; every existing ChemFM-1B checkpoint depends on
                             # that. Do not "tidy" this into a startswith("chemfm") test — it
                             # silently flips 1B behaviour. The Qwen Base models take True.
                             add_special_tokens=(self.kind != "chemfm_3b"), padding=True,
                             return_tensors="pt")
        return {k: v.to(self.device_str) for k, v in enc.items() if k in ("input_ids", "attention_mask")}

    def token_length_report(self, smiles_list):
        from rdkit import Chem
        n = []
        for s in smiles_list:
            try:
                s = Chem.MolToSmiles(Chem.MolFromSmiles(s))
            except Exception:
                pass
            n.append(len(self.tokenizer(self.prompt_format.format(smiles=s),
                                        add_special_tokens=not self.kind.startswith("chemfm")
                                        )["input_ids"]))
        n = np.array(n)
        return {"max": int(n.max()), "mean": round(float(n.mean()), 1), "p99": int(np.percentile(n, 99)),
                "n_over_max_len": int((n > self.cfg.max_sequence_length).sum()),
                "max_sequence_length": self.cfg.max_sequence_length}

    def forward(self, batch):
        """(B,) for a single endpoint, (B, n_tasks) for the multitask head."""
        lg = self.model(**batch).logits.float()
        return lg.squeeze(-1) if lg.shape[-1] == 1 else lg

    # ---------- state ----------
    def trainable_state(self):
        return {n: p.detach().cpu().clone() for n, p in self.model.named_parameters() if p.requires_grad}

    def load_trainable(self, state):
        """Copy a trainable-parameter state into the model.

        fedformula speed-up: load_state_dict(strict=False) walked every module of the 1B model
        (~1 s per call, 8 calls per FedFORMULA round) to set 25M LoRA/head values. Copying straight into
        the named parameters is the same operation (load_state_dict also ends in param.copy_). Unknown
        keys still raise, as the old assert did."""
        if not hasattr(self, "_pmap"):
            self._pmap = dict(self.model.named_parameters())
        unexpected = [k for k in state if k not in self._pmap]
        assert not unexpected, unexpected[:3]
        with torch.no_grad():
            for k, v in state.items():
                self._pmap[k].copy_(v.to(self.device_str))

    def n_trainable(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
