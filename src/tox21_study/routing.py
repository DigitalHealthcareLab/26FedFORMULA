"""CleaR routing on the LoRA path (study_adaptation of clear@ac37b1ce for decoder + peft).

Difference from the public CleaR code, recorded in reference_differences.md:
  C4  public code swaps the WHOLE layer output against a duplicated frozen BertLayer
        (`hidden = mask*layer_out + (1-mask)*fixed_layer_out`).
      Here the mask multiplies ONLY the LoRA delta, keeping the base path intact:
        y = base_layer(x) + z_{i,l} * scaling * B(A(dropout(x)))
      This is what the execution prompt §9.7 requires ("해당 layer의 LoRA 경로에 적용... Base 경로는 유지").
  C5  public warm-up mask collapses to one scalar per batch; here every mask is per (sample, layer).
  C6  public `routing_probs` is a fixed size-32 buffer; here p_i is carried per molecule id.

One mask per (sample, transformer layer) is shared by all target modules of that layer.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

SPLITMIX_A = np.uint64(0x9E3779B97F4A7C15)
SPLITMIX_B = np.uint64(0xBF58476D1CE4E5B9)
SPLITMIX_C = np.uint64(0x94D049BB133111EB)


def _splitmix64(x: np.ndarray) -> np.ndarray:
    """Counter-based hash -> uniform uint64. Used so an inference mask depends only on
    (molecule hash, layer, mc index, base seed) and never on batch composition or order."""
    with np.errstate(over="ignore"):
        z = (x + SPLITMIX_A).astype(np.uint64)
        z = ((z ^ (z >> np.uint64(30))) * SPLITMIX_B).astype(np.uint64)
        z = ((z ^ (z >> np.uint64(27))) * SPLITMIX_C).astype(np.uint64)
        return (z ^ (z >> np.uint64(31))).astype(np.uint64)


def stable_molecule_hash(molecule_ids) -> np.ndarray:
    """'NRER01234' -> uint64. Stable across processes (no Python hash randomisation)."""
    out = np.empty(len(molecule_ids), dtype=np.uint64)
    for i, m in enumerate(molecule_ids):
        h = np.uint64(1469598103934665603)
        for ch in str(m).encode():
            h = np.uint64((int(h) ^ ch) * 1099511628211 % (1 << 64))
        out[i] = h
    return out


def deterministic_mask(mol_hash: np.ndarray, n_layers: int, mc_index: int, base_seed: int,
                       gamma: float) -> np.ndarray:
    """Inference mask bank: z[i, l] ~ Bernoulli(gamma), reproducible from ids alone (p_i = 1)."""
    i = mol_hash.reshape(-1, 1).astype(np.uint64)
    l = np.arange(n_layers, dtype=np.uint64).reshape(1, -1)
    key = (i * np.uint64(0x100000001B3)) ^ (l * np.uint64(0x9E3779B1)) ^ np.uint64(
        (int(mc_index) << 32) ^ (int(base_seed) & 0xFFFFFFFF))
    u = _splitmix64(key).astype(np.float64) / float(1 << 64)
    return (u < gamma).astype(np.float32)


@dataclass
class RoutingContext:
    """Holds the current batch's per-(sample, layer) LoRA mask.

    enabled=False  -> vanilla LoRA (delta always fully applied): used for M0 and for warm-up-free baselines.
    mask=None with enabled=True is an error; callers must set a mask for every forward.
    """
    n_layers: int
    enabled: bool = False
    mask: torch.Tensor | None = None            # [B, n_layers] float {0,1}
    _force: float | None = field(default=None, repr=False)   # verification override: 1.0 / 0.0

    def set_mask(self, mask: torch.Tensor) -> None:
        assert mask.dim() == 2 and mask.shape[1] == self.n_layers, mask.shape
        self.mask = mask

    def force(self, value: float | None) -> None:
        """Verification hook: force all-one (1.0) or all-zero (0.0) routing."""
        self._force = value

    def factor(self, layer_idx: int, batch: int, device, dtype):
        if self._force is not None:
            return torch.full((batch, 1, 1), float(self._force), device=device, dtype=dtype)
        if not self.enabled:
            return None                                  # vanilla LoRA: no multiplication at all
        assert self.mask is not None, "routing enabled but no mask set for this batch"
        assert self.mask.shape[0] == batch, f"mask batch {self.mask.shape[0]} != input batch {batch}"
        return self.mask[:, layer_idx].to(device=device, dtype=dtype).view(batch, 1, 1)


class RoutedLoraLinear(nn.Module):
    """Wraps a peft `lora.Linear` so the LoRA delta is scaled by the layer's routing factor.

    Reproduces peft's own forward for the active adapter, then applies z. The base weight path,
    its gradient status (frozen) and the adapter parameters are untouched.
    """

    def __init__(self, lora_layer: nn.Module, ctx: RoutingContext, layer_idx: int, adapter: str):
        super().__init__()
        self.lora_layer = lora_layer
        self.ctx = ctx
        self.layer_idx = layer_idx
        self.adapter = adapter

    def forward(self, x, *args, **kwargs):
        ll = self.lora_layer
        base_out = ll.base_layer(x, *args, **kwargs)
        a = self.adapter
        if a not in ll.lora_A:
            return base_out
        dropout = ll.lora_dropout[a]
        lora_A, lora_B, scaling = ll.lora_A[a], ll.lora_B[a], ll.scaling[a]
        xa = x.to(lora_A.weight.dtype)
        delta = lora_B(lora_A(dropout(xa))) * scaling
        z = self.ctx.factor(self.layer_idx, x.shape[0], delta.device, delta.dtype)
        if z is not None:
            delta = delta * z
        return base_out + delta.to(base_out.dtype)


def _layer_index_from_name(name: str) -> int | None:
    """'...model.layers.17.self_attn.q_proj' -> 17 (Llama/Qwen decoder naming)."""
    parts = name.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
            return int(parts[i + 1])
    return None


def install_routing(model: nn.Module, adapter: str, n_layers: int) -> RoutingContext:
    """Replace every peft LoRA linear inside decoder layers with a routed wrapper.

    Returns the shared RoutingContext. Modules outside `model.layers.<i>` (e.g. a LoRA-wrapped
    score head, if any) are left untouched so that routing applies only to transformer-layer LoRA.
    """
    from peft.tuners.lora import Linear as LoraLinear

    ctx = RoutingContext(n_layers=n_layers)
    targets = []
    for name, mod in model.named_modules():
        if isinstance(mod, LoraLinear):
            li = _layer_index_from_name(name)
            if li is None:
                continue
            targets.append((name, mod, li))
    for name, mod, li in targets:
        parent = model.get_submodule(name.rsplit(".", 1)[0])
        setattr(parent, name.rsplit(".", 1)[1], RoutedLoraLinear(mod, ctx, li, adapter))
    ctx.n_wrapped = len(targets)                                   # type: ignore[attr-defined]
    ctx.wrapped_names = [n for n, _, _ in targets]                 # type: ignore[attr-defined]
    ctx.wrapped_layer_indices = sorted({li for _, _, li in targets})  # type: ignore[attr-defined]
    return ctx


def sample_training_mask(p_i: torch.Tensor, n_layers: int, gamma: float,
                         generator: torch.Generator) -> torch.Tensor:
    """z[i, l] ~ Bernoulli(gamma * p_i) — independent per sample AND per layer."""
    b = p_i.shape[0]
    u = torch.rand((b, n_layers), generator=generator, device=p_i.device)
    return (u < (gamma * p_i).clamp(0, 1).view(b, 1)).float()
