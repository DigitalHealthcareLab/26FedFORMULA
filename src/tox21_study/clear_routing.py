"""CleaR score pass, GMM posterior and consistency target (execution prompt §9).

Order fixed by the prompt, per period (FL: round start, centralized: epoch start):
  1. freeze the just-received / current checkpoint
  2. ordinary dropout OFF, routing active with p_i = 1 and probability gamma, canonical inputs only
  3. N = consistency_mc_forwards stochastic forwards -> b_i = mean of the N LOGITS
  4. per-sample BCEWithLogits(b_i, observed_label, reduction='none') is the GMM input
     (mean-logit-then-BCE, NOT the mean of per-forward BCE)
  5. M1 fits one GMM on all samples; M2 fits one GMM per OBSERVED (noisy) class
     p_i = posterior of the component with the SMALLER mean loss
  6. cache sigmoid(b_i / temperature).detach() as the self-consistency target
  7. p_i and target stay fixed for the whole period
No fallback is invented: an impossible or non-finite fit is recorded and raises, so the run is marked
failed (spec `clear.abstention_fallback = null`, `clear.extra_min_sample_rule = null`).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture


class GMMFitError(RuntimeError):
    """Raised when a required GMM cannot be estimated; the run must be marked failed."""


def _fit_gmm(losses: np.ndarray, cfg, tag: str, audit: list) -> np.ndarray:
    n = len(losses)
    rec = {"group": tag, "n_samples": int(n)}
    if n < cfg.gmm_n_components:
        rec.update({"status": "failed_too_few_samples"})
        audit.append(rec)
        raise GMMFitError(f"{tag}: {n} samples < {cfg.gmm_n_components} components")
    x = losses.reshape(-1, 1).astype(np.float64)
    if not np.isfinite(x).all():
        rec.update({"status": "failed_non_finite_loss"})
        audit.append(rec)
        raise GMMFitError(f"{tag}: non-finite per-sample loss")
    gm = GaussianMixture(n_components=cfg.gmm_n_components, random_state=cfg.gmm_random_state,
                         covariance_type=cfg.gmm_covariance_type, tol=cfg.gmm_tol,
                         reg_covar=cfg.gmm_reg_covar, max_iter=cfg.gmm_max_iter,
                         n_init=cfg.gmm_n_init, init_params=cfg.gmm_init_params)
    try:
        gm.fit(x)
    except Exception as e:
        rec.update({"status": f"failed_fit_exception:{type(e).__name__}"})
        audit.append(rec)
        raise GMMFitError(f"{tag}: {e}") from e
    clean = int(np.argmin(gm.means_.ravel()))          # lower mean loss = clean (CleaR estimate_MM)
    p = gm.predict_proba(x)[:, clean]
    if not np.isfinite(p).all():
        rec.update({"status": "failed_non_finite_posterior"})
        audit.append(rec)
        raise GMMFitError(f"{tag}: non-finite posterior")
    rec.update({"status": "ok", "converged": bool(gm.converged_), "n_iter": int(gm.n_iter_),
                "component_means": [round(float(v), 6) for v in gm.means_.ravel()],
                "component_weights": [round(float(v), 6) for v in gm.weights_.ravel()],
                "clean_component": clean, "degenerate_means": bool(
                    abs(gm.means_.ravel()[0] - gm.means_.ravel()[1]) < 1e-12),
                "p_mean": round(float(p.mean()), 6), "p_min": round(float(p.min()), 6),
                "p_max": round(float(p.max()), 6),
                "frac_p_gt_threshold": round(float((p > cfg.diagnostic_clean_threshold).mean()), 6)})
    audit.append(rec)
    return p


@torch.no_grad()
def score_pass(backbone, ctx, smiles, labels, mol_hash, cfg, period: int, micro_batch: int,
               device) -> tuple[np.ndarray, np.ndarray]:
    """N stochastic forwards at p_i=1 with gamma routing; returns (mean logits b_i, per-sample BCE)."""
    was_training = backbone.model.training
    backbone.model.eval()                              # ordinary dropout off
    prev_enabled = ctx.enabled
    ctx.enabled = True
    from .routing import deterministic_mask
    from .batching import eval_batches
    n = len(smiles)
    acc = np.zeros((cfg.consistency_mc_forwards, n), dtype=np.float64)
    for m in range(cfg.consistency_mc_forwards):
        # p_i = 1 for the score pass -> mask depends only on gamma; seed separates period and forward
        mask_np = deterministic_mask(mol_hash, ctx.n_layers, mc_index=m,
                                     base_seed=(cfg.repeat_seed * 1000 + period), gamma=cfg.routing_gamma)
        for pos in eval_batches(backbone, smiles, micro_batch, cfg.sort_eval_by_length):
            batch = backbone.encode([smiles[i] for i in pos], augment=False)   # canonical input
            ctx.set_mask(torch.from_numpy(mask_np[pos]).to(device))
            acc[m, pos] = backbone.forward(batch).float().cpu().numpy()
    ctx.enabled = prev_enabled
    if was_training:
        backbone.model.train()
    b = acc.mean(axis=0)                                                # mean of LOGITS
    bce = F.binary_cross_entropy_with_logits(torch.tensor(b), torch.tensor(labels, dtype=torch.float64),
                                             reduction="none").numpy()
    return b, bce


@torch.no_grad()
def score_pass_mt(backbone, ctx, smiles, Y, Mk, mol_hash, cfg, period: int,
                  micro_batch: int, device):
    """멀티태스크 CleaR 채점. (n, T) 평균 로짓과 (n, T) per-cell BCE 를 돌려준다.

    태스크마다 score_pass 를 부르면 MC 전방계산이 12 배가 된다. 헤드가 이미 12 로짓을
    한 번에 내므로, MC 는 한 번만 돌리고 태스크별로 열을 나누면 단일 태스크와 같은 비용이다.
    미관측 칸의 BCE 는 NaN 으로 둔다 — 라벨이 없으니 값이 없는 게 맞고, 호출부가 마스크로
    거른다."""
    was_training = backbone.model.training
    backbone.model.eval()
    prev_enabled = ctx.enabled
    ctx.enabled = True
    from .routing import deterministic_mask
    from .batching import eval_batches
    n, T = len(smiles), Y.shape[1]
    acc = np.zeros((cfg.consistency_mc_forwards, n, T), dtype=np.float64)
    for mc in range(cfg.consistency_mc_forwards):
        mask_np = deterministic_mask(mol_hash, ctx.n_layers, mc_index=mc,
                                     base_seed=(cfg.repeat_seed * 1000 + period),
                                     gamma=cfg.routing_gamma)
        for pos in eval_batches(backbone, smiles, micro_batch, cfg.sort_eval_by_length):
            batch = backbone.encode([smiles[i] for i in pos], augment=False)
            ctx.set_mask(torch.from_numpy(mask_np[pos]).to(device))
            acc[mc, pos] = backbone.forward(batch).float().cpu().numpy()
    ctx.enabled = prev_enabled
    if was_training:
        backbone.model.train()
    b = acc.mean(axis=0)
    bce = F.binary_cross_entropy_with_logits(
        torch.tensor(b), torch.tensor(Y, dtype=torch.float64), reduction="none").numpy()
    bce[Mk == 0] = np.nan
    return b, bce


def period_state(method: str, b_logits: np.ndarray, bce: np.ndarray, observed: np.ndarray,
                 cfg, period: int, client_id, audit: list) -> tuple[np.ndarray, np.ndarray]:
    """Returns (p_i, consistency_target). M1 pooled GMM; M2 and M2F one GMM per observed class."""
    p = np.ones(len(bce), dtype=np.float64)
    if method == "M1":
        p = _fit_gmm(bce, cfg, f"period{period}|client{client_id}|pooled", audit)
    elif method in ("M2", "M2F"):
        for c in (0, 1):
            idx = np.where(observed == c)[0]
            if len(idx) == 0:
                audit.append({"group": f"period{period}|client{client_id}|class{c}",
                              "n_samples": 0, "status": "skipped_empty_class"})
                continue                                  # nothing to fit: spec says empty class needs no fit
            p[idx] = _fit_gmm(bce[idx], cfg, f"period{period}|client{client_id}|class{c}", audit)
    else:
        raise ValueError(method)
    target = 1.0 / (1.0 + np.exp(-b_logits / cfg.consistency_temperature))   # detached soft target
    return p, target


def consistency_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """lambda is applied by the caller. Soft-target BCE against the cached sigmoid(b_i/T)."""
    return F.binary_cross_entropy_with_logits(logits, target, reduction="mean")
