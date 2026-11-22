"""Evaluation metrics. Primary = sklearn average_precision_score (never called PR-AUC)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, roc_auc_score, balanced_accuracy_score,
                             brier_score_loss, confusion_matrix)


def threshold_by_validation_ba(y: np.ndarray, p: np.ndarray) -> float:
    """maximize_validation_balanced_accuracy; ties -> closest to 0.5, then the smaller value."""
    cands = np.unique(np.concatenate([p, [0.5]]))
    best, best_t = -1.0, 0.5
    for t in cands:
        ba = balanced_accuracy_score(y, (p >= t).astype(int))
        if ba > best + 1e-12:
            best, best_t = ba, float(t)
        elif abs(ba - best) <= 1e-12:
            if (abs(t - 0.5), t) < (abs(best_t - 0.5), best_t):
                best_t = float(t)
    return best_t


def evaluate(y: np.ndarray, p: np.ndarray, threshold: float | None = None) -> dict:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    single_class = len(np.unique(y)) < 2
    out = {
        "n": int(len(y)), "n_pos": int(y.sum()), "prevalence": round(float(y.mean()), 6),
        "average_precision": None if single_class else float(average_precision_score(y, p)),
        "roc_auc": None if single_class else float(roc_auc_score(y, p)),
        "brier_score": float(brier_score_loss(y, p)),
        "single_class": bool(single_class),
    }
    if threshold is not None:
        yh = (p >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
        out.update({
            "threshold": float(threshold),
            "balanced_accuracy": float(balanced_accuracy_score(y, yh)),
            "sensitivity": float(tp / (tp + fn)) if (tp + fn) else None,
            "specificity": float(tn / (tn + fp)) if (tn + fp) else None,
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        })
    return out


def detection_metrics(is_flip: np.ndarray, p_clean: np.ndarray, clean_threshold: float = 0.5) -> dict:
    """Injected-flip detection from 1 - p_i. Analysis only; never fed back into training/HPO."""
    is_flip = np.asarray(is_flip).astype(int)
    score = 1.0 - np.asarray(p_clean, dtype=float)
    if is_flip.sum() == 0 or is_flip.sum() == len(is_flip):
        return {"n": int(len(is_flip)), "n_flip": int(is_flip.sum()),
                "detection_auroc": None, "detection_ap": None,
                "note": "no positive noise class -> N/A"}
    pred_clean = (np.asarray(p_clean) > clean_threshold).astype(int)
    truly_clean = 1 - is_flip
    tp = int(((pred_clean == 1) & (truly_clean == 1)).sum())
    fp = int(((pred_clean == 1) & (truly_clean == 0)).sum())
    fn = int(((pred_clean == 0) & (truly_clean == 1)).sum())
    return {"n": int(len(is_flip)), "n_flip": int(is_flip.sum()),
            "detection_auroc": float(roc_auc_score(is_flip, score)),
            "detection_ap": float(average_precision_score(is_flip, score)),
            "clean_precision": tp / (tp + fp) if (tp + fp) else None,
            "clean_recall": tp / (tp + fn) if (tp + fn) else None}
