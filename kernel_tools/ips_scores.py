# Copyright (C) 2026 Ryan Pyle
# SPDX-License-Identifier: MIT
"""IPS-based scalar scores and their evaluation (paper Table 1).

Every function takes numpy arrays with the following shape convention:

    IPS            (n_total, n_total)
    y_target_full  (n_total,)                 true class labels
    y_pred_full    (n_total,)                 model's argmax predictions
    y_loss_full    (n_total,)                 per-example cross-entropy loss
    y_hidden_full  (n_total, hidden_dim)      penultimate activations
    y_logits_full  (n_total, n_classes)       raw logits
    y_correct_full (n_total,)                 1 if correct, 0 if error
    train_idx      (n_train_samples,)         indices of training samples in IPS
    target_idx     (n_target_samples,)        indices of the audit / test samples

Row order across all arrays must match the IPS matrix's row order — this is
satisfied by ``kernel_tools.train.collect_full_eval`` and by IPSAccumulator's
sample layout.
"""
from __future__ import annotations

import numpy as np

# Optional deps are imported lazily inside the functions that use them so a
# minimal audit-only workflow doesn't pay the sklearn import cost.


def get_train_target_indices(batch_size: int, train_batches: int, target_batches: int):
    n_train = batch_size * train_batches
    n_target = batch_size * target_batches
    return np.arange(0, n_train), np.arange(n_train, n_train + n_target)


def compute_ips_scores(
    IPS: np.ndarray,
    y_target_full: np.ndarray,
    y_pred_full: np.ndarray,
    train_idx: np.ndarray,
    target_idx: np.ndarray,
    n_classes: int = 10,
    topk: int = 10,
) -> dict[str, np.ndarray]:
    """Compute IPS-derived scalar scores for each audit example.

    Returns per-example arrays for:
      - ``ips_max``    max IPS to any training sample
      - ``ips_topk``   mean IPS over top-k training samples
      - ``ips_l1``     L1 norm of IPS to training samples
      - ``ips_l2``     L2 norm of IPS to training samples (paper "IPS ell_2 norm")
      - ``ips_margin`` class-margin: max IPS to same-predicted-class training
                       samples minus max IPS to any other class
    """
    IPS_tt = IPS[np.ix_(target_idx, train_idx)]
    n_target, n_train = IPS_tt.shape

    ips_max = IPS_tt.max(axis=1)

    k = min(topk, n_train)
    topk_values = np.partition(IPS_tt, -k, axis=1)[:, -k:]
    ips_topk = topk_values.mean(axis=1)

    ips_l1 = np.sum(np.abs(IPS_tt), axis=1)
    ips_l2 = np.linalg.norm(IPS_tt, axis=1)

    train_labels = y_target_full[train_idx].astype(int)
    y_pred_target = y_pred_full[target_idx].astype(int)

    ips_max_per_class = np.full((n_target, n_classes), -np.inf, dtype=np.float32)
    for c in range(n_classes):
        mask_c = train_labels == c
        if not mask_c.any():
            continue
        ips_c = IPS_tt[:, mask_c]
        ips_max_per_class[:, c] = ips_c.max(axis=1)

    same_class_max = ips_max_per_class[np.arange(n_target), y_pred_target]
    ips_other = ips_max_per_class.copy()
    ips_other[np.arange(n_target), y_pred_target] = -np.inf
    other_class_max = ips_other.max(axis=1)
    ips_margin = same_class_max - other_class_max

    return {
        "ips_max": ips_max,
        "ips_topk": ips_topk,
        "ips_margin": ips_margin,
        "ips_l1": ips_l1,
        "ips_l2": ips_l2,
    }


def compute_hidden_scores(
    y_hidden_full: np.ndarray,
    train_idx: np.ndarray,
    target_idx: np.ndarray,
    topk: int = 10,
) -> dict[str, np.ndarray]:
    """Baseline: cosine similarity between penultimate activations of
    audit examples and training examples (mean top-k).
    """
    hidden_train = y_hidden_full[train_idx]
    hidden_target = y_hidden_full[target_idx]

    train_norm = hidden_train / np.linalg.norm(hidden_train, axis=1, keepdims=True)
    target_norm = hidden_target / np.linalg.norm(hidden_target, axis=1, keepdims=True)
    cos_sim = target_norm @ train_norm.T

    _, n_train = cos_sim.shape
    k = min(topk, n_train)
    hid_max = cos_sim.max(axis=1)
    topk_values = np.partition(cos_sim, -k, axis=1)[:, -k:]
    hid_topk = topk_values.mean(axis=1)

    return {"hid_max": hid_max, "hid_topk": hid_topk}


def compute_confidence_scores(
    y_logits_full: np.ndarray,
    target_idx: np.ndarray,
) -> dict[str, np.ndarray]:
    """Standard logit-based confidence scores (MSP, logit margin)."""
    from scipy.special import logsumexp

    logits_target = y_logits_full[target_idx]
    logsumexp_logits = logsumexp(logits_target, axis=1)
    log_probs = logits_target - logsumexp_logits[:, None]
    probs = np.exp(log_probs)
    msp = probs.max(axis=1)

    sorted_logits = np.sort(logits_target, axis=1)
    margin = sorted_logits[:, -1] - sorted_logits[:, -2]

    return {"msp": msp, "logit_margin": margin}


def evaluate_error_detection(
    score_dict: dict[str, np.ndarray],
    y_correct_full: np.ndarray,
    target_idx: np.ndarray,
) -> dict[str, dict[str, float]]:
    """AUROC + AUPR for error detection.

    Convention: scores where LARGER means "more likely correct". We flip
    signs so misclassification is the positive class.
    """
    from sklearn.metrics import roc_auc_score, average_precision_score

    y_correct_target = y_correct_full[target_idx].astype(int)
    y_error = 1 - y_correct_target

    results: dict[str, dict[str, float]] = {}
    for name, s in score_dict.items():
        scores_for_error = -s
        results[name] = {
            "AUROC_error": float(roc_auc_score(y_error, scores_for_error)),
            "AUPR_error": float(average_precision_score(y_error, scores_for_error)),
        }
    return results


def evaluate_loss_prediction(
    score_dict: dict[str, np.ndarray],
    y_loss_full: np.ndarray,
    target_idx: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Spearman + Pearson between scalar score s and -per-example loss."""
    from scipy.stats import spearmanr, pearsonr

    loss_target = y_loss_full[target_idx]
    minus_loss = -loss_target

    results: dict[str, dict[str, float]] = {}
    for name, s in score_dict.items():
        rho_s, _ = spearmanr(s, minus_loss)
        rho_p, _ = pearsonr(s, minus_loss)
        results[name] = {
            "Spearman_s_vs_-loss": float(rho_s),
            "Pearson_s_vs_-loss": float(rho_p),
        }
    return results


def run_ips_utility_experiments(
    *,
    batch_size: int,
    train_batches: int,
    target_batches: int,
    IPS,
    y_target_full,
    y_pred_full,
    y_loss_full,
    y_hidden_full,
    y_logits_full,
    y_correct_full=None,
    n_classes: int = 10,
    topk: int = 10,
    verbose: bool = True,
):
    """Table 1 orchestrator: compute all score families + both evaluations.

    Accepts either torch tensors or numpy arrays. Returns
    ``(error_metrics, loss_metrics)`` dicts keyed by score name.
    """
    # Coerce to numpy (accept torch tensors transparently)
    def _to_np(x):
        if x is None:
            return None
        try:
            return np.asarray(x.data if hasattr(x, "data") else x)
        except Exception:
            return np.asarray(x)

    IPS = _to_np(IPS)
    y_target_full = _to_np(y_target_full)
    y_pred_full = _to_np(y_pred_full)
    y_loss_full = _to_np(y_loss_full)
    y_hidden_full = _to_np(y_hidden_full)
    y_logits_full = _to_np(y_logits_full)
    if y_correct_full is None:
        y_correct_full = (y_target_full == y_pred_full).astype(int)
    else:
        y_correct_full = _to_np(y_correct_full)

    train_idx, target_idx = get_train_target_indices(batch_size, train_batches, target_batches)

    ips_scores = compute_ips_scores(
        IPS, y_target_full, y_pred_full, train_idx, target_idx,
        n_classes=n_classes, topk=topk,
    )
    hidden_scores = compute_hidden_scores(y_hidden_full, train_idx, target_idx, topk=topk)
    conf_scores = compute_confidence_scores(y_logits_full, target_idx)

    all_scores: dict[str, np.ndarray] = {}
    all_scores.update({f"IPS_{k}": v for k, v in ips_scores.items()})
    all_scores.update({f"HID_{k}": v for k, v in hidden_scores.items()})
    all_scores.update({f"CONF_{k}": v for k, v in conf_scores.items()})

    error_metrics = evaluate_error_detection(all_scores, y_correct_full, target_idx)
    loss_metrics = evaluate_loss_prediction(all_scores, y_loss_full, target_idx)

    if verbose:
        print("=== Error Detection (misclassification, higher = better detector) ===")
        for name, m in error_metrics.items():
            print(f"{name:20s} | AUROC: {m['AUROC_error']:.4f} | AUPR: {m['AUPR_error']:.4f}")
        print("\n=== Loss Prediction (per-example, higher Spearman = better) ===")
        for name, m in loss_metrics.items():
            print(
                f"{name:20s} | Spearman(s,-L): {m['Spearman_s_vs_-loss']:.4f} | "
                f"Pearson(s,-L): {m['Pearson_s_vs_-loss']:.4f}"
            )

    return error_metrics, loss_metrics
