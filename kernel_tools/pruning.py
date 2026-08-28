# Copyright (C) 2026 Ryan Pyle
# SPDX-License-Identifier: MIT
"""Data-pruning score functions and index-set builders (paper Appendix A.4).

Given the audit ``PNTK`` matrix (n_train × n_audit) and optionally the
train-train IPS block, compute the paper's 8 pruning score families:

  most_harmful         top-k+ influence sum (drives loss UP)
  least_impact_abs     lowest sum-of-abs-influence
  least_impact_signed  lowest |sum-of-influence|
  most_redundant       highest cosine sim with mean-influence direction
  cluster_prune        cluster-balanced pruning via IPS spectral clusters
  cluster_upweight     cluster_prune + upweight retained samples
  class_bal_least      class-balanced least_impact_abs
  random / random_class_bal   controls

Consumers pass an artifact (from Group 1 with ``--do-ips``) that contains
``PNTK`` and ``IPS``, plus the training-set labels.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np


@dataclass
class PruningScores:
    """Per-sample scores used by the various pruning methods.

    All fields have shape ``(n_train,)`` unless noted.
    """
    harm_scores: np.ndarray                   # signed sum of influence — high = harmful
    abs_impact: np.ndarray                    # sum of |influence| — low = disposable
    signed_impact: np.ndarray                 # |sum of influence| — low = cancels out
    redundancy_scores: np.ndarray             # cosine sim with mean influence
    train_labels: np.ndarray                  # true class labels (n_train,)


def compute_pruning_scores(PNTK: np.ndarray, train_labels: np.ndarray) -> PruningScores:
    """Compute all per-sample pruning scores from the audit matrix ``PNTK``.

    PNTK convention: ``PNTK[m, n]`` > 0 means training sample m drove
    loss on audit sample n UPWARD (harmful to that test point).
    """
    harm_scores = PNTK.sum(axis=1)                   # (n_train,)
    abs_impact = np.abs(PNTK).sum(axis=1)
    signed_impact = np.abs(harm_scores)
    mean_influence = PNTK.mean(axis=0)
    mean_norm = np.linalg.norm(mean_influence)
    row_norms = np.linalg.norm(PNTK, axis=1)
    row_norms_safe = np.where(row_norms > 0, row_norms, 1.0)
    redundancy_scores = (PNTK @ mean_influence) / (row_norms_safe * mean_norm)
    return PruningScores(
        harm_scores=harm_scores, abs_impact=abs_impact,
        signed_impact=signed_impact, redundancy_scores=redundancy_scores,
        train_labels=train_labels,
    )


def compute_ips_spectral_clusters(
    IPS_train: np.ndarray,
    n_clusters: int = 100,
    seed: int = 42,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (cluster_labels, embedding_np, cluster_sizes).

    Spectral clustering on the (shifted-to-nonneg) IPS matrix via low-rank
    SVD + KMeans on the row-normalized U√S embedding.
    """
    import torch
    from sklearn.cluster import KMeans

    IPS_nn = (IPS_train + 1.0) / 2.0
    IPS_nn_t = torch.as_tensor(IPS_nn, dtype=torch.float32)
    if verbose:
        print(f"Computing rank-{n_clusters} SVD of IPS_train ({IPS_nn.shape[0]}x{IPS_nn.shape[1]}) ...")
    t0 = time.time()
    U, S, V = torch.svd_lowrank(IPS_nn_t, q=n_clusters)
    if verbose:
        print(f"  SVD done in {time.time()-t0:.1f}s. Top 5 singular values: {S[:5].numpy()}")
    embedding = U * S.sqrt().unsqueeze(0)
    row_norms = embedding.norm(dim=1, keepdim=True).clamp(min=1e-8)
    embedding = (embedding / row_norms).numpy()
    if verbose:
        print(f"Running KMeans with {n_clusters} clusters ...")
    t0 = time.time()
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=5, max_iter=300)
    cluster_labels = km.fit_predict(embedding)
    if verbose:
        print(f"  KMeans done in {time.time()-t0:.1f}s")
    cluster_sizes = np.bincount(cluster_labels, minlength=n_clusters)
    return cluster_labels, embedding, cluster_sizes


# --- Index-set builders ---
# Each returns ``keep_indices`` (np.ndarray of length n_train - n_remove).

def _keep_after_remove(n_train: int, remove_idx: np.ndarray) -> np.ndarray:
    return np.setdiff1d(np.arange(n_train), np.asarray(remove_idx))


def get_harmful_indices(scores: PruningScores, k_pct: float, n_train: int) -> np.ndarray:
    n_remove = int(n_train * k_pct / 100)
    remove_idx = np.argsort(scores.harm_scores)[-n_remove:]
    return _keep_after_remove(n_train, remove_idx)


def get_least_impactful_abs_indices(scores: PruningScores, k_pct: float, n_train: int) -> np.ndarray:
    n_remove = int(n_train * k_pct / 100)
    remove_idx = np.argsort(scores.abs_impact)[:n_remove]
    return _keep_after_remove(n_train, remove_idx)


def get_least_impactful_signed_indices(scores: PruningScores, k_pct: float, n_train: int) -> np.ndarray:
    n_remove = int(n_train * k_pct / 100)
    remove_idx = np.argsort(scores.signed_impact)[:n_remove]
    return _keep_after_remove(n_train, remove_idx)


def get_most_redundant_indices(scores: PruningScores, k_pct: float, n_train: int) -> np.ndarray:
    n_remove = int(n_train * k_pct / 100)
    remove_idx = np.argsort(scores.redundancy_scores)[-n_remove:]
    return _keep_after_remove(n_train, remove_idx)


def get_class_balanced_least_impactful_indices(scores: PruningScores, k_pct: float, n_train: int) -> np.ndarray:
    n_remove = int(n_train * k_pct / 100)
    n_classes = int(scores.train_labels.max()) + 1
    per_class_remove = n_remove // n_classes
    remove_idx: list[int] = []
    for c in range(n_classes):
        class_mask = scores.train_labels == c
        class_indices = np.where(class_mask)[0]
        class_impact = scores.abs_impact[class_mask]
        least = np.argsort(class_impact)[:per_class_remove]
        remove_idx.extend(class_indices[least])
    return _keep_after_remove(n_train, np.asarray(remove_idx))


def get_random_indices(k_pct: float, n_train: int, rng_seed: int) -> np.ndarray:
    n_remove = int(n_train * k_pct / 100)
    rng = np.random.RandomState(rng_seed)
    remove_idx = rng.choice(n_train, size=n_remove, replace=False)
    return _keep_after_remove(n_train, remove_idx)


def get_random_class_balanced_indices(
    k_pct: float, n_train: int, train_labels: np.ndarray, rng_seed: int,
) -> np.ndarray:
    n_remove = int(n_train * k_pct / 100)
    n_classes = int(train_labels.max()) + 1
    per_class_remove = n_remove // n_classes
    rng = np.random.RandomState(rng_seed)
    remove_idx: list[int] = []
    for c in range(n_classes):
        class_indices = np.where(train_labels == c)[0]
        chosen = rng.choice(len(class_indices), size=min(per_class_remove, len(class_indices)), replace=False)
        remove_idx.extend(class_indices[chosen])
    return _keep_after_remove(n_train, np.asarray(remove_idx))


# --- Destructive counterparts (Fig 22): remove the MOST important data ---

def get_most_helpful_indices(scores: PruningScores, k_pct: float, n_train: int) -> np.ndarray:
    """Remove k% most-helpful samples (most-negative harm_scores)."""
    n_remove = int(n_train * k_pct / 100)
    remove_idx = np.argsort(scores.harm_scores)[:n_remove]
    return _keep_after_remove(n_train, remove_idx)


def get_highest_impact_abs_indices(scores: PruningScores, k_pct: float, n_train: int) -> np.ndarray:
    n_remove = int(n_train * k_pct / 100)
    remove_idx = np.argsort(scores.abs_impact)[-n_remove:]
    return _keep_after_remove(n_train, remove_idx)


def get_highest_impact_signed_indices(scores: PruningScores, k_pct: float, n_train: int) -> np.ndarray:
    n_remove = int(n_train * k_pct / 100)
    remove_idx = np.argsort(scores.signed_impact)[-n_remove:]
    return _keep_after_remove(n_train, remove_idx)


def get_class_bal_highest_impact_indices(scores: PruningScores, k_pct: float, n_train: int) -> np.ndarray:
    n_remove = int(n_train * k_pct / 100)
    n_classes = int(scores.train_labels.max()) + 1
    per_class_remove = n_remove // n_classes
    remove_idx: list[int] = []
    for c in range(n_classes):
        class_mask = scores.train_labels == c
        class_indices = np.where(class_mask)[0]
        class_impact = scores.abs_impact[class_mask]
        highest = np.argsort(class_impact)[-per_class_remove:]
        remove_idx.extend(class_indices[highest])
    return _keep_after_remove(n_train, np.asarray(remove_idx))


def get_smallest_clusters_indices(
    cluster_labels: np.ndarray, cluster_sizes: np.ndarray, embedding: np.ndarray,
    k_pct: float, n_train: int,
) -> np.ndarray:
    """Remove k% by eliminating the SMALLEST clusters first (destroys rare structure)."""
    n_remove = int(n_train * k_pct / 100)
    sorted_clusters = np.argsort(cluster_sizes)
    remove_idx: list[int] = []
    for c in sorted_clusters:
        members = np.where(cluster_labels == c)[0]
        if len(remove_idx) + len(members) <= n_remove:
            remove_idx.extend(members)
        else:
            needed = n_remove - len(remove_idx)
            centroid = embedding[members].mean(axis=0)
            dists = np.linalg.norm(embedding[members] - centroid, axis=1)
            closest = np.argsort(dists)[:needed]
            remove_idx.extend(members[closest])
            break
    return _keep_after_remove(n_train, np.asarray(remove_idx))


def get_cluster_prune_indices(
    cluster_labels: np.ndarray,
    cluster_sizes: np.ndarray,
    embedding: np.ndarray,
    k_pct: float,
    n_train: int,
    with_upweight: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Remove k% from over-represented clusters.

    Within each cluster, removes samples closest to the cluster centroid
    (most redundant). If ``with_upweight=True``, also returns per-sample
    weights that scale each cluster's total weight back to its original size.
    """
    n_clusters = len(cluster_sizes)
    uniform_size = n_train / n_clusters
    n_remove = int(n_train * k_pct / 100)

    excess = np.maximum(cluster_sizes - uniform_size, 0)
    total_excess = excess.sum()
    if total_excess < 1:
        excess = cluster_sizes.astype(float)
        total_excess = excess.sum()
    per_cluster_remove = (excess / total_excess * n_remove).astype(int)
    remainder = n_remove - per_cluster_remove.sum()
    if remainder > 0:
        priority = np.argsort(-excess)
        for i in range(remainder):
            per_cluster_remove[priority[i % n_clusters]] += 1
    per_cluster_remove = np.minimum(per_cluster_remove, cluster_sizes)

    remove_idx: list[int] = []
    for c in range(n_clusters):
        if per_cluster_remove[c] == 0:
            continue
        members = np.where(cluster_labels == c)[0]
        centroid = embedding[members].mean(axis=0)
        dists = np.linalg.norm(embedding[members] - centroid, axis=1)
        closest = np.argsort(dists)[:per_cluster_remove[c]]
        remove_idx.extend(members[closest])
    keep_idx = _keep_after_remove(n_train, np.asarray(remove_idx))

    if not with_upweight:
        return keep_idx

    new_cluster_sizes = np.bincount(cluster_labels[keep_idx], minlength=n_clusters)
    weights = np.ones(len(keep_idx))
    keep_cluster_labels = cluster_labels[keep_idx]
    for c in range(n_clusters):
        if new_cluster_sizes[c] > 0:
            w = cluster_sizes[c] / new_cluster_sizes[c]
            weights[keep_cluster_labels == c] = w
    weights *= len(keep_idx) / weights.sum()
    return keep_idx, weights
