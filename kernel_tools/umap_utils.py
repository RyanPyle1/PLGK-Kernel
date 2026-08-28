# Copyright (C) 2026 Ryan Pyle
# SPDX-License-Identifier: MIT
"""Helpers for building UMAP embeddings from the IPS similarity matrix.

The paper's Section 6.2 figures (Figs 5, 6, 7) project the IPS matrix into
2D via UMAP on precomputed distances (``d = 1 - IPS`` or shifted variant).
"""
from __future__ import annotations

import numpy as np


def ips_to_distance(IPS) -> np.ndarray:
    """Convert IPS similarity to a UMAP-compatible non-negative distance.

    Uses ``D = max(0, 1 - IPS)`` as a precomputed distance for UMAP. Since
    IPS ∈ [-1, 1] for a cosine-normalized Gram, ``1 - IPS`` lands in [0, 2].
    """
    IPS = np.asarray(IPS.data if hasattr(IPS, "data") else IPS, dtype=np.float32)
    D = 1.0 - IPS
    D[D < 0] = 0.0
    # Force exact-zero diagonal (UMAP is sensitive to floating jitter here)
    np.fill_diagonal(D, 0.0)
    # Symmetrize (float rounding can leave tiny asymmetry)
    D = 0.5 * (D + D.T)
    return D


def umap_from_ips(
    IPS,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 0,
) -> np.ndarray:
    """Fit a UMAP embedding on distances derived from the IPS matrix.

    Returns ``X_umap`` of shape (n_total, n_components).
    Depends on the ``umap-learn`` package.
    """
    import umap

    D = ips_to_distance(IPS)
    reducer = umap.UMAP(
        n_components=n_components,
        metric="precomputed",
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    return reducer.fit_transform(D)
