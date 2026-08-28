# Copyright (C) 2026 Ryan Pyle
# SPDX-License-Identifier: MIT
"""SVD / mode-decomposition utilities for the paper's Section 6.3.1.

Two shapes of ``V_modes`` are used downstream:

  * "flat" — a numpy array of shape ``(K, P)`` where each row is a
    parameter-space mode (a right-singular vector of Φtrain). Convenient
    for storage and for input-space Jacobian projections.

  * "param-list" — a Python list of length K, where each entry is a list
    of tensors with the same shapes as the model's parameters (recovered
    by unflattening the flat mode). Required by the cure algorithm
    (``kernel_tools.adversarial_cure``), which iterates parameter tensors.

Convenience:
  ``lowrank_svd_train_side(train_mat, k)`` — truncated SVD wrapper with
    a randomized fallback for large matrices.
  ``flat_modes_to_param_list(V_flat, template_params)`` — unflatten.
  ``param_list_to_flat_modes(V_list)`` — flatten.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
import torch.nn as nn


def lowrank_svd_train_side(train_mat: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Truncated SVD ``train_mat = U diag(S) V^T``.

    Prefers ``torch.linalg.svd`` (exact) when the matrix is small enough,
    falls back to ``torch.svd_lowrank`` (randomized) when either dimension
    exceeds 4096 to keep memory manageable.

    Returns:
      U: (n, k)
      S: (k,)
      V: (p, k)
    """
    n, p = train_mat.shape
    if n * p <= 4096 * 4096:
        U, S, Vh = torch.linalg.svd(train_mat, full_matrices=False)
        Vt = Vh                        # torch.linalg.svd returns Vh
        U_k = U[:, :k].contiguous()
        S_k = S[:k].contiguous()
        V_k = Vt[:k, :].transpose(0, 1).contiguous()   # (p, k)
        return U_k, S_k, V_k
    # Randomized fallback
    q = min(2 * k + 10, min(n, p))
    U, S, V = torch.svd_lowrank(train_mat, q=q, niter=4)
    return U[:, :k].contiguous(), S[:k].contiguous(), V[:, :k].contiguous()


def compute_param_shapes(model: nn.Module) -> list[torch.Size]:
    """Return list of parameter shapes for non-meta params, in named order."""
    return [p.shape for _, p in model.named_parameters() if not getattr(p, "is_meta", False)]


def flat_modes_to_param_list(
    V_flat: np.ndarray | torch.Tensor,
    param_shapes: list[torch.Size],
) -> list[list[torch.Tensor]]:
    """Convert ``V_flat`` (K, P) → K-length list of param-shaped mode-tensor-lists.

    Assumes the flat-order matches ``compute_param_shapes(model)`` — i.e.
    concatenation follows ``model.named_parameters()`` iteration order,
    which is stable across a single Python process.
    """
    if isinstance(V_flat, np.ndarray):
        V_flat = torch.from_numpy(V_flat)
    V_flat = V_flat.detach().cpu().float()
    K, P = V_flat.shape
    P_check = int(sum(int(np.prod(s)) for s in param_shapes))
    assert P == P_check, f"flat dim {P} does not match model param total {P_check}"
    modes: list[list[torch.Tensor]] = []
    for k in range(K):
        row = V_flat[k]                # (P,)
        parts: list[torch.Tensor] = []
        offset = 0
        for shape in param_shapes:
            n = int(np.prod(shape))
            parts.append(row[offset:offset + n].reshape(shape).contiguous())
            offset += n
        modes.append(parts)
    return modes


def param_list_to_flat_modes(V_list: list[list[torch.Tensor]]) -> torch.Tensor:
    """Inverse of ``flat_modes_to_param_list``. Returns (K, P) float tensor."""
    return torch.stack(
        [torch.cat([p.reshape(-1) for p in v], dim=0) for v in V_list],
        dim=0,
    )


def modal_cancellation_ratio(
    C_signed: np.ndarray,
    Lambda: np.ndarray,
    K_grid: list[int] | np.ndarray | None = None,
) -> np.ndarray:
    """Compute the paper's CR(K) = |Σ_k^K Λ_k C_k| / Σ_k^K |Λ_k C_k|
    for each K in ``K_grid`` (defaults to 1..r).

    ``C_signed`` shape: (n_examples, r); ``Lambda`` shape: (r,).
    Returns array of shape (n_examples, len(K_grid)).
    """
    r = C_signed.shape[1]
    if K_grid is None:
        K_grid = np.arange(1, r + 1)
    K_grid = np.asarray(K_grid, dtype=int)
    weighted = C_signed * Lambda[None, :]        # (n_ex, r)
    abs_weighted = np.abs(weighted)
    out = np.zeros((C_signed.shape[0], len(K_grid)))
    for j, K in enumerate(K_grid):
        num = np.abs(weighted[:, :K].sum(axis=1))
        denom = abs_weighted[:, :K].sum(axis=1) + 1e-30
        out[:, j] = num / denom
    return out
