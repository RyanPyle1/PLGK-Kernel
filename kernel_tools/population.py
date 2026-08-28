# Copyright (C) 2026 Ryan Pyle
# SPDX-License-Identifier: MIT
"""Population-level distances in the path-integrated loss-gradient geometry.

Supports the paper's population-similarity appendix. The similarity kernel of
Section 4 is defined between pairs of points; this module supplies the
population-to-population counterparts and the inference machinery they need.

Two distances, reported separately rather than combined:

  ``mmd``            first-order: the distance between kernel mean embeddings.
                     With the linear kernel k(x,y) = <Phi(x), Phi(y)> used
                     throughout, MMD^2 reduces exactly to ||mu_S - mu_S'||^2.
                     The UNBIASED estimator is used (self-pair terms omitted);
                     the biased plug-in reports positive discrepancy even for
                     two samples of the same distribution. Returns MMD, not
                     MMD^2, so it shares units with the Bures term.

  ``bures_distance`` splits the kernel Bures-Wasserstein distance into its
                     mean part (recovering MMD) and its covariance part. The
                     paper's prediction is an asymmetry between the two, so
                     they must be reported separately.

Inference (``permutation_test``) operates on BASE EXAMPLES, not individual
audit points. The adversarial audit set of Section 6.3 nests all three groups
within the same source examples -- 1 clean, 4 adversarial and 8 random points
per 16-slot block -- and writes the clean image unchanged into all four attack
slots. Point-level permutation would shuffle exact duplicates and same-image
siblings across groups, violating exchangeability and inflating significance.
``adversarial_group_masks`` exposes the slot layout so callers can construct
the blocks.

Numerical notes that are load-bearing rather than cosmetic:

  * ``bures_distance`` computes in the joint centered span of the two groups.
    With n points the centered data spans at most n-1 dimensions, so this is
    exact, not an approximation, and avoids eigendecomposing a rank-deficient
    matrix at ambient width (which is 51200 for attribution signatures).
  * ``_psd_sqrt`` clips eigenvalues at exactly zero. A small positive floor
    would add sqrt(floor) per null direction to the trace, making the result
    depend on ambient dimension and silently breaking that projection.
  * ``project_pooled`` hoists the dimensionality reduction out of resampling
    loops. It centers on the POOLED mean; centering each group separately
    would project out mu_A - mu_B and annihilate the MMD term.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ----------------------------------------------------------------------
# Group masks for the paper's adversarial audit-set slot layout
# ----------------------------------------------------------------------

def adversarial_group_masks(n_audit: int) -> dict[str, np.ndarray]:
    """Return boolean masks for the clean / adv / rand groups.

    Mirrors the slot layout built by
    ``kernel_tools.adversarial_data.build_adversarial_audit_sets`` and used
    by ``scripts/mnist_mode_svd_snapshots.py``: blocks of 16 slots per
    successful example, each block being 4 attacks x
    ``[clean, adv, rand1, rand2]``.
    """
    if n_audit % 16 != 0:
        raise ValueError(
            f"n_audit={n_audit} is not a multiple of 16; this does not look like "
            "an adversarial audit set built by build_adversarial_audit_sets()."
        )
    n_examples = n_audit // 16
    clean = [16 * i + off for i in range(n_examples) for off in (0, 4, 8, 12)]
    adv = [16 * i + off for i in range(n_examples) for off in (1, 5, 9, 13)]
    rand = [16 * i + off for i in range(n_examples)
            for off in (2, 3, 6, 7, 10, 11, 14, 15)]
    out = {}
    for name, pos in (("clean", clean), ("adv", adv), ("rand", rand)):
        m = np.zeros(n_audit, dtype=bool)
        m[pos] = True
        out[name] = m
    return out


# ----------------------------------------------------------------------
# Core distances
# ----------------------------------------------------------------------

@dataclass
class PopulationDistance:
    """Decomposition of a population-to-population distance.

    ``mean_term`` is the first-order (MMD) part, ``cov_term`` the
    second-order (Bures) part, and ``total`` the combined kernel
    Bures-Wasserstein distance. All are distances, not squared distances.
    """
    mean_term: float
    cov_term: float
    total: float

    def as_dict(self) -> dict[str, float]:
        return {"mean_term": self.mean_term, "cov_term": self.cov_term,
                "total": self.total}


def mmd(A: np.ndarray, B: np.ndarray, *, unbiased: bool = True) -> float:
    """Maximum mean discrepancy between two point sets, linear kernel.

    ``A`` is ``(n_a, d)``, ``B`` is ``(n_b, d)``. With a linear kernel the
    MMD is just the distance between mean embeddings,
    ``||mean(A) - mean(B)||``, which is the quantity the PLGK first-order
    bound speaks about.

    ``unbiased=True`` applies the standard correction for the bias that the
    plug-in estimator carries at small n. The biased plug-in systematically
    reports a positive distance even for two samples from the same
    distribution, which at n~16 is exactly the regime where a spurious
    "separation" would appear.
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    n_a, n_b = A.shape[0], B.shape[0]
    if not unbiased:
        return float(np.linalg.norm(A.mean(axis=0) - B.mean(axis=0)))
    # Unbiased MMD^2 with linear kernel k(x,y) = <x,y>:
    #   E[k(a,a')] + E[k(b,b')] - 2 E[k(a,b)]  with self-terms excluded.
    if n_a < 2 or n_b < 2:
        return float(np.linalg.norm(A.mean(axis=0) - B.mean(axis=0)))
    Kaa = A @ A.T
    Kbb = B @ B.T
    Kab = A @ B.T
    term_a = (Kaa.sum() - np.trace(Kaa)) / (n_a * (n_a - 1))
    term_b = (Kbb.sum() - np.trace(Kbb)) / (n_b * (n_b - 1))
    term_ab = Kab.mean()
    mmd2 = term_a + term_b - 2.0 * term_ab
    # Unbiased estimator can go slightly negative when the true MMD is ~0.
    return float(np.sqrt(max(mmd2, 0.0)))


def _psd_sqrt(M: np.ndarray) -> np.ndarray:
    """Symmetric PSD square root via eigendecomposition.

    Eigenvalues are clipped at exactly zero, not at a small positive floor.
    A positive floor would add ``sqrt(floor)`` per null direction to the
    trace, making the result depend on the ambient dimension -- which
    silently breaks the (otherwise exact) subspace projection in
    ``bures_distance`` and would make distances incomparable across feature
    spaces of different width.
    """
    M = 0.5 * (M + M.T)
    w, V = np.linalg.eigh(M)
    w = np.clip(w, 0.0, None)
    return (V * np.sqrt(w)) @ V.T


def _shrink(C: np.ndarray, shrinkage: float) -> np.ndarray:
    """Ledoit-Wolf-style shrinkage toward a scaled identity.

    At n comparable to (or below) d the sample covariance is rank-deficient
    and its small eigenvalues are pure noise; the Bures term is dominated by
    exactly those directions. Shrinkage is not cosmetic here -- without it
    the covariance distance at n=16 mostly measures sampling noise.
    """
    if shrinkage <= 0.0:
        return C
    d = C.shape[0]
    target = np.trace(C) / d * np.eye(d)
    return (1.0 - shrinkage) * C + shrinkage * target


def _joint_subspace(A: np.ndarray, B: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """Orthonormal basis (d, r) for the span of both groups' centered data.

    With ``n_a + n_b`` points the centered data occupies a subspace of
    dimension at most ``n_a + n_b - 2``, which in this application is tiny
    (~48) next to the ambient dimension (~1024 for attribution signatures,
    up to 44k for raw path features). Both the mean difference and the
    Bures term live entirely inside this subspace: the mean difference is a
    combination of the data points, and each covariance has range contained
    in its group's centered span. So projecting is exact, not an
    approximation -- it just avoids eigendecomposing a rank-deficient
    ambient-sized matrix.
    """
    stacked = np.vstack([A - A.mean(axis=0), B - B.mean(axis=0)])
    # SVD rather than QR: the R-diagonal is not a reliable rank indicator
    # for a spread spectrum (an unpivoted QR can put a small value on the
    # diagonal for a direction that carries real variance), and silently
    # dropping such a direction would make the projection lossy. Singular
    # values give the correct rank test.
    U, s, _ = np.linalg.svd(stacked.T, full_matrices=False)
    if s.size == 0:
        return U[:, :0]
    keep = s > tol * s[0] * max(stacked.shape)
    return U[:, keep]


def bures_distance(
    A: np.ndarray,
    B: np.ndarray,
    *,
    shrinkage: float = 0.1,
    unbiased_mean: bool = True,
) -> PopulationDistance:
    """Kernel Bures-Wasserstein distance between two populations.

    ``W^2 = ||mu_A - mu_B||^2 + Tr(C_A + C_B - 2 (C_A^{1/2} C_B C_A^{1/2})^{1/2})``

    Returns the mean and covariance terms separately (as distances, i.e.
    square roots of the respective squared terms) alongside the total. The
    separation between the two is the point of the exercise: the PLGK
    cancellation result predicts the mean term is blind to the clean/adv
    distinction while the covariance term is not.

    Computation happens in the joint centered span of the two groups (see
    ``_joint_subspace``), which leaves both terms unchanged while keeping
    the eigendecompositions at ``(n_a + n_b)`` scale rather than ambient
    dimension.

    ``shrinkage`` regularizes both covariances toward a scaled identity
    *within that subspace*; see ``_shrink``. Note this makes the regularizer
    subspace-relative -- a scaled identity is not a projection-invariant
    target -- which is the sensible choice here (the ambient null directions
    carry no data and should not dilute the shrinkage), but it does mean the
    value is only comparable across runs with similar group sizes. Report
    the value used and sweep it; results at n~16 are sensitive to it.
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if A.ndim != 2 or B.ndim != 2 or A.shape[1] != B.shape[1]:
        raise ValueError(f"shape mismatch: A {A.shape}, B {B.shape}")

    mean_sq = mmd(A, B, unbiased=unbiased_mean) ** 2

    Q = _joint_subspace(A, B)
    if Q.shape[1] == 0:
        return PopulationDistance(
            mean_term=float(np.sqrt(max(mean_sq, 0.0))),
            cov_term=0.0,
            total=float(np.sqrt(max(mean_sq, 0.0))),
        )
    Ap = A @ Q
    Bp = B @ Q

    C_A = _shrink(np.atleast_2d(np.cov(Ap, rowvar=False, ddof=1)), shrinkage)
    C_B = _shrink(np.atleast_2d(np.cov(Bp, rowvar=False, ddof=1)), shrinkage)

    sA = _psd_sqrt(C_A)
    cross = _psd_sqrt(sA @ C_B @ sA)
    cov_sq = float(np.trace(C_A) + np.trace(C_B) - 2.0 * np.trace(cross))
    cov_sq = max(cov_sq, 0.0)

    return PopulationDistance(
        mean_term=float(np.sqrt(max(mean_sq, 0.0))),
        cov_term=float(np.sqrt(cov_sq)),
        total=float(np.sqrt(max(mean_sq + cov_sq, 0.0))),
    )

def project_pooled(A: np.ndarray, B: np.ndarray):
    """Project both groups onto the span of the pooled, GLOBALLY centered data.

    Returns ``(Ap, Bp)`` on which ``mmd`` and ``bures_distance`` give the
    same values as on the originals -- including under bootstrap resampling
    and label permutation -- so this can be hoisted out of a resampling loop.

    At real scale that matters a lot: attribution signatures live in
    R^n_train (51200 for the paper's MNIST config) while two 64-point groups
    span at most 128 dimensions, and the naive code re-runs a
    51200-dimensional SVD inside every one of ~1000 permutations.

    WARNING -- do NOT use ``_joint_subspace`` for this. That helper centers
    each group about *its own* mean before spanning, which deliberately
    projects out the between-group mean difference: it is correct inside
    ``bures_distance`` (where the mean term is computed separately, before
    projection) but it annihilates MMD if applied up front. Measured on
    random data: MMD 7.073 -> 0.000, and under permutation the discrepancy
    blows up entirely. Hence the global centering here, which keeps
    ``mu_A - mu_B`` inside the retained span.
    """
    pooled = np.vstack([A, B])
    mu = pooled.mean(axis=0)
    centered = pooled - mu
    U, s, _ = np.linalg.svd(centered.T, full_matrices=False)
    if s.size == 0:
        return A[:, :0], B[:, :0]
    keep = s > 1e-10 * s[0] * max(centered.shape)
    Q = U[:, keep]
    return (A - mu) @ Q, (B - mu) @ Q


# ----------------------------------------------------------------------
# Uncertainty: bootstrap CIs and permutation nulls
# ----------------------------------------------------------------------

def permutation_test(
    A: np.ndarray,
    B: np.ndarray,
    stat_fn,
    *,
    n_perm: int = 1000,
    seed: int = 0,
) -> tuple[float, float]:
    """Label-permutation test for a two-sample statistic.

    Returns ``(observed, p_value)`` where the p-value is the fraction of
    permutations whose statistic is >= observed (with the standard +1
    correction). This is the load-bearing check at small n: it answers
    "would I see a separation this large if the group labels were
    meaningless?", which a bare distance cannot.
    """
    rng = np.random.default_rng(seed)
    observed = float(stat_fn(A, B))
    pooled = np.vstack([A, B])
    n_a = A.shape[0]
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(pooled.shape[0])
        shuffled = pooled[perm]
        if stat_fn(shuffled[:n_a], shuffled[n_a:]) >= observed:
            count += 1
    p = (count + 1) / (n_perm + 1)
    return observed, float(p)
