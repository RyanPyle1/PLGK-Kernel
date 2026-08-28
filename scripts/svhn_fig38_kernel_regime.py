# Copyright (C) 2026 Ryan Pyle
# SPDX-License-Identifier: MIT
"""Figure 38 (Appendix A.17.3): SVHN kernel-regime metrics (Rich -> Lazy stability).

STATUS: NOT INCLUDED. This figure is the one item in the paper without a
producing script in this repository. The specification below is complete
enough to reimplement it; it is stated here rather than omitted so the gap
is explicit.

The figure requires a separate re-training pass that tracks feature,
activation, and subspace overlap of the train-side loss-gradient matrix Phi
and the hidden-representation matrix H across epochs, each measured against
three reference frames (t=0, t-1, and T_final).

Per epoch:
  * feature_overlap_prev[t]   = <Phi_t, Phi_{t-1}>_F / (||Phi_t||_F ||Phi_{t-1}||_F)
  * feature_overlap_0[t]      = same, against Phi_0
  * feature_overlap_T[t]      = same, against Phi_final
  * subspace_overlap_{prev,0,T} = top-K right-singular-subspace overlap,
                                  obtainable from ``lowrank_svd_train_side``
  * activation_overlap_{prev,0,T} = the feature-overlap formula applied to H,
                                    the last hidden layer's activations

The MNIST counterpart, ``mnist_fig14_kernel_regime.py``, computes the same
family of metrics and is the natural template.

This diagnostic must stay separate from the Task-A audit driver so that
cross-run comparability is preserved, and costs roughly 2x the audit
driver's wall time.
"""
raise SystemExit(
    "\n[svhn_fig38_kernel_regime] Not included in this repository.\n"
    "This figure needs a separate re-training pass with Phi/H/subspace\n"
    "overlap tracking. The module docstring gives the full metric\n"
    "specification; scripts/mnist_fig14_kernel_regime.py is the template.\n"
)
