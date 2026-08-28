# Copyright (C) 2026 Ryan Pyle
# SPDX-License-Identifier: MIT
"""Figures 39, 40 (Appendix A.17.3): SVHN mode activity.

At the final training checkpoint we take the train-side SVD Phi_train = U S V^T
(rank K = --svd-k in the driver, paper: 128). For each mode k we compute
C_k[audit_sample] = <V_k, grad_theta L(x_audit_sample)>.

--compare adv   → Fig 39: mean |C_k| adv vs mean |C_k| clean
--compare rand  → Fig 40: mean |C_k| random vs mean |C_k| clean

The paper shows scatter of (sigma_k, a_k) colored by Delta = |C_k^X| - |C_k^clean|,
demonstrating that adversarial mass shifts into small-sigma modes while random
perturbations do not shift as sharply.

Consumes ``data/svhn_adv_task_a.pt`` (produced by ``svhn_adversarial_audit.py``
Step 5, which emits the ``svd_final`` block).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kernel_tools.console import enable_unicode_stdout

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def main() -> None:
    enable_unicode_stdout()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task-a", type=str, default="data/svhn_adv_task_a.pt")
    ap.add_argument("--compare", choices=["adv", "rand"], default="adv",
                    help="Compare clean against adv (Fig 39) or random (Fig 40).")
    ap.add_argument("--out", type=str, default=None,
                    help="Auto-picked from --compare if omitted.")
    ap.add_argument("--rank-tol", type=float, default=1e-3,
                    help="Mask modes with sigma < rank_tol * sigma_max (paper: 1e-3)")
    args = ap.parse_args()
    if args.out is None:
        args.out = {
            "adv":  "figures/fig39_svhn_mode_activity_adv_vs_clean.png",
            "rand": "figures/fig40_svhn_mode_activity_rand_vs_clean.png",
        }[args.compare]

    art = torch.load(args.task_a, weights_only=False)
    if "svd_final" not in art:
        raise RuntimeError(
            "Task A artifact missing 'svd_final'. Re-run svhn_adversarial_audit.py "
            "(its Step 5 emits svd_final block)."
        )
    svd = art["svd_final"]
    S = svd["S"].numpy()      # (K,)
    a = svd["a"].numpy()      # (K,)   per-mode scalar coefficient
    C = svd["C_audit"].numpy()  # (n_audit, K)
    K = svd["K"]
    n_audit = C.shape[0]
    n_ex = n_audit // 16
    print(f"K={K}   sigma range [{S.min():.4g}, {S.max():.4g}]   audit slots={n_audit}")

    # Slot layout — mirror driver
    clean_positions = [16 * i + off for i in range(n_ex) for off in (0, 4, 8, 12)]
    adv_positions   = [16 * i + off for i in range(n_ex) for off in (1, 5, 9, 13)]
    rand_positions  = [16 * i + off for i in range(n_ex) for off in (2, 3, 6, 7, 10, 11, 14, 15)]

    C_abs = np.abs(C)
    C_clean = C_abs[clean_positions].mean(axis=0)  # (K,)
    C_adv   = C_abs[adv_positions].mean(axis=0)
    C_rand  = C_abs[rand_positions].mean(axis=0)
    if args.compare == "adv":
        C_other = C_adv;  other_label = "adv";  fig_label = "Fig 39 — adv vs clean"
    else:
        C_other = C_rand; other_label = "rand"; fig_label = "Fig 40 — rand vs clean"
    delta = C_other - C_clean  # positive -> perturbation activates mode more

    # Rank mask: drop very small-sigma modes as unreliable
    mask = S >= args.rank_tol * S.max()
    Sm, am, dm = S[mask], a[mask], delta[mask]
    print(f"kept {mask.sum()}/{K} modes after rank_tol={args.rank_tol}")

    # Marker size proportional to |Delta|
    size = 12 + 200 * (np.abs(dm) / max(np.max(np.abs(dm)), 1e-12))

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    vmax = float(np.max(np.abs(dm)))
    sc = ax.scatter(Sm, am, c=dm, s=size, cmap="RdBu_r",
                     vmin=-vmax, vmax=vmax, edgecolor="black", linewidth=0.3)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\sigma_k$  (singular value)")
    ax.set_ylabel(r"$a_k$  (mode coefficient in loss-gradient direction)")
    ax.set_title(
        f"SVHN — mode activity: {other_label} vs clean ({fig_label})\n"
        rf"color: $\Delta = \langle|C_k|\rangle_{{\rm {other_label}}} - \langle|C_k|\rangle_{{\rm clean}}$;   "
        r"marker size $\propto |\Delta|$"
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(rf"$\Delta$ ({other_label} - clean)")
    ax.grid(True, alpha=0.25, which="both")

    fig.tight_layout()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200); plt.close(fig)
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
