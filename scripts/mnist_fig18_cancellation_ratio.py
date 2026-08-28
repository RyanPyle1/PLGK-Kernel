# Copyright (C) 2026 Ryan Pyle
# SPDX-License-Identifier: MIT
"""Figure 18: modal cancellation ratio CR(K, t) over training + across K.

Paper's Section 6.3.1 definition:
    CR(K, t) = |Σ_{k=1..K} Λ_k(t) C_{k,n}(t)| / Σ_{k=1..K} |Λ_k(t) C_{k,n}(t)|

CR=1 → no cancellation (all mode contributions share sign);
CR≈0 → near-perfect cancellation.

Two panels (paper layout):
  Top:  CR(K=K_max, t) vs training epoch, mean±sd per group.
  Bot:  CR(K, t=final) vs number of retained modes K, mean±sd per group.

Consumes ``data/mnist_expc_svd_full.pt`` (Group 3 snapshot artifact).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def _band(ax, x, mu, sd, color, label):
    ax.plot(x, mu, color=color, lw=1.4, label=label)
    ax.fill_between(x, mu - sd, mu + sd, alpha=0.18, color=color)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--svd-full", type=str, default="data/mnist_expc_svd_full.pt")
    ap.add_argument("--out", type=str, default="figures/fig18_cancellation_ratio.png")
    ap.add_argument("--K-grid-log-points", type=int, default=40)
    args = ap.parse_args()

    art = torch.load(args.svd_full, weights_only=False)
    S_per_epoch = art["S_per_epoch"]
    a_per_epoch = art["a_per_epoch"]
    C_signed = {
        "clean": art["C_signed_clean_per_epoch"],
        "adv":   art["C_signed_adv_per_epoch"],
        "rand":  art["C_signed_rand_per_epoch"],
    }
    epochs = np.asarray(art["epochs"])
    K = art["K"]
    n_epochs = len(epochs)
    print(f"epochs={n_epochs}  K={K}")

    # Per (t, group) compute mean CR(K_max, t) across examples of the group.
    eps = 1e-12
    CR_time = {g: np.zeros((n_epochs,)) for g in C_signed}
    CR_time_sd = {g: np.zeros((n_epochs,)) for g in C_signed}
    for t in range(n_epochs):
        S = S_per_epoch[t].numpy()
        a = a_per_epoch[t].numpy()
        Lambda = S * a                                        # (K,)
        for g, mats in C_signed.items():
            C_g = mats[t].cpu().numpy() if hasattr(mats[t], "cpu") else np.asarray(mats[t])  # (n_g, K)
            weighted = C_g * Lambda[None, :]                  # (n_g, K)
            num = np.abs(weighted.sum(axis=1))
            denom = np.abs(weighted).sum(axis=1) + eps
            ratio = num / denom
            CR_time[g][t] = ratio.mean()
            CR_time_sd[g][t] = ratio.std()

    # CR(K, t=final) sweep
    K_grid = np.unique(
        np.round(np.geomspace(1, K, args.K_grid_log_points)).astype(int)
    )
    CR_K = {g: np.zeros(len(K_grid)) for g in C_signed}
    CR_K_sd = {g: np.zeros(len(K_grid)) for g in C_signed}
    S_final = S_per_epoch[-1].numpy()
    a_final = a_per_epoch[-1].numpy()
    Lambda_final = S_final * a_final
    for g, mats in C_signed.items():
        C_g = mats[-1].cpu().numpy() if hasattr(mats[-1], "cpu") else np.asarray(mats[-1])
        weighted = C_g * Lambda_final[None, :]
        for j, Kj in enumerate(K_grid):
            num = np.abs(weighted[:, :Kj].sum(axis=1))
            denom = np.abs(weighted[:, :Kj]).sum(axis=1) + eps
            ratio = num / denom
            CR_K[g][j] = ratio.mean()
            CR_K_sd[g][j] = ratio.std()

    fig, (ax_t, ax_k) = plt.subplots(2, 1, figsize=(7, 6))
    colors = {"clean": "#4c78a8", "adv": "#e45756", "rand": "#59a14f"}
    for g in ("clean", "adv", "rand"):
        _band(ax_t, epochs + 1, CR_time[g], CR_time_sd[g], colors[g], g)
        _band(ax_k, K_grid,     CR_K[g],   CR_K_sd[g],   colors[g], g)
    ax_t.set_xlabel("epoch")
    ax_t.set_ylabel(f"CR(K={K}, t)")
    ax_t.set_title(f"Top: cancellation ratio over training (K = {K})")
    ax_t.set_ylim(-0.05, 1.05)
    ax_t.grid(True, alpha=0.25); ax_t.legend(loc="best")

    ax_k.set_xlabel("K (modes retained)")
    ax_k.set_ylabel(f"CR(K, t=final)")
    ax_k.set_title("Bottom: cancellation ratio at final epoch vs mode count")
    ax_k.set_xscale("log")
    ax_k.set_ylim(-0.05, 1.05)
    ax_k.grid(True, alpha=0.25, which="both"); ax_k.legend(loc="best")

    fig.suptitle(
        "Fig 18: adversarials show stronger cancellation once kernel regime engages\n"
        "(low CR = mode contributions cancel; saturation around K ≈ 125 in paper)",
        fontsize=11,
    )
    fig.tight_layout()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200); plt.close(fig)
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
