# Copyright (C) 2026 Ryan Pyle
# SPDX-License-Identifier: MIT
"""Figure 11: temporal contributions + total influence along the adv ray.

Task B audits ``adv_data2`` = 128 points linearly interpolated from a clean
image (α=0) to its adversarial counterpart (α=1), followed by 128 points on
a ray from the same clean image to a matched-eps random perturbation.

Left panel: per-α curves of (per-epoch influence magnitude) / (total per-α
final influence) for a set of α values along the adversarial ray. Shows
that early-training influence dominates near α=0 (clean) but late-training
influence dominates near α=1 (adversarial).

Right panel: total influence magnitude summed over training, as a function
of α — for both the adv ray and the random ray. Shows super-linear increase
toward the adversarial endpoint (paper's finding).

Consumes ``per_epoch_pntk`` from ``data/mnist_adv_task_b.pt``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm  # noqa: F401 — ensures matplotlib.colormaps is populated
import numpy as np
import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-b", type=str, default="data/mnist_adv_task_b.pt")
    ap.add_argument("--out", type=str, default="figures/fig11_perturbation_ray.png")
    ap.add_argument("--n-alpha-lines", type=int, default=8,
                    help="Number of alpha positions to trace in the left panel")
    args = ap.parse_args()

    art = torch.load(args.task_b, weights_only=False)
    if "per_epoch_pntk" not in art or not art["per_epoch_pntk"]:
        raise RuntimeError(
            "Task B artifact missing per_epoch_pntk. Rerun mnist_adversarial_audit.py "
            "without --skip-task-b, and be sure step/epoch callbacks are active."
        )
    pntk_snaps = art["per_epoch_pntk"]
    n_epochs = len(pntk_snaps)
    n_audit = pntk_snaps[0].shape[1]
    half = n_audit // 2
    print(f"epochs={n_epochs}  n_audit={n_audit}  alpha_steps per ray = {half}")

    # gross influence per (epoch, audit_slot) = sum_m |P'_{m,slot}|
    G = np.stack(
        [pntk.abs().sum(dim=0).numpy() for pntk in pntk_snaps], axis=0
    )  # (n_epochs, n_audit)

    epochs = np.arange(1, n_epochs + 1)
    adv_ray_G = G[:, :half]       # (n_epochs, half)
    rand_ray_G = G[:, half:]      # (n_epochs, half)

    # Total per-alpha over full training (summed over epochs)
    total_adv_per_alpha = adv_ray_G.sum(axis=0)     # (half,)
    total_rand_per_alpha = rand_ray_G.sum(axis=0)   # (half,)

    # For the temporal-fraction lines: pick K alpha positions evenly along adv ray.
    K = min(args.n_alpha_lines, half)
    alpha_positions = np.linspace(0, half - 1, K, dtype=int)
    alphas = alpha_positions / max(half - 1, 1)     # 0..1 fraction along ray

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: per-alpha, per-epoch fractional contribution
    cmap = matplotlib.colormaps["viridis"]
    for k, apos in enumerate(alpha_positions):
        total = max(total_adv_per_alpha[apos], 1e-30)
        frac = adv_ray_G[:, apos] / total
        color = cmap(k / max(K - 1, 1))
        ax_l.plot(epochs, frac * 100.0, color=color, lw=1.3, label=f"α={alphas[k]:.2f}")
    ax_l.set_xlabel("epoch")
    ax_l.set_ylabel("per-epoch influence, as % of total")
    ax_l.set_title("(a) temporal contribution along the adv ray")
    ax_l.grid(True, alpha=0.25)
    ax_l.legend(fontsize=7, loc="upper right", title="ray position")

    # Right: total influence magnitude vs alpha (both rays)
    alpha_axis = np.linspace(0, 1, half)
    ax_r.plot(alpha_axis, total_adv_per_alpha, color="#e45756", lw=1.6, label="adv ray")
    ax_r.plot(alpha_axis, total_rand_per_alpha, color="#59a14f", lw=1.4, label="random ray", ls="--")
    ax_r.set_xlabel("α  (0 = clean,  1 = ray endpoint)")
    ax_r.set_ylabel(r"total influence magnitude $\sum_t G_n(t)$")
    ax_r.set_title("(b) total influence vs ray position")
    ax_r.grid(True, alpha=0.25)
    ax_r.legend(loc="best")

    fig.suptitle(
        "Fig 11: influence along interpolation ray from clean → adversarial\n"
        "left: early-training dominates near clean, late-training dominates near adv;  "
        "right: super-linear rise toward adv endpoint.",
        fontsize=10,
    )
    fig.tight_layout()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200); plt.close(fig)
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
