# Copyright (C) 2026 Ryan Pyle
# SPDX-License-Identifier: MIT
"""Figure 12: per-epoch mean influence magnitude + per-epoch mean loss,
across clean / adversarial / random-perturbation groups.

Top panel: mean_m |sum_n |P'[m,n]|| for each group per epoch.
Bottom panel: per-example CE loss averaged over each group per epoch.

Consumes ``per_epoch_group_metrics`` from ``data/mnist_adv_task_a.pt``.
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-a", type=str, default="data/mnist_adv_task_a.pt")
    ap.add_argument("--out", type=str, default="figures/fig12_per_epoch_influence_loss.png")
    args = ap.parse_args()

    art = torch.load(args.task_a, weights_only=False)
    mets = art["per_epoch_group_metrics"]
    if not mets:
        raise RuntimeError("Empty per_epoch_group_metrics.")

    epochs = np.arange(1, len(mets) + 1)
    def _series(k): return np.array([m[k] for m in mets])

    fig, (ax_i, ax_l) = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
    colors = {"clean": "#4c78a8", "adv": "#e45756", "rand": "#59a14f"}

    # Top: per-epoch influence
    for g in ("clean", "adv", "rand"):
        mu = _series(f"I_mean_{g}")
        sd = _series(f"I_sd_{g}")
        ax_i.plot(epochs, mu, color=colors[g], label=g, lw=1.4)
        ax_i.fill_between(epochs, mu - sd, mu + sd, alpha=0.18, color=colors[g])
    ax_i.set_ylabel("mean per-training-sample |influence|")
    ax_i.set_title("Per-epoch influence magnitude, by audit group")
    ax_i.grid(True, alpha=0.25)
    ax_i.legend(loc="best")

    # Bottom: per-epoch loss
    for g in ("clean", "adv", "rand"):
        mu = _series(f"loss_mean_{g}")
        sd = _series(f"loss_sd_{g}")
        ax_l.plot(epochs, mu, color=colors[g], label=g, lw=1.4)
        ax_l.fill_between(epochs, mu - sd, mu + sd, alpha=0.18, color=colors[g])
    ax_l.set_ylabel("mean per-example cross-entropy loss")
    ax_l.set_xlabel("epoch")
    ax_l.set_title("Per-epoch loss, by audit group")
    ax_l.grid(True, alpha=0.25)
    ax_l.legend(loc="best")

    fig.suptitle(
        "Fig 12: adversarial examples receive elevated influence throughout training\n"
        "yet loss plateaus (cancellation, cf. Fig 13)",
        fontsize=11,
    )
    fig.tight_layout()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200); plt.close(fig)
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
