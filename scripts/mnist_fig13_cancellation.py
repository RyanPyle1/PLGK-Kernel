# Copyright (C) 2026 Ryan Pyle
# SPDX-License-Identifier: MIT
"""Figure 13: cumulative net (S_n) vs gross (G_n) influence per group.

Per §6.3, define for each audit point n and epoch t:
    S_n(t) = sum_m P'_{m,n}(t)          (signed total)
    G_n(t) = sum_m |P'_{m,n}(t)|        (absolute magnitude)

Top panel: mean S_n across each group vs epoch.
Bottom panel: mean G_n across each group vs epoch.

If cancellation occurs, G_n grows while S_n plateaus — the paper's finding
for adversarial examples once the network enters the kernel regime.

Consumes ``per_epoch_pntk`` from ``data/mnist_adv_task_a.pt``.
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


def _group_positions(n_examples: int) -> dict[str, list[int]]:
    return {
        "clean": [16 * i + off for i in range(n_examples) for off in (0, 4, 8, 12)],
        "adv":   [16 * i + off for i in range(n_examples) for off in (1, 5, 9, 13)],
        "rand":  [16 * i + off for i in range(n_examples) for off in (2, 3, 6, 7, 10, 11, 14, 15)],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-a", type=str, default="data/mnist_adv_task_a.pt")
    ap.add_argument("--out", type=str, default="figures/fig13_cancellation.png")
    args = ap.parse_args()

    art = torch.load(args.task_a, weights_only=False)
    pntk_snaps = art["per_epoch_pntk"]
    if not pntk_snaps:
        raise RuntimeError("Task A artifact missing per_epoch_pntk.")

    n_epochs = len(pntk_snaps)
    n_audit = pntk_snaps[0].shape[1]
    n_examples = n_audit // 16
    positions = _group_positions(n_examples)
    epochs = np.arange(1, n_epochs + 1)

    # Compute S_n(t) and G_n(t) per group
    S = {g: np.zeros((n_epochs, len(positions[g]))) for g in positions}
    G = {g: np.zeros((n_epochs, len(positions[g]))) for g in positions}
    for t, PNTK in enumerate(pntk_snaps):
        Sn_all = PNTK.sum(dim=0).numpy()               # (n_audit,)
        Gn_all = PNTK.abs().sum(dim=0).numpy()         # (n_audit,)
        for g, cols in positions.items():
            S[g][t] = Sn_all[cols]
            G[g][t] = Gn_all[cols]

    fig, (ax_s, ax_g) = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
    colors = {"clean": "#4c78a8", "adv": "#e45756", "rand": "#59a14f"}

    for g in ("clean", "adv", "rand"):
        mu_s = S[g].mean(axis=1); sd_s = S[g].std(axis=1)
        mu_g = G[g].mean(axis=1); sd_g = G[g].std(axis=1)
        ax_s.plot(epochs, mu_s, color=colors[g], label=g, lw=1.4)
        ax_s.fill_between(epochs, mu_s - sd_s, mu_s + sd_s, alpha=0.18, color=colors[g])
        ax_g.plot(epochs, mu_g, color=colors[g], label=g, lw=1.4)
        ax_g.fill_between(epochs, mu_g - sd_g, mu_g + sd_g, alpha=0.18, color=colors[g])

    ax_s.set_ylabel(r"$S_n = \sum_m P'_{m,n}$  (signed)")
    ax_s.set_title(r"Cumulative net influence — plateau for adv indicates cancellation")
    ax_s.grid(True, alpha=0.25)
    ax_s.legend(loc="best")

    ax_g.set_ylabel(r"$G_n = \sum_m |P'_{m,n}|$  (gross)")
    ax_g.set_title("Cumulative gross influence — continues to grow for adv")
    ax_g.set_xlabel("epoch")
    ax_g.grid(True, alpha=0.25)
    ax_g.legend(loc="best")

    fig.suptitle(
        "Fig 13: net (top) vs gross (bottom) cumulative influence per group",
        fontsize=11,
    )
    fig.tight_layout()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200); plt.close(fig)
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
