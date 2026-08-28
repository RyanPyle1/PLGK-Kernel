# Copyright (C) 2026 Ryan Pyle
# SPDX-License-Identifier: MIT
"""Figure 10: per-layer × per-epoch influence decomposition on adv/clean/random.

For each of the 5 LeNet-5 layers (conv1, conv2, fc1, fc2, fc3) and each
epoch, show:
  Top row:    signed sum of influence per audit-group (clean/adv/rand),
              averaged across group members.
  Bottom row: absolute sum, same protocol.

Two rows × 5 layer-columns × 3 group-lines-per-panel = 10 panels.

Consumes ``data/mnist_adv_task_a.pt`` (contains ``per_epoch_layer_contribs``
list of dicts, one per epoch; each dict maps layer_name → (2, n_audit) with
row 0 = signed_sum, row 1 = abs_sum).

Slot layout in the adv audit set (matches ``build_adversarial_audit_sets``):
  within each 16-slot block for a successful example:
    offsets 0,4,8,12  = clean
    offsets 1,5,9,13  = adversarial
    offsets 2,3,6,7,10,11,14,15 = random
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
    ap.add_argument("--out", type=str, default="figures/fig10_per_layer_temporal.png")
    args = ap.parse_args()

    art = torch.load(args.task_a, weights_only=False)
    contribs = art["per_epoch_layer_contribs"]
    layers = art["layer_prefixes"]
    if not contribs:
        raise RuntimeError("Task A artifact has empty per_epoch_layer_contribs.")

    n_epochs = len(contribs)
    n_audit = contribs[0][layers[0]].shape[1]
    n_examples = n_audit // 16
    print(f"epochs={n_epochs}  layers={layers}  n_audit={n_audit}  n_examples={n_examples}")

    positions = _group_positions(n_examples)
    epochs = np.arange(1, n_epochs + 1)

    # For each (layer, epoch, group) compute mean+sd across the group's audit slots,
    # for both signed and abs.
    group_colors = {"clean": "#4c78a8", "adv": "#e45756", "rand": "#59a14f"}
    fig, axes = plt.subplots(2, len(layers), figsize=(3 * len(layers), 5),
                              sharex=True)
    if len(layers) == 1:
        axes = axes[:, None]

    for li, layer in enumerate(layers):
        signed = np.zeros((n_epochs, 3))  # mean per group
        signed_sd = np.zeros((n_epochs, 3))
        absum = np.zeros((n_epochs, 3))
        absum_sd = np.zeros((n_epochs, 3))
        for ep in range(n_epochs):
            c = contribs[ep][layer]  # (2, n_audit)
            for gi, g in enumerate(("clean", "adv", "rand")):
                cols = positions[g]
                signed[ep, gi] = c[0, cols].mean().item()
                signed_sd[ep, gi] = c[0, cols].std().item()
                absum[ep, gi] = c[1, cols].mean().item()
                absum_sd[ep, gi] = c[1, cols].std().item()

        for gi, g in enumerate(("clean", "adv", "rand")):
            axes[0, li].plot(epochs, signed[:, gi], color=group_colors[g], label=g, lw=1.2)
            axes[0, li].fill_between(
                epochs, signed[:, gi] - signed_sd[:, gi], signed[:, gi] + signed_sd[:, gi],
                alpha=0.15, color=group_colors[g],
            )
            axes[1, li].plot(epochs, absum[:, gi], color=group_colors[g], label=g, lw=1.2)
            axes[1, li].fill_between(
                epochs, absum[:, gi] - absum_sd[:, gi], absum[:, gi] + absum_sd[:, gi],
                alpha=0.15, color=group_colors[g],
            )
        axes[0, li].set_title(layer)
        axes[0, li].grid(True, alpha=0.25)
        axes[1, li].grid(True, alpha=0.25)
        axes[1, li].set_xlabel("epoch")

    axes[0, 0].set_ylabel("signed sum of influence")
    axes[1, 0].set_ylabel(r"$\sum |$influence$|$")
    axes[0, 0].legend(loc="best", fontsize=8)

    fig.suptitle(
        "Fig 10: per-layer × per-epoch influence on adv/clean/random audit groups\n"
        "top = signed net contribution; bottom = absolute magnitude",
        fontsize=11,
    )
    fig.tight_layout()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200); plt.close(fig)
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
