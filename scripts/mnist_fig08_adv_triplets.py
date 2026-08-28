# Copyright (C) 2026 Ryan Pyle
# SPDX-License-Identifier: MIT
"""Figure 8: sample clean / adversarial / random-perturbation image triplets.

Reads ``data/mnist_adv_data.pt`` (produced by ``mnist_adversarial_audit.py``)
and produces 3 rows × 4 columns of adv-data quadruplets: clean, adv, rand1,
rand2 for each of the 3 example blocks. Rows are the three test images the
paper illustrates in Fig 8; columns are the four attacks (FGSM, PGD,
BIM, DeepFool) — actually the paper shows clean/adv/rand1/rand2 per row.

We follow the paper's layout: 3 rows × 4 columns showing example i =
(0, 16, 32) → i.e. the first three "successful" examples, each shown with
its FGSM triplet (attack index 0, slot layout [clean, adv, rand1, rand2]).
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
    ap.add_argument("--adv-bundle", type=str, default="data/mnist_adv_data.pt")
    ap.add_argument("--out", type=str, default="figures/fig08_adv_triplets.png")
    ap.add_argument("--rows", type=int, default=3,
                    help="Number of example blocks to show (paper uses 3)")
    ap.add_argument("--attack-index", type=int, default=0,
                    help="0=FGSM, 1=PGD-Linf, 2=BIM-Linf, 3=DeepFool-Linf")
    args = ap.parse_args()

    bundle = torch.load(args.adv_bundle, weights_only=False)
    adv = bundle["adv_data"]  # (N, 1, 28, 28)

    # Slot layout: 16 slots per successful example, each of 4 attacks contributes
    # [clean, adv, rand1, rand2] starting at 4*attack_index within the 16-slot block.
    cols = ["clean", "adv", "rand1", "rand2"]

    fig, axes = plt.subplots(args.rows, 4, figsize=(9, 2.2 * args.rows))
    if args.rows == 1:
        axes = axes[None, :]
    for r in range(args.rows):
        block_start = r * 16 + 4 * args.attack_index
        for c in range(4):
            ax = axes[r, c]
            ax.imshow(adv[block_start + c].squeeze().numpy(), cmap="gray", vmin=0, vmax=1)
            if r == 0:
                ax.set_title(cols[c])
            ax.set_xticks([]); ax.set_yticks([]); ax.set_frame_on(False)
    fig.suptitle(
        f"Fig 8: clean / adversarial / random-perturbation triplets "
        f"(attack: {['FGSM','PGD','BIM','DeepFool'][args.attack_index]}-L∞)",
        fontsize=11,
    )
    fig.tight_layout()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200); plt.close(fig)
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
