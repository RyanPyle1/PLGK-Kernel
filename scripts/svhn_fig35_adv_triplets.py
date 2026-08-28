# Copyright (C) 2026 Ryan Pyle
# SPDX-License-Identifier: MIT
"""Figure 35 (Appendix A.17.3): SVHN clean / adversarial / random triplets.

Analogous to MNIST Fig 8 but on the SVHN adversarial bundle. SVHN images
are RGB and were normalized by the loader (with the SVHN train-split mean/
std), so we per-image min-max renormalize for display.

Layout: ``--rows`` example blocks × [clean, adv, rand1, rand2].
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


def _img_for_display(chw: torch.Tensor) -> np.ndarray:
    arr = chw.detach().cpu().numpy()
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    lo, hi = float(np.min(arr)), float(np.max(arr))
    if hi > lo:
        arr = (arr - lo) / (hi - lo)
    return arr.squeeze()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adv-bundle", type=str, default="data/svhn_adv_data.pt")
    ap.add_argument("--out", type=str, default="figures/fig35_svhn_adv_triplets.png")
    ap.add_argument("--rows", type=int, default=3,
                    help="Number of example blocks to show (paper uses 3)")
    ap.add_argument("--attack-index", type=int, default=0,
                    help="0=FGSM, 1=PGD-Linf, 2=BIM-Linf, 3=DeepFool-Linf")
    args = ap.parse_args()

    bundle = torch.load(args.adv_bundle, weights_only=False)
    adv = bundle["adv_data"]  # (N, 3, 32, 32)

    cols = ["clean", "adv", "rand1", "rand2"]

    fig, axes = plt.subplots(args.rows, 4, figsize=(9, 2.4 * args.rows))
    if args.rows == 1:
        axes = axes[None, :]
    for r in range(args.rows):
        block_start = r * 16 + 4 * args.attack_index
        for c in range(4):
            ax = axes[r, c]
            img = _img_for_display(adv[block_start + c])
            if img.ndim == 2:
                ax.imshow(img, cmap="gray")
            else:
                ax.imshow(img)
            if r == 0:
                ax.set_title(cols[c])
            ax.set_xticks([]); ax.set_yticks([]); ax.set_frame_on(False)
    fig.suptitle(
        f"Fig 35: SVHN — clean / adversarial / random-perturbation triplets "
        f"(attack: {['FGSM','PGD','BIM','DeepFool'][args.attack_index]}-Linf)",
        fontsize=11,
    )
    fig.tight_layout()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200); plt.close(fig)
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
