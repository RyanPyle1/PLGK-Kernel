# Copyright (C) 2026 Ryan Pyle
# SPDX-License-Identifier: MIT
"""Figure 37 (Appendix A.17.3): SVHN per-epoch influence magnitude + loss,
by clean / adv / random audit group.

Top panel: mean per-training-sample |influence|, averaged across each
audit group's slots, per epoch.
Bottom panel: per-example CE loss averaged across each group's slots.

Directly mirrors MNIST Fig 12; the underlying ``per_epoch_group_metrics``
list has the same schema.

Consumes ``data/svhn_adv_task_a.pt`` (from ``svhn_adversarial_audit.py``).
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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task-a", type=str, default="data/svhn_adv_task_a.pt")
    ap.add_argument("--out", type=str, default="figures/fig37_svhn_per_epoch_influence_loss.png")
    args = ap.parse_args()

    art = torch.load(args.task_a, weights_only=False)
    mets = art["per_epoch_group_metrics"]
    if not mets:
        raise RuntimeError("Empty per_epoch_group_metrics.")

    epochs = np.arange(1, len(mets) + 1)
    def _series(k): return np.array([m[k] for m in mets])

    fig, (ax_i, ax_l) = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
    colors = {"clean": "#4c78a8", "adv": "#e45756", "rand": "#59a14f"}

    for g in ("clean", "adv", "rand"):
        mu = _series(f"I_mean_{g}"); sd = _series(f"I_sd_{g}")
        ax_i.plot(epochs, mu, color=colors[g], label=g, lw=1.4)
        ax_i.fill_between(epochs, mu - sd, mu + sd, alpha=0.18, color=colors[g])
    ax_i.set_ylabel("mean per-training-sample |influence|")
    ax_i.set_title("SVHN — per-epoch influence magnitude, by audit group")
    ax_i.grid(True, alpha=0.25)
    ax_i.legend(loc="best")

    for g in ("clean", "adv", "rand"):
        mu = _series(f"loss_mean_{g}"); sd = _series(f"loss_sd_{g}")
        ax_l.plot(epochs, mu, color=colors[g], label=g, lw=1.4)
        ax_l.fill_between(epochs, mu - sd, mu + sd, alpha=0.18, color=colors[g])
    ax_l.set_ylabel("mean per-example CE loss")
    ax_l.set_xlabel("epoch")
    ax_l.set_title("SVHN — per-epoch loss, by audit group")
    ax_l.grid(True, alpha=0.25)
    ax_l.legend(loc="best")

    fig.suptitle(
        "Fig 37: SVHN — adversarial examples receive elevated influence throughout training\n"
        "yet loss plateaus (cancellation)",
        fontsize=11,
    )
    fig.tight_layout()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200); plt.close(fig)
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
