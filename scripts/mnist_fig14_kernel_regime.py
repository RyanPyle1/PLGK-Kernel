# Copyright (C) 2026 Ryan Pyle
# SPDX-License-Identifier: MIT
"""Figure 14: kernel-regime metrics vs influence/loss over training.

Three "kernel-regime" metrics (paper §6.3, Section 6.3.1 setup) computed
in the Group 3 snapshot driver (each epoch's value = cosine vs FINAL epoch):

  feat_overlap      — mean cos(feat_t[i], feat_T[i]) over train samples
  subspace_overlap  — mean |cos(V_top10[k], V_top10_T[k])| over top-10 modes
  act_overlap       — mean cos(hidden_t[i], hidden_T[i]) over adv-audit samples

All three saturate near 1.0 when the network enters the kernel regime.
Paper's finding: all three approach 1 between epochs 5 and 10, coinciding
with the knee in loss and the peak of clean/random gross-influence sums.

Two panels:
  Left:  three metrics (left y-axis) vs group influence (right y-axis)
  Right: three metrics (left y-axis) vs group losses (right y-axis)

Consumes ``data/mnist_expc_svd_full.pt``.
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
    ap.add_argument("--svd-full", type=str, default="data/mnist_expc_svd_full.pt")
    ap.add_argument("--out", type=str, default="figures/fig14_kernel_regime.png")
    args = ap.parse_args()

    art = torch.load(args.svd_full, weights_only=False)
    epochs = np.asarray(art["epochs"]) + 1
    km = art["kernel_regime"]
    feat = np.asarray(km["feat_overlap"])
    sub = np.asarray(km["subspace_overlap"])
    act = np.asarray(km["act_overlap"])
    L = art["loss_by_group"]; I = art["I_by_group"]

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)

    # Left panel: metrics + influence
    ax_l.plot(epochs, feat, color="#4c78a8", label="feat overlap", lw=1.3)
    ax_l.plot(epochs, sub,  color="#f58518", label="subspace overlap (top-10)", lw=1.3)
    ax_l.plot(epochs, act,  color="#54a24b", label="activation overlap", lw=1.3)
    ax_l.set_xlabel("epoch"); ax_l.set_ylabel("overlap vs final epoch")
    ax_l.set_ylim(-0.05, 1.05); ax_l.grid(True, alpha=0.25)
    ax_l.legend(loc="upper left")
    ax_lr = ax_l.twinx()
    for g, c in [("clean", "#b279a2"), ("adv", "#e45756"), ("rand", "#59a1cc")]:
        mu = np.asarray(I[f"{g}_mean"]); sd = np.asarray(I[f"{g}_sd"])
        ax_lr.plot(epochs, mu, color=c, ls="--", lw=1.2, label=f"{g} |infl|")
        ax_lr.fill_between(epochs, mu - sd, mu + sd, color=c, alpha=0.15)
    ax_lr.set_ylabel(r"per-sample $\overline{|\mathrm{infl}|}$")
    ax_lr.legend(loc="upper right", fontsize=8)
    ax_l.set_title("(a) kernel-regime metrics vs influence")

    # Right panel: metrics + losses
    ax_r.plot(epochs, feat, color="#4c78a8", label="feat overlap", lw=1.3)
    ax_r.plot(epochs, sub,  color="#f58518", label="subspace overlap (top-10)", lw=1.3)
    ax_r.plot(epochs, act,  color="#54a24b", label="activation overlap", lw=1.3)
    ax_r.set_xlabel("epoch"); ax_r.set_ylabel("overlap vs final epoch")
    ax_r.set_ylim(-0.05, 1.05); ax_r.grid(True, alpha=0.25)
    ax_r.legend(loc="upper left")
    ax_rr = ax_r.twinx()
    for g, c in [("clean", "#b279a2"), ("adv", "#e45756"), ("rand", "#59a1cc")]:
        mu = np.asarray(L[f"{g}_mean"]); sd = np.asarray(L[f"{g}_sd"])
        ax_rr.plot(epochs, mu, color=c, ls="--", lw=1.2, label=f"{g} loss")
        ax_rr.fill_between(epochs, mu - sd, mu + sd, color=c, alpha=0.15)
    ax_rr.set_ylabel("mean CE loss")
    ax_rr.legend(loc="upper right", fontsize=8)
    ax_r.set_title("(b) kernel-regime metrics vs loss")

    fig.suptitle(
        "Fig 14: as the three overlap metrics approach 1 (network enters kernel regime),\n"
        "clean/random influence peaks and losses reach their knee; adversarial loss then plateaus.",
        fontsize=11,
    )
    fig.tight_layout()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200); plt.close(fig)
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
