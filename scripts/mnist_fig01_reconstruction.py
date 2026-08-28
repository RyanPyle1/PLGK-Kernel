# Copyright (C) 2026 Ryan Pyle
# SPDX-License-Identifier: MIT
"""Figure 1: audit reconstruction MRE and loss correlation over training.

Consumes ``data/mnist_experiment1.pt`` (from ``mnist_train_and_audit.py``).

Writes:
  figures/fig01_reconstruction_mre.png
  figures/fig01_reconstruction_corr.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", type=str, default="data/mnist_experiment1.pt")
    ap.add_argument("--out-dir", type=str, default="figures")
    args = ap.parse_args()

    art = torch.load(args.artifact, weights_only=False)
    mles = np.asarray(art["mles"])
    logitcorrs = np.asarray(art["logitcorrs"])
    epochs = np.arange(1, len(mles) + 1)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # -- MRE panel
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.plot(epochs, mles, color="#4c78a8", lw=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean loss reconstruction error")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    p_mre = out_dir / "fig01_reconstruction_mre.png"
    fig.savefig(p_mre, dpi=200); plt.close(fig)

    # -- Corr panel (Y-axis zoom near 1)
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.plot(epochs, np.abs(logitcorrs), color="#e45756", lw=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss reconstruction correlation")
    ymin = max(0.999, float(np.min(np.abs(logitcorrs))) - 1e-4)
    ax.set_ylim([ymin, 1.0002])
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    p_corr = out_dir / "fig01_reconstruction_corr.png"
    fig.savefig(p_corr, dpi=200); plt.close(fig)

    print(f"Wrote: {p_mre}")
    print(f"       {p_corr}")
    print(f"Final MRE={mles[-1]:.4g}  Final |Corr|={np.abs(logitcorrs[-1]):.6f}")


if __name__ == "__main__":
    main()
