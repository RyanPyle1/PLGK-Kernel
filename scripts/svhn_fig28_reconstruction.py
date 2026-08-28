# Copyright (C) 2026 Ryan Pyle
# SPDX-License-Identifier: MIT
"""Figure 28 (Appendix A.17.1): SVHN audit reconstruction over training.

Emits four training-curve panels; the paper's Fig 28 is the MRE + Corr
pair. The test-accuracy and training-loss panels are supplementary and are
not numbered figures in the paper.

Emits:
  fig28_svhn_reconstruction_mre.png    (paper's Fig 28: MRE)
  fig28_svhn_reconstruction_corr.png   (paper's Fig 28: Corr)
  svhn_testacc.png                     (supplementary)
  svhn_trainloss.png                   (supplementary)

Consumes ``data/svhn_experiment1.pt`` (from ``svhn_train_and_audit.py``).
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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifact", type=str, default="data/svhn_experiment1.pt")
    ap.add_argument("--out-dir", type=str, default="figures")
    args = ap.parse_args()

    art = torch.load(args.artifact, weights_only=False)
    testacc = np.asarray(art["testacc"])
    trainloss = np.asarray(art["trainloss"])
    logitcorrs = np.asarray(art["logitcorrs"])
    mles = np.asarray(art["mles"])
    epochs = np.arange(1, len(testacc) + 1)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    def _save(name: str, fig):
        p = out_dir / name
        fig.tight_layout()
        fig.savefig(p, dpi=200)
        plt.close(fig)
        print(f"Wrote: {p}")

    # Fig 28 — test accuracy
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.plot(epochs, testacc, color="#4c78a8", lw=1.5)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Test Accuracy")
    ax.set_title("SVHN — Test Accuracy over Training")
    ax.grid(True, alpha=0.25)
    _save("svhn_testacc.png", fig)

    # Fig 29 — training loss
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.plot(np.arange(len(trainloss)), trainloss, color="#4c78a8", lw=0.7)
    ax.set_xlabel("Iteration"); ax.set_ylabel("Training Loss")
    ax.set_title("SVHN — Training Loss over Training")
    ax.grid(True, alpha=0.25)
    _save("svhn_trainloss.png", fig)

    # Fig 30 — reconstruction correlation
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    corr_abs = np.abs(logitcorrs)
    ax.plot(epochs, corr_abs, color="#e45756", lw=1.5)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Correlation")
    ax.set_title("SVHN — Loss Reconstruction Correlation over Training")
    ymin = max(0.99, float(np.min(corr_abs)) - 1e-3)
    ax.set_ylim([ymin, 1.001])
    ax.grid(True, alpha=0.25)
    _save("fig28_svhn_reconstruction_corr.png", fig)

    # Fig 31 — mean loss reconstruction error
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.plot(epochs, mles, color="#54a24b", lw=1.5)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Mean Absolute Error")
    ax.set_title("SVHN — Mean Loss Reconstruction Error over Training")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    _save("fig28_svhn_reconstruction_mre.png", fig)

    print(f"\nFinal acc: {testacc[-1]:.4f}   Final MRE: {mles[-1]:.4g}   Final |Corr|: {np.abs(logitcorrs[-1]):.6f}")


if __name__ == "__main__":
    main()
