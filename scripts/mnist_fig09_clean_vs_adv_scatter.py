# Copyright (C) 2026 Ryan Pyle
# SPDX-License-Identifier: MIT
"""Figure 9: per-training-sample influence, clean vs adversarial.

For each training subject m, we compute:
  - influence-on-clean-audit = mean over "clean" audit slots of |PNTK[m, slot]|
  - influence-on-adv-audit   = mean over "adv"   audit slots of |PNTK[m, slot]|
and scatter one against the other. The paper's finding (Section 6.3,
correlation > 0.6): adversarial influences are broadly ~linear scalings
of clean influences, not a re-concentration on a few points.

Consumes ``data/mnist_adv_task_a.pt`` (Task A adv audit).
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
    ap.add_argument("--out", type=str, default="figures/fig09_clean_vs_adv_scatter.png")
    ap.add_argument("--attack-labels", type=str, default="FGSM,PGD,BIM,DeepFool")
    args = ap.parse_args()

    art = torch.load(args.task_a, weights_only=False)
    PNTK = art["PNTK"]
    if PNTK is None:
        raise RuntimeError("Task A artifact missing PNTK.")
    PNTK_abs = PNTK.abs().numpy()   # (n_train, n_audit)
    n_audit = PNTK.shape[1]
    n_examples = n_audit // 16      # 16 slots per successful example (paper's default)
    print(f"n_train={PNTK.shape[0]}, n_audit={n_audit}, n_examples={n_examples}")

    # Slot layout per example (16 slots): 4 attacks × [clean, adv, rand1, rand2]
    # → clean slots: {4*a + 0 for a in 0..3}
    # → adv   slots: {4*a + 1 for a in 0..3}
    # → rand  slots: {4*a + 2 or 3 for a in 0..3}
    clean_offsets = [4 * a + 0 for a in range(4)]
    adv_offsets = [4 * a + 1 for a in range(4)]

    clean_cols = [16 * i + off for i in range(n_examples) for off in clean_offsets]
    adv_cols = [16 * i + off for i in range(n_examples) for off in adv_offsets]

    infl_clean = PNTK_abs[:, clean_cols].mean(axis=1)
    infl_adv = PNTK_abs[:, adv_cols].mean(axis=1)

    corr = float(np.corrcoef(infl_clean, infl_adv)[0, 1])
    slope, intercept = np.polyfit(infl_clean, infl_adv, 1)
    print(f"Pearson corr(|clean|, |adv|) = {corr:.4f}")
    print(f"Best-fit slope (linear scaling): {slope:.3f}")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(infl_clean, infl_adv, s=8, alpha=0.35, color="#4c78a8", edgecolor="none")
    # Diagonal + best-fit line
    lim = max(infl_clean.max(), infl_adv.max()) * 1.05
    ax.plot([0, lim], [0, lim], ls="--", c="black", lw=0.7, alpha=0.6, label="y = x")
    xf = np.linspace(0, lim, 100)
    ax.plot(xf, slope * xf + intercept, ls="-", c="#e45756", lw=1.2,
             label=f"fit slope {slope:.2f}, corr {corr:.3f}")
    ax.set_xlabel("|clean-example influence|  (per training sample, mean over clean audit slots)")
    ax.set_ylabel("|adversarial-example influence|  (per training sample, mean over adv slots)")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200); plt.close(fig)
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
