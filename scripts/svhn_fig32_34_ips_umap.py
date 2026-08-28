# Copyright (C) 2026 Ryan Pyle
# SPDX-License-Identifier: MIT
"""SVHN IPS UMAP figures (Appendix A.17.2), companion to MNIST Figs 5-6.

The paper covers three UMAP panels on SVHN:
  * colored by true class (all 10 digits)
  * colored by predicted class
  * colored by misclassification (correct vs wrong overlaid)

Filenames use ``svhn_fig`` prefix so they don't collide with the MNIST
Fig 5 / 6 outputs in the same ``figures/`` directory.

Consumes ``data/svhn_experiment1.pt`` produced with ``--do-ips``.

Wall time: UMAP fit ~2-4 min for the paper's default point set. The fit
is cached to ``figures/_svhn_umap_cache.npy`` for reuse.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import torch

from kernel_tools import umap_from_ips


def _load_or_fit_umap(IPS, cache_path: Path, seed: int) -> np.ndarray:
    if cache_path.exists():
        X = np.load(cache_path)
        if X.shape[0] == IPS.shape[0]:
            print(f"Loaded cached UMAP: {cache_path}")
            return X
    print("Fitting UMAP (this takes a few minutes)...")
    X = umap_from_ips(IPS, random_state=seed)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, X)
    print(f"Cached UMAP: {cache_path}")
    return X


def _to_np(x):
    return np.asarray(x.data if hasattr(x, "data") else x)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifact", type=str, default="data/svhn_experiment1.pt")
    ap.add_argument("--out-dir", type=str, default="figures")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    art = torch.load(args.artifact, weights_only=False)
    if art.get("IPS") is None:
        raise RuntimeError(
            "Artifact missing IPS. Rerun svhn_train_and_audit.py with --do-ips."
        )
    IPS = _to_np(art["IPS"])
    y_target_full = _to_np(art["y_target_full"]).astype(int)
    y_pred_full = _to_np(art["y_pred_full"]).astype(int)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    X_umap = _load_or_fit_umap(IPS, out_dir / "_svhn_umap_cache.npy", args.seed)

    N = 10
    cmap = ListedColormap(matplotlib.colormaps["tab10"].colors[:N])
    bounds = np.arange(-0.5, N + 0.5, 1)
    norm = BoundaryNorm(bounds, N)

    # -- panel: colored by true class --
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(X_umap[:, 0], X_umap[:, 1], s=8, c=y_target_full, cmap=cmap, norm=norm)
    ax.set_title("SVHN — UMAP of IPS similarity (by true class)")
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    cbar = fig.colorbar(sc, ax=ax, ticks=np.arange(N))
    cbar.set_label("Class")
    fig.tight_layout()
    p = out_dir / "svhn_fig_ips_umap_by_class.png"
    fig.savefig(p, dpi=200); plt.close(fig)
    print(f"Wrote: {p}")

    # -- panel: colored by predicted class --
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(X_umap[:, 0], X_umap[:, 1], s=8, c=y_pred_full, cmap=cmap, norm=norm)
    ax.set_title("SVHN — UMAP of IPS similarity (by predicted class)")
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    cbar = fig.colorbar(sc, ax=ax, ticks=np.arange(N))
    cbar.set_label("Predicted class")
    fig.tight_layout()
    p = out_dir / "svhn_fig_ips_umap_by_pred.png"
    fig.savefig(p, dpi=200); plt.close(fig)
    print(f"Wrote: {p}")

    # -- panel: misclassified overlay --
    correct_mask = y_target_full == y_pred_full
    wrong_mask = ~correct_mask
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(X_umap[correct_mask, 0], X_umap[correct_mask, 1], s=8,
                c="#7f7f7f", label=f"Correct ({correct_mask.sum()})")
    ax.scatter(X_umap[wrong_mask, 0], X_umap[wrong_mask, 1], s=20, c="#e45756",
                alpha=0.85, edgecolor="black", linewidth=0.3,
                label=f"Misclassified ({wrong_mask.sum()})")
    ax.set_title("SVHN — UMAP of IPS similarity (misclassifications overlaid)")
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.legend(loc="best")
    fig.tight_layout()
    p = out_dir / "svhn_fig_ips_umap_misclassified.png"
    fig.savefig(p, dpi=200); plt.close(fig)
    print(f"Wrote: {p}")


if __name__ == "__main__":
    main()
