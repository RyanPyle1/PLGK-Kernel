# Copyright (C) 2026 Ryan Pyle
# SPDX-License-Identifier: MIT
"""Figures 5 & 6: 2D UMAP of the IPS similarity matrix (Section 6.2).

Fig 5 = two panels:
  (a) UMAP colored by true class (all 10 classes)
  (b) UMAP with misclassified points overlaid on correct points

Fig 6 = two panels:
  (a) UMAP colored by IPS similarity to a chosen target point (all classes)
  (b) Same, zoomed to only the target point's class

Consumes ``data/mnist_experiment1.pt`` produced with ``--do-ips``.
Writes: figures/fig05a, fig05b, fig06a, fig06b.

Wall time: UMAP fit ~2-4 min for the paper's default ~53k points; scripts
cache the fit to ``figures/_umap_cache.npy`` for re-use across Fig 5/6/7.
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", type=str, default="data/mnist_experiment1.pt")
    ap.add_argument("--out-dir", type=str, default="figures")
    ap.add_argument("--target-index", type=int, default=0,
                    help="Fig 6: which point to use as similarity-target")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    art = torch.load(args.artifact, weights_only=False)
    if art.get("IPS") is None:
        raise RuntimeError(
            "Artifact is missing IPS matrix. Rerun mnist_train_and_audit.py with --do-ips."
        )
    IPS = np.asarray(art["IPS"].data if hasattr(art["IPS"], "data") else art["IPS"])
    y_target_full = np.asarray(art["y_target_full"].data if hasattr(art["y_target_full"], "data") else art["y_target_full"]).astype(int)
    y_pred_full = np.asarray(art["y_pred_full"].data if hasattr(art["y_pred_full"], "data") else art["y_pred_full"]).astype(int)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    X_umap = _load_or_fit_umap(IPS, out_dir / "_umap_cache.npy", args.seed)

    # -- Fig 5a: UMAP colored by class --
    N = 10
    labels = [str(i) for i in range(N)]
    cmap = ListedColormap(matplotlib.colormaps["tab10"].colors[:N])
    bounds = np.arange(-0.5, N + 0.5, 1)
    norm = BoundaryNorm(bounds, N)

    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(X_umap[:, 0], X_umap[:, 1], s=8, c=y_target_full, cmap=cmap, norm=norm)
    ax.set_title("UMAP of IPS similarity — colored by true class")
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    cbar = fig.colorbar(sc, ax=ax, ticks=np.arange(N))
    cbar.ax.set_yticklabels(labels)
    cbar.set_label("Class")
    fig.tight_layout()
    p = out_dir / "fig05a_ips_umap_by_class.png"
    fig.savefig(p, dpi=200); plt.close(fig)
    print(f"Wrote: {p}")

    # -- Fig 5b: UMAP with misclassified overlay --
    correct_mask = y_target_full == y_pred_full
    wrong_mask = ~correct_mask
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(X_umap[correct_mask, 0], X_umap[correct_mask, 1], s=8, c="#7f7f7f", label="Correct")
    ax.scatter(X_umap[wrong_mask, 0], X_umap[wrong_mask, 1], s=20, c="#e45756",
                alpha=0.85, edgecolor="black", linewidth=0.3, label="Misclassified")
    ax.set_title("UMAP of IPS similarity — misclassifications on cluster edges")
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.legend(loc="best")
    fig.tight_layout()
    p = out_dir / "fig05b_ips_umap_misclassified.png"
    fig.savefig(p, dpi=200); plt.close(fig)
    print(f"Wrote: {p}")

    # -- Fig 6: similarity-to-target --
    ind_choose = args.target_index
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(X_umap[:, 0], X_umap[:, 1], s=8, c=IPS[ind_choose, :], cmap="viridis")
    ax.scatter(X_umap[ind_choose, 0], X_umap[ind_choose, 1], s=80, c="red",
                edgecolor="black", linewidth=0.6, marker="*", label=f"Target (idx {ind_choose})")
    ax.set_title("UMAP of IPS similarity — colored by similarity to target point")
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.legend(loc="best")
    fig.colorbar(sc, ax=ax, label="IPS similarity")
    fig.tight_layout()
    p = out_dir / "fig06a_ips_umap_similarity_to_target.png"
    fig.savefig(p, dpi=200); plt.close(fig)
    print(f"Wrote: {p}")

    class_of_target = int(y_target_full[ind_choose])
    inds_class = np.where(y_target_full == class_of_target)[0]
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(X_umap[inds_class, 0], X_umap[inds_class, 1], s=8,
                     c=IPS[ind_choose, inds_class], cmap="viridis")
    ax.scatter(X_umap[ind_choose, 0], X_umap[ind_choose, 1], s=80, c="red",
                edgecolor="black", linewidth=0.6, marker="*", label=f"Target (class {class_of_target})")
    ax.set_title(f"UMAP of IPS similarity — zoomed to class {class_of_target} only")
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.legend(loc="best")
    fig.colorbar(sc, ax=ax, label="IPS similarity")
    fig.tight_layout()
    p = out_dir / "fig06b_ips_umap_similarity_zoom_class.png"
    fig.savefig(p, dpi=200); plt.close(fig)
    print(f"Wrote: {p}")


if __name__ == "__main__":
    main()
