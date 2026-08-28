# Copyright (C) 2026 Ryan Pyle
# SPDX-License-Identifier: MIT
"""Figure 7: 25 lowest/highest UMAP dim1/dim2 class-7 training samples.

Consumes ``data/mnist_experiment1.pt`` and the UMAP cache produced by
``mnist_fig05_06_ips_umap.py``. Writes four 5×5 grids to figures/.
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
from torch.utils.data import Subset

from kernel_tools import LNModel, mnist_loaders, umap_from_ips


def _show_5x5_grid(indices, labels, dataset, suptitle: str, out_path: Path) -> None:
    fig, axes = plt.subplots(5, 5, figsize=(12, 12))
    for k, (idx, title) in enumerate(zip(indices, labels)):
        r, c = divmod(k, 5)
        ax = axes[r, c]
        img = np.array(dataset[int(idx)][0]).squeeze()
        ax.imshow(img, cmap="gray")
        ax.set_title(str(title), fontsize=9)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_frame_on(False)
    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", type=str, default="data/mnist_experiment1.pt")
    ap.add_argument("--data-root", type=str, default="data/MNIST")
    ap.add_argument("--out-dir", type=str, default="figures")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--train-batches", type=int, default=200)
    ap.add_argument("--target-class", type=int, default=7)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    art = torch.load(args.artifact, weights_only=False)
    if art.get("IPS") is None:
        raise RuntimeError("Artifact is missing IPS matrix. Rerun with --do-ips.")
    IPS = art["IPS"]
    y_target_full = np.asarray(
        art["y_target_full"].data if hasattr(art["y_target_full"], "data") else art["y_target_full"]
    ).astype(int)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Load or fit UMAP
    cache = out_dir / "_umap_cache.npy"
    if cache.exists() and np.load(cache).shape[0] == IPS.shape[0]:
        X_umap = np.load(cache)
        print(f"Loaded cached UMAP: {cache}")
    else:
        print("Fitting UMAP...")
        X_umap = umap_from_ips(IPS, random_state=args.seed)
        np.save(cache, X_umap)

    # Locate class-target-class training samples (train samples come first in the union)
    n_train = args.train_batches * args.batch_size
    inds_class_train = np.where(y_target_full[:n_train] == args.target_class)[0]

    # Sort within class along each UMAP dim
    Umap1 = np.argsort(X_umap[inds_class_train, 0])
    Umap2 = np.argsort(X_umap[inds_class_train, 1])

    # Reload train dataset for image display
    train_loader, _test, _audit = mnist_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        train_batches=args.train_batches,
        audit_batches=1,
    )
    dataset_class = Subset(train_loader.dataset, inds_class_train.tolist())

    labels_1lo = [f"Dim1 = {X_umap[inds_class_train[Umap1[i]], 0]:.2f}" for i in range(25)]
    labels_1hi = [f"Dim1 = {X_umap[inds_class_train[Umap1[-1-i]], 0]:.2f}" for i in range(25)]
    labels_2lo = [f"Dim2 = {X_umap[inds_class_train[Umap2[i]], 1]:.2f}" for i in range(25)]
    labels_2hi = [f"Dim2 = {X_umap[inds_class_train[Umap2[-1-i]], 1]:.2f}" for i in range(25)]

    _show_5x5_grid(
        Umap1[:25], labels_1lo, dataset_class,
        suptitle=f"Fig 7 top-left: 25 lowest UMAP Dim 1 (class {args.target_class})",
        out_path=out_dir / f"fig07a_class{args.target_class}_dim1_low.png",
    )
    _show_5x5_grid(
        Umap1[-25:], labels_1hi, dataset_class,
        suptitle=f"Fig 7 top-right: 25 highest UMAP Dim 1 (class {args.target_class})",
        out_path=out_dir / f"fig07b_class{args.target_class}_dim1_high.png",
    )
    _show_5x5_grid(
        Umap2[:25], labels_2lo, dataset_class,
        suptitle=f"Fig 7 bot-left: 25 lowest UMAP Dim 2 (class {args.target_class})",
        out_path=out_dir / f"fig07c_class{args.target_class}_dim2_low.png",
    )
    _show_5x5_grid(
        Umap2[-25:], labels_2hi, dataset_class,
        suptitle=f"Fig 7 bot-right: 25 highest UMAP Dim 2 (class {args.target_class})",
        out_path=out_dir / f"fig07d_class{args.target_class}_dim2_high.png",
    )
    print(f"Wrote 4 grids to {out_dir}")


if __name__ == "__main__":
    main()
