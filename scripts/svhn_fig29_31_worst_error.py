# Copyright (C) 2026 Ryan Pyle
# SPDX-License-Identifier: MIT
"""Figures 29-31: SVHN worst-error decomposition (Appendix A.17.1).

Coverage matrix mapping:
  Fig 29 — single-mistake audit: the misclassified test digit + sorted
           per-training-sample influences.
  Fig 30 — 25 most helpful + 25 most harmful training samples (any class).
  Fig 31 — 25 most helpful + 25 most harmful training samples restricted
           to the wrongly-predicted class.

Analogous to MNIST's Figs 2-4 but on SVHN: RGB display (permute CHW->HWC,
per-image renormalize to [0,1] since the loader normalizes with SVHN mean/std)
and using SVHN's numpy .labels attribute for training-label lookup.

Produces (into ``figures/``):
  fig29_svhn_worst_error.png              — worst-error image + sorted influence
  fig30a_svhn_top25_helpful.png           — 25 most helpful (any class)
  fig30b_svhn_top25_harmful.png           — 25 most harmful (any class)
  fig31a_svhn_top25_helpful_class<K>.png  — restricted to wrongly predicted class
  fig31b_svhn_top25_harmful_class<K>.png
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

from kernel_tools import SVHNModel, svhn_loaders


def _img_for_display(sample_tuple) -> np.ndarray:
    """Take a ``(tensor CHW, label)`` sample and return a HWC uint-safe
    RGB array in [0, 1]. Handles the fact the loader normalized with
    SVHN mean/std, so we per-image min-max renormalize for display."""
    img = sample_tuple[0]
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = np.transpose(img, (1, 2, 0))
    lo, hi = float(np.min(img)), float(np.max(img))
    if hi > lo:
        img = (img - lo) / (hi - lo)
    return img.squeeze()


def _show_5x5_grid(indices, labels, dataset, suptitle: str, out_path: Path) -> None:
    fig, axes = plt.subplots(5, 5, figsize=(12, 12))
    for k, (idx, title) in enumerate(zip(indices, labels)):
        r, c = divmod(k, 5)
        ax = axes[r, c]
        img = _img_for_display(dataset[int(idx)])
        if img.ndim == 2:
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(img)
        ax.set_title(str(title), fontsize=9)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_frame_on(False)
    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200); plt.close(fig)
    print(f"Wrote: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifact", type=str, default="data/svhn_experiment1.pt")
    ap.add_argument("--data-root", type=str, default="data/SVHN")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--train-batches", type=int, default=200)
    ap.add_argument("--out-dir", type=str, default="figures")
    args = ap.parse_args()

    art = torch.load(args.artifact, weights_only=False)
    PNTK = art["PNTK"]
    y_target = art["y_target"]
    loss_final = art["loss_final"]
    if PNTK is None:
        raise RuntimeError("Artifact missing PNTK — rerun svhn_train_and_audit.py with --do-audit")

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Rehydrate model + loaders (need images for display + train labels for Fig 34)
    device = torch.device("cpu")  # display-only
    model = SVHNModel(device=device)
    model.load_state_dict(art["model_final_state"])
    train_loader, _test_loader, audit_loader = svhn_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        train_batches=args.train_batches,
        audit_batches=1,
    )

    # Predictions on the audit set to identify the worst error
    ypred_final = torch.zeros(len(audit_loader.dataset))
    with torch.no_grad():
        for idy, (test_x, _tl) in enumerate(audit_loader):
            ypred_final[idy * args.batch_size:(idy + 1) * args.batch_size] = torch.argmax(model(test_x), 1)
    worst_err_ind = int(torch.argmax(loss_final))
    worst_true = int(y_target[worst_err_ind])
    worst_pred = int(ypred_final[worst_err_ind])
    print(f"Worst error: audit-idx {worst_err_ind}  true={worst_true}  predicted={worst_pred}")

    # -- Fig 32: worst input + sorted influence --
    fig, (ax_img, ax_bar) = plt.subplots(1, 2, figsize=(11, 4))
    img = _img_for_display(audit_loader.dataset[worst_err_ind])
    if img.ndim == 2:
        ax_img.imshow(img, cmap="gray")
    else:
        ax_img.imshow(img)
    ax_img.set_title(f"Predicted: {worst_pred}, True: {worst_true}")
    ax_img.set_xticks([]); ax_img.set_yticks([])
    worst_influences = PNTK[:, worst_err_ind]
    sorted_influences = torch.sort(worst_influences)[0]
    ax_bar.plot(sorted_influences.numpy())
    ax_bar.set_xlabel("Sorted training-sample index")
    ax_bar.set_ylabel("Training influence")
    ax_bar.grid(True, alpha=0.25)
    fig.tight_layout()
    p = out_dir / "fig29_svhn_worst_error.png"
    fig.savefig(p, dpi=200); plt.close(fig)
    print(f"Wrote: {p}")

    # -- Fig 33: 25 most helpful / harmful (any class) --
    sorted_inds = torch.sort(worst_influences)[1]
    helpful_inds = sorted_inds[:25]
    harmful_inds = sorted_inds[-25:]
    helpful_labels = [f"Inf: {worst_influences[int(i)]:.3f}" for i in helpful_inds]
    harmful_labels = [f"Inf: {worst_influences[int(i)]:.3f}" for i in harmful_inds]

    _show_5x5_grid(
        helpful_inds, helpful_labels, train_loader.dataset,
        suptitle="Figure 30a: SVHN — 25 Most Helpful",
        out_path=out_dir / "fig30a_svhn_top25_helpful.png",
    )
    _show_5x5_grid(
        harmful_inds, harmful_labels, train_loader.dataset,
        suptitle="Figure 30b: SVHN — 25 Most Harmful",
        out_path=out_dir / "fig30b_svhn_top25_harmful.png",
    )

    # -- Fig 34: restrict to wrongly-predicted class --
    # SVHN train labels are in .labels (numpy int array).
    train_labels = np.asarray(train_loader.dataset.labels)[: args.train_batches * args.batch_size]
    inds_wpred = np.where(train_labels == worst_pred)[0]
    if len(inds_wpred) < 25:
        print(f"[warn] fewer than 25 samples of class {worst_pred} in first "
              f"{args.train_batches * args.batch_size} training samples; skipping Fig 31.")
        return
    infl_wpred = PNTK[torch.as_tensor(inds_wpred, dtype=torch.long), worst_err_ind]
    sorted_wpred = torch.sort(infl_wpred)[1]
    dataset_wpred = Subset(train_loader.dataset, inds_wpred.tolist())
    helpful_wpred_inds = sorted_wpred[:25]
    harmful_wpred_inds = sorted_wpred[-25:]
    helpful_wpred_labels = [f"Inf: {infl_wpred[int(i)]:.3f}" for i in helpful_wpred_inds]
    harmful_wpred_labels = [f"Inf: {infl_wpred[int(i)]:.3f}" for i in harmful_wpred_inds]

    _show_5x5_grid(
        helpful_wpred_inds, helpful_wpred_labels, dataset_wpred,
        suptitle=f"Figure 31a: SVHN — 25 Most Helpful (Class {worst_pred})",
        out_path=out_dir / f"fig31a_svhn_top25_helpful_class{worst_pred}.png",
    )
    _show_5x5_grid(
        harmful_wpred_inds, harmful_wpred_labels, dataset_wpred,
        suptitle=f"Figure 31b: SVHN — 25 Most Harmful (Class {worst_pred})",
        out_path=out_dir / f"fig31b_svhn_top25_harmful_class{worst_pred}.png",
    )


if __name__ == "__main__":
    main()
