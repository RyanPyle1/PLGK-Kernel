# Copyright (C) 2026 Ryan Pyle
# SPDX-License-Identifier: MIT
"""Figures 2, 3, 4: audit one wrong prediction (worst-error decomposition).

Fig 2: the misclassified test digit + sorted per-training-sample influences.
Fig 3: 25 most helpful + 25 most harmful training samples (any class).
Fig 4: 25 most helpful + 25 most harmful training samples restricted to
       class 6 (the class the network wrongly predicted).

Consumes ``data/mnist_experiment1.pt``. Writes into ``figures/``.

Note: the "worst error" is chosen as ``argmax(loss_final)`` over the audit
set, matching the paper's protocol. On the paper's seed=4 run this is a
class-4 example the model calls a 6, giving rise to the class-6 restriction
in Fig 4.
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
from torch.utils.data import DataLoader, Subset

from kernel_tools import LNModel, mnist_loaders


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
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--train-batches", type=int, default=200)
    ap.add_argument("--out-dir", type=str, default="figures")
    args = ap.parse_args()

    art = torch.load(args.artifact, weights_only=False)
    PNTK = art["PNTK"]
    y_target = art["y_target"]
    loss_final = art["loss_final"]
    if PNTK is None:
        raise RuntimeError("Artifact missing PNTK — rerun mnist_train_and_audit.py with --do-audit")

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Reload the model + audit loader (need images for display)
    device = torch.device("cpu")  # display-only, cpu is fine
    model = LNModel(device=device)
    model.load_state_dict(art["model_final_state"])
    train_loader, _test_loader, audit_loader = mnist_loaders(
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

    # -- Fig 2: worst input + sorted influence bar --
    fig, (ax_img, ax_bar) = plt.subplots(1, 2, figsize=(11, 4))
    img = np.array(audit_loader.dataset[worst_err_ind][0]).squeeze()
    ax_img.imshow(img, cmap="gray")
    ax_img.set_title(f"Predicted: {worst_pred}, True: {worst_true}")
    ax_img.set_xticks([]); ax_img.set_yticks([])
    worst_influences = PNTK[:, worst_err_ind]
    sorted_influences = torch.sort(worst_influences)[0]
    ax_bar.plot(sorted_influences.numpy())
    ax_bar.set_xlabel("Sorted training-sample index")
    ax_bar.set_ylabel("Training influence")
    ax_bar.grid(True, alpha=0.25)
    fig.tight_layout()
    p = out_dir / "fig02_worst_error.png"
    fig.savefig(p, dpi=200); plt.close(fig)
    print(f"Wrote: {p}")

    # -- Fig 3: 25 most helpful / harmful (any class) --
    # Sort by influence: negative influence = helpful (drove loss DOWN);
    # positive influence = harmful (drove loss UP).
    sorted_inds = torch.sort(worst_influences)[1]  # ascending
    helpful_inds = sorted_inds[:25]
    harmful_inds = sorted_inds[-25:]

    helpful_labels = [f"Inf: {worst_influences[int(i)]:.3f}" for i in helpful_inds]
    harmful_labels = [f"Inf: {worst_influences[int(i)]:.3f}" for i in harmful_inds]

    _show_5x5_grid(
        helpful_inds, helpful_labels, train_loader.dataset,
        suptitle="Figure 3a: 25 Most Helpful",
        out_path=out_dir / "fig03a_top25_helpful.png",
    )
    _show_5x5_grid(
        harmful_inds, harmful_labels, train_loader.dataset,
        suptitle="Figure 3b: 25 Most Harmful",
        out_path=out_dir / "fig03b_top25_harmful.png",
    )
    print(f"Wrote: {out_dir}/fig03a_top25_helpful.png")
    print(f"       {out_dir}/fig03b_top25_harmful.png")

    # -- Fig 4: 25 most helpful/harmful RESTRICTED to the wrongly-predicted class --
    y_traintarget = torch.zeros(args.train_batches * args.batch_size)
    for idx, (_train_x, train_label) in enumerate(train_loader):
        if idx >= args.train_batches:
            break
        y_traintarget[idx * args.batch_size:(idx + 1) * args.batch_size] = train_label

    inds_wpred = torch.where(y_traintarget == worst_pred)[0]
    infl_wpred = PNTK[inds_wpred, worst_err_ind]
    sorted_wpred = torch.sort(infl_wpred)[1]
    dataset_wpred = Subset(train_loader.dataset, inds_wpred.tolist())
    helpful_wpred_inds = sorted_wpred[:25]
    harmful_wpred_inds = sorted_wpred[-25:]
    helpful_wpred_labels = [f"Inf: {infl_wpred[int(i)]:.3f}" for i in helpful_wpred_inds]
    harmful_wpred_labels = [f"Inf: {infl_wpred[int(i)]:.3f}" for i in harmful_wpred_inds]

    _show_5x5_grid(
        helpful_wpred_inds, helpful_wpred_labels, dataset_wpred,
        suptitle=f"Figure 4a: 25 Most Helpful (Class {worst_pred})",
        out_path=out_dir / f"fig04a_top25_helpful_class{worst_pred}.png",
    )
    _show_5x5_grid(
        harmful_wpred_inds, harmful_wpred_labels, dataset_wpred,
        suptitle=f"Figure 4b: 25 Most Harmful (Class {worst_pred})",
        out_path=out_dir / f"fig04b_top25_harmful_class{worst_pred}.png",
    )
    print(f"Wrote: fig04a/b_top25_*_class{worst_pred}.png")


if __name__ == "__main__":
    main()
