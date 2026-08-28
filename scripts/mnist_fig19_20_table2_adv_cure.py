# Copyright (C) 2026 Ryan Pyle
# SPDX-License-Identifier: MIT
"""Figures 19, 20 + Table 2: mode-aware adversarial cure.

Runs the label-free (T2) cure algorithm (paper §6.3) using:
  - a trained MNIST/LeNet-5 model from Group 1 (``data/mnist_experiment1.pt``)
  - a fresh adversarial audit set from Group 2's Task A (``data/mnist_adv_data.pt``)
  - V_modes + Lambda from Group 3's SVD snapshot (``data/mnist_expc_svd_epoch49.pt``)
    ⚠  Group 3 not yet landed. This script errors out with clear guidance
       if the SVD artifact is missing.

Outputs:
  - Table 2 (CSV + markdown): final accuracy per data type (Clean 100%,
    Adv 0%, Cured Adv ~87.5%, Cured Random ~100%, Adv + Random ~15.6%)
  - Figure 19: per-group loss over training with cure comparison
  - Figure 20: side-by-side clean / adv / cured-adv / adv+random for a
    single example
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn.functional import cross_entropy

from kernel_tools.console import enable_unicode_stdout
from kernel_tools import LNModel, build_cure_data


def _accuracy(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    with torch.no_grad():
        preds = model(x).argmax(dim=1)
    return float((preds == y).float().mean().item())


def main() -> None:
    enable_unicode_stdout()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-artifact", type=str, default="data/mnist_experiment1.pt")
    ap.add_argument("--adv-bundle", type=str, default="data/mnist_adv_data.pt")
    ap.add_argument("--svd-artifact", type=str, default="data/mnist_expc_svd_epoch49.pt",
                    help="Group 3 output containing V_modes (list) + Lambda (tensor). "
                         "Load .pt via torch.load; expects dict keys 'V_modes' and 'Lambda'.")
    ap.add_argument("--out-fig19", type=str, default="figures/fig19_cure_losses.png")
    ap.add_argument("--out-fig20", type=str, default="figures/fig20_cure_examples.png")
    ap.add_argument("--out-table2-csv", type=str, default="figures/table2_cure_accuracy.csv")
    ap.add_argument("--out-table2-md", type=str, default="figures/table2_cure_accuracy.md")
    ap.add_argument("--eps", type=float, default=0.1, help="L∞ radius for cure PGD")
    ap.add_argument("--tau", type=float, default=0.5, help="Mode-target shift magnitude")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--step-size", type=float, default=0.005)
    ap.add_argument("--k-modes", type=int, default=10)
    ap.add_argument("--use-t1", action="store_true",
                    help="Use label-aware T1 variant instead of the paper's T2 (label-free).")
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    print(f"device: {device}")

    # --- Load model ---
    src = torch.load(args.source_artifact, weights_only=False)
    model = LNModel(device=device)
    model.load_state_dict(src["model_final_state"])
    model.eval()

    # --- Load adversarial audit set ---
    adv_bundle = torch.load(args.adv_bundle, weights_only=False)
    adv_data = adv_bundle["adv_data"]
    adv_label = adv_bundle["adv_label"]
    print(f"adv_data shape: {tuple(adv_data.shape)}")

    # --- Load V_modes + Lambda from Group 3 (SVD snapshot) ---
    svd_path = Path(args.svd_artifact)
    if not svd_path.exists():
        raise SystemExit(
            f"SVD artifact not found at {svd_path}.\n"
            "This script depends on Group 3's mnist_mode_svd_snapshots.py output.\n"
            "Run: python scripts/mnist_mode_svd_snapshots.py "
            "--source-artifact data/mnist_experiment1.pt --k 128 "
            "--out data/mnist_expc_svd_epoch49.pt"
        )
    svd = torch.load(svd_path, weights_only=False)
    if "V_modes" not in svd or "Lambda" not in svd:
        raise SystemExit(
            f"SVD artifact {svd_path} is missing required keys 'V_modes' or 'Lambda'. "
            f"Got keys: {list(svd.keys())}"
        )
    V_modes = svd["V_modes"]  # list-of-param-lists
    Lambda = svd["Lambda"]
    print(f"V_modes: {len(V_modes)}   Lambda shape: {Lambda.shape}")

    # --- Pre-cure accuracy sanity check ---
    idxs = torch.arange(adv_data.size(0))
    clean_mask = (idxs % 4 == 0)
    adv_mask = (idxs % 4 == 1)
    rand_mask = (idxs % 4 >= 2)

    adv_data_dev = adv_data.to(device); adv_label_dev = adv_label.to(device).long()
    acc_clean_pre = _accuracy(model, adv_data_dev[clean_mask], adv_label_dev[clean_mask])
    acc_adv_pre = _accuracy(model, adv_data_dev[adv_mask], adv_label_dev[adv_mask])
    acc_rand_pre = _accuracy(model, adv_data_dev[rand_mask], adv_label_dev[rand_mask])
    print(f"Pre-cure -- Clean: {acc_clean_pre:.4f}  Adv: {acc_adv_pre:.4f}  Rand: {acc_rand_pre:.4f}")

    # --- Run cure ---
    variant = "T1" if args.use_t1 else "T2"
    print(f"\n=== Running {variant} cure ===")
    cure_data, stats = build_cure_data(
        model=model, adv_data=adv_data, adv_label=adv_label,
        V_modes=V_modes, Lambda=Lambda,
        tau=args.tau, eps=args.eps, p="linf",
        steps=args.steps, step_size=args.step_size,
        k_modes=args.k_modes, use_t1=args.use_t1,
    )
    print(f"stats: {stats}")

    # --- Post-cure accuracy ---
    cure_dev = cure_data.to(device)
    acc_clean = _accuracy(model, cure_dev[clean_mask], adv_label_dev[clean_mask])
    acc_adv = _accuracy(model, cure_dev[adv_mask], adv_label_dev[adv_mask])
    acc_rand = _accuracy(model, cure_dev[rand_mask], adv_label_dev[rand_mask])

    # Control: random perturbation of matched magnitude (not mode-directed)
    # Use adv_data with the second random slot (mask rand's slot=3, i.e. idx % 4 == 3)
    ctrl_rand_slot = (idxs % 4 == 3)
    acc_adv_plus_rand = _accuracy(
        model, adv_data_dev[ctrl_rand_slot], adv_label_dev[ctrl_rand_slot],
    )

    # --- Table 2 ---
    table_rows = [
        ("Clean",                     acc_clean_pre),
        ("Random (matched-eps)",      acc_rand_pre),
        ("Adversarial",               acc_adv_pre),
        (f"Cured Adversarial ({variant})", acc_adv),
        (f"Cured Random ({variant})",      acc_rand),
        ("Adversarial + Random",      acc_adv_plus_rand),
    ]
    out_csv = Path(args.out_table2_csv); out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Data Type", "Accuracy"])
        for name, acc in table_rows:
            w.writerow([name, f"{acc:.4f}"])
    out_md = Path(args.out_table2_md); out_md.parent.mkdir(parents=True, exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(f"# Table 2 — Final accuracy per data type (cure variant: {variant})\n\n")
        f.write("| Data Type | Accuracy |\n|---|---:|\n")
        for name, acc in table_rows:
            f.write(f"| {name} | {acc*100:.1f}% |\n")
    print(f"\nWrote CSV: {out_csv}")
    print(f"Wrote MD:  {out_md}")

    # --- Fig 19: grouped example losses ---
    with torch.no_grad():
        def _loss_per(x, y):
            return float(cross_entropy(model(x.to(device)), y.to(device).long()).item())
        losses = {
            "Clean":            _loss_per(adv_data_dev[clean_mask], adv_label_dev[clean_mask]),
            "Adversarial":      _loss_per(adv_data_dev[adv_mask], adv_label_dev[adv_mask]),
            "Random":           _loss_per(adv_data_dev[rand_mask], adv_label_dev[rand_mask]),
            f"Cured Adv ({variant})":    _loss_per(cure_dev[adv_mask], adv_label_dev[adv_mask]),
            f"Cured Random ({variant})": _loss_per(cure_dev[rand_mask], adv_label_dev[rand_mask]),
            "Adv + Random":     _loss_per(adv_data_dev[ctrl_rand_slot], adv_label_dev[ctrl_rand_slot]),
        }
    fig, ax = plt.subplots(figsize=(7, 4))
    xs = list(losses.keys()); ys = list(losses.values())
    ax.bar(xs, ys, color=["#4c78a8", "#e45756", "#59a14f", "#54a24b", "#b279a2", "#bab0ab"])
    ax.set_ylabel("mean cross-entropy loss on adv set")
    ax.set_title(f"Fig 19: per-group loss (cure variant: {variant})")
    ax.grid(True, alpha=0.25, axis="y")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    out19 = Path(args.out_fig19); out19.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out19, dpi=200); plt.close(fig)
    print(f"Wrote: {out19}")

    # --- Fig 20: side-by-side single-example perturbations ---
    ex_idx = 0
    variants = [
        ("clean",             adv_data[ex_idx + 0]),
        ("adversarial",       adv_data[ex_idx + 1]),
        (f"cured adv ({variant})", cure_data[ex_idx + 1]),
        ("adv + random",      adv_data[ex_idx + 3]),
    ]
    fig, axes = plt.subplots(1, len(variants), figsize=(3 * len(variants), 3.2))
    for ax, (name, img) in zip(axes, variants):
        ax.imshow(img.squeeze().cpu().numpy(), cmap="gray", vmin=0, vmax=1)
        ax.set_title(name)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_frame_on(False)
    fig.suptitle(
        f"Fig 20: cure does not merely undo adv "
        f"(mean cos(cure_δ, adv_δ) = {stats['mean_cos_with_adv_delta']:.3f})",
        fontsize=11,
    )
    fig.tight_layout()
    out20 = Path(args.out_fig20); out20.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out20, dpi=200); plt.close(fig)
    print(f"Wrote: {out20}")


if __name__ == "__main__":
    main()
