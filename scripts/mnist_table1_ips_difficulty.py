# Copyright (C) 2026 Ryan Pyle
# SPDX-License-Identifier: MIT
"""Table 1: IPS-based scores as supervised difficulty predictors on MNIST.

Consumes ``data/mnist_experiment1.pt`` (produced with ``--do-ips``) and
emits a CSV + markdown of the paper's Table 1: for each scalar score,
AUROC and AUPR for misclassification detection + Spearman correlation
with per-example loss.

Score families:
  IPS_*   : IPS-max, IPS-topk, IPS class-margin, IPS L1, IPS L2 norm
  HID_*   : penultimate-hidden max cosine, top-k cosine (baseline)
  CONF_*  : max-softmax, logit margin (standard confidence baselines)
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from kernel_tools import run_ips_utility_experiments


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", type=str, default="data/mnist_experiment1.pt")
    ap.add_argument("--out-csv", type=str, default="figures/table1_ips_difficulty.csv")
    ap.add_argument("--out-md", type=str, default="figures/table1_ips_difficulty.md")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--train-batches", type=int, default=200)
    ap.add_argument("--target-batches", type=int, default=1)
    ap.add_argument("--n-classes", type=int, default=10)
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()

    art = torch.load(args.artifact, weights_only=False)
    for k in ("IPS", "y_target_full", "y_pred_full", "y_loss_full", "y_hidden_full", "y_logits_full"):
        if art.get(k) is None:
            raise RuntimeError(
                f"Artifact missing '{k}'. Rerun mnist_train_and_audit.py with --do-ips."
            )

    error_metrics, loss_metrics = run_ips_utility_experiments(
        batch_size=args.batch_size,
        train_batches=args.train_batches,
        target_batches=args.target_batches,
        IPS=art["IPS"],
        y_target_full=art["y_target_full"],
        y_pred_full=art["y_pred_full"],
        y_loss_full=art["y_loss_full"],
        y_hidden_full=art["y_hidden_full"],
        y_logits_full=art["y_logits_full"],
        n_classes=args.n_classes,
        topk=args.topk,
        verbose=True,
    )

    # --- Write CSV ---
    out_csv = Path(args.out_csv); out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["score", "AUROC_misclass", "AUPR_misclass", "Spearman(s,-L)", "Pearson(s,-L)"])
        for name in error_metrics.keys():
            e = error_metrics[name]
            l = loss_metrics[name]
            w.writerow([
                name,
                f"{e['AUROC_error']:.4f}",
                f"{e['AUPR_error']:.4f}",
                f"{l['Spearman_s_vs_-loss']:.4f}",
                f"{l['Pearson_s_vs_-loss']:.4f}",
            ])
    print(f"\nWrote CSV: {out_csv}")

    # --- Write markdown table (paper format) ---
    out_md = Path(args.out_md); out_md.parent.mkdir(parents=True, exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Table 1 — IPS-based scores as supervised difficulty predictors on MNIST\n\n")
        f.write("| Score | AUROC (misclass) | AUPR (misclass) | Spearman(s, -L) | Pearson(s, -L) |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for name in error_metrics.keys():
            e = error_metrics[name]; l = loss_metrics[name]
            f.write(
                f"| {name} | {e['AUROC_error']:.4f} | {e['AUPR_error']:.4f} | "
                f"{l['Spearman_s_vs_-loss']:.4f} | {l['Pearson_s_vs_-loss']:.4f} |\n"
            )
    print(f"Wrote markdown: {out_md}")


if __name__ == "__main__":
    main()
