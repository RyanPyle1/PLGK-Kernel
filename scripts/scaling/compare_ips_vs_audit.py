# Copyright (C) 2026 Ryan Pyle
# SPDX-License-Identifier: MIT
"""Table 6 (Appendix A.18.1): per-sample IPS-vs-audit correlation on CIFAR-10.

For each of {20, 50, 80, 120}-epoch checkpoints of ResNet-18-GN, compute the
per-audit-sample Pearson correlation between:
  * audit column-sum: sum_m PNTK[m, n]     (predicted per-sample loss change)
  * IPS diagonal (or aggregated audit-to-train IPS): sum_m IPS[m, n]

The paper reports 0.941 -> 0.945 across those four checkpoints (climbing
modestly as the network settles into a kernel regime).

Assumes each checkpoint run was invoked as:
  python scripts/scaling/run_cifar10_audit.py --model resnet18gn \\
      --epochs <N> --do-ips --out data/cifar10_resnet18gn_audit_ep<N>.pt

and produces ``figures/table6_ips_vs_audit_cifar10.{csv,md}``.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, nargs="+", default=[20, 50, 80, 120])
    ap.add_argument("--artifact-template", type=str,
                    default="data/cifar10_resnet18gn_audit_ep{ep}.pt",
                    help="Format string with '{ep}' placeholder")
    ap.add_argument("--out-csv", type=str,
                    default="figures/table6_ips_vs_audit_cifar10.csv")
    ap.add_argument("--out-md", type=str,
                    default="figures/table6_ips_vs_audit_cifar10.md")
    args = ap.parse_args()

    rows = []
    for ep in args.epochs:
        path = Path(args.artifact_template.format(ep=ep))
        if not path.exists():
            print(f"[skip] {path} not found")
            continue
        art = torch.load(path, weights_only=False)
        PNTK = art.get("PNTK")
        IPS = art.get("IPS")
        if PNTK is None:
            print(f"[skip] {path} has no PNTK (audit not enabled)")
            continue
        if IPS is None:
            print(f"[skip] {path} has no IPS (rerun with --do-ips)")
            continue

        # Audit per-audit-point predicted loss change: sum over training rows
        audit_pred = PNTK.sum(dim=0).numpy()  # (n_audit,)

        # IPS: pick the train->audit block. IPSAccumulator stores an (n_total x
        # n_total) matrix where n_total = n_train + n_audit; the audit rows
        # start at n_train. We want per-audit-point aggregated train-side
        # similarity: sum of IPS entries between audit-point n and each train
        # sample m.
        n_train = art.get("n_train_samples") or (PNTK.shape[0])
        n_audit = PNTK.shape[1]
        if IPS.shape[0] != n_train + n_audit:
            # Fallback: assume IPS is the (train+audit)-square matrix even if
            # metadata is missing; use the last n_audit rows as audit-side.
            n_train = IPS.shape[0] - n_audit
        train_to_audit_block = IPS[:n_train, n_train:]  # (n_train, n_audit)
        ips_pred = train_to_audit_block.sum(dim=0).numpy()  # (n_audit,)

        corr = float(np.corrcoef(audit_pred, ips_pred)[0, 1])
        rows.append({"epoch": ep, "corr": corr, "n_audit": n_audit, "n_train": int(n_train)})
        print(f"  ep {ep:4d}   n_audit={n_audit}   n_train={n_train}   corr={corr:.4f}")

    out_csv = Path(args.out_csv); out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "n_train", "n_audit", "ips_audit_corr"])
        for r in rows:
            w.writerow([r["epoch"], r["n_train"], r["n_audit"], f"{r['corr']:.6f}"])
    print(f"Wrote CSV: {out_csv}")

    out_md = Path(args.out_md); out_md.parent.mkdir(parents=True, exist_ok=True)
    with open(out_md, "w") as f:
        f.write("# Table 6 — CIFAR-10 ResNet-18-GN IPS-vs-audit per-sample correlation\n\n")
        f.write("| Epoch | n_train | n_audit | Pearson (IPS vs Audit) |\n|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(f"| {r['epoch']} | {r['n_train']} | {r['n_audit']} | {r['corr']:.4f} |\n")
    print(f"Wrote MD: {out_md}")


if __name__ == "__main__":
    main()
