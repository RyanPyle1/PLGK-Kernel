# Copyright (C) 2026 Ryan Pyle
# SPDX-License-Identifier: MIT
"""Table 5 (Appendix A.18): scaling audit / IPS costs across three rows.

Reads JSON summaries emitted by:
  * ``run_cifar10_audit.py --model resnet18gn``  -> row 1
  * ``run_cifar10_audit.py --model vittiny4``    -> row 2
  * ``run_ips_only.py --dataset imagenet``       -> row 3

Emits ``figures/table5_scaling.csv`` and ``.md``.

Paper reference values:
  Row 1 (CIFAR / ResNet18-GN):    audit 14.9 -> 126s/epoch  (8.5x)  6.0 GiB  56.6%
  Row 2 (CIFAR / ViT-Tiny/4):     audit 15.8 -> 178s/epoch (11.3x)  4.62 GiB 46.8%
  Row 3 (ImageNet / ResNet18-GN): IPS  2489 -> 6772s        (2.7x)  22.4 GiB 41.3%
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROW_ORDER = [
    ("row1_cifar_resnet18gn",       "CIFAR / ResNet-18-GN",  "data/cifar10_resnet18gn_audit.json"),
    ("row2_cifar_vittiny4",         "CIFAR / ViT-Tiny/4",    "data/cifar10_vittiny4_audit.json"),
    ("row3_imagenet_resnet18gn_ips","ImageNet / ResNet-18-GN (IPS-only)",
                                                              "data/imagenet_resnet18gn_ips_only.json"),
]


def _fmt(v, spec: str = ".1f") -> str:
    if v is None:
        return "—"
    try:
        return format(float(v), spec)
    except (ValueError, TypeError):
        return str(v)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-csv", type=str, default="figures/table5_scaling.csv")
    ap.add_argument("--out-md", type=str, default="figures/table5_scaling.md")
    args = ap.parse_args()

    rows = []
    for key, label, default_path in ROW_ORDER:
        p = Path(default_path)
        if not p.exists():
            print(f"[skip] {p} missing")
            continue
        with open(p) as f:
            d = json.load(f)
        rows.append({
            "key": key,
            "label": label,
            "params": d.get("params"),
            "epochs": d.get("epochs"),
            "wall_seconds": d.get("wall_seconds"),
            "peak_cuda_mb": d.get("peak_cuda_mb"),
            "testacc_final": d.get("testacc_final"),
            "final_mre": d.get("final_mre"),
            "final_corr": d.get("final_corr"),
        })

    out_csv = Path(args.out_csv); out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "label", "params", "epochs", "wall_seconds",
                     "peak_cuda_mb", "testacc_final", "final_mre", "final_corr"])
        for r in rows:
            w.writerow([
                r["key"], r["label"],
                r["params"], r["epochs"],
                _fmt(r["wall_seconds"], ".1f"),
                _fmt(r["peak_cuda_mb"], ".1f"),
                _fmt(r["testacc_final"], ".4f"),
                _fmt(r["final_mre"], ".6f"),
                _fmt(r["final_corr"], ".6f"),
            ])
    print(f"Wrote CSV: {out_csv}")

    out_md = Path(args.out_md); out_md.parent.mkdir(parents=True, exist_ok=True)
    with open(out_md, "w") as f:
        f.write("# Table 5 — Scaling: audit / IPS overhead at CIFAR + ImageNet scale\n\n")
        f.write("| Config | Params | Epochs | Wall (s) | Peak CUDA (MB) | Test Acc | Final MRE | Final Corr |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            params_cell = f"{r['params']:,}" if r["params"] else "—"
            f.write(
                f"| {r['label']} "
                f"| {params_cell} "
                f"| {r['epochs']} "
                f"| {_fmt(r['wall_seconds'], '.1f')} "
                f"| {_fmt(r['peak_cuda_mb'], '.1f')} "
                f"| {_fmt(r['testacc_final'], '.4f')} "
                f"| {_fmt(r['final_mre'], '.6f')} "
                f"| {_fmt(r['final_corr'], '.6f')} |\n"
            )
    print(f"Wrote MD: {out_md}")


if __name__ == "__main__":
    main()
