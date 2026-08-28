# Copyright (C) 2026 Ryan Pyle
# SPDX-License-Identifier: MIT
"""Table A.5: MNIST audit profiling summary (CSV + markdown).

Reads ``data/mnist_profiling_<variant>.json`` files produced by
``mnist_audit_profiling.py`` and emits ``figures/table_a5_profiling.csv``
and ``.md``.

Paper's Table A.5 shape:
  Method             Time(s)  Memory(GiB)  Final MLE   Final Corr
  Baseline Train     226.5    0.435        —           —
  Audit             4040.0    2.915        0.010386    0.999998
  Audit (Non-trap)  3953.5    2.859        0.090122    0.990681
  Supersample (2x)  4162.7    3.039        0.010401    0.999998
  Subsample (2x)    3159.2    4.991        0.013338    0.999976
  Subsample (5x)    2035.2    5.084        0.021712    0.999786
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


ROW_ORDER = [
    ("baseline",       "Baseline Train"),
    ("audit",          "Audit"),
    ("non_trap",       "Audit (Non-trapezoidal)"),
    ("supersample_2",  "Supersample (2x)"),
    ("subsample_2",    "Subsample (2x)"),
    ("subsample_5",    "Subsample (5x)"),
]


def _fmt_num(v, fmt: str = "{:.4f}"):
    if v is None:
        return "—"
    try:
        return fmt.format(float(v))
    except (ValueError, TypeError):
        return str(v)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=str, default="data")
    ap.add_argument("--out-csv", type=str, default="figures/table_a5_profiling.csv")
    ap.add_argument("--out-md", type=str, default="figures/table_a5_profiling.md")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    rows = []
    for variant, label in ROW_ORDER:
        p = data_dir / f"mnist_profiling_{variant}.json"
        if not p.exists():
            print(f"[skip] missing {p}")
            continue
        with open(p) as f:
            d = json.load(f)
        rows.append({
            "variant": variant,
            "label": label,
            "wall_seconds": d.get("wall_seconds"),
            "peak_rss_mb": d.get("peak_rss_mb"),
            "peak_cuda_mb": d.get("peak_cuda_mb"),
            "final_mre": d.get("final_mre"),
            "final_corr": d.get("final_corr"),
            "testacc_final": d.get("testacc_final"),
        })

    # -- CSV --
    out_csv = Path(args.out_csv); out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "label", "wall_seconds", "peak_rss_mb",
                     "peak_cuda_mb", "final_mre", "final_corr", "testacc_final"])
        for r in rows:
            w.writerow([
                r["variant"], r["label"],
                _fmt_num(r["wall_seconds"], "{:.1f}"),
                _fmt_num(r["peak_rss_mb"], "{:.1f}"),
                _fmt_num(r["peak_cuda_mb"], "{:.1f}"),
                _fmt_num(r["final_mre"], "{:.6f}"),
                _fmt_num(r["final_corr"], "{:.6f}"),
                _fmt_num(r["testacc_final"], "{:.4f}"),
            ])
    print(f"Wrote CSV: {out_csv}")

    # -- Markdown --
    out_md = Path(args.out_md); out_md.parent.mkdir(parents=True, exist_ok=True)
    with open(out_md, "w") as f:
        f.write("# Table A.5 — MNIST audit profiling\n\n")
        f.write("| Method | Time (s) | Peak RSS (MB) | Peak CUDA (MB) | Final MLE | Final Corr | Test Acc |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(
                f"| {r['label']} "
                f"| {_fmt_num(r['wall_seconds'], '{:.1f}')} "
                f"| {_fmt_num(r['peak_rss_mb'], '{:.1f}')} "
                f"| {_fmt_num(r['peak_cuda_mb'], '{:.1f}')} "
                f"| {_fmt_num(r['final_mre'], '{:.6f}')} "
                f"| {_fmt_num(r['final_corr'], '{:.6f}')} "
                f"| {_fmt_num(r['testacc_final'], '{:.4f}')} |\n"
            )
    print(f"Wrote MD: {out_md}")


if __name__ == "__main__":
    main()
