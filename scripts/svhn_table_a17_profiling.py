# Copyright (C) 2026 Ryan Pyle
# SPDX-License-Identifier: MIT
"""Table A.17: SVHN audit profiling summary (CSV + markdown).

Reads ``data/svhn_profiling_<variant>.json`` files produced by
``svhn_audit_profiling.py`` and emits ``figures/table_a17_svhn_profiling.csv``
and ``.md``.

Table shape:
  Method             Time(s)  Memory(MB)  Final MRE   Final Corr
  Baseline Train     ...      ...
  Audit              ...      ...        ...          ...
  Supersample (2x)   ...      ...        ...          ...
  Subsample (2x)     ...      ...        ...          ...
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROW_ORDER = [
    ("baseline",       "Baseline Train"),
    ("audit",          "Audit"),
    ("supersample_2",  "Supersample (2x)"),
    ("subsample_2",    "Subsample (2x)"),
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
    ap.add_argument("--out-csv", type=str, default="figures/table_a17_svhn_profiling.csv")
    ap.add_argument("--out-md", type=str, default="figures/table_a17_svhn_profiling.md")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    rows = []
    for variant, label in ROW_ORDER:
        p = data_dir / f"svhn_profiling_{variant}.json"
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

    out_md = Path(args.out_md); out_md.parent.mkdir(parents=True, exist_ok=True)
    with open(out_md, "w") as f:
        f.write("# Table A.17 — SVHN audit profiling\n\n")
        f.write("| Method | Time (s) | Peak RSS (MB) | Peak CUDA (MB) | Final MRE | Final Corr | Test Acc |\n")
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
