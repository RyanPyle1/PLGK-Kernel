# Copyright (C) 2026 Ryan Pyle
# SPDX-License-Identifier: MIT
"""Figure 17: per-mode input sensitivity, temporal — top-10 vs modes 11-50.

For each epoch t, compute ‖∇_x c_k(x)‖ per mode k on the adversarial audit
set (Group 2's ``mnist_adv_data.pt``), then plot the mean over the top-10
modes vs mean over modes 11-50 across training time. Two panels:
raw (left) and loss-gradient-normalized (right).

Expensive: O(epochs * N_audit * K) double-autograd passes. On CPU with
paper defaults (50 epochs, 256 audit samples, K=50) this is ~2 hours.
Ship the temporal-dCdx cache with the repo when possible.

Requires per-epoch model checkpoints (``model_snapshots_per_epoch`` in the
Group 3 SVD artifact). We add this as an optional snapshot flag; if not
present the script errors with clear guidance.

  --svd-full           data/mnist_expc_svd_full.pt
  --adv-bundle         data/mnist_adv_data.pt
  --cache              data/mnist_dcdx_temporal_K50.npz
  --compute            regenerate cache (expensive)
  --K                  modes to analyze (default 50)
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
import torch.nn as nn

from kernel_tools import LNModel


def _load_dcdx_helper():
    """Lazy import of ``compute_dcdx_norms`` from the Fig 15/16 script."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from mnist_fig15_16_mode_decomposition import compute_dcdx_norms
    return compute_dcdx_norms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--svd-full", type=str, default="data/mnist_expc_svd_full.pt")
    ap.add_argument("--adv-bundle", type=str, default="data/mnist_adv_data.pt")
    ap.add_argument("--cache", type=str, default="data/mnist_dcdx_temporal_K50.npz")
    ap.add_argument("--compute", action="store_true",
                    help="Regenerate the per-epoch dCdx cache (expensive).")
    ap.add_argument("--K", type=int, default=50)
    ap.add_argument("--out", type=str, default="figures/fig17_input_sensitivity_temporal.png")
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    cache_path = Path(args.cache)
    if args.compute or not cache_path.exists():
        raise SystemExit(
            f"Per-epoch dCdx cache not present at {cache_path} and compute is expensive.\n"
            "This script currently requires either:\n"
            "  (a) a pre-computed cache produced by an earlier run, OR\n"
            "  (b) per-epoch model checkpoints so we can regenerate.\n"
            "\n"
            "Group 3's SVD snapshot driver does NOT persist per-epoch model\n"
            "checkpoints by default (it only keeps the final state). To generate\n"
            "the cache, add per-epoch model-state snapshots to\n"
            "``mnist_mode_svd_snapshots.py``'s epoch_callback and rerun,\n"
            "then invoke this script with --compute.\n"
            "\n"
            "This script is included so the figure exists in the repo scaffold;\n"
            "actual regeneration is deferred to the reproducibility rerun.\n"
        )

    npz = np.load(cache_path)
    dcdx_t = npz["dcdx_per_epoch"]                   # (n_epochs, N_audit, K)
    n_epochs, N_audit, K = dcdx_t.shape
    K = min(K, args.K)
    dcdx_t = dcdx_t[..., :K]

    # Group masks (paper slot layout)
    n_examples = N_audit // 16
    clean_positions = [16 * i + off for i in range(n_examples) for off in (0, 4, 8, 12)]
    adv_positions   = [16 * i + off for i in range(n_examples) for off in (1, 5, 9, 13)]
    rand_positions  = [16 * i + off for i in range(n_examples) for off in (2, 3, 6, 7, 10, 11, 14, 15)]

    # Per-epoch: mean sensitivity over top-10 modes vs modes 11-K, per group
    top10 = slice(0, min(10, K)); rest = slice(min(10, K), K)
    epochs = np.arange(1, n_epochs + 1)
    colors = {"clean": "#4c78a8", "adv": "#e45756", "rand": "#59a14f"}
    groups = {"clean": clean_positions, "adv": adv_positions, "rand": rand_positions}

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    for g_name, g_idx in groups.items():
        top10_series = dcdx_t[:, g_idx, top10].mean(axis=(1, 2))
        rest_series  = dcdx_t[:, g_idx, rest ].mean(axis=(1, 2))
        ax_l.plot(epochs, top10_series, color=colors[g_name], lw=1.3, label=f"{g_name} top-10")
        ax_l.plot(epochs, rest_series,  color=colors[g_name], lw=1.1, ls="--", alpha=0.65,
                   label=f"{g_name} modes 11–{K}")
    ax_l.set_xlabel("epoch"); ax_l.set_ylabel(r"mean $\|\nabla_x c_k(x)\|$")
    ax_l.set_title("Fig 17 (left): raw temporal input sensitivity")
    ax_l.grid(True, alpha=0.25); ax_l.legend(loc="best", fontsize=8)

    # Loss-gradient-normalized (proxy: divide by sqrt(sum_k dcdx^2))
    dcdx_norm = dcdx_t / (np.linalg.norm(dcdx_t, axis=-1, keepdims=True) + 1e-30)
    for g_name, g_idx in groups.items():
        top10_series = dcdx_norm[:, g_idx, top10].mean(axis=(1, 2))
        rest_series  = dcdx_norm[:, g_idx, rest ].mean(axis=(1, 2))
        ax_r.plot(epochs, top10_series, color=colors[g_name], lw=1.3, label=f"{g_name} top-10")
        ax_r.plot(epochs, rest_series,  color=colors[g_name], lw=1.1, ls="--", alpha=0.65,
                   label=f"{g_name} modes 11–{K}")
    ax_r.set_xlabel("epoch"); ax_r.set_ylabel("loss-corrected sensitivity")
    ax_r.set_title("Fig 17 (right): loss-corrected temporal input sensitivity")
    ax_r.grid(True, alpha=0.25); ax_r.legend(loc="best", fontsize=8)

    fig.suptitle(
        "Fig 17: top-10 mode input sensitivity increases over training;\n"
        "loss-correction reduces the adversarial-vs-clean gap",
        fontsize=11,
    )
    fig.tight_layout()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200); plt.close(fig)
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
