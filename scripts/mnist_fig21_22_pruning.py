# Copyright (C) 2026 Ryan Pyle
# SPDX-License-Identifier: MIT
"""Figures 21 and 22: MNIST data-pruning accuracy curves (Appendix A.4).

Fig 21 = "constructive" pruning — removing the LEAST important data:
  most methods stay near baseline until 40-60% pruned; some (esp.
  class-balanced least-impact) even improve slightly.

Fig 22 = "destructive" pruning — removing the MOST important data:
  informed pruning drops accuracy much faster than random, confirming
  that the audit + IPS successfully identify high-value samples.

Consumes ``data/mnist_pruning_results.pkl`` (from ``mnist_data_pruning.py``).
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CONSTRUCTIVE_METHODS = {
    "least_impact_abs":    ("Cut Least Impactful (|I|)",         "#1f77b4"),
    "least_impact_signed": ("Cut Least Impactful (signed)",      "#2ca02c"),
    "cluster_prune":       ("IPS Cluster Prune",                 "#ff7f0e"),
    "cluster_upweight":    ("IPS Cluster (upweight)",            "#e377c2"),
    "class_bal_least":     ("Class-Bal Least Impactful",         "#8c564b"),
}
DESTRUCTIVE_METHODS = {
    "cut_most_helpful":      ("Cut Most Helpful",             "#d62728"),
    "cut_highest_abs":       ("Cut Highest Impact (|I|)",     "#1f77b4"),
    "cut_highest_signed":    ("Cut Highest Impact (signed)",  "#2ca02c"),
    "cut_smallest_clusters": ("Cut Smallest Clusters (IPS)",  "#ff7f0e"),
    "cut_classbal_highest":  ("Cut Class-Bal Highest Impact", "#9467bd"),
    "most_harmful":          ("Cut Most Harmful",             "#e377c2"),
    "most_redundant":        ("Cut Most Redundant",           "#8c564b"),
}
RANDOM_METHODS = [
    ("random",           "#7f7f7f", "Random (seeds)"),
    ("random_class_bal", "#bcbd22", "Random class-bal (seeds)"),
]


def _plot_deterministic(ax, results, ks, method_dict):
    for key, (label, color) in method_dict.items():
        if key not in results:
            continue
        try:
            accs = [results[key][k]["final_acc"] for k in ks]
        except KeyError:
            continue
        ax.plot(ks, accs, "o-", color=color, label=label, linewidth=2, markersize=5)


def _plot_random(ax, results, ks):
    for key, color, label in RANDOM_METHODS:
        if key not in results:
            continue
        means = np.array([
            np.mean([r["final_acc"] for r in results[key][k]]) for k in ks
        ])
        stds = np.array([
            np.std([r["final_acc"] for r in results[key][k]]) for k in ks
        ])
        ax.plot(ks, means, "s--", color=color, label=label, linewidth=1.5, markersize=5)
        ax.fill_between(ks, means - stds, means + stds, alpha=0.15, color=color)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pruning-results", type=str, default="data/mnist_pruning_results.pkl")
    ap.add_argument("--out-fig21", type=str, default="figures/fig21_pruning_constructive.png")
    ap.add_argument("--out-fig22", type=str, default="figures/fig22_pruning_destructive.png")
    args = ap.parse_args()

    with open(args.pruning_results, "rb") as f:
        payload = pickle.load(f)
    results = payload["results"]
    ks = payload["ks"]

    if "baseline" in results:
        baseline_acc = results["baseline"][0]["final_acc"]
    else:
        baseline_acc = None

    # -- Fig 21: constructive pruning --
    fig, ax = plt.subplots(figsize=(10, 6))
    if baseline_acc is not None:
        ax.axhline(baseline_acc, color="black", linestyle="--", linewidth=1.5,
                    label="Baseline (no pruning)")
    _plot_deterministic(ax, results, ks, CONSTRUCTIVE_METHODS)
    _plot_random(ax, results, ks)
    ax.set_xlabel("Data Pruned (%)", fontsize=13)
    ax.set_ylabel("Test Accuracy", fontsize=13)
    ax.set_title(
        "Fig 21 — MNIST constructive pruning: Test Accuracy vs Fraction Removed\n"
        "(excludes most_harmful and most_redundant which degrade sharply)",
        fontsize=12,
    )
    ax.set_xticks(ks); ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    out21 = Path(args.out_fig21); out21.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out21, dpi=200); plt.close(fig)
    print(f"Wrote: {out21}")

    # -- Fig 22: destructive pruning --
    fig, ax = plt.subplots(figsize=(10, 6))
    if baseline_acc is not None:
        ax.axhline(baseline_acc, color="black", linestyle="--", linewidth=1.5,
                    label="Baseline (no pruning)")
    _plot_deterministic(ax, results, ks, DESTRUCTIVE_METHODS)
    _plot_random(ax, results, ks)
    ax.set_xlabel("Data Pruned (%)", fontsize=13)
    ax.set_ylabel("Test Accuracy", fontsize=13)
    ax.set_title(
        "Fig 22 — MNIST destructive pruning: removing the MOST important data",
        fontsize=13,
    )
    ax.set_xticks(ks); ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="lower left")
    fig.tight_layout()
    out22 = Path(args.out_fig22); out22.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out22, dpi=200); plt.close(fig)
    print(f"Wrote: {out22}")


if __name__ == "__main__":
    main()
