# Copyright (C) 2026 Ryan Pyle
# SPDX-License-Identifier: MIT
"""Group 4 headline: data-pruning retrain experiments (Appendix A.4).

For each (method × k-percent) combination, load Group 1's model
initialization, prune the training set per the method's index selector,
retrain for exactly ``total_updates`` steps (matching Group 1 without
audit), and record final test accuracy.

Methods:
  - baseline (no pruning)
  - most_harmful, least_impact_abs, least_impact_signed,
    most_redundant, cluster_prune, cluster_upweight, class_bal_least
  - random, random_class_bal (each run over 5 seeds for confidence bands)

Cross-group dependency: needs Group 1's artifact WITH ``--do-ips``
(``data/mnist_experiment1.pt``) — the pruning script uses both ``PNTK``
and ``IPS``.

Wall time: roughly (n_methods × n_ks + 2 × n_ks × 5_seeds) × Group-1
train time. With paper defaults (6 det methods × 7 k-values + 2 random ×
7 × 5 seeds = 112 retrains × ~15 min / GPU each) = ~28 hours on a single
consumer GPU. Reduce via ``--ks``, ``--methods``, ``--random-seeds`` for
smoke tests.
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset

from kernel_tools import (
    LNModel, mnist_loaders,
    compute_pruning_scores, compute_ips_spectral_clusters,
    get_harmful_indices, get_least_impactful_abs_indices,
    get_least_impactful_signed_indices, get_most_redundant_indices,
    get_class_balanced_least_impactful_indices,
    get_random_indices, get_random_class_balanced_indices,
    get_cluster_prune_indices,
    # destructive (Fig 22)
    get_most_helpful_indices, get_highest_impact_abs_indices,
    get_highest_impact_signed_indices, get_class_bal_highest_impact_indices,
    get_smallest_clusters_indices,
)


DEFAULT_KS = [5, 10, 20, 30, 40, 50, 60]
DEFAULT_DET_METHODS = [
    "most_harmful", "least_impact_abs", "least_impact_signed",
    "most_redundant", "cluster_prune", "class_bal_least",
]
DEFAULT_DESTRUCT_METHODS = [
    "cut_most_helpful", "cut_highest_abs", "cut_highest_signed",
    "cut_smallest_clusters", "cut_classbal_highest",
]
DEFAULT_RANDOM_SEEDS = [0, 1, 2, 3, 4]


def _retrain(
    train_dataset, test_dataset, model_init, *,
    keep_indices, sample_weights=None,
    batch_size, lr, total_updates, eval_every, device,
):
    """Single retrain run. Returns (trainloss_arr, testacc_arr, final_acc, wall_seconds)."""
    model = LNModel(device=device)
    model.load_state_dict(deepcopy(model_init))
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = CrossEntropyLoss(reduction="none")

    subset = Subset(train_dataset, list(keep_indices))
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    trainloss_curve: list[float] = []
    testacc_curve: list[float] = []
    n_keep = len(keep_indices)

    update_count = 0
    epoch_count = 0
    t0 = time.time()
    while update_count < total_updates:
        for idx, (train_x, train_label) in enumerate(loader):
            if update_count >= total_updates:
                break
            optimizer.zero_grad()
            pred = model(train_x.to(device).float())
            losses = loss_fn(pred, train_label.to(device).long())
            if sample_weights is not None:
                batch_start = idx * batch_size
                batch_end = min(batch_start + len(train_x), n_keep)
                w = torch.tensor(sample_weights[batch_start:batch_end],
                                  device=device, dtype=torch.float32)
                loss = (losses * w).mean()
            else:
                loss = losses.mean()
            loss.backward()
            optimizer.step()
            trainloss_curve.append(loss.item())
            update_count += 1
        epoch_count += 1
        is_last = update_count >= total_updates
        if epoch_count % eval_every == 0 or is_last:
            correct = 0; total = 0
            with torch.no_grad():
                for test_x, test_label in test_loader:
                    pred = model(test_x.to(device).float()).argmax(dim=1)
                    correct += (pred.cpu() == test_label).sum().item()
                    total += len(test_label)
            testacc_curve.append(correct / total)
    wall = time.time() - t0
    final_acc = testacc_curve[-1] if testacc_curve else 0.0
    return np.array(trainloss_curve), np.array(testacc_curve), final_acc, wall


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-artifact", type=str, default="data/mnist_experiment1.pt",
                    help="Group 1 artifact WITH IPS (run mnist_train_and_audit.py --do-ips)")
    ap.add_argument("--data-root", type=str, default="data/MNIST")
    ap.add_argument("--out", type=str, default="data/mnist_pruning_results.pkl")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--train-batches", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1.5e-5)
    ap.add_argument("--epoch-base", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--ks", type=int, nargs="+", default=DEFAULT_KS,
                    help="Pruning percentages")
    ap.add_argument("--methods", type=str, nargs="+",
                    default=DEFAULT_DET_METHODS + ["cluster_upweight"] + DEFAULT_DESTRUCT_METHODS,
                    help="Methods to run (constructive + destructive + cluster_upweight)")
    ap.add_argument("--random-seeds", type=int, nargs="+", default=DEFAULT_RANDOM_SEEDS)
    ap.add_argument("--n-clusters", type=int, default=100)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--skip-baseline", action="store_true")
    ap.add_argument("--skip-random", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device(args.device) if args.device else (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    print(f"device: {device}")

    n_train = args.train_batches * args.batch_size
    total_updates = args.train_batches * args.epoch_base
    print(f"n_train={n_train}  total_updates={total_updates}")

    # --- Load Group 1 artifact ---
    src = torch.load(args.source_artifact, weights_only=False)
    if src.get("PNTK") is None or src.get("IPS") is None:
        raise SystemExit(
            f"{args.source_artifact} is missing PNTK or IPS. "
            "Rerun Group 1's mnist_train_and_audit.py with --do-ips."
        )
    PNTK = src["PNTK"].numpy()                    # (n_train, n_audit)
    IPS_full = src["IPS"]
    IPS_full_np = IPS_full.numpy() if hasattr(IPS_full, "numpy") else np.asarray(IPS_full)
    IPS_train = IPS_full_np[:n_train, :n_train]
    model_init = src["model_init_state"]
    print(f"PNTK shape: {PNTK.shape}   IPS_train shape: {IPS_train.shape}")

    # --- Load data ---
    train_loader_full, test_loader, _ = mnist_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        train_batches=args.train_batches,
        audit_batches=1,
    )
    train_dataset = train_loader_full.dataset
    test_dataset = test_loader.dataset
    train_labels = np.array([train_dataset.targets[i].item() for i in range(n_train)])

    # --- Compute pruning scores + IPS clusters ---
    scores = compute_pruning_scores(PNTK, train_labels)
    print(f"Score summaries: harm min={scores.harm_scores.min():.4f} max={scores.harm_scores.max():.4f}")
    cluster_labels, embedding, cluster_sizes = compute_ips_spectral_clusters(
        IPS_train, n_clusters=args.n_clusters, seed=42,
    )

    det_dispatchers = {
        # Constructive (Fig 21)
        "most_harmful":        lambda k: get_harmful_indices(scores, k, n_train),
        "least_impact_abs":    lambda k: get_least_impactful_abs_indices(scores, k, n_train),
        "least_impact_signed": lambda k: get_least_impactful_signed_indices(scores, k, n_train),
        "most_redundant":      lambda k: get_most_redundant_indices(scores, k, n_train),
        "cluster_prune":       lambda k: get_cluster_prune_indices(
            cluster_labels, cluster_sizes, embedding, k, n_train, with_upweight=False),
        "class_bal_least":     lambda k: get_class_balanced_least_impactful_indices(scores, k, n_train),
        # Destructive (Fig 22)
        "cut_most_helpful":      lambda k: get_most_helpful_indices(scores, k, n_train),
        "cut_highest_abs":       lambda k: get_highest_impact_abs_indices(scores, k, n_train),
        "cut_highest_signed":    lambda k: get_highest_impact_signed_indices(scores, k, n_train),
        "cut_smallest_clusters": lambda k: get_smallest_clusters_indices(
            cluster_labels, cluster_sizes, embedding, k, n_train),
        "cut_classbal_highest":  lambda k: get_class_bal_highest_impact_indices(scores, k, n_train),
    }

    results: dict = {}

    if not args.skip_baseline:
        print("=" * 60)
        print("Baseline (no pruning) ...")
        tl, ta, acc, sec = _retrain(
            train_dataset, test_dataset, model_init,
            keep_indices=np.arange(n_train), sample_weights=None,
            batch_size=args.batch_size, lr=args.lr,
            total_updates=total_updates, eval_every=args.eval_every, device=device,
        )
        results["baseline"] = {0: {"trainloss": tl, "testacc": ta, "final_acc": acc, "time": sec}}
        print(f"  baseline: acc={acc:.4f}, time={sec:.1f}s")

    for method in args.methods:
        if method == "cluster_upweight":
            # Handled separately (returns weights too)
            results[method] = {}
            for k in args.ks:
                print(f"cluster_upweight k={k}% ...")
                keep, weights = get_cluster_prune_indices(
                    cluster_labels, cluster_sizes, embedding, k, n_train, with_upweight=True,
                )
                tl, ta, acc, sec = _retrain(
                    train_dataset, test_dataset, model_init,
                    keep_indices=keep, sample_weights=weights,
                    batch_size=args.batch_size, lr=args.lr,
                    total_updates=total_updates, eval_every=args.eval_every, device=device,
                )
                results[method][k] = {"trainloss": tl, "testacc": ta, "final_acc": acc, "time": sec}
                print(f"  cluster_upweight k={k}%: acc={acc:.4f} time={sec:.1f}s")
            continue
        if method not in det_dispatchers:
            print(f"[warn] unknown method '{method}' — skipping")
            continue
        results[method] = {}
        for k in args.ks:
            print(f"{method} k={k}% ...")
            keep = det_dispatchers[method](k)
            tl, ta, acc, sec = _retrain(
                train_dataset, test_dataset, model_init,
                keep_indices=keep, sample_weights=None,
                batch_size=args.batch_size, lr=args.lr,
                total_updates=total_updates, eval_every=args.eval_every, device=device,
            )
            results[method][k] = {"trainloss": tl, "testacc": ta, "final_acc": acc, "time": sec}
            print(f"  {method} k={k}%: acc={acc:.4f}, time={sec:.1f}s, kept={len(keep)}")

    if not args.skip_random:
        for variant_name, idx_fn in [
            ("random",           lambda k, s: get_random_indices(k, n_train, s)),
            ("random_class_bal", lambda k, s: get_random_class_balanced_indices(k, n_train, train_labels, s)),
        ]:
            results[variant_name] = {}
            for k in args.ks:
                seed_runs = []
                for seed_i in args.random_seeds:
                    keep = idx_fn(k, seed_i)
                    tl, ta, acc, sec = _retrain(
                        train_dataset, test_dataset, model_init,
                        keep_indices=keep, sample_weights=None,
                        batch_size=args.batch_size, lr=args.lr,
                        total_updates=total_updates, eval_every=args.eval_every, device=device,
                    )
                    seed_runs.append({"trainloss": tl, "testacc": ta, "final_acc": acc, "time": sec})
                results[variant_name][k] = seed_runs
                accs = [r["final_acc"] for r in seed_runs]
                print(f"  {variant_name} k={k}%: acc={np.mean(accs):.4f} +/- {np.std(accs):.4f}")

    # --- Save ---
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump({
            "results": results, "ks": args.ks,
            "n_train": n_train, "total_updates": total_updates,
        }, f)
    print(f"\nWrote pruning results to {out}")


if __name__ == "__main__":
    main()
