# Copyright (C) 2026 Ryan Pyle
# SPDX-License-Identifier: MIT
"""Table 4 (Appendix A.12): audit vs TRAK reconstruction fidelity.

Trains the same LeNet-5 model as Group 1, saves 5 checkpoints (every 10
epochs), then runs TRAK attribution in two configurations:

  1-ckpt: single (final) checkpoint — cheapest
  5-ckpt: TRAK's recommended multi-checkpoint average

Both are compared to the actual loss change on the audit set via MRE and
Pearson correlation. Reference row is Group 1's audit MLE/Corr (loaded
from its artifact).

Wall time on CPU: ~2 hr training + ~2 min TRAK.
GPU: ~15 min training + ~1 min TRAK.

Depends on the ``traker`` package (``pip install traker``).
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import tempfile
import time
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset

from kernel_tools import LNModel, mnist_loaders


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference-audit", type=str, default="data/mnist_experiment1.pt",
                    help="Group 1 artifact — supplies audit reference MRE/Corr")
    ap.add_argument("--data-root", type=str, default="data/MNIST")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--train-batches", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1.5e-5)
    ap.add_argument("--proj-dim", type=int, default=2048)
    ap.add_argument("--checkpoint-epochs", type=int, nargs="+",
                    default=[9, 19, 29, 39, 49])
    ap.add_argument("--out-csv", type=str, default="figures/table4_trak_comparison.csv")
    ap.add_argument("--out-md", type=str, default="figures/table4_trak_comparison.md")
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    try:
        from trak import TRAKer
    except ImportError:
        raise SystemExit(
            "TRAK is required for this comparison but not installed.\n"
            "Install with: pip install traker\n"
        )

    device = torch.device(args.device) if args.device else (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    print(f"device: {device}")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    model = LNModel(device=device)
    train_loader, test_loader, audit_loader = mnist_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        train_batches=args.train_batches,
        audit_batches=1,
    )
    train_set_size = args.train_batches * args.batch_size
    train_subset = Subset(train_loader.dataset, list(range(train_set_size)))
    trak_train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=False)

    optimizer = Adam(model.parameters(), lr=args.lr)
    loss_fn = CrossEntropyLoss()

    ntarget = args.batch_size  # target_batches=1
    y_init = torch.zeros(ntarget, 10)
    y_target_labels = torch.zeros(ntarget)
    loss_init = torch.zeros(ntarget)
    with torch.no_grad():
        for idy, (test_x, test_label) in enumerate(audit_loader):
            if idy < 1:
                y_init[idy * args.batch_size:(idy + 1) * args.batch_size] = model(test_x.to(device).float()).cpu()
                y_target_labels[idy * args.batch_size:(idy + 1) * args.batch_size] = test_label
        for i in range(len(y_init)):
            loss_init[i] = loss_fn(y_init[i], y_target_labels[i].long())

    # --- Train + save checkpoints ---
    checkpoints: dict[int, dict] = {}
    print(f"Training for {args.epochs} epochs, saving checkpoints at {args.checkpoint_epochs} ...")
    t0 = time.time()
    for ep in range(args.epochs):
        for idx, (train_x, train_label) in enumerate(train_loader):
            if idx >= args.train_batches:
                break
            optimizer.zero_grad()
            pred = model(train_x.to(device).float())
            loss = loss_fn(pred, train_label.to(device).long())
            loss.backward()
            optimizer.step()
        if ep in args.checkpoint_epochs:
            checkpoints[ep] = deepcopy(model.state_dict())
        # test acc (light)
        correct = 0; total = 0
        with torch.no_grad():
            for x, y in test_loader:
                p = model(x.to(device).float()).argmax(dim=1)
                correct += int((p.cpu() == y).sum()); total += len(y)
        print(f"  epoch {ep:02d}: acc={correct/total:.4f}")
    train_time = time.time() - t0
    print(f"Training done in {train_time:.1f}s")

    # --- Final losses ---
    y_final = torch.zeros(ntarget, 10)
    loss_final = torch.zeros(ntarget)
    with torch.no_grad():
        for idy, (test_x, test_label) in enumerate(audit_loader):
            if idy < 1:
                y_final[idy * args.batch_size:(idy + 1) * args.batch_size] = model(test_x.to(device).float()).cpu()
        for i in range(len(y_final)):
            loss_final[i] = loss_fn(y_final[i], y_target_labels[i].long())
    actual_loss_change = (loss_init - loss_final).cpu().numpy()

    # --- TRAK attribution: 1-ckpt and 5-ckpt configs ---
    target_subset = Subset(test_loader.dataset, list(range(ntarget)))
    target_loader = DataLoader(target_subset, batch_size=args.batch_size, shuffle=False)

    trak_rows = []
    for config_name, ckpt_idxs in [("1ckpt", [args.checkpoint_epochs[-1]]),
                                    ("5ckpt", args.checkpoint_epochs)]:
        print(f"\n=== TRAK {config_name} ({len(ckpt_idxs)} checkpoints) ===")
        save_dir = Path(tempfile.gettempdir()) / f"trak_results_{config_name}"
        if save_dir.exists():
            shutil.rmtree(save_dir)
        trak_start = time.time()

        traker = TRAKer(
            model=LNModel(device=device),
            task="image_classification",
            train_set_size=train_set_size,
            save_dir=str(save_dir),
            load_from_save_dir=False,
            device=device,
            proj_dim=args.proj_dim,
            use_half_precision=False,
        )
        for model_id, ckpt_ep in enumerate(ckpt_idxs):
            traker.load_checkpoint(checkpoints[ckpt_ep], model_id=model_id)
            for batch in trak_train_loader:
                traker.featurize(batch=batch, num_samples=batch[0].shape[0])
        traker.finalize_features()

        for model_id, ckpt_ep in enumerate(ckpt_idxs):
            traker.start_scoring_checkpoint(
                exp_name=config_name, checkpoint=checkpoints[ckpt_ep],
                model_id=model_id, num_targets=ntarget,
            )
            for batch in target_loader:
                traker.score(batch=batch, num_samples=batch[0].shape[0])
        scores = traker.finalize_scores(exp_name=config_name)
        trak_time = time.time() - trak_start

        trak_sum = scores.sum(axis=0)
        # Least-squares rescale so TRAK and the audit are compared on shape
        # rather than absolute magnitude.
        alpha = -np.dot(actual_loss_change, trak_sum) / (np.dot(trak_sum, trak_sum) + 1e-12)
        scaled_trak = alpha * trak_sum
        scaled_mle = float(np.mean(np.abs(actual_loss_change - scaled_trak)))
        corr = float(np.corrcoef(actual_loss_change, -trak_sum)[0, 1])
        trak_rows.append({
            "method": f"TRAK ({config_name})",
            "mle": scaled_mle,
            "corr": corr,
            "time_s": train_time + trak_time,
        })
        print(f"  scaled MLE={scaled_mle:.6f}   corr={corr:.6f}   time={trak_time:.1f}s")

    # --- Reference: Group 1's audit MLE/Corr ---
    ref_row = {"method": "Audit (trapezoidal)", "mle": None, "corr": None, "time_s": None}
    if Path(args.reference_audit).exists():
        ref = torch.load(args.reference_audit, weights_only=False)
        ref_row["mle"] = float(ref["mles"][-1])
        ref_row["corr"] = float(abs(ref["logitcorrs"][-1]))
    else:
        print(f"[warn] reference audit not found at {args.reference_audit} — MLE/Corr left blank")

    rows = [ref_row] + trak_rows
    out_csv = Path(args.out_csv); out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "MLE", "correlation", "total_time_s"])
        for r in rows:
            w.writerow([r["method"],
                         "" if r["mle"] is None else f"{r['mle']:.6f}",
                         "" if r["corr"] is None else f"{r['corr']:.6f}",
                         "" if r["time_s"] is None else f"{r['time_s']:.1f}"])
    print(f"\nWrote CSV: {out_csv}")

    def _fmt(v, spec):
        return "" if v is None else format(v, spec)
    out_md = Path(args.out_md); out_md.parent.mkdir(parents=True, exist_ok=True)
    with open(out_md, "w") as f:
        f.write("# Table 4 — TRAK vs Audit reconstruction (MNIST, LeNet-5)\n\n")
        f.write("| Method | MLE ↓ | Correlation ↑ | Total Time (s) |\n|---|---:|---:|---:|\n")
        for r in rows:
            f.write(f"| {r['method']} | {_fmt(r['mle'], '.4f')} | "
                     f"{_fmt(r['corr'], '.4f')} | {_fmt(r['time_s'], '.1f')} |\n")
    print(f"Wrote MD: {out_md}")


if __name__ == "__main__":
    main()
