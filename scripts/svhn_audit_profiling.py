# Copyright (C) 2026 Ryan Pyle
# SPDX-License-Identifier: MIT
"""SVHN audit profiling table (Appendix A.17).

SVHN counterpart of ``mnist_audit_profiling.py``. The paper's SVHN table
has 4 variants (fewer than MNIST's 6):

  baseline        — training only, no audit
  audit           — standard trapezoidal audit (Group 5 default)
  supersample_2   — 2 gradient evaluations per step (higher fidelity)
  subsample_2     — audit every 2 steps (~1.3x speedup)

Same JSON schema as MNIST — ``svhn_table_a17_profiling.py`` collects them.

Wall time per variant on GPU: ~15-30 min at paper defaults. On CPU: several
hours per variant; reduce with ``--epochs`` / ``--train-batches`` for smoke
tests.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import psutil
import torch

from kernel_tools import SVHNModel, svhn_loaders, train_with_audit


def _measure_memory(device: torch.device) -> tuple[float, float]:
    proc = psutil.Process(os.getpid())
    rss_mb = proc.memory_info().rss / (1024 * 1024)
    cuda_mb = 0.0
    if device.type == "cuda":
        cuda_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    return rss_mb, cuda_mb


def _run_baseline(model, train_loader, test_loader, *, epochs, train_batches, batch_size, lr, device):
    from torch.nn import CrossEntropyLoss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = CrossEntropyLoss()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()  # no arg = current device
    t0 = time.time()
    for _ in range(epochs):
        for idx, (train_x, train_label) in enumerate(train_loader):
            if idx >= train_batches:
                break
            optimizer.zero_grad()
            pred = model(train_x.to(device).float())
            loss = loss_fn(pred, train_label.to(device).long())
            loss.backward()
            optimizer.step()
    wall = time.time() - t0
    correct = 0; total = 0
    with torch.no_grad():
        for x, y in test_loader:
            p = model(x.to(device).float()).argmax(dim=1)
            correct += int((p.cpu() == y).sum()); total += len(y)
    return {
        "variant": "baseline",
        "wall_seconds": wall,
        "testacc_final": correct / total,
        "final_mre": None,
        "final_corr": None,
    }


def _run_audit_variant(
    variant: str, model, train_loader, test_loader, audit_loader,
    *, epochs, train_batches, batch_size, lr, device,
):
    if variant == "audit":
        artifact = train_with_audit(
            model=model, train_loader=train_loader, test_loader=test_loader,
            audit_loader=audit_loader,
            epochs=epochs, train_batches=train_batches, batch_size=batch_size,
            lr=lr, device=device, do_audit=True, do_ips=False,
            target_batches=1, verbose=False,
        )
    elif variant == "supersample_2":
        # Same heuristic used in mnist_audit_profiling.py: rerun update()
        # once more per step, doubling accumulation cost. Approximation of
        # the paper's true supersample-2 variant.
        def _ss_step_cb(*, audit, ips, optimizer, epoch, idx, update, lr, **_):
            if audit is None:
                return
            for tx, ty in train_loader:
                audit.update(train_x=tx, train_label=ty, idx=idx,
                             audit_loader=audit_loader, optimizer=optimizer,
                             epoch=epoch, train_batches=train_batches, lr_use=lr)
                break
        artifact = train_with_audit(
            model=model, train_loader=train_loader, test_loader=test_loader,
            audit_loader=audit_loader,
            epochs=epochs, train_batches=train_batches, batch_size=batch_size,
            lr=lr, device=device, do_audit=True, do_ips=False,
            target_batches=1, verbose=False,
            step_callback=_ss_step_cb,
        )
    elif variant.startswith("subsample_"):
        N = int(variant.split("_")[-1])
        state = {"prev_pntk": None}
        def _sub_step_cb(*, audit, ips, optimizer, epoch, idx, update, lr, **_):
            if audit is None:
                return
            if (update + 1) % N != 0:
                if state["prev_pntk"] is not None:
                    audit.PNTK[:] = state["prev_pntk"]
            state["prev_pntk"] = audit.PNTK.clone()
        artifact = train_with_audit(
            model=model, train_loader=train_loader, test_loader=test_loader,
            audit_loader=audit_loader,
            epochs=epochs, train_batches=train_batches, batch_size=batch_size,
            lr=lr, device=device, do_audit=True, do_ips=False,
            target_batches=1, verbose=False,
            step_callback=_sub_step_cb,
        )
    else:
        raise ValueError(f"Unknown variant '{variant}'")
    return artifact


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variants", type=str, nargs="+",
                    default=["baseline", "audit", "supersample_2", "subsample_2"])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--train-batches", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--data-root", type=str, default="data/SVHN")
    ap.add_argument("--out-dir", type=str, default="data")
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    print(f"device: {device}")

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    for variant in args.variants:
        print(f"\n=== profiling variant: {variant} ===")
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.cuda.reset_peak_memory_stats()

        model = SVHNModel(device=device)
        train_loader, test_loader, audit_loader = svhn_loaders(
            data_root=args.data_root,
            batch_size=args.batch_size,
            train_batches=args.train_batches,
            audit_batches=1,
        )

        if variant == "baseline":
            row = _run_baseline(model, train_loader, test_loader,
                                 epochs=args.epochs, train_batches=args.train_batches,
                                 batch_size=args.batch_size, lr=args.lr, device=device)
        else:
            t0 = time.time()
            artifact = _run_audit_variant(
                variant, model, train_loader, test_loader, audit_loader,
                epochs=args.epochs, train_batches=args.train_batches,
                batch_size=args.batch_size, lr=args.lr, device=device,
            )
            wall = time.time() - t0
            row = {
                "variant": variant,
                "wall_seconds": wall,
                "testacc_final": float(artifact["testacc"][-1]),
                "final_mre": float(artifact["mles"][-1]),
                "final_corr": float(artifact["logitcorrs"][-1]),
            }
        rss_mb, cuda_mb = _measure_memory(device)
        row["peak_rss_mb"] = rss_mb
        row["peak_cuda_mb"] = cuda_mb
        out = out_dir / f"svhn_profiling_{variant}.json"
        with open(out, "w") as f:
            json.dump(row, f, indent=2)
        print(f"  wrote {out}")
        print(f"  {row}")


if __name__ == "__main__":
    main()
