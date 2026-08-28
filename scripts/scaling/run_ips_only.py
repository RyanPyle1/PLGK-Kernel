# Copyright (C) 2026 Ryan Pyle
# SPDX-License-Identifier: MIT
"""Table 5 row 3 (Appendix A.18.1): ImageNet ResNet-18-GN IPS-only.

Full audit is prohibitive at ImageNet scale (M x N = 1.28M x 10k). IPS-only
is tractable at ~2.7x baseline overhead because the JL random projection
compresses the per-sample gradient tensor to ``ips_proj_dim`` features.

Paper's row 3 targets:
  ~2489s baseline -> ~6772s with IPS  (~2.7x)
  ~22.4 GiB peak CUDA
  ~41.3% top-1

Bring-your-own ImageNet:
  --data-root <PATH>       expected layout: <PATH>/train and <PATH>/val
                           (ImageFolder-style class subdirectories)

For a CIFAR sanity check without ImageNet, set --dataset cifar10.

Wall time: multi-GPU day at ImageNet scale. Do not attempt on CPU.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import psutil
import torch

from kernel_tools import (
    ResNet18GN, cifar10_loaders, imagenet_loaders, train_with_audit,
)


def _measure_memory(device: torch.device) -> tuple[float, float]:
    proc = psutil.Process(os.getpid())
    rss_mb = proc.memory_info().rss / (1024 * 1024)
    cuda_mb = 0.0
    if device.type == "cuda":
        cuda_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    return rss_mb, cuda_mb


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=["imagenet", "cifar10"], default="imagenet")
    ap.add_argument("--data-root", type=str, default=None,
                    help="Path to dataset root (ImageNet: train/val ImageFolder). "
                         "Required for --dataset imagenet.")
    ap.add_argument("--epochs", type=int, default=90)
    ap.add_argument("--train-batches", type=int, default=5005,
                    help="~1.28M / batch_size for full-epoch ImageNet. "
                         "Paper: 5005 at bs 256.")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--ips-proj-dim", type=int, default=1024)
    ap.add_argument("--audit-batches", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--out-json", type=str, default=None)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    tag = "imagenet" if args.dataset == "imagenet" else "cifar10"
    if args.out is None:
        args.out = f"data/{tag}_resnet18gn_ips_only.pt"
    if args.out_json is None:
        args.out_json = f"data/{tag}_resnet18gn_ips_only.json"

    if args.dataset == "imagenet" and args.data_root is None:
        raise SystemExit(
            "--data-root is required for --dataset imagenet. Point at a folder\n"
            "containing 'train' and 'val' subdirectories (ImageFolder layout)."
        )

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.reset_peak_memory_stats()

    device = torch.device(args.device) if args.device else (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    print(f"device: {device}   dataset: {args.dataset}")

    if args.dataset == "imagenet":
        model = ResNet18GN(device=device, num_classes=1000, variant="imagenet")
        train_loader, test_loader, audit_loader = imagenet_loaders(
            data_root=args.data_root,
            batch_size=args.batch_size,
            train_batches=args.train_batches,
            audit_batches=args.audit_batches,
            num_workers=args.num_workers,
        )
    else:
        model = ResNet18GN(device=device, num_classes=10, variant="cifar")
        train_loader, test_loader, audit_loader = cifar10_loaders(
            data_root=args.data_root or "data/CIFAR10",
            batch_size=args.batch_size,
            train_batches=args.train_batches,
            audit_batches=args.audit_batches,
            num_workers=args.num_workers,
            augment=False,
        )
    print(f"params: {model.numparams:,}")

    t0 = time.time()
    artifact = train_with_audit(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        audit_loader=audit_loader,
        epochs=args.epochs,
        train_batches=args.train_batches,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        # IPS-only: no full audit accumulator (would blow memory at 11M params
        # x 1.28M train samples). IPS's JL projection keeps this tractable.
        do_audit=False,
        do_ips=True,
        ips_proj_dim=args.ips_proj_dim,
        target_batches=args.audit_batches,
        verbose=True,
    )
    wall = time.time() - t0
    rss_mb, cuda_mb = _measure_memory(device)

    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, out_path)
    print(f"\nSaved artifact: {out_path}")

    summary = {
        "dataset": args.dataset,
        "model": "resnet18gn",
        "params": int(model.numparams),
        "epochs": args.epochs,
        "train_batches": args.train_batches,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "ips_proj_dim": args.ips_proj_dim,
        "wall_seconds": wall,
        "peak_rss_mb": rss_mb,
        "peak_cuda_mb": cuda_mb,
        "testacc_final": float(artifact["testacc"][-1]),
    }
    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary: {args.out_json}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
