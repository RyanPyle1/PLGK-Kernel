# Copyright (C) 2026 Ryan Pyle
# SPDX-License-Identifier: MIT
"""Table 5 rows 1 & 2 (Appendix A.18.1 & A.18.2): CIFAR-10 audit at scale.

  --model resnet18gn     -> Table 5 row 1  (11.2M params, ~56.6% top-1 at 120ep,
                           audit ~8.5x baseline, ~6.0 GiB CUDA peak)
  --model vittiny4       -> Table 5 row 2  (2.70M params, ~46.8% top-1 at 120ep,
                           audit ~11.3x baseline, ~4.6 GiB CUDA peak)

Wall-time targets (single A100, paper's setup):
  resnet18gn baseline ~15s/epoch, audit ~126s/epoch  -> ~4h at 120 epochs
  vittiny4   baseline ~16s/epoch, audit ~178s/epoch  -> ~6h at 120 epochs

On a consumer GPU (RTX 3060/4070 tier), multiply by ~3-5x.
On CPU, this is impractical (~50-100x); use --epochs 1 --train-batches 5
for smoke tests only.

Wraps ``train_with_audit`` with a full audit accumulator + optional IPS
(JL-projected to keep memory in check for large models).

Emits an artifact ``.pt`` with the standard train_with_audit schema plus
wall-time and peak-memory JSON at ``--out-json``.
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
    ResNet18GN, ViTTiny4, cifar10_loaders, train_with_audit,
)


def _measure_memory(device: torch.device) -> tuple[float, float]:
    proc = psutil.Process(os.getpid())
    rss_mb = proc.memory_info().rss / (1024 * 1024)
    cuda_mb = 0.0
    if device.type == "cuda":
        cuda_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    return rss_mb, cuda_mb


def _build_model(name: str, device: torch.device) -> torch.nn.Module:
    if name == "resnet18gn":
        return ResNet18GN(device=device, num_classes=10, variant="cifar")
    if name == "vittiny4":
        return ViTTiny4(device=device, num_classes=10)
    raise ValueError(f"Unknown model {name!r}. Choose resnet18gn or vittiny4.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=["resnet18gn", "vittiny4"], required=True)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--train-batches", type=int, default=390,
                    help="~50k / batch_size for full-epoch. Paper: 390 at bs 128.")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=None,
                    help="Adam LR. Default: 1e-3 for resnet, 1.5e-6 for ViT "
                         "(paper's chosen ViT LR; DeiT-standard 5e-4 lowers "
                         "audit corr to ~0.28).")
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--do-ips", action="store_true",
                    help="also accumulate IPS (adds ~1.5-2x wall time)")
    ap.add_argument("--ips-proj-dim", type=int, default=1024,
                    help="JL projection dim for IPS. Paper: 1024 for scale.")
    ap.add_argument("--audit-batches", type=int, default=1,
                    help="How many test batches form the audit set (paper: 1).")
    ap.add_argument("--data-root", type=str, default="data/CIFAR10")
    ap.add_argument("--out", type=str, default=None,
                    help="Artifact .pt path (default: data/cifar10_<model>_audit.pt)")
    ap.add_argument("--out-json", type=str, default=None,
                    help="Timing / memory JSON (default: data/cifar10_<model>_audit.json)")
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    if args.lr is None:
        args.lr = 1e-3 if args.model == "resnet18gn" else 1.5e-6
    if args.out is None:
        args.out = f"data/cifar10_{args.model}_audit.pt"
    if args.out_json is None:
        args.out_json = f"data/cifar10_{args.model}_audit.json"

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.reset_peak_memory_stats()

    device = torch.device(args.device) if args.device else (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    print(f"device: {device}  model: {args.model}  lr: {args.lr}")

    model = _build_model(args.model, device)
    print(f"params: {model.numparams:,}   non-meta tensors: {model.num_nonmeta}")

    train_loader, test_loader, audit_loader = cifar10_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        train_batches=args.train_batches,
        audit_batches=args.audit_batches,
        augment=False,
    )

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
        do_audit=True,
        do_ips=args.do_ips,
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
        "model": args.model,
        "params": int(model.numparams),
        "epochs": args.epochs,
        "train_batches": args.train_batches,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "wall_seconds": wall,
        "peak_rss_mb": rss_mb,
        "peak_cuda_mb": cuda_mb,
        "testacc_final": float(artifact["testacc"][-1]),
        "final_mre": float(artifact["mles"][-1]),
        "final_corr": float(artifact["logitcorrs"][-1]),
    }
    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary: {args.out_json}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
