# Copyright (C) 2026 Ryan Pyle
# SPDX-License-Identifier: MIT
"""Verify GPUAuditAccumulator reproduces the CPU reference exactly.

Runs identical short training loops -- same seed, same init, same data
order -- with the reference CPU ``AuditAccumulator`` and with
``GPUAuditAccumulator``, then compares the resulting PNTK matrices
elementwise.

This gate exists because the GPU variant is a device-placement change to
the accumulator that produces the paper's core object. A silent mismatch
would corrupt every downstream population distance without any obvious
symptom, so it must be checked numerically rather than argued.

Tolerance: fp32 matmul on CPU (MKL/OpenBLAS) and on CUDA (cuBLAS) reduce in
different orders, so bitwise equality is not expected. We require agreement
to a relative tolerance appropriate for accumulated fp32 GEMM, and report
the actual observed deviation so drift is visible rather than hidden by a
loose threshold.

Usage:
    python scripts/verify_gpu_audit.py --steps 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from kernel_tools import LNModel, mnist_loaders
from kernel_tools.audit import AuditAccumulator
from kernel_tools.audit_gpu import GPUAuditAccumulator


def run(accum_cls, *, device, steps, train_batches, batch_size, lr, seed,
        data_root):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = LNModel(device=device)
    train_loader, _test, audit_loader = mnist_loaders(
        data_root=data_root, batch_size=batch_size,
        train_batches=train_batches, audit_batches=1)
    P = int(sum(p.numel() for p in model.parameters() if not p.is_meta))

    acc = accum_cls(
        model=model, numparams=P, ndat_train=train_batches * batch_size,
        n_audit=batch_size, batch_size=batch_size, device=device, nahead=50,
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lf = torch.nn.CrossEntropyLoss()

    acc.prime_test_gradients(audit_loader, 1)
    for idx, (x, y) in enumerate(train_loader):
        if idx >= steps:
            break
        opt.zero_grad()
        loss = lf(model(x.to(device).float()), y.to(device).long())
        loss.backward()
        acc.capture_train_gradients(x, y)
        opt.step()
        acc.update(train_x=x, train_label=y, idx=idx, audit_loader=audit_loader,
                   optimizer=opt, epoch=0, train_batches=train_batches,
                   lr_use=lr)
    return acc.PNTK.detach().cpu()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--train-batches", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1.5e-5)
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--data-root", type=str, default="data/MNIST")
    ap.add_argument("--rtol", type=float, default=2e-3,
                    help="Relative tolerance on max |PNTK| deviation")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable -- nothing to verify.")

    common = dict(steps=args.steps, train_batches=args.train_batches,
                  batch_size=args.batch_size, lr=args.lr, seed=args.seed,
                  data_root=args.data_root)

    print(f"Running CPU reference ({args.steps} steps) ...")
    pntk_cpu = run(AuditAccumulator, device=torch.device("cpu"), **common)
    print(f"Running GPU variant   ({args.steps} steps) ...")
    pntk_gpu = run(GPUAuditAccumulator, device=torch.device("cuda:0"), **common)

    if pntk_cpu.shape != pntk_gpu.shape:
        raise SystemExit(f"SHAPE MISMATCH: {pntk_cpu.shape} vs {pntk_gpu.shape}")

    diff = (pntk_cpu - pntk_gpu).abs()
    scale = pntk_cpu.abs().max().item()
    max_abs = diff.max().item()
    rel = max_abs / scale if scale > 0 else 0.0
    nz_cpu = int((pntk_cpu != 0).sum())

    print()
    print(f"  PNTK shape        : {tuple(pntk_cpu.shape)}")
    print(f"  nonzero entries   : {nz_cpu:,}")
    print(f"  max |PNTK|        : {scale:.6e}")
    print(f"  max abs deviation : {max_abs:.6e}")
    print(f"  max rel deviation : {rel:.3e}   (tolerance {args.rtol:.0e})")
    print(f"  mean abs deviation: {diff.mean().item():.6e}")
    # Correlation across all entries -- catches structural divergence that a
    # max-deviation check on a near-zero matrix could miss.
    a = pntk_cpu.flatten().double()
    b = pntk_gpu.flatten().double()
    if a.std() > 0 and b.std() > 0:
        corr = float(torch.corrcoef(torch.stack([a, b]))[0, 1])
        print(f"  corr(CPU, GPU)    : {corr:.10f}")
    else:
        corr = 1.0
        print("  corr(CPU, GPU)    : n/a (degenerate)")

    print()
    if nz_cpu == 0:
        raise SystemExit("FAIL: CPU PNTK is all zeros -- the test did no work.")
    if rel <= args.rtol and corr > 0.9999:
        print("PASS -- GPU accumulator matches the CPU reference.")
    else:
        raise SystemExit(
            f"FAIL -- deviation {rel:.3e} exceeds tolerance {args.rtol:.0e} "
            f"(corr {corr:.6f}). Do NOT use the GPU variant for real runs."
        )


if __name__ == "__main__":
    main()
