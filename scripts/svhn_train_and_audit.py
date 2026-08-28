# Copyright (C) 2026 Ryan Pyle
# SPDX-License-Identifier: MIT
"""Group 5 headline: train small CNN on SVHN + accumulate audit (+ optional IPS).

Reproduces the training + reconstruction pipeline behind Appendix A.16 and
A.17: Figures 28-31 (audit MRE + LC over training on SVHN), Figures 32-34
(SVHN IPS UMAP + difficulty), and Table A.17 (SVHN reconstruction summary).

Save format (torch.save, .pt): a single dict — see kernel_tools.train
docstring for keys.

Notes vs the MNIST driver:
  * Model is ``SVHNModel`` (~140k params — ~3x LeNet).
  * IPS uses a Johnson-Lindenstrauss random projection to ``ips_proj_dim``
    features (paper default 512) so that the (n_total, feat_dim) grad
    scratch fits in memory. Without this, SVHN's per-sample gradient matrix
    would be ~30 GiB.
  * Learning rate is 5e-5 (the paper's SVHN baseline).

Wall time on CPU: ~3-4 hr audit-only, ~5-6 hr with --do-ips.
Single consumer GPU: ~30 min audit-only, ~60 min with --do-ips.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from kernel_tools import SVHNModel, svhn_loaders, train_with_audit


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--train-batches", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-5,
                    help="Adam learning rate (paper's SVHN default: 5e-5)")
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--device", type=str, default=None,
                    help="'cpu', 'cuda:0', etc. Default: cuda:0 if available else cpu.")
    ap.add_argument("--do-ips", action="store_true",
                    help="also accumulate IPS (~2x wall time)")
    ap.add_argument("--ips-proj-dim", type=int, default=512,
                    help="JL projection dim for IPS (paper: 512). "
                         "Ignored unless --do-ips is set.")
    ap.add_argument("--out", type=str, default="data/svhn_experiment1.pt")
    ap.add_argument("--data-root", type=str, default="data/SVHN")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device(args.device) if args.device else (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    print(f"device: {device}")

    model = SVHNModel(device=device)
    print(f"params: {model.numparams}  non-meta tensors: {model.num_nonmeta}")

    train_loader, test_loader, audit_loader = svhn_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        train_batches=args.train_batches,
        audit_batches=1,
    )

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
        target_batches=1,
        verbose=True,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, out_path)
    print(f"\nSaved: {out_path}")
    print(f"Final acc: {artifact['testacc'][-1]:.4f}")
    print(f"Final MRE: {artifact['mles'][-1]:.4g}   Final Corr: {artifact['logitcorrs'][-1]:.6f}")


if __name__ == "__main__":
    main()
