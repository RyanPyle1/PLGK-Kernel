# Copyright (C) 2026 Ryan Pyle
# SPDX-License-Identifier: MIT
"""Group 2 headline: train + audit MNIST with an ADVERSARIAL audit set.

Section 6.3 pipeline. Loads the Group 1 trained model checkpoint, generates
Task A (paired clean/adv/random) and Task B (interpolation-ray) audit sets
via foolbox, then retrains the same model from scratch with the adversarial
audit set replacing the standard test-batch audit set. Also records:

  - per-epoch snapshots of NTKtrain_store (for §6.3 mode-SVD analysis;
    Figs 12, 13, 14)
  - per-epoch per-layer contribution accumulators (Fig 10)

Produces ``data/mnist_adv_audit.pt`` and ``data/mnist_adv_data.pt``
(the adv-data itself, for downstream figure scripts).

Wall time: ~2x Group 1 training (audit set is larger + we run twice for
Task A and Task B). CPU: ~3 hours. Single GPU: ~30 min.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from copy import deepcopy
import torch
from torch.utils.data import DataLoader, TensorDataset

from kernel_tools import (
    LNModel, mnist_loaders, train_with_audit,
    build_adversarial_audit_sets, DEFAULT_EPSILONS,
)


def _build_audit_loader(adv_x: torch.Tensor, adv_y: torch.Tensor, batch_size: int):
    """Wrap an (N, C, H, W) tensor + labels in a DataLoader with fixed order.

    We force the DataLoader batch size = N so the audit set is exactly one
    batch (matches ``target_batches=1``); the accumulator's per-batch
    indexing assumes each batch is full-size.
    """
    N = adv_x.shape[0]
    return DataLoader(
        TensorDataset(adv_x, adv_y), batch_size=N, shuffle=False, drop_last=False,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-artifact", type=str, default="data/mnist_experiment1.pt",
                    help="Group 1 artifact to load the trained model from")
    ap.add_argument("--out-adv-data", type=str, default="data/mnist_adv_data.pt")
    ap.add_argument("--out-task-a", type=str, default="data/mnist_adv_task_a.pt")
    ap.add_argument("--out-task-b", type=str, default="data/mnist_adv_task_b.pt")
    ap.add_argument("--data-root", type=str, default="data/MNIST")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--train-batches", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1.5e-5)
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--n-successful", type=int, default=16,
                    help="Number of successfully-attacked test examples to include")
    ap.add_argument("--n-source-test-batches", type=int, default=1,
                    help="Test batches to scan when hunting for successful examples")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--gpu-audit", action="store_true",
                    help="Keep the audit accumulator's large buffers on the "
                         "compute device. ~70x faster per step at MNIST scale "
                         "and numerically equivalent (see "
                         "scripts/verify_gpu_audit.py), but needs ~2.2 GB of "
                         "VRAM. Off by default: the CPU path is the reference.")
    ap.add_argument("--skip-task-b", action="store_true", help="skip the Task B ray audit")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device(args.device) if args.device else (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )

    # Optional accelerated accumulator. Equivalence to the CPU reference is
    # gated by scripts/verify_gpu_audit.py; run that before trusting results
    # produced with --gpu-audit on new hardware.
    if args.gpu_audit:
        if device.type != "cuda":
            raise SystemExit("--gpu-audit requires a CUDA device")
        from kernel_tools.train_gpu import train_with_audit_gpu as _train
        print("audit accumulator: GPU-resident (--gpu-audit)")
    else:
        _train = train_with_audit
    print(f"device: {device}")

    # -- Step 1: load the Group 1 trained model to attack --
    src = torch.load(args.source_artifact, weights_only=False)
    model = LNModel(device=device)
    model.load_state_dict(src["model_final_state"])
    print(f"loaded model from {args.source_artifact}")

    # -- Step 2: collect enough test images to hunt for successful examples --
    _train_loader, test_loader, _audit_loader = mnist_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        train_batches=args.train_batches,
        audit_batches=1,
    )
    imgs = []
    lbls = []
    for i, (x, y) in enumerate(test_loader):
        if i >= args.n_source_test_batches:
            break
        imgs.append(x); lbls.append(y)
    test_x = torch.cat(imgs, dim=0)
    test_label = torch.cat(lbls, dim=0)
    print(f"test batch for adversarial hunt: shape={tuple(test_x.shape)}")

    # -- Step 3: generate Task A + Task B audit sets --
    print("generating adversarial audit sets via foolbox ...")
    adv_bundle = build_adversarial_audit_sets(
        model=model,
        test_x=test_x,
        test_label=test_label,
        n_successful=args.n_successful,
        epsilons=DEFAULT_EPSILONS,
        device=device,
        seed=args.seed,
    )
    print(f"success_inds: {adv_bundle['success_inds']}")
    print(f"adv_data shape: {tuple(adv_bundle['adv_data'].shape)}")

    out_adv_path = Path(args.out_adv_data)
    out_adv_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(adv_bundle, out_adv_path)
    print(f"saved adv_data bundle to {out_adv_path}")

    # -- Step 4: Task A audit --
    print("\n=== Task A: retraining with adv_data as audit target ===")
    adv_a_loader = _build_audit_loader(
        adv_bundle["adv_data"], adv_bundle["adv_label"], args.batch_size,
    )
    torch.manual_seed(args.seed)  # reset RNG so we get the SAME training trajectory as Group 1
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    model_a = LNModel(device=device)
    train_loader_a, test_loader_a, _ = mnist_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        train_batches=args.train_batches,
        audit_batches=1,
    )
    # Audit loader returns a single batch of size N = adv_data.shape[0],
    # so target_batches=1 with audit_batch_size = N.
    n_audit = adv_bundle["adv_data"].shape[0]
    target_batches_a = 1
    audit_bs_a = n_audit

    # Per-epoch snapshots for downstream Fig 10, 12, 13:
    #   layer_contribs_per_epoch: list of dicts {layer_name: (2, n_audit)}
    #                              (2 = [signed_sum, abs_sum]); accumulated
    #                              *within* the epoch by summing step contribs.
    #   pntk_snapshots_per_epoch: list of tensors (n_train_samples, n_audit)
    #                              copied from audit.PNTK at end of each epoch.
    #   per_epoch_group_losses:   list of dicts {clean_mean/sd, adv_mean/sd, ...}
    layer_prefixes = ["conv1", "conv2", "fc1", "fc2", "fc3"]
    layer_indices = None
    per_epoch_layer_contribs: list[dict[str, torch.Tensor]] = []
    per_epoch_pntk: list[torch.Tensor] = []
    per_epoch_group_metrics: list[dict[str, float]] = []
    epoch_accum: dict[str, torch.Tensor] = {}

    def _step_cb(*, audit, ips, optimizer, epoch, idx, update, lr, **_):
        nonlocal layer_indices, epoch_accum
        if audit is None:
            return
        if layer_indices is None:
            layer_indices = audit.compute_layer_indices(layer_prefixes)
            # Announce the discovered layer indices exactly once
            print(f"  layer_indices: {[(k, len(v)) for k, v in layer_indices.items()]}")
        # Compute eta for this step
        eta = audit.adam_effective_lr(optimizer, epoch=epoch, idx=idx, train_batches=args.train_batches)
        # Per-layer step contribution (2, n_audit) per layer
        contribs = audit.last_step_contribution_by_layer(layer_indices, lr_use=lr, eta=eta)
        # Accumulate within this epoch (create tensors on first step of epoch)
        if not epoch_accum:
            for name, tensor in contribs.items():
                epoch_accum[name] = torch.zeros_like(tensor)
        for name, tensor in contribs.items():
            epoch_accum[name] += tensor

    def _epoch_cb(*, audit, ips, model, epoch, epochs, **_):
        nonlocal epoch_accum
        if audit is None:
            return
        # Snapshot per-layer accumulator + full PNTK
        per_epoch_layer_contribs.append({k: v.clone() for k, v in epoch_accum.items()})
        per_epoch_pntk.append(audit.PNTK.clone())
        # Group loss/influence per epoch: compute audit-set losses on the current model
        with torch.no_grad():
            adv_x_dev = adv_bundle["adv_data"].to(device)
            adv_y_dev = adv_bundle["adv_label"].to(device)
            logits = model(adv_x_dev)
            per_ex_loss = torch.nn.functional.cross_entropy(logits, adv_y_dev.long(), reduction="none").cpu()
        # Slot layout: within each 16-slot block, positions 0/4/8/12=clean,
        # 1/5/9/13=adv, {2,3,6,7,10,11,14,15}=random.
        n_ex = adv_bundle["adv_data"].shape[0] // 16
        clean_positions = [16 * i + off for i in range(n_ex) for off in (0, 4, 8, 12)]
        adv_positions = [16 * i + off for i in range(n_ex) for off in (1, 5, 9, 13)]
        rand_positions = [16 * i + off for i in range(n_ex) for off in (2, 3, 6, 7, 10, 11, 14, 15)]
        # Per-training-sample influence into each group (mean-abs over group's audit cols)
        pntk_abs = audit.PNTK.abs()
        I_clean = pntk_abs[:, clean_positions].mean(dim=1)
        I_adv = pntk_abs[:, adv_positions].mean(dim=1)
        I_rand = pntk_abs[:, rand_positions].mean(dim=1)
        per_epoch_group_metrics.append({
            "loss_mean_clean": float(per_ex_loss[clean_positions].mean()),
            "loss_sd_clean":   float(per_ex_loss[clean_positions].std()),
            "loss_mean_adv":   float(per_ex_loss[adv_positions].mean()),
            "loss_sd_adv":     float(per_ex_loss[adv_positions].std()),
            "loss_mean_rand":  float(per_ex_loss[rand_positions].mean()),
            "loss_sd_rand":    float(per_ex_loss[rand_positions].std()),
            "I_mean_clean":    float(I_clean.mean()),
            "I_sd_clean":      float(I_clean.std()),
            "I_mean_adv":      float(I_adv.mean()),
            "I_sd_adv":        float(I_adv.std()),
            "I_mean_rand":     float(I_rand.mean()),
            "I_sd_rand":       float(I_rand.std()),
        })
        # Reset epoch accumulator for next epoch
        epoch_accum = {}
        print(f"    epoch {epoch}: layer snapshots + PNTK snapshot + group losses recorded")

    artifact_a = _train(
        model=model_a,
        train_loader=train_loader_a,
        test_loader=test_loader_a,
        audit_loader=adv_a_loader,
        epochs=args.epochs,
        train_batches=args.train_batches,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        do_audit=True,
        do_ips=False,
        target_batches=target_batches_a,
        audit_batch_size=audit_bs_a,
        verbose=True,
        step_callback=_step_cb,
        epoch_callback=_epoch_cb,
    )
    artifact_a["per_epoch_layer_contribs"] = per_epoch_layer_contribs
    artifact_a["per_epoch_pntk"] = per_epoch_pntk
    artifact_a["per_epoch_group_metrics"] = per_epoch_group_metrics
    artifact_a["layer_prefixes"] = layer_prefixes
    artifact_a["adv_bundle_path"] = args.out_adv_data
    torch.save(artifact_a, args.out_task_a)
    print(f"saved Task A artifact to {args.out_task_a}")

    # -- Step 5: Task B audit (interpolation rays) --
    if not args.skip_task_b:
        print("\n=== Task B: retraining with adv_data2 as audit target ===")
        adv_b_loader = _build_audit_loader(
            adv_bundle["adv_data2"], adv_bundle["adv_label2"], args.batch_size,
        )
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
        model_b = LNModel(device=device)
        train_loader_b, test_loader_b, _ = mnist_loaders(
            data_root=args.data_root,
            batch_size=args.batch_size,
            train_batches=args.train_batches,
            audit_batches=1,
        )
        n_audit_b = adv_bundle["adv_data2"].shape[0]
        # Task B needs per-epoch PNTK snapshots (for Fig 11 ray analysis) but
        # no per-layer or per-group-slot decomposition (adv_data2 is a ray,
        # not a paired group set).
        per_epoch_pntk_b: list[torch.Tensor] = []
        def _epoch_cb_b(*, audit, ips, model, epoch, epochs, **_):
            if audit is None:
                return
            per_epoch_pntk_b.append(audit.PNTK.clone())
            print(f"    epoch {epoch}: PNTK snapshot recorded (Task B)")

        artifact_b = _train(
            model=model_b,
            train_loader=train_loader_b,
            test_loader=test_loader_b,
            audit_loader=adv_b_loader,
            epochs=args.epochs,
            train_batches=args.train_batches,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
            do_audit=True,
            do_ips=False,
            target_batches=1,
            audit_batch_size=n_audit_b,
            verbose=True,
            epoch_callback=_epoch_cb_b,
        )
        artifact_b["per_epoch_pntk"] = per_epoch_pntk_b
        artifact_b["adv_bundle_path"] = args.out_adv_data
        torch.save(artifact_b, args.out_task_b)
        print(f"saved Task B artifact to {args.out_task_b}")


if __name__ == "__main__":
    main()
