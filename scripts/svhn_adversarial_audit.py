# Copyright (C) 2026 Ryan Pyle
# SPDX-License-Identifier: MIT
"""Group 5 SVHN adversarial driver (Appendix A.17.3).

Analogous to Group 2 (MNIST adversarial audit) but simplified per the paper:
SVHN's section runs only "Task A" (paired clean/adv/random audit) — no Task B
interpolation rays and no mode-directed cure.

Pipeline:
  1. Load the Group 5 pre-trained SVHN model.
  2. Use foolbox to generate FGSM / PGD-Linf / DeepFool / CW attacks on the
     first test batch; keep N successful examples in the paper's slot layout
     (16 slots per example: 4 attacks x [clean, adv, rand1, rand2]).
  3. Retrain the SVHN model from scratch with the adversarial audit set as
     the audit target, capturing per-epoch snapshots for downstream figures.

Per-epoch snapshots recorded (for Figs 37, 38, 39, 40):
  * per-layer contribution accumulators (per-layer breakdown of PNTK)
  * full PNTK snapshot
  * per-group (clean/adv/random) mean loss + mean-abs influence

Outputs:
  data/svhn_adv_data.pt          — adv_bundle from build_adversarial_audit_sets
  data/svhn_adv_task_a.pt        — Task-A artifact + snapshots

Wall time: ~2x Group 5 training. GPU: ~1 hr; CPU: several hours.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader, TensorDataset

from kernel_tools import (
    SVHNModel, svhn_loaders, train_with_audit,
    build_adversarial_audit_sets, DEFAULT_EPSILONS,
    lowrank_svd_train_side,
)


SVHN_LAYER_PREFIXES = ["conv1", "conv2", "conv3", "conv4", "conv5", "fc"]


def _build_audit_loader(adv_x: torch.Tensor, adv_y: torch.Tensor):
    N = adv_x.shape[0]
    return DataLoader(
        TensorDataset(adv_x, adv_y), batch_size=N, shuffle=False, drop_last=False,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-artifact", type=str, default="data/svhn_experiment1.pt",
                    help="Group 5 artifact to load the trained model from")
    ap.add_argument("--out-adv-data", type=str, default="data/svhn_adv_data.pt")
    ap.add_argument("--out-task-a", type=str, default="data/svhn_adv_task_a.pt")
    ap.add_argument("--data-root", type=str, default="data/SVHN")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--train-batches", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--n-successful", type=int, default=16,
                    help="Number of successfully-attacked test examples to include")
    ap.add_argument("--n-source-test-batches", type=int, default=1,
                    help="Test batches to scan when hunting for successful examples")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--svd-k", type=int, default=128,
                    help="Rank of final-checkpoint train-side SVD (paper: 128). "
                         "Used by Fig 39 mode-activity analysis (adv vs clean).")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device(args.device) if args.device else (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    print(f"device: {device}")

    # -- Step 1: load pre-trained SVHN model --
    src = torch.load(args.source_artifact, weights_only=False)
    model = SVHNModel(device=device)
    model.load_state_dict(src["model_final_state"])
    print(f"loaded model from {args.source_artifact}")

    # -- Step 2: gather a batch of test images --
    _train_loader, test_loader, _audit_loader = svhn_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        train_batches=args.train_batches,
        audit_batches=1,
    )
    imgs, lbls = [], []
    for i, (x, y) in enumerate(test_loader):
        if i >= args.n_source_test_batches:
            break
        imgs.append(x); lbls.append(y)
    test_x = torch.cat(imgs, dim=0)
    test_label = torch.cat(lbls, dim=0)
    print(f"test batch for adversarial hunt: shape={tuple(test_x.shape)}")

    # -- Step 3: generate Task A audit set --
    # NOTE: build_adversarial_audit_sets always emits both adv_data (Task A
    # slots) and adv_data2 (Task B ray). The SVHN section uses only Task A;
    # we still keep the whole bundle in the .pt for reproducibility.
    # SVHN images come normalized (mean/std) so pixel values are not in
    # [0, 1]. foolbox verifies inputs against model.bounds — pass the actual
    # observed range from the current batch, padded a hair for safety.
    lo = float(test_x.min()) - 1e-3
    hi = float(test_x.max()) + 1e-3
    print(f"observed input bounds: ({lo:.3f}, {hi:.3f})")
    print("generating adversarial audit sets via foolbox ...")
    adv_bundle = build_adversarial_audit_sets(
        model=model,
        test_x=test_x,
        test_label=test_label,
        n_successful=args.n_successful,
        epsilons=DEFAULT_EPSILONS,
        bounds=(lo, hi),
        device=device,
        seed=args.seed,
    )
    print(f"success_inds: {adv_bundle['success_inds']}")
    print(f"adv_data shape: {tuple(adv_bundle['adv_data'].shape)}")

    out_adv = Path(args.out_adv_data)
    out_adv.parent.mkdir(parents=True, exist_ok=True)
    torch.save(adv_bundle, out_adv)
    print(f"saved adv bundle to {out_adv}")

    # -- Step 4: Task A audit --
    print("\n=== Task A: retraining SVHN with adv_data as audit target ===")
    adv_loader = _build_audit_loader(adv_bundle["adv_data"], adv_bundle["adv_label"])
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    model_a = SVHNModel(device=device)
    train_loader_a, test_loader_a, _ = svhn_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        train_batches=args.train_batches,
        audit_batches=1,
    )
    n_audit = adv_bundle["adv_data"].shape[0]

    per_epoch_layer_contribs: list[dict[str, torch.Tensor]] = []
    per_epoch_pntk: list[torch.Tensor] = []
    per_epoch_group_metrics: list[dict[str, float]] = []
    layer_indices = None
    epoch_accum: dict[str, torch.Tensor] = {}

    def _step_cb(*, audit, ips, optimizer, epoch, idx, update, lr, **_):
        nonlocal layer_indices, epoch_accum
        if audit is None:
            return
        if layer_indices is None:
            layer_indices = audit.compute_layer_indices(SVHN_LAYER_PREFIXES)
            print(f"  layer_indices: {[(k, len(v)) for k, v in layer_indices.items()]}")
        eta = audit.adam_effective_lr(optimizer, epoch=epoch, idx=idx, train_batches=args.train_batches)
        contribs = audit.last_step_contribution_by_layer(layer_indices, lr_use=lr, eta=eta)
        if not epoch_accum:
            for name, tensor in contribs.items():
                epoch_accum[name] = torch.zeros_like(tensor)
        for name, tensor in contribs.items():
            epoch_accum[name] += tensor

    def _epoch_cb(*, audit, ips, model, epoch, epochs, **_):
        nonlocal epoch_accum
        if audit is None:
            return
        per_epoch_layer_contribs.append({k: v.clone() for k, v in epoch_accum.items()})
        per_epoch_pntk.append(audit.PNTK.clone())
        with torch.no_grad():
            adv_x_dev = adv_bundle["adv_data"].to(device)
            adv_y_dev = adv_bundle["adv_label"].to(device)
            logits = model(adv_x_dev)
            per_ex_loss = torch.nn.functional.cross_entropy(
                logits, adv_y_dev.long(), reduction="none",
            ).cpu()
        n_ex = adv_bundle["adv_data"].shape[0] // 16
        clean_positions = [16 * i + off for i in range(n_ex) for off in (0, 4, 8, 12)]
        adv_positions   = [16 * i + off for i in range(n_ex) for off in (1, 5, 9, 13)]
        rand_positions  = [16 * i + off for i in range(n_ex) for off in (2, 3, 6, 7, 10, 11, 14, 15)]
        pntk_abs = audit.PNTK.abs()
        I_clean = pntk_abs[:, clean_positions].mean(dim=1)
        I_adv   = pntk_abs[:, adv_positions].mean(dim=1)
        I_rand  = pntk_abs[:, rand_positions].mean(dim=1)
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
        epoch_accum = {}
        print(f"    epoch {epoch}: layer + PNTK + group snapshots recorded")

    artifact_a = train_with_audit(
        model=model_a,
        train_loader=train_loader_a,
        test_loader=test_loader_a,
        audit_loader=adv_loader,
        epochs=args.epochs,
        train_batches=args.train_batches,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        do_audit=True,
        do_ips=False,
        target_batches=1,
        audit_batch_size=n_audit,
        verbose=True,
        step_callback=_step_cb,
        epoch_callback=_epoch_cb,
    )
    artifact_a["per_epoch_layer_contribs"] = per_epoch_layer_contribs
    artifact_a["per_epoch_pntk"] = per_epoch_pntk
    artifact_a["per_epoch_group_metrics"] = per_epoch_group_metrics
    artifact_a["layer_prefixes"] = SVHN_LAYER_PREFIXES
    artifact_a["adv_bundle_path"] = args.out_adv_data

    # -- Step 5: final-checkpoint SVD of Phi_train for Fig 39 mode analysis --
    # Paper's SVHN uses a single final-checkpoint SVD: for each SVD mode k,
    # compare mean |C_k| between adv and clean audit slots.
    # C_k = <V_k, grad_theta L(x_audit)> = (per-audit-sample loss-grad) @ V_k.
    print("\n=== Step 5: final-checkpoint SVD of Phi_train (for Fig 39) ===")
    from kernel_tools.audit import _flatten_per_sample_grads
    from torch.func import functional_call, vmap, grad as func_grad
    # Recompute per-training-sample gradients on the final model
    Phi_train = torch.zeros(
        args.train_batches * args.batch_size, model_a.numparams,
    )
    loss_fn = torch.nn.CrossEntropyLoss()
    params_dict = {k: v.detach() for k, v in model_a.named_parameters()}
    buffers_dict = {k: v.detach() for k, v in model_a.named_buffers()}
    def _loss_fn_functional(params, buffers, sample, target):
        inputs = sample.unsqueeze(0); targets = target.unsqueeze(0)
        outputs = functional_call(model_a, (params, buffers), (inputs,))
        return loss_fn(outputs, targets)
    _grad_fn = vmap(func_grad(_loss_fn_functional), in_dims=(None, None, 0, 0))
    for idx, (train_x, train_label) in enumerate(train_loader_a):
        if idx >= args.train_batches:
            break
        x = train_x.to(device); y = train_label.to(device)
        Phi_train[idx * args.batch_size:(idx + 1) * args.batch_size] = \
            _flatten_per_sample_grads(_grad_fn(params_dict, buffers_dict, x, y)).cpu()

    # Per-audit-sample gradients (on the adv audit set)
    Phi_audit = torch.zeros(n_audit, model_a.numparams)
    for idy, (adv_x, adv_y) in enumerate(adv_loader):
        x = adv_x.to(device); y = adv_y.to(device)
        Phi_audit[idy * n_audit:(idy + 1) * n_audit] = \
            _flatten_per_sample_grads(_grad_fn(params_dict, buffers_dict, x, y)).cpu()

    print(f"Phi_train: {tuple(Phi_train.shape)}   Phi_audit: {tuple(Phi_audit.shape)}")
    print(f"Running rank-{args.svd_k} SVD ...")
    U, S, V = lowrank_svd_train_side(Phi_train, k=args.svd_k)
    # Signed per-mode coefficient under a reference direction:
    # a_k = <V_k, grad_theta L_hat>, where L_hat is the mean loss-gradient
    # direction over the audit set. This gives a signed scalar per mode for
    # the (sigma_k, a_k) scatter plot.
    grad_ref = Phi_audit.mean(dim=0)  # (P,)
    a = (V.T @ grad_ref)              # (K,)  <V_k, grad_ref>
    # C_k[audit_sample] = <V_k, grad_theta L(x_audit_sample)>
    C = Phi_audit @ V                  # (n_audit, K)
    artifact_a["svd_final"] = {
        "U": U, "S": S, "V": V, "a": a, "C_audit": C, "K": args.svd_k,
    }
    print(f"SVD stored: sigma range [{float(S.min()):.4g}, {float(S.max()):.4g}]")

    torch.save(artifact_a, args.out_task_a)
    print(f"saved Task A artifact to {args.out_task_a}")


if __name__ == "__main__":
    main()
