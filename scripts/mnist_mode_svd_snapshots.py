# Copyright (C) 2026 Ryan Pyle
# SPDX-License-Identifier: MIT
"""Group 3 headline: retrain MNIST + capture per-epoch SVD of Φ_train.

A fresh training run that (a) uses the same trained model as Group 1 as the
attack source, (b) reuses Group 2's Task A adversarial audit set, and
(c) at every epoch computes the truncated SVD of the eta-weighted rolling
train-gradient buffer and records a rich set of per-mode statistics.

Outputs a compact ``.pt`` artifact with the following keys:

  epochs                : list[int]  the epoch each snapshot came from
  S_per_epoch           : list[Tensor (K,)]  singular values
  V_final_flat          : Tensor (K, P)  right-singular vectors at epoch T
  V_modes_paramlist     : list of K param-shaped lists (for cure)
  Lambda_final          : Tensor (K,)  S_k * a_k at epoch T (cure drive)
  a_per_epoch           : list[Tensor (K,)]  net U-column sum per epoch
  C_signed_{clean,adv,rand}_per_epoch : list[Tensor (N_g, K)]
                          per-example mode-coefficients each epoch
  frac_basis_{clean,adv,rand}_per_epoch : list[ndarray (K,)]  |C|/Σ|C|
  loss_mean_{group}_per_epoch, loss_sd_{group}_per_epoch
  I_mean_{group}_per_epoch, I_sd_{group}_per_epoch

The saved artifact ALSO includes ``V_modes`` (paramlist) and ``Lambda`` at
the final epoch — this is what Group 2's cure script consumes.

Wall time: ~2× Group 1 training (extra epoch-end SVD + eval on adv audit).
CPU: ~3 hours. Single GPU: ~30 min.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from copy import deepcopy
import torch
from torch.utils.data import DataLoader, TensorDataset

from kernel_tools.console import enable_unicode_stdout
from kernel_tools import (
    LNModel, mnist_loaders, train_with_audit,
    lowrank_svd_train_side, compute_param_shapes,
    flat_modes_to_param_list, param_list_to_flat_modes,
)


def _build_audit_loader(adv_x: torch.Tensor, adv_y: torch.Tensor):
    N = adv_x.shape[0]
    return DataLoader(TensorDataset(adv_x, adv_y), batch_size=N, shuffle=False, drop_last=False)


def main() -> None:
    enable_unicode_stdout()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adv-bundle", type=str, default="data/mnist_adv_data.pt",
                    help="Group 2 output — adversarial audit bundle")
    ap.add_argument("--data-root", type=str, default="data/MNIST")
    ap.add_argument("--out", type=str, default="data/mnist_expc_svd_full.pt",
                    help="Full per-epoch SVD stats artifact")
    ap.add_argument("--out-cure", type=str, default="data/mnist_expc_svd_epoch49.pt",
                    help="Final-epoch V_modes + Lambda for Group 2 cure")
    ap.add_argument("--K", type=int, default=128,
                    help="Number of modes to keep per SVD snapshot (paper: 128)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--train-batches", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1.5e-5)
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--gpu-audit", action="store_true",
                    help="Keep the audit accumulator's large buffers on the "
                         "compute device. ~70x faster per step at MNIST scale "
                         "and numerically equivalent (see "
                         "scripts/verify_gpu_audit.py), but needs ~2.2 GB of "
                         "VRAM. Off by default: the CPU path is the reference.")
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

    # --- Load adversarial audit set (from Group 2) ---
    adv_bundle = torch.load(args.adv_bundle, weights_only=False)
    adv_data = adv_bundle["adv_data"]
    adv_label = adv_bundle["adv_label"]
    N_audit = adv_data.shape[0]
    n_examples = N_audit // 16
    print(f"adv_data shape: {tuple(adv_data.shape)}  ({n_examples} success examples × 16 slots)")

    # Slot layout (paper): 4 attacks × [clean, adv, rand1, rand2] within each 16-slot block
    clean_positions = [16 * i + off for i in range(n_examples) for off in (0, 4, 8, 12)]
    adv_positions = [16 * i + off for i in range(n_examples) for off in (1, 5, 9, 13)]
    rand_positions = [16 * i + off for i in range(n_examples) for off in (2, 3, 6, 7, 10, 11, 14, 15)]
    mask_clean = torch.zeros(N_audit, dtype=torch.bool); mask_clean[clean_positions] = True
    mask_adv = torch.zeros(N_audit, dtype=torch.bool); mask_adv[adv_positions] = True
    mask_rand = torch.zeros(N_audit, dtype=torch.bool); mask_rand[rand_positions] = True

    # --- Setup model + loaders ---
    model = LNModel(device=device)
    train_loader, test_loader, _ = mnist_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        train_batches=args.train_batches,
        audit_batches=1,
    )
    audit_loader = _build_audit_loader(adv_data, adv_label)
    param_shapes = compute_param_shapes(model)

    # --- Per-epoch SVD stats accumulators ---
    epochs_recorded: list[int] = []
    S_per_epoch: list[torch.Tensor] = []
    a_per_epoch: list[torch.Tensor] = []
    V_per_epoch: list[torch.Tensor] = []   # will only keep the LAST one to save disk
    C_signed = {"clean": [], "adv": [], "rand": []}
    # Slot-ordered copy of the same coefficients. The per-group split above
    # discards which audit slot each row came from, and slot identity is what
    # identifies the source example a point belongs to. The population
    # analyses (mnist_population_temporal.py) group audit points by source
    # example, so they need this; keeping both costs one (N_audit, K) tensor
    # per epoch, which is negligible beside the SVD itself.
    C_all_per_epoch: list[torch.Tensor] = []
    frac_basis = {"clean": [], "adv": [], "rand": []}
    loss_by_group = {"clean_mean": [], "clean_sd": [],
                     "adv_mean": [], "adv_sd": [],
                     "rand_mean": [], "rand_sd": []}
    I_by_group = {"clean_mean": [], "clean_sd": [],
                  "adv_mean": [], "adv_sd": [],
                  "rand_mean": [], "rand_sd": []}
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

    # Fig 14 support: store per-epoch feature-space + activation-space
    # snapshots; the paper's kernel-regime metrics (feat overlap, subspace
    # overlap, activation overlap) compare each epoch against the FINAL
    # epoch T, so we build them in a post-training pass below.
    train_features_per_epoch: list[torch.Tensor] = []      # (n, P) each
    V_top10_per_epoch: list[torch.Tensor] = []             # (10, P) each
    activations_per_epoch: list[torch.Tensor] = []         # (N_audit, H) each

    def _epoch_cb(*, audit, ips, model, epoch, epochs, optimizer, train_batches, lr, **_):
        if audit is None:
            return
        # Snapshot per-epoch SVD of eta-weighted NTKtrain_store using the
        # most-recent step's effective per-parameter learning rate.
        with torch.no_grad():
            eta = audit.adam_effective_lr(
                optimizer, epoch=epoch, idx=train_batches - 1, train_batches=train_batches,
            )
            train_mat = audit.train_features_snapshot(apply_eta=eta)   # (nahead*B, P) on CPU
            U, S, V = lowrank_svd_train_side(train_mat, args.K)          # U (n,K), S (K,), V (P,K)
            V_flat = V.transpose(0, 1)                                    # (K, P)
            a = U.transpose(0, 1) @ torch.ones(U.shape[0], dtype=U.dtype)

            # --- Fig 14 support: snapshot per-epoch state ---
            # Row-normalized features so post-training cosines are cheap.
            feats = train_mat
            feat_norm = feats / (feats.norm(dim=1, keepdim=True) + 1e-30)
            train_features_per_epoch.append(feat_norm.clone())
            V_top10 = V_flat[:min(10, V_flat.shape[0])]
            V_top10_norm = V_top10 / (V_top10.norm(dim=1, keepdim=True) + 1e-30)
            V_top10_per_epoch.append(V_top10_norm.clone())
            adv_x_dev = adv_data.to(device)
            hidden, _ = model(adv_x_dev, doHidden=True)
            hidden_flat = hidden.reshape(hidden.shape[0], -1).detach().cpu()
            hidden_norm = hidden_flat / (hidden_flat.norm(dim=1, keepdim=True) + 1e-30)
            activations_per_epoch.append(hidden_norm.clone())

            # Per-example C_k on the adversarial audit set
            # V comes from the CPU-side SVD of the (CPU) feature snapshot,
            # while NTKtest may live on the GPU under --gpu-audit. Match
            # devices explicitly; results are identical either way.
            C_mat = audit.NTKtest @ V.to(audit.NTKtest.device)             # (N_audit, K)
            C_all_per_epoch.append(C_mat.detach().cpu().clone())
            for g_name, g_mask in [("clean", mask_clean), ("adv", mask_adv), ("rand", mask_rand)]:
                C_signed[g_name].append(C_mat[g_mask].clone())
                absC = C_mat[g_mask].abs()
                frac = absC / (absC.sum(dim=1, keepdim=True) + 1e-30)
                frac_basis[g_name].append(frac.mean(dim=0).cpu().numpy())

            # Snapshot loss + influence per group
            adv_x_dev = adv_data.to(device); adv_y_dev = adv_label.to(device).long()
            logits = model(adv_x_dev)
            per_ex_loss = loss_fn(logits, adv_y_dev).cpu()
            # PNTK may be GPU-resident under --gpu-audit; the group masks
            # below are CPU boolean tensors, so bring it back for indexing.
            pntk_abs = audit.PNTK.abs().cpu()
            for g_name, g_mask in [("clean", mask_clean), ("adv", mask_adv), ("rand", mask_rand)]:
                # per-training-sample mean-abs influence into this group
                I_g = pntk_abs[:, g_mask].mean(dim=1)
                loss_by_group[f"{g_name}_mean"].append(float(per_ex_loss[g_mask].mean()))
                loss_by_group[f"{g_name}_sd"].append(float(per_ex_loss[g_mask].std()))
                I_by_group[f"{g_name}_mean"].append(float(I_g.mean()))
                I_by_group[f"{g_name}_sd"].append(float(I_g.std()))

            epochs_recorded.append(int(epoch))
            S_per_epoch.append(S.cpu().clone())
            a_per_epoch.append(a.cpu().clone())
            V_per_epoch.append(V_flat.cpu().clone())     # keep all for now; caller can trim
            print(f"    epoch {epoch}: SVD (K={args.K}) + mode stats recorded  "
                  f"(sigma_1={S[0].item():.4e}, sigma_K={S[-1].item():.4e})")

    print("Training + SVD snapshots ...")
    artifact = _train(
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
        do_ips=False,
        target_batches=1,
        audit_batch_size=N_audit,
        verbose=True,
        epoch_callback=_epoch_cb,
    )

    # -- Post-training: kernel-regime metrics vs final epoch --
    feat_overlap = []; subspace_overlap = []; act_overlap = []
    F_T = train_features_per_epoch[-1]
    V10_T = V_top10_per_epoch[-1]
    H_T = activations_per_epoch[-1]
    for t in range(len(epochs_recorded)):
        # feat: mean over samples of cos(feat_t[i], feat_T[i])
        feat_overlap.append(float((train_features_per_epoch[t] * F_T).sum(dim=1).mean().item()))
        # subspace: mean over top-10 of |cos(V10_t[k], V10_T[k])|
        subspace_overlap.append(float((V_top10_per_epoch[t] * V10_T).sum(dim=1).abs().mean().item()))
        # activation: mean over adv samples of cos(H_t[i], H_T[i])
        act_overlap.append(float((activations_per_epoch[t] * H_T).sum(dim=1).mean().item()))

    # Assemble the full artifact
    V_final_flat = V_per_epoch[-1]                          # (K, P)
    S_final = S_per_epoch[-1]                               # (K,)
    a_final = a_per_epoch[-1]                               # (K,)
    Lambda_final = S_final * a_final                        # (K,)  drive per mode
    V_modes_paramlist = flat_modes_to_param_list(V_final_flat, param_shapes)

    full_out = {
        "epochs": epochs_recorded,
        "S_per_epoch": S_per_epoch,
        "a_per_epoch": a_per_epoch,
        "V_final_flat": V_final_flat,
        "Lambda_final": Lambda_final,
        "V_modes": V_modes_paramlist,           # for cure
        "Lambda": Lambda_final,                 # for cure
        "C_all_per_epoch": C_all_per_epoch,   # slot-ordered; see note above
        "C_signed_clean_per_epoch": C_signed["clean"],
        "C_signed_adv_per_epoch": C_signed["adv"],
        "C_signed_rand_per_epoch": C_signed["rand"],
        "frac_basis_clean_per_epoch": frac_basis["clean"],
        "frac_basis_adv_per_epoch": frac_basis["adv"],
        "frac_basis_rand_per_epoch": frac_basis["rand"],
        "loss_by_group": loss_by_group,
        "I_by_group": I_by_group,
        "K": args.K,
        "adv_bundle_path": args.adv_bundle,
        "model_final_state": artifact["model_final_state"],
        "trainloss": artifact["trainloss"],
        "testacc": artifact["testacc"],
        # Fig 14 kernel-regime metrics (each epoch vs final)
        "kernel_regime": {
            "feat_overlap": feat_overlap,
            "subspace_overlap": subspace_overlap,
            "act_overlap": act_overlap,
        },
    }
    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(full_out, out_path)
    print(f"Saved full SVD artifact to {out_path}")

    # Compact cure artifact (just V_modes + Lambda + shapes)
    cure_out = {
        "V_modes": V_modes_paramlist,
        "Lambda": Lambda_final,
        "V_final_flat": V_final_flat,
        "S_final": S_final,
        "a_final": a_final,
        "K": args.K,
    }
    out_cure = Path(args.out_cure); out_cure.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cure_out, out_cure)
    print(f"Saved cure artifact (V_modes + Lambda) to {out_cure}")


if __name__ == "__main__":
    main()
