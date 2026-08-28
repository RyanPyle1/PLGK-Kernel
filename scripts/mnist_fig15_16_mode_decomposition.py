# Copyright (C) 2026 Ryan Pyle
# SPDX-License-Identifier: MIT
"""Figures 15 & 16: mode-space decomposition of the adversarial signal.

Fig 15: for each mode k, the mean change in mode-coefficient c_k between
        adversarial and clean audit points, scattered against
          x-axis = singular value S_k
          y-axis = signed train aggregation a_k
        with point size/color proportional to |mean(c_adv[k] - c_clean[k])|.
        The paper's finding: adversarial-vs-clean difference concentrates
        in the top-9 largest-σ modes.

Fig 16: per-mode input-space Jacobian norm ‖∇_x c_k(x)‖ averaged over each
        example group, then ALSO the loss-gradient-normalized version.

Consumes:
  --svd-full   data/mnist_expc_svd_full.pt         (Group 3 headline)
  --model      data/mnist_experiment1.pt           (Group 1 checkpoint)
  --adv-bundle data/mnist_adv_data.pt              (Group 2 adv audit)
  --dcdx-cache data/mnist_dcdx_norms_adv_K100.npz  (cached; --compute-dcdx to regenerate)

Scope: this script reproduces Figures 15 and 16 from the Group 3 headline
artifact. The additional per-mode statistics quoted in §6.3.1 (per-mode
loss contribution, cumulative explained fraction) are not recomputed here.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from kernel_tools.console import enable_unicode_stdout
from kernel_tools import LNModel


def _param_list_dot_flat(param_list, flat: torch.Tensor) -> torch.Tensor:
    """<flat, [p0 p1 ... concat flat]> for a list of param tensors.
    Returns a scalar tensor. Both sides must be same total dim.
    """
    offset = 0
    total = None
    for p in param_list:
        n = p.numel()
        term = (p.reshape(-1) * flat[offset:offset + n]).sum()
        total = term if total is None else total + term
        offset += n
    return total


def compute_dcdx_norms(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    V_top: torch.Tensor,     # (K, P) each row is a parameter-space mode
    loss_fn,
    device: torch.device,
    chunk_size: int = 16,
    verbose: bool = True,
) -> np.ndarray:
    """Return ``||∇_x c_k(x)||`` for each (sample, mode). Shape: (N, K).

    Implementation: c_k(x) = V_k^T ∇_θ L(f(x;θ), y). For each sample x we
    compute the loss gradient w.r.t. parameters, dot with V_k to get a
    scalar c_k, then take its gradient w.r.t. x, and record its L2 norm.
    Runs in chunks to control memory. Uses ``create_graph=True`` on the
    param-side gradient so the chained ``autograd.grad`` w.r.t. x works.
    """
    model = model.to(device).eval()
    K, P = V_top.shape
    V_top = V_top.to(device).float()
    N = x.shape[0]

    # Unflatten V_top into per-param-tensor mode representations once.
    param_shapes = [p.shape for p in model.parameters() if not getattr(p, "is_meta", False)]
    param_numels = [int(np.prod(s)) for s in param_shapes]
    V_modes: list[list[torch.Tensor]] = []
    for k in range(K):
        row = V_top[k]
        parts, off = [], 0
        for shape, n in zip(param_shapes, param_numels):
            parts.append(row[off:off + n].reshape(shape).contiguous())
            off += n
        V_modes.append(parts)

    params = list(model.parameters())
    out = np.zeros((N, K), dtype=np.float32)

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        xb = x[start:end].detach().to(device).float()
        yb = y[start:end].detach().to(device).long()
        # Per-sample loop is needed because we want per-sample ||∇_x c_k||,
        # and functorch/vmap composition with double autograd is finicky.
        for i in range(xb.shape[0]):
            xi = xb[i:i + 1].detach().clone().requires_grad_(True)
            yi = yb[i:i + 1]
            logits = model(xi)
            loss = loss_fn(logits, yi)
            grad_theta = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
            for k in range(K):
                c_k = _param_list_dot_flat(V_modes[k], V_top[k])  # scalar; both are the same mode
                # Actually we want c_k = <V_k, grad_theta>
                c_k = sum((V_modes[k][j] * grad_theta[j]).sum() for j in range(len(grad_theta)))
                dcdx_i = torch.autograd.grad(c_k, xi, retain_graph=(k < K - 1))[0]
                out[start + i, k] = float(dcdx_i.detach().reshape(-1).norm().item())
        if verbose:
            print(f"  dCdx chunk {start:4d}/{N} done")
    return out


def main() -> None:
    enable_unicode_stdout()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--svd-full", type=str, default="data/mnist_expc_svd_full.pt")
    ap.add_argument("--model", type=str, default="data/mnist_experiment1.pt")
    ap.add_argument("--adv-bundle", type=str, default="data/mnist_adv_data.pt")
    ap.add_argument("--dcdx-cache", type=str, default="data/mnist_dcdx_norms_adv_K100.npz")
    ap.add_argument("--compute-dcdx", action="store_true",
                    help="Regenerate dCdx norms cache (skipped by default if cache exists)")
    ap.add_argument("--K", type=int, default=100, help="Number of modes to analyze")
    ap.add_argument("--out-fig15", type=str, default="figures/fig15_mode_activity_adv_clean.png")
    ap.add_argument("--out-fig16", type=str, default="figures/fig16_input_sensitivity.png")
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )

    # --- Load SVD artifact + adv bundle ---
    svd = torch.load(args.svd_full, weights_only=False)
    S_final = svd["S_per_epoch"][-1].numpy()          # (K_total,)
    a_final = svd["a_per_epoch"][-1].numpy()          # (K_total,)
    V_final = svd["V_final_flat"].numpy()             # (K_total, P)
    C_clean = svd["C_signed_clean_per_epoch"][-1]     # tensor (N_clean, K_total)
    C_adv   = svd["C_signed_adv_per_epoch"][-1]
    C_rand  = svd["C_signed_rand_per_epoch"][-1]
    K = min(args.K, S_final.shape[0])
    print(f"K available: {S_final.shape[0]}   using K={K}")

    S_k = S_final[:K]; a_k = a_final[:K]
    Lambda_k = S_k * a_k
    C_clean_np = (C_clean.cpu().numpy() if hasattr(C_clean, "cpu") else np.asarray(C_clean))[:, :K]
    C_adv_np   = (C_adv.cpu().numpy()   if hasattr(C_adv, "cpu")   else np.asarray(C_adv))[:, :K]
    C_rand_np  = (C_rand.cpu().numpy()  if hasattr(C_rand, "cpu")  else np.asarray(C_rand))[:, :K]

    # ------------------ Fig 15 ------------------
    # For each mode: mean(|C_adv[k]|) - mean(|C_clean[k]|). Absolute
    # ratio-like value characterizes "adv shifts mode k more than clean does".
    diff_abs = np.abs(C_adv_np).mean(axis=0) - np.abs(C_clean_np).mean(axis=0)   # (K,)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    size = 40 + 400 * (np.abs(diff_abs) / (np.max(np.abs(diff_abs)) + 1e-30))
    sc = ax.scatter(S_k, a_k, s=size, c=diff_abs, cmap="coolwarm",
                     edgecolor="black", linewidth=0.4, alpha=0.85)
    # Annotate top-9 modes with their index
    top9 = np.argsort(-np.abs(diff_abs))[:9]
    for k in top9:
        ax.annotate(f"k={k+1}", (S_k[k], a_k[k]), fontsize=8,
                     xytext=(6, 4), textcoords="offset points")
    ax.set_xlabel(r"singular value $\sigma_k$")
    ax.set_ylabel(r"signed train aggregation $a_k = u_k^\top 1$")
    ax.set_title("Fig 15: adv-vs-clean per-mode activity difference (size/color = |Δ|)")
    fig.colorbar(sc, ax=ax, label=r"mean $|c_{k,\mathrm{adv}}| - |c_{k,\mathrm{clean}}|$")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out15 = Path(args.out_fig15); out15.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out15, dpi=200); plt.close(fig)
    print(f"Wrote: {out15}")

    # ------------------ Fig 16 ------------------
    # Load or compute dCdx norms
    dcdx_path = Path(args.dcdx_cache)
    if args.compute_dcdx or not dcdx_path.exists():
        print(f"Computing dCdx norms (K={K}) — this is O(N * K) autograd passes ...")
        # Load model + adv batch to compute against
        m_art = torch.load(args.model, weights_only=False)
        model = LNModel(device=device)
        model.load_state_dict(m_art["model_final_state"])
        adv = torch.load(args.adv_bundle, weights_only=False)
        adv_data = adv["adv_data"]; adv_label = adv["adv_label"]
        loss_fn = nn.CrossEntropyLoss()
        V_top = torch.from_numpy(V_final[:K])
        dcdx = compute_dcdx_norms(
            model=model, x=adv_data, y=adv_label, V_top=V_top,
            loss_fn=loss_fn, device=device, chunk_size=8, verbose=True,
        )
        dcdx_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dcdx_path, dcdx=dcdx, K=np.array([K]))
        print(f"Cached dCdx norms to {dcdx_path}")
    else:
        dcdx = np.load(dcdx_path)["dcdx"]                # (N_audit, K)
        print(f"Loaded cached dCdx norms: {dcdx.shape}")

    N_audit = dcdx.shape[0]
    n_examples = N_audit // 16
    clean_positions = [16 * i + off for i in range(n_examples) for off in (0, 4, 8, 12)]
    adv_positions   = [16 * i + off for i in range(n_examples) for off in (1, 5, 9, 13)]
    rand_positions  = [16 * i + off for i in range(n_examples) for off in (2, 3, 6, 7, 10, 11, 14, 15)]

    dcdx_by_group = {
        "clean": dcdx[clean_positions],
        "adv":   dcdx[adv_positions],
        "rand":  dcdx[rand_positions],
    }
    # Loss-corrected: divide by per-example loss-gradient norm (proxy: ||∇_θL|| via
    # the summed |c_k| across all modes — cheap proxy of the true norm).
    # More faithful: use per-example ||∇_θL||; we can compute it from
    # sum_k c_k^2 = ||projection onto V||^2 <= ||∇_θL||^2. For the paper's
    # normalization we use ||∇_θL|| as the divisor. We approximate this by
    # sqrt(sum_k c_k^2) using the full C matrix.
    C_full = np.concatenate([C_clean_np, C_adv_np, C_rand_np], axis=0)
    # But we lost the row ordering. Recompute over dcdx-aligned order:
    C_full_dcdx_order = np.zeros((N_audit, K), dtype=np.float32)
    C_full_dcdx_order[clean_positions] = C_clean_np
    C_full_dcdx_order[adv_positions] = C_adv_np
    C_full_dcdx_order[rand_positions] = C_rand_np
    grad_norm_proxy = np.sqrt((C_full_dcdx_order ** 2).sum(axis=1) + 1e-30)  # (N_audit,)
    dcdx_corr = dcdx / grad_norm_proxy[:, None]                              # (N_audit, K)
    dcdx_corr_by_group = {
        "clean": dcdx_corr[clean_positions],
        "adv":   dcdx_corr[adv_positions],
        "rand":  dcdx_corr[rand_positions],
    }

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = {"clean": "#4c78a8", "adv": "#e45756", "rand": "#59a14f"}
    modes_axis = np.arange(1, K + 1)
    for g in ("clean", "adv", "rand"):
        mu = dcdx_by_group[g].mean(axis=0); sd = dcdx_by_group[g].std(axis=0)
        ax_l.plot(modes_axis, mu, color=colors[g], lw=1.3, label=g)
        ax_l.fill_between(modes_axis, mu - sd, mu + sd, alpha=0.18, color=colors[g])
        mu = dcdx_corr_by_group[g].mean(axis=0); sd = dcdx_corr_by_group[g].std(axis=0)
        ax_r.plot(modes_axis, mu, color=colors[g], lw=1.3, label=g)
        ax_r.fill_between(modes_axis, mu - sd, mu + sd, alpha=0.18, color=colors[g])
    ax_l.set_xlabel("mode rank $k$"); ax_l.set_ylabel(r"$\|\nabla_x c_k(x)\|$")
    ax_l.set_title("Fig 16 (left): raw input-space sensitivity per mode")
    ax_l.grid(True, alpha=0.25); ax_l.legend(loc="best")

    ax_r.set_xlabel("mode rank $k$")
    ax_r.set_ylabel(r"$\|\nabla_x c_k(x)\| \, / \, \|\nabla_\theta L\|$")
    ax_r.set_title("Fig 16 (right): loss-corrected input-space sensitivity")
    ax_r.grid(True, alpha=0.25); ax_r.legend(loc="best")

    fig.suptitle(
        "Fig 16: per-mode input sensitivity. Top-K modes are the most\n"
        "input-sensitive (loss-corrected inverts the raw ordering)",
        fontsize=11,
    )
    fig.tight_layout()
    out16 = Path(args.out_fig16); out16.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out16, dpi=200); plt.close(fig)
    print(f"Wrote: {out16}")


if __name__ == "__main__":
    main()
