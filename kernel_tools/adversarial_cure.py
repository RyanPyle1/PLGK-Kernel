# Copyright (C) 2026 Ryan Pyle
# SPDX-License-Identifier: MIT
"""Mode-aware adversarial cure (paper Algorithm 2).

Given a trained model, its Φ_train SVD modes (``V_modes``) and per-mode
drive weights (``Lambda``), this module produces a bounded input-space
perturbation that pushes each adversarial example's mode-coefficients
against the training-consensus drive direction. The result is a "cured"
adversarial that restores the model's original correct classification.

Two variants (paper §6.3):
  T1 = label-aware: uses ground-truth labels + loss term (``lam_loss>0``)
  T2 = label-free: uses model's own pseudo-label + no loss term
        (this is the paper's demonstrated method; T1 is a control)

Both are exposed via ``run_cure`` with the ``use_t1`` flag.

Depends on V_modes/Lambda produced by ``mnist_mode_svd_snapshots.py``
(Group 3 output). Shared parameter-list helpers stay in this module; the
driver script is separate.
"""
from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss


# ---------- small helpers ----------

def dot_param_lists(a_tensors: Iterable[torch.Tensor], b_tensors: Iterable[torch.Tensor]) -> torch.Tensor:
    return sum((ai * bi).sum() for ai, bi in zip(a_tensors, b_tensors))


def project_grad_to_modes(grad_list, V_modes_slice) -> torch.Tensor:
    """Return length-K vector of ⟨grad, V_j⟩ for j in 0..K-1."""
    return torch.stack(
        [dot_param_lists(grad_list, V_modes_slice[j]) for j in range(len(V_modes_slice))],
        dim=0,
    )


def modes_linear_combo(coeffs: torch.Tensor, V_modes_slice) -> list[torch.Tensor]:
    """Sum coeffs[j] * V_modes_slice[j] elementwise, returning a param-list."""
    out = [torch.zeros_like(t) for t in V_modes_slice[0]]
    for j, cj in enumerate(coeffs):
        Vj = V_modes_slice[j]
        for i in range(len(out)):
            out[i] = out[i] + cj * Vj[i]
    return out


def cosine_sq_batch(U: torch.Tensor, V: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    if U.dim() == 1: U = U.unsqueeze(0)
    if V.dim() == 1: V = V.unsqueeze(0).expand_as(U)
    num = (U * V).sum(dim=1)
    den = (U.norm(dim=1) * V.norm(dim=1) + eps)
    return ((num / den).pow(2)).mean()


# ---------- core cure PGD loop ----------

def adversarial_cure(
    model: nn.Module,
    loss_fn: nn.Module,
    x_adv: torch.Tensor,
    V_modes,
    *,
    c_target: torch.Tensor | None = None,
    Lambda: torch.Tensor | None = None,
    tau: float = 0.5,
    y_true: torch.Tensor | None = None,
    x_clean: torch.Tensor | None = None,
    delta_adv: torch.Tensor | None = None,
    eps: float = 0.1,
    p: str = "linf",
    steps: int = 20,
    step_size: float = 0.005,
    lam_loss: float = 0.0,
    lam_match: float = 1.0,
    lam_ortho: float = 0.1,
    k_modes: int | None = None,
    ortho_space: str = "input",
) -> torch.Tensor:
    """One batch of adversarial cures via PGD.

    Returns the cured adversarial tensor with the same shape as ``x_adv``.
    See module docstring for T1 vs T2 selection.
    """
    device = next(model.parameters()).device
    model.eval()
    params = list(model.parameters())

    if Lambda is not None:
        Lambda = torch.as_tensor(Lambda, device=device, dtype=torch.float32).detach()
    V_modes = [[t.detach().to(device) for t in mode] for mode in V_modes]
    K_total = len(V_modes)
    if k_modes is None:
        k_modes = K_total
    k = int(min(k_modes, K_total))
    Vm_k = V_modes[:k]

    @torch.no_grad()
    def clamp_and_project(x0: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(0, 1)
        if p == "linf":
            delta = (x - x0).clamp_(-eps, eps)
            return (x0 + delta).clamp(0, 1)
        if p == "l2":
            delta = (x - x0).view(x.size(0), -1)
            nrm = delta.norm(dim=1, keepdim=True) + 1e-12
            scale = (eps / nrm).clamp(max=1.0)
            delta = (delta * scale).view_as(x)
            return (x0 + delta).clamp(0, 1)
        raise ValueError("p must be 'linf' or 'l2'")

    def param_grad_list(x_in: torch.Tensor):
        logits = model(x_in)
        need_labels = lam_loss > 0.0
        if need_labels:
            assert y_true is not None, "T1 (lam_loss>0) requires y_true"
            loss = loss_fn(logits, y_true)
        else:
            with torch.no_grad():
                pseudo = logits.argmax(dim=1)
            loss = loss_fn(logits, pseudo)
        grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
        return grads, logits

    x0 = x_adv.detach()
    x = x_adv.clone().detach().requires_grad_(True)

    g_adv, _ = param_grad_list(x)
    c_adv_k = project_grad_to_modes(g_adv, Vm_k).detach()

    if c_target is None:
        assert (Lambda is not None) and (tau > 0.0), "T2 needs Lambda + tau"
        Lambda_k = Lambda[:k] if Lambda is not None and Lambda.numel() >= k else Lambda
        c_target_k = (c_adv_k + -1.0 * tau * torch.sign(Lambda_k)).detach()
    else:
        ct = torch.as_tensor(c_target, device=device, dtype=torch.float32)
        c_target_k = ct[:k].detach()

    if ortho_space == "input":
        if delta_adv is None:
            delta_adv = (x_adv - x_clean).detach() if x_clean is not None else torch.zeros_like(x_adv)
    elif ortho_space == "modes":
        c_ref_k = c_adv_k
    else:
        raise ValueError("ortho_space must be 'input' or 'modes'")

    for _ in range(steps):
        g_list, logits = param_grad_list(x)
        c_now_k = project_grad_to_modes(g_list, Vm_k)
        r_k = c_now_k - c_target_k

        match_term = 0.5 * (r_k.pow(2).sum())
        if ortho_space == "input":
            delta_now = (x - x0).view(x.size(0), -1)
            delta_ref = delta_adv.view(x.size(0), -1)
            ortho_term = cosine_sq_batch(delta_now, delta_ref)
        else:
            ortho_term = cosine_sq_batch(c_now_k, c_ref_k)

        if lam_loss > 0.0:
            loss_term = loss_fn(logits, y_true)
            total = lam_loss * loss_term + lam_match * match_term + lam_ortho * ortho_term
        else:
            total = lam_match * match_term + lam_ortho * ortho_term

        grad_x = torch.autograd.grad(total, x, retain_graph=False)[0]
        if p == "linf":
            x = (x - step_size * grad_x.sign()).detach().requires_grad_(True)
        else:
            gx = grad_x / (
                grad_x.view(grad_x.size(0), -1).norm(dim=1, keepdim=True) + 1e-12
            ).view_as(grad_x)
            x = (x - step_size * gx).detach().requires_grad_(True)
        x = clamp_and_project(x0, x).detach().requires_grad_(True)

    return x.detach()


def build_cure_data(
    model: nn.Module,
    adv_data: torch.Tensor,
    adv_label: torch.Tensor,
    V_modes,
    Lambda,
    *,
    tau: float = 0.5,
    eps: float = 0.1,
    p: str = "linf",
    steps: int = 20,
    step_size: float = 0.005,
    lam_match: float = 1.0,
    lam_ortho: float = 0.1,
    chunk: int = 32,
    k_modes: int = 10,
    use_t1: bool = False,
) -> tuple[torch.Tensor, dict]:
    """Wrapper: cure all adversarial + random items in a 4-per-group bundle.

    ``adv_data`` follows the paper's 4-item slot pattern:
      slot 0 = clean, slot 1 = adv, slots 2,3 = random.
    Repeats every 4 rows. Clean rows pass through unchanged; slots 1,2,3
    are all pushed through the cure PGD loop.

    Returns ``(cure_data, stats)``. ``stats`` contains counts + mean cosine
    between the cure delta and the (adv - clean) delta (paper-reported).
    """
    device = next(model.parameters()).device
    loss_fn = CrossEntropyLoss()
    B = adv_data.size(0)
    assert B % 4 == 0, f"adv_data batch size must be a multiple of 4; got {B}"

    adv_data = adv_data.to(device).float()
    adv_label = adv_label.to(device).long()
    cure_data = adv_data.clone()

    idxs = torch.arange(B, device=device)
    mask_adv = (idxs % 4 == 1)
    mask_rand = (idxs % 4 >= 2)

    def _group_base_indices(sel):
        return (sel // 4) * 4

    def _cure_indices(sel):
        nonlocal cure_data
        if sel.numel() == 0:
            return
        for s in range(0, sel.numel(), chunk):
            sub = sel[s:s + chunk]
            base = _group_base_indices(sub)
            x_adv = adv_data[sub]
            x_clean = adv_data[base]
            y_true = adv_label[sub]
            delta_adv = x_adv - x_clean
            cure_kwargs = dict(
                model=model, loss_fn=loss_fn,
                x_adv=x_adv, V_modes=V_modes,
                Lambda=Lambda, tau=tau,
                x_clean=x_clean, delta_adv=delta_adv,
                eps=eps, p=p, steps=steps, step_size=step_size,
                lam_match=lam_match, lam_ortho=lam_ortho,
                k_modes=k_modes, ortho_space="input",
            )
            if use_t1:
                x_cure = adversarial_cure(
                    **cure_kwargs, c_target=None,
                    y_true=y_true, lam_loss=1.0,
                )
            else:
                x_cure = adversarial_cure(
                    **cure_kwargs, c_target=None,
                    y_true=None, lam_loss=0.0,
                )
            cure_data[sub] = x_cure

    _cure_indices(idxs[mask_adv])
    _cure_indices(idxs[mask_rand])

    with torch.no_grad():
        delta_cure = (cure_data - adv_data).view(B, -1)
        base = _group_base_indices(idxs)
        delta_ref = (adv_data - adv_data[base]).view(B, -1)
        cos = (delta_cure * delta_ref).sum(dim=1) / (
            delta_cure.norm(dim=1) * delta_ref.norm(dim=1) + 1e-12
        )
        stats = dict(
            mean_cure_norm=float(delta_cure.norm(dim=1).mean().item()),
            mean_cos_with_adv_delta=float(cos.mean().item()),
            n_adv_cured=int(mask_adv.sum().item()),
            n_rand_cured=int(mask_rand.sum().item()),
        )
    return cure_data, stats
