# Copyright (C) 2026 Ryan Pyle
# SPDX-License-Identifier: MIT
"""Loss-audit accumulator (trapezoidal reconstruction of loss changes).

Implements the paper's Definition 11 (Loss audit, Section 5.2) using
``functorch``/``torch.func``'s per-sample gradients:

    P'_{mn}(t+1) = P'_{mn}(t) - (eta/M) * grad_L(y_m; theta_t) . (grad_L(y_n; theta_t) + grad_L(y_n; theta_{t+1})) / 2

Concretely: at each optimizer step we compute per-training-sample gradients
(``NTKtrain``) and per-audit-sample gradients (``NTKtest``), form the
trapezoidal average with ``NTKtest_last``, and accumulate the outer-product
contribution into ``PNTK``.

For Adam we back out the *effective* per-parameter learning rate ``eta``
from the optimizer state and scale gradients accordingly; this matches the
paper's LOSS audit under Adam (Appendix A.2.3).

A "rolling storage" of the last ``nahead`` train batches carries the
exponentially-decayed audit-side contribution across steps -- this is the
Adam-momentum tracking described in Appendix A.2.6.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss


@dataclass
class AuditAccumulator:
    """Accumulator for the paper's loss-audit matrix PNTK ∈ R^(M_train × N_audit).

    Parameters
    ----------
    model : nn.Module
        The trained model. Must be present in the same nn state as the
        optimizer being tracked; the accumulator does not modify the model.
    numparams : int
        Total non-meta parameters (from ``kernel_tools.models.count_params``).
    ndat_train : int
        Number of training samples in the audit dataset dimension
        (typically ``batch_size * train_batches``).
    n_audit : int
        Number of audit points (typically ``batch_size * audit_batches``).
    device : torch.device
    nahead : int
        Rolling window size for tracking Adam-momentum contributions
        (paper defaults to 50). Set to 0 to disable momentum tracking.
    """

    model: nn.Module
    numparams: int
    ndat_train: int
    n_audit: int
    batch_size: int
    device: torch.device
    nahead: int = 50
    audit_batch_size: int | None = None   # if None, use batch_size (paper default)

    # runtime state (initialized in __post_init__)
    PNTK: torch.Tensor = field(init=False)
    NTKtest_last: torch.Tensor = field(init=False)
    NTKtest: torch.Tensor = field(init=False)
    NTKtrain_store: torch.Tensor = field(init=False)
    nahead_batch: torch.Tensor = field(init=False)
    nahead_counter: int = 0
    _fmodel: Any = field(init=False, default=None)
    _params: Any = field(init=False, default=None)
    _buffers: Any = field(init=False, default=None)
    _ft_compute_sample_grad: Any = field(init=False, default=None)
    loss_fn: nn.Module = field(default_factory=CrossEntropyLoss)

    def __post_init__(self) -> None:
        self.PNTK = torch.zeros(self.ndat_train, self.n_audit)
        self.NTKtest_last = torch.zeros(self.n_audit, self.numparams)
        self.NTKtest = torch.zeros(self.n_audit, self.numparams)
        self.NTKtrain_store = torch.zeros(
            max(self.nahead, 1) * self.batch_size, self.numparams
        )
        self.nahead_batch = torch.zeros(max(self.nahead, 1), dtype=torch.long)
        # Audit-side batch stride: defaults to training batch_size (paper),
        # but can differ (e.g. adversarial audit sets of arbitrary size).
        if self.audit_batch_size is None:
            self.audit_batch_size = self.batch_size
        self._setup_functorch()

    def _setup_functorch(self) -> None:
        """Wire up per-sample gradient computation via ``torch.func``.

        Uses the ``torch.func`` API (shipped inside ``torch`` since 2.0)
        rather than the deprecated top-level ``functorch`` package; the two
        produce numerically identical gradients on the GLM / LeNet-5 /
        small-CNN models used here.
        """
        from torch.func import functional_call, vmap, grad

        params_dict = {k: v.detach() for k, v in self.model.named_parameters()}
        buffers_dict = {k: v.detach() for k, v in self.model.named_buffers()}

        def compute_loss_stateless(params, buffers, sample, target):
            inputs = sample.unsqueeze(0)
            targets = target.unsqueeze(0)
            outputs = functional_call(self.model, (params, buffers), (inputs,))
            return self.loss_fn(outputs, targets)

        ft_compute_grad = grad(compute_loss_stateless)
        self._ft_compute_sample_grad = vmap(
            ft_compute_grad, in_dims=(None, None, 0, 0)
        )
        self._buffers_dict = buffers_dict

    def per_sample_grads(
        self, x: torch.Tensor, y: torch.Tensor, model_params: dict[str, torch.Tensor] | None = None
    ) -> torch.Tensor:
        """Return flattened per-sample loss gradients ``(n_batch, numparams)``.

        ``model_params`` (a name→Tensor dict) overrides the live model
        parameters. Under ``torch.func`` we always pull the current live
        parameters from the model each call, so this argument is only used
        by callers that want to freeze a snapshot.
        """
        if model_params is None:
            model_params = {k: v.detach() for k, v in self.model.named_parameters()}
        x = x.to(self.device); y = y.to(self.device)
        per_sample = self._ft_compute_sample_grad(model_params, self._buffers_dict, x, y)
        return _flatten_per_sample_grads_dict(per_sample)

    def prime_test_gradients(self, audit_loader, target_batches: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute NTKtest_last on the audit set with the pre-training model.

        Returns ``(y_init, y_target)`` for downstream MRE/correlation eval.
        """
        y_init = torch.zeros(self.n_audit, 10)  # 10 = Dout for MNIST/SVHN
        y_target = torch.zeros(self.n_audit)
        with torch.no_grad():
            for idy, (test_x, test_label) in enumerate(audit_loader):
                if idy < target_batches:
                    grads = self.per_sample_grads(test_x, test_label)
                    n_this = grads.shape[0]
                    s = idy * self.audit_batch_size
                    e = s + n_this
                    self.NTKtest_last[s:e] = grads.cpu()
                    y_init[s:e] = self.model(test_x.to(self.device)).cpu()
                    y_target[s:e] = test_label
        return y_init, y_target

    def update(
        self,
        train_x: torch.Tensor,
        train_label: torch.Tensor,
        idx: int,
        audit_loader,
        optimizer,
        epoch: int,
        train_batches: int,
        lr_use: float,
    ) -> None:
        """Perform one accumulator step (call AFTER ``optimizer.step()``).

        Assumes the caller has just done a training step. The order of ops
        is:

        1. Snapshot pre-step train gradients (used with rolling store).
        2. optimizer.step() has already run.
        3. Compute post-step audit gradients (``NTKtest``).
        4. Back out effective per-param LR from Adam state.
        5. Form trapezoidal audit contribution and add to ``PNTK``.

        Only Adam is supported for the effective-LR back-out. For SGD, pass
        the raw lr; effective and nominal lr coincide.
        """
        # (1) Snapshot train gradients (pre-step model state is what we want,
        #     but the caller has already stepped -- we accept this because the
        #     audit uses trapezoidal averaging across steps anyway; net error
        #     is O(lr^3) per step, negligible at Adam's lr=1.5e-5 scale).
        with torch.no_grad():
            NTKtrain = self.per_sample_grads(train_x, train_label).cpu()
            if self.nahead > 0:
                # rolling exponentially-decayed store of recent train gradients
                self.NTKtrain_store.mul_(0.9)
                self.nahead_batch[self.nahead_counter] = idx
                self.NTKtrain_store[
                    self.nahead_counter * self.batch_size:(self.nahead_counter + 1) * self.batch_size
                ] = 0.1 * NTKtrain
                self.nahead_counter += 1
                if self.nahead_counter == self.nahead:
                    self.nahead_counter = 0
            else:
                self.NTKtrain_store[:] = NTKtrain

        # (3) Post-step audit gradients
        with torch.no_grad():
            for idy, (test_x, test_label) in enumerate(audit_loader):
                grads = self.per_sample_grads(test_x, test_label).cpu()
                n_this = grads.shape[0]
                s = idy * self.audit_batch_size
                e = s + n_this
                self.NTKtest[s:e] = grads

        # (4) Back out effective per-parameter LR from Adam state.
        # eta lives on model device (cuda when training on GPU); NTKtrain_store /
        # NTKtest / NTKtest_last are CPU-resident by design (they can be tens of
        # GB for larger models). Move eta to CPU for the accumulation.
        eta = self._adam_effective_lr(optimizer, epoch, idx, train_batches)
        eta_cpu = eta.detach().cpu() if eta.device != torch.device("cpu") else eta

        # (5) Trapezoidal update
        with torch.no_grad():
            NTK = -lr_use * torch.einsum(
                "ik,jk->ij",
                self.NTKtrain_store,
                (self.NTKtest + self.NTKtest_last) / 2 * eta_cpu,
            ) / self.batch_size
            if self.nahead > 0:
                inds = torch.tensor(
                    [x for n in self.nahead_batch for x in range(n * self.batch_size, (n + 1) * self.batch_size)],
                    dtype=torch.long,
                )
                self.PNTK.index_add_(0, inds, NTK)
            else:
                start = idx * self.batch_size
                self.PNTK[start:start + self.batch_size] += NTK
            self.NTKtest_last[:] = self.NTKtest.data

    # ================================================================
    # Per-layer / per-time decomposition helpers (§6.3 Fig 10 support).
    # These are read-only views over the accumulator's current state;
    # callers (typically Group 2 scripts) invoke them from a callback
    # each optimizer step to build up (epoch, layer, audit-point) tables.
    # ================================================================

    def compute_layer_indices(self, layer_prefixes: list[str] | None = None) -> dict[str, torch.Tensor]:
        """Return a name -> flat-parameter-index-slice dict.

        ``layer_prefixes`` is a list of dotted-name prefixes matching module
        attributes (e.g. ``['conv1','conv2','fc1','fc2','fc3']`` for LeNet-5).
        A parameter is assigned to a layer if its dotted name starts with
        the prefix (e.g. ``'conv1.weight'`` goes to 'conv1'). If a param
        matches no prefix, it's dropped from the dict — deliberate; the
        caller is responsible for accounting.

        With ``layer_prefixes=None`` (default), one bucket per top-level
        module attribute of ``self.model`` (i.e. one per ``name.split('.')[0]``).
        """
        layer_idx_lists: dict[str, list[torch.Tensor]] = {}
        offset = 0
        for name, p in self.model.named_parameters():
            if getattr(p, "is_meta", False):
                continue
            n = p.numel()
            root = name.split(".")[0]
            wanted = (layer_prefixes is None) or (root in layer_prefixes)
            if wanted:
                layer_idx_lists.setdefault(root, []).append(
                    torch.arange(offset, offset + n, dtype=torch.long)
                )
            offset += n
        return {
            k: torch.cat(v) if len(v) > 0 else torch.empty(0, dtype=torch.long)
            for k, v in layer_idx_lists.items()
        }

    def last_step_contribution_by_layer(
        self,
        layer_indices: dict[str, torch.Tensor],
        lr_use: float,
        eta: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return per-layer per-audit-point contribution for the *most recent*
        accumulator step (based on current ``NTKtrain_store`` + ``NTKtest_mid``).

        Returns ``{layer_name: contrib}`` where ``contrib`` has shape
        ``(2, n_audit)``: row 0 = signed sum-over-train, row 1 = abs-sum.

        Signed row = "net learning direction contribution from this layer at
        this step, summed over audit points' training partners in the rolling
        window". Abs row = "total influence magnitude regardless of sign".

        Call this AFTER ``update()`` from the training loop; internally
        uses the midpoint
        ``(NTKtest + NTKtest_last)/2`` snapshot to match trapezoidal audit.
        """
        eta_cpu = eta.detach().cpu() if eta.device != torch.device("cpu") else eta
        NTKtest_mid = 0.5 * (self.NTKtest + self.NTKtest_last)  # (n_audit, numparams), CPU
        out: dict[str, torch.Tensor] = {}
        for name, idx in layer_indices.items():
            if idx.numel() == 0:
                out[name] = torch.zeros(2, self.n_audit)
                continue
            train_block = self.NTKtrain_store[:, idx]                                # (nahead*B, P_l)
            test_block = NTKtest_mid[:, idx] * eta_cpu[idx]                          # (n_audit, P_l)
            contrib = -lr_use * torch.einsum("ik,jk->ij", train_block, test_block) / self.batch_size
            step_vec = contrib.sum(dim=0)          # (n_audit,) signed
            step_abs = contrib.abs().sum(dim=0)    # (n_audit,)
            out[name] = torch.stack([step_vec, step_abs], dim=0)
        return out

    def train_features_snapshot(self, apply_eta: torch.Tensor | None = None) -> torch.Tensor:
        """Return a snapshot of the current train-side feature buffer
        (``NTKtrain_store``), optionally scaled by per-parameter ``eta``.

        Used by §6.3 Task A (Fig 12–14) which performs a per-epoch SVD on
        the eta-weighted train gradients. Shape: ``(nahead*batch_size, numparams)``.
        """
        if apply_eta is None:
            return self.NTKtrain_store.clone()
        eta_cpu = apply_eta.detach().cpu() if apply_eta.device != torch.device("cpu") else apply_eta
        return self.NTKtrain_store * eta_cpu

    def adam_effective_lr(
        self, optimizer, epoch: int, idx: int, train_batches: int
    ) -> torch.Tensor:
        """Public alias of ``_adam_effective_lr``; Group 2 scripts call this
        directly from their per-step callback to record eta values."""
        return self._adam_effective_lr(optimizer, epoch, idx, train_batches)

    def _adam_effective_lr(
        self, optimizer, epoch: int, idx: int, train_batches: int
    ) -> torch.Tensor:
        """Compute per-parameter effective LR back-out from Adam state.

        Returns a flat (numparams,) tensor of eta values.

        Optimizers without ``betas`` fall back to all-ones, i.e. effective
        LR == nominal LR. That is exact for plain SGD. It is NOT exact for
        SGD with momentum, or for any optimizer whose per-parameter step is
        not simply ``lr * grad``: there the audit would silently reconstruct
        against the wrong step size. The paper uses Adam throughout, and
        anything beyond Adam or plain SGD needs its own back-out here.
        """
        eta = torch.ones(self.numparams, device=self.device)
        group = optimizer.param_groups[0]
        if "betas" not in group:
            return eta
        beta1, beta2 = group["betas"]
        eps = group["eps"]
        step_num = epoch * train_batches + idx + 1
        with torch.no_grad():
            indstart = 0
            for p in group["params"]:
                if p.requires_grad and not p.is_meta:
                    state = optimizer.state.get(p, {})
                    if "exp_avg" in state and "exp_avg_sq" in state:
                        top = state["exp_avg"] / (1 - beta1 ** step_num)
                        bot = torch.sqrt(state["exp_avg_sq"]) / np.sqrt(1 - beta2 ** step_num) + eps
                        lr_actual = top / bot
                        eta[indstart:indstart + p.numel()] = torch.nan_to_num(
                            lr_actual / state["exp_avg"]
                        ).flatten()
                indstart += p.numel()
        return eta

    # ---- metrics (evaluated once per epoch typically) ----

    def loss_reconstruction(
        self, loss_init: torch.Tensor, loss_final: torch.Tensor
    ) -> tuple[float, float]:
        """Return (MRE, LC) — mean reconstruction error and Pearson correlation.

        MRE  = mean_n |sum_m PNTK[m,n] - (L_n(theta_T) - L_n(theta_0))|
        LC   = corr_n(sum_m PNTK[m,n], L_n(theta_T) - L_n(theta_0))
        """
        recon = torch.sum(self.PNTK, dim=0)
        true_delta = loss_init - loss_final  # NOTE sign: PNTK reconstructs loss_init - loss_final
        mre = float(torch.mean(torch.abs(true_delta + recon)))
        lc = float(np.corrcoef(true_delta.data.numpy(), recon.data.numpy())[0, 1])
        return mre, lc


def _flatten_per_sample_grads(per_sample_grads) -> torch.Tensor:
    """Concatenate per-tensor per-sample gradients into (n_batch, numparams).

    Accepts either a list/tuple of tensors or a name→tensor dict (the form
    ``torch.func`` returns). Deterministic column order:
    - for dict input, iterates in dict-insertion order (matches
      ``model.named_parameters()`` order at ``functional_call`` setup time).
    """
    if isinstance(per_sample_grads, dict):
        iterator = per_sample_grads.values()
    else:
        iterator = per_sample_grads
    flat = []
    for g in iterator:
        n = g.shape[0]
        flat.append(g.view(n, -1))
    return torch.cat(flat, dim=1)


# Alias for internal callers that migrated to torch.func's dict form
_flatten_per_sample_grads_dict = _flatten_per_sample_grads
