# Copyright (C) 2026 Ryan Pyle
# SPDX-License-Identifier: MIT
"""IPS (Intrinsic Perceptual Similarity) accumulator.

Implements the paper's Section 4 similarity kernel via cosine-normalized
loss-gradient inner products, accumulated across training checkpoints:

    IPS[m, n] = <sum_t sqrt(eta_t) g_m(t), sum_t sqrt(eta_t) g_n(t)>
              / sqrt(||sum_t sqrt(eta_t) g_m(t)||^2 * ||...||^2)

The eta-weighting matches the paper's per-parameter learning-rate back-out
under Adam (Appendix A.2.3). Updates run once every ``update_every`` steps.

The accumulator's row index runs over the union of training samples AND
audit samples, so IPS can be queried across the train-to-test similarity
block after training.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .audit import _flatten_per_sample_grads


@dataclass
class IPSAccumulator:
    """IPS accumulator over training + audit union.

    Parameters
    ----------
    model : nn.Module
    numparams : int
    n_train_samples : int
        ``batch_size * train_batches``
    n_audit_samples : int
        ``batch_size * audit_batches``
    batch_size : int
    device : torch.device
    update_every : int
        Update frequency in optimizer steps (paper uses 200).
    """

    model: nn.Module
    numparams: int
    n_train_samples: int
    n_audit_samples: int
    batch_size: int
    device: torch.device
    update_every: int = 200
    proj_dim: int | None = None  # None = no projection (use full numparams);
                                  # int k = JL random projection to k dims
    proj_seed: int = 0

    # runtime state
    n_total: int = field(init=False)
    IPS_top: torch.Tensor = field(init=False)   # unnormalized Gram
    IPS_bot: torch.Tensor = field(init=False)   # per-sample squared norms
    NTKf: torch.Tensor = field(init=False)      # scratch (n_total, feat_dim)
    R: torch.Tensor | None = field(init=False, default=None)  # projection matrix
    feat_dim: int = field(init=False)
    _ft_compute_sample_grad: Any = field(init=False, default=None)
    loss_fn: nn.Module = field(default_factory=nn.CrossEntropyLoss)
    n_updates: int = 0

    def __post_init__(self) -> None:
        import math as _math
        self.n_total = self.n_train_samples + self.n_audit_samples
        self.IPS_top = torch.zeros(self.n_total, self.n_total)
        self.IPS_bot = torch.zeros(self.n_total)
        if self.proj_dim is not None and self.proj_dim < self.numparams:
            self.feat_dim = int(self.proj_dim)
            g = torch.Generator(device="cpu").manual_seed(int(self.proj_seed))
            # Gaussian JL: entries ~ N(0, 1/k), so <Rx, Ry> ~ <x, y>.
            self.R = torch.randn(
                self.numparams, self.feat_dim, generator=g
            ) / _math.sqrt(self.feat_dim)
        else:
            self.feat_dim = int(self.numparams)
            self.R = None
        self.NTKf = torch.zeros(self.n_total, self.feat_dim)
        self._setup_functorch()

    def _setup_functorch(self) -> None:
        from torch.func import functional_call, vmap, grad

        buffers_dict = {k: v.detach() for k, v in self.model.named_buffers()}

        def compute_loss_stateless(params, buffers, sample, target):
            inputs = sample.unsqueeze(0)
            targets = target.unsqueeze(0)
            outputs = functional_call(self.model, (params, buffers), (inputs,))
            return self.loss_fn(outputs, targets)

        self._ft_compute_sample_grad = vmap(
            grad(compute_loss_stateless), in_dims=(None, None, 0, 0)
        )
        self._buffers_dict = buffers_dict

    def maybe_update(
        self,
        step: int,
        train_loader,
        audit_loader,
        optimizer,
        epoch: int,
        train_batches: int,
        eta_scratch: torch.Tensor,
    ) -> bool:
        """Update accumulator if it's time (returns True if updated).

        ``eta_scratch`` is a flat (numparams,) tensor the caller has
        pre-allocated (reused across steps).
        """
        if (step + 1) % self.update_every != 0:
            return False
        with torch.no_grad():
            # Back out per-param effective LR from Adam state (same as audit)
            eta = _adam_eta(optimizer, eta_scratch, epoch, step, train_batches).cpu()

            # Compute per-sample grads across train + audit (torch.func dict style).
            # When self.R is set, project grads to feat_dim BEFORE storing so we
            # never materialize the full (n_total, numparams) matrix.
            params_dict = {k: v.detach() for k, v in self.model.named_parameters()}
            for idx2, (train_x, train_label) in enumerate(train_loader):
                if idx2 >= train_batches:
                    break
                x = train_x.to(self.device); y = train_label.to(self.device)
                grads = _flatten_per_sample_grads(
                    self._ft_compute_sample_grad(params_dict, self._buffers_dict, x, y)
                ).cpu()
                if self.R is not None:
                    grads = grads @ self.R
                self.NTKf[idx2 * self.batch_size:(idx2 + 1) * self.batch_size] = grads
            for idy, (test_x, test_label) in enumerate(audit_loader):
                x = test_x.to(self.device); y = test_label.to(self.device)
                grads = _flatten_per_sample_grads(
                    self._ft_compute_sample_grad(params_dict, self._buffers_dict, x, y)
                ).cpu()
                if self.R is not None:
                    grads = grads @ self.R
                start = self.n_train_samples + idy * self.batch_size
                self.NTKf[start:start + self.batch_size] = grads

            # Accumulate weighted Gram + squared norms.
            # eta is a (numparams,) per-parameter weighting. Under the JL
            # projection we approximate with the mean weight (scalar), matching
            # the paper's projected variant. Without projection, per-parameter
            # eta is applied exactly.
            if self.R is None:
                self.IPS_top += self.NTKf @ (self.NTKf * eta).T
                self.IPS_bot += (self.NTKf * (self.NTKf * eta)).sum(dim=1)
            else:
                eta_bar = float(eta.mean())
                self.IPS_top += eta_bar * (self.NTKf @ self.NTKf.T)
                self.IPS_bot += eta_bar * (self.NTKf * self.NTKf).sum(dim=1)
        self.n_updates += 1
        return True

    def finalize(self, eps: float = 1e-30) -> torch.Tensor:
        """Return the cosine-normalized IPS matrix, shape (n_total, n_total)."""
        s = self.IPS_bot.clamp_min(eps).sqrt()
        s_inv = s.reciprocal()
        IPS = self.IPS_top.clone()
        IPS.mul_(s_inv[:, None])
        IPS.mul_(s_inv[None, :])
        return IPS


def _adam_eta(
    optimizer, eta: torch.Tensor, epoch: int, idx: int, train_batches: int
) -> torch.Tensor:
    """Back-out per-parameter effective LR from Adam state. Reuses provided
    ``eta`` (flat numparams tensor) as the output buffer."""
    group = optimizer.param_groups[0]
    if "betas" not in group:
        eta.fill_(1.0)
        return eta
    beta1, beta2 = group["betas"]
    eps_o = group["eps"]
    step_num = epoch * train_batches + idx + 1
    indstart = 0
    with torch.no_grad():
        for p in group["params"]:
            if p.requires_grad and not p.is_meta:
                state = optimizer.state.get(p, {})
                if "exp_avg" in state and "exp_avg_sq" in state:
                    top = state["exp_avg"] / (1 - beta1 ** step_num)
                    bot = torch.sqrt(state["exp_avg_sq"]) / np.sqrt(1 - beta2 ** step_num) + eps_o
                    lr_actual = top / bot
                    eta[indstart:indstart + p.numel()] = torch.nan_to_num(
                        lr_actual / state["exp_avg"]
                    ).flatten()
            indstart += p.numel()
    return eta
