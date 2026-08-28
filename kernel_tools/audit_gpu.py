# Copyright (C) 2026 Ryan Pyle
# SPDX-License-Identifier: MIT
"""Optional GPU-resident variant of ``AuditAccumulator``.

``kernel_tools.audit.AuditAccumulator`` keeps its large buffers
(``NTKtrain_store``, ``NTKtest``, ``NTKtest_last``, ``PNTK``) on the CPU by
design: at ImageNet/ResNet scale they can be tens of GB and will not fit in
VRAM. At MNIST/LeNet scale they total ~2.2 GB and the CPU residency is a
severe pessimization, because the dominant per-step operation is a dense
einsum over them.

Measured on an A10G against 8 CPU threads at the paper's MNIST defaults:

    per-step einsum, (12800, 44426) x (256, 44426):
        CPU, 8 threads :  1250 ms
        GPU (A10G)     :    17 ms      (~73x)

    50-epoch audit run:
        CPU  : ~7 h
        GPU  : ~12 min

This subclass changes DEVICE PLACEMENT ONLY. Dtype (fp32), operation order,
trapezoidal averaging, the Adam effective-LR back-out and the ``index_add_``
scatter are all identical to the parent, so results agree to floating-point
reassociation noise rather than approximately. ``scripts/verify_gpu_audit.py``
asserts that against the CPU reference and is the gate for using this at all.

Use it via ``--device cuda`` on the audit-producing scripts. It is opt-in:
the CPU path remains the default so a reproduction needs no GPU and no trust
in a second implementation.

WHAT "EQUIVALENT" MEANS HERE. The gate compares the two accumulators on
IDENTICAL inputs and finds them equivalent to floating-point reassociation
noise (corr 0.99999999, mean absolute deviation ~1e-9). It does NOT claim
that a full training run reproduces bit-for-bit with the flag on or off:
cuDNN kernel selection is nondeterministic, so two runs at the same seed
diverge slightly regardless of this class, and downstream quantities such as
the per-epoch mode coefficients can differ by a few percent as a result. That
is a property of GPU training, not of this accumulator. If you need run-to-run
determinism, that is a separate concern from the choice made here.

WHEN NOT TO USE THIS: ``buffer_bytes()`` reports the footprint. If
``nahead * batch_size * numparams * 4`` approaches available VRAM, stay on
the CPU implementation -- that is the case the parent's design protects.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .audit import AuditAccumulator


@dataclass
class GPUAuditAccumulator(AuditAccumulator):
    """``AuditAccumulator`` with its large buffers resident on ``device``.

    Drop-in for the parent: same constructor arguments, same public API.
    ``update()`` is re-implemented rather than inherited because the parent
    hardcodes ``.cpu()`` on the per-sample-gradient results.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        dev = self.device
        # Move the four large buffers onto the compute device. Everything
        # else (nahead_batch, counters) is tiny and stays where it is.
        self.PNTK = self.PNTK.to(dev)
        self.NTKtest = self.NTKtest.to(dev)
        self.NTKtest_last = self.NTKtest_last.to(dev)
        self.NTKtrain_store = self.NTKtrain_store.to(dev)

    def buffer_bytes(self) -> int:
        """Total bytes held by the four large device-resident buffers."""
        return sum(
            t.numel() * t.element_size()
            for t in (self.PNTK, self.NTKtest, self.NTKtest_last,
                      self.NTKtrain_store)
        )

    def prime_test_gradients(self, audit_loader, target_batches: int = 1):
        """Device-resident counterpart of the parent's priming pass.

        The parent writes ``grads.cpu()`` into ``NTKtest_last``; here the
        buffer already lives on ``self.device``, so the transfer is dropped.
        Returned ``y_init`` / ``y_target`` stay on the CPU, matching the
        parent's contract (the trainer builds ``loss_init`` from them on CPU).
        """
        y_init = torch.zeros(self.n_audit, 10)
        y_target = torch.zeros(self.n_audit)
        with torch.no_grad():
            for idy, (test_x, test_label) in enumerate(audit_loader):
                if idy < target_batches:
                    grads = self.per_sample_grads(test_x, test_label)
                    n_this = grads.shape[0]
                    s = idy * self.audit_batch_size
                    e = s + n_this
                    self.NTKtest_last[s:e] = grads.to(self.device)
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
        """Device-resident counterpart of ``AuditAccumulator.update``.

        Mirrors the parent step-for-step; the only differences are that
        gradients are kept on ``self.device`` instead of being moved to the
        CPU, and ``eta`` is used directly (the parent moves it to the CPU to
        meet its CPU-resident buffers).
        """
        dev = self.device

        # (1) Train-side gradients into the rolling store.
        with torch.no_grad():
            NTKtrain = self.per_sample_grads(train_x, train_label).to(dev)
            if self.nahead > 0:
                self.NTKtrain_store.mul_(0.9)
                self.nahead_batch[self.nahead_counter] = idx
                self.NTKtrain_store[
                    self.nahead_counter * self.batch_size:
                    (self.nahead_counter + 1) * self.batch_size
                ] = 0.1 * NTKtrain
                self.nahead_counter += 1
                if self.nahead_counter == self.nahead:
                    self.nahead_counter = 0
            else:
                self.NTKtrain_store[:] = NTKtrain

        # (3) Post-step audit gradients.
        with torch.no_grad():
            for idy, (test_x, test_label) in enumerate(audit_loader):
                grads = self.per_sample_grads(test_x, test_label).to(dev)
                n_this = grads.shape[0]
                s = idy * self.audit_batch_size
                e = s + n_this
                self.NTKtest[s:e] = grads

        # (4) Effective per-parameter LR. Already on self.device.
        eta = self._adam_effective_lr(optimizer, epoch, idx, train_batches)

        # (5) Trapezoidal update -- the expensive einsum, now on device.
        with torch.no_grad():
            NTK = -lr_use * torch.einsum(
                "ik,jk->ij",
                self.NTKtrain_store,
                (self.NTKtest + self.NTKtest_last) / 2 * eta,
            ) / self.batch_size
            if self.nahead > 0:
                inds = torch.tensor(
                    [x for n in self.nahead_batch
                     for x in range(n * self.batch_size, (n + 1) * self.batch_size)],
                    dtype=torch.long, device=dev,
                )
                self.PNTK.index_add_(0, inds, NTK)
            else:
                start = idx * self.batch_size
                self.PNTK[start:start + self.batch_size] += NTK
            self.NTKtest_last[:] = self.NTKtest.data

    def last_step_contribution_by_layer(self, layer_indices, lr_use, eta):
        """Device-resident counterpart of the parent's per-layer decomposition.

        The parent moves ``eta`` to the CPU to match its CPU-resident
        buffers; here everything is already on ``self.device``. Layer index
        tensors are moved to the device once per call. Results come back on
        the CPU, matching the parent's contract -- callers accumulate them
        into per-epoch tables.
        """
        dev = self.device
        NTKtest_mid = 0.5 * (self.NTKtest + self.NTKtest_last)
        out: dict[str, torch.Tensor] = {}
        for name, idx in layer_indices.items():
            if idx.numel() == 0:
                out[name] = torch.zeros(2, self.n_audit)
                continue
            idx_d = idx.to(dev)
            train_block = self.NTKtrain_store[:, idx_d]
            test_block = NTKtest_mid[:, idx_d] * eta[idx_d]
            contrib = -lr_use * torch.einsum(
                "ik,jk->ij", train_block, test_block) / self.batch_size
            out[name] = torch.stack(
                [contrib.sum(dim=0), contrib.abs().sum(dim=0)], dim=0
            ).cpu()
        return out

    def train_features_snapshot(self, apply_eta=None):
        """Device-resident counterpart; returns a CPU tensor.

        Group 3's SVD consumes this on the CPU (``lowrank_svd_train_side``),
        and at 2.12 GB it is the largest single transfer in the pipeline --
        but it happens once per epoch, not once per step, so the copy is
        not on the hot path.
        """
        if apply_eta is None:
            return self.NTKtrain_store.detach().cpu().clone()
        return (self.NTKtrain_store * apply_eta.to(self.device)).detach().cpu()

    def loss_reconstruction(self, loss_init, loss_final):
        """CPU-side metric; parent assumes ``PNTK`` is already on the CPU."""
        import numpy as np
        recon = torch.sum(self.PNTK, dim=0).cpu()
        true_delta = loss_init - loss_final
        mre = float(torch.mean(torch.abs(true_delta + recon)))
        lc = float(np.corrcoef(true_delta.data.numpy(), recon.data.numpy())[0, 1])
        return mre, lc
