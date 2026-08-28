# Copyright (C) 2026 Ryan Pyle
# SPDX-License-Identifier: MIT
"""``train_with_audit`` variant that uses the GPU-resident accumulator.

``kernel_tools.train.train_with_audit`` constructs an ``AuditAccumulator``
internally with no injection point, so this is a copy of that function with
the accumulator class swapped and the CPU-specific bits adjusted. Behavior
is otherwise identical -- same step order, same callbacks, same returned
artifact dict -- so downstream consumers do not care which was used.

See ``src/audit_gpu.py`` for why (73x on the dominant einsum at MNIST
scale) and ``scripts/verify_gpu_audit.py`` for the equivalence gate.
"""
from __future__ import annotations

import time
from copy import deepcopy
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from .ips import IPSAccumulator
from .train import collect_full_eval

from .audit_gpu import GPUAuditAccumulator


def train_with_audit_gpu(
    model: nn.Module,
    train_loader,
    test_loader,
    audit_loader,
    *,
    epochs: int = 50,
    train_batches: int = 200,
    batch_size: int = 256,
    lr: float = 1.5e-5,
    device: torch.device | str = "cuda:0",
    do_audit: bool = True,
    do_ips: bool = False,
    ips_update_every: int = 200,
    ips_proj_dim: int | None = None,
    nahead: int = 50,
    target_batches: int = 1,
    audit_batch_size: int | None = None,
    verbose: bool = True,
    step_callback=None,
    epoch_callback=None,
    progress_every: int = 1,
) -> dict[str, Any]:
    device = torch.device(device)
    model = model.to(device)
    numparams = int(sum(p.numel() for p in model.parameters() if not p.is_meta))

    audit_bs = audit_batch_size if audit_batch_size is not None else batch_size
    n_train_samples = train_batches * batch_size
    n_audit_samples = target_batches * audit_bs

    audit = None
    ips = None
    if do_audit:
        audit = GPUAuditAccumulator(
            model=model, numparams=numparams, ndat_train=n_train_samples,
            n_audit=n_audit_samples, batch_size=batch_size, device=device,
            nahead=nahead, audit_batch_size=audit_bs,
        )
        if verbose:
            print(f"audit buffers on {device}: "
                  f"{audit.buffer_bytes()/1024**3:.2f} GB", flush=True)
    if do_ips:
        ips = IPSAccumulator(
            model=model, numparams=numparams, n_train_samples=n_train_samples,
            n_audit_samples=n_audit_samples, batch_size=batch_size,
            device=device, update_every=ips_update_every, proj_dim=ips_proj_dim,
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = CrossEntropyLoss()

    num_updates = train_batches * epochs
    trainloss = np.zeros(num_updates)
    testacc = np.zeros(epochs)
    logitcorrs = np.zeros(epochs)
    mles = np.zeros(epochs)
    model_init_state = deepcopy(model.state_dict())

    y_init, y_target = audit.prime_test_gradients(audit_loader, target_batches)
    loss_init = torch.zeros(n_audit_samples)
    for i in range(len(y_init)):
        loss_init[i] = loss_fn(y_init[i], y_target[i].long())

    eta_scratch = torch.zeros(numparams, device=device)
    update = 0
    t0 = time.time()
    loss_final = torch.zeros(n_audit_samples)

    for _epoch in range(epochs):
        for idx, (train_x, train_label) in enumerate(train_loader):
            if idx >= train_batches:
                break
            optimizer.zero_grad()
            predict_y = model(train_x.to(device).float())
            _error = loss_fn(predict_y, train_label.to(device).long())
            _error.backward()
            optimizer.step()

            if audit is not None:
                audit.update(
                    train_x=train_x, train_label=train_label, idx=idx,
                    audit_loader=audit_loader, optimizer=optimizer,
                    epoch=_epoch, train_batches=train_batches, lr_use=lr,
                )
            if ips is not None:
                ips.maybe_update(
                    step=update, train_loader=train_loader,
                    audit_loader=audit_loader, optimizer=optimizer,
                    epoch=_epoch, train_batches=train_batches,
                    eta_scratch=eta_scratch,
                )
            if step_callback is not None:
                step_callback(audit=audit, ips=ips, optimizer=optimizer,
                              epoch=_epoch, idx=idx, update=update, lr=lr)
            trainloss[update] = _error.item()
            update += 1

        if epoch_callback is not None:
            epoch_callback(audit=audit, ips=ips, model=model, epoch=_epoch,
                           epochs=epochs, optimizer=optimizer,
                           train_batches=train_batches, lr=lr)

        correct = total = 0
        with torch.no_grad():
            for test_x, test_label in test_loader:
                pred = model(test_x.to(device).float())
                correct += int((pred.argmax(dim=-1).cpu() == test_label).sum())
                total += int(test_label.shape[0])
        testacc[_epoch] = correct / total

        y_final = torch.zeros(n_audit_samples, 10)
        with torch.no_grad():
            for idy, (test_x, _tl) in enumerate(audit_loader):
                if idy < target_batches:
                    preds = model(test_x.to(device).float())
                    s = idy * audit_bs
                    y_final[s:s + preds.shape[0]] = preds.cpu()
        for i in range(len(y_final)):
            loss_final[i] = loss_fn(y_final[i], y_target[i].long())

        if audit is not None:
            mre, lc = audit.loss_reconstruction(loss_init, loss_final)
            logitcorrs[_epoch] = lc
            mles[_epoch] = mre
            if verbose and (_epoch % progress_every == 0 or _epoch == epochs - 1):
                el = time.time() - t0
                left = el / (_epoch + 1) * (epochs - _epoch - 1)
                print(f"Epoch {_epoch:3d}: acc={testacc[_epoch]:.4f}  "
                      f"MRE={mre:.4g}  Corr={lc:.6f}  "
                      f"[{el/60:.1f} min, ~{left/60:.1f} left]", flush=True)

    if verbose:
        print(f"Total train time: {time.time() - t0:.1f}s", flush=True)

    out = {
        "trainloss": trainloss, "testacc": testacc, "logitcorrs": logitcorrs,
        "mles": mles, "loss_init": loss_init, "loss_final": loss_final,
        "y_target": y_target, "model_init_state": model_init_state,
        "model_final_state": deepcopy(model.state_dict()),
        "PNTK": audit.PNTK.detach().cpu() if audit is not None else None,
        "IPS": ips.finalize() if ips is not None else None,
        "_gpu_audit_variant": True,
    }
    if ips is not None:
        out.update(collect_full_eval(
            model=model, train_loader=train_loader, test_loader=test_loader,
            train_batches=train_batches, target_batches=target_batches,
            batch_size=batch_size, audit_batch_size=audit_bs, device=device,
        ))
    return out
