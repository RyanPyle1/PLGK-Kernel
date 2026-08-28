# Copyright (C) 2026 Ryan Pyle
# SPDX-License-Identifier: MIT
"""Training orchestrator for the paper's MNIST/SVHN experiments.

``train_with_audit`` runs the model's standard training loop and, optionally,
maintains an ``AuditAccumulator`` and/or ``IPSAccumulator`` in lockstep. The
returned dict contains everything downstream figure-plotting scripts need:

    {
        "trainloss": ndarray of shape (num_updates,),
        "testacc":   ndarray of shape (epochs,),
        "logitcorrs": ndarray of shape (epochs,),
        "mles":      ndarray of shape (epochs,),
        "loss_init": tensor of shape (n_audit,),
        "loss_final": tensor of shape (n_audit,),
        "y_target":  tensor of shape (n_audit,),
        "model_init_state": OrderedDict,
        "model_final_state": OrderedDict,
        "PNTK":      tensor of shape (n_train_samples, n_audit) or None,
        "IPS":       tensor of shape (n_total, n_total) or None,

        # Populated when do_ips=True (needed by IPS-consuming figures):
        "y_target_full":  tensor of shape (n_total,)  int labels
        "y_pred_full":    tensor of shape (n_total,)  argmax predictions
        "y_loss_full":    tensor of shape (n_total,)  per-example CE loss
        "y_logits_full":  tensor of shape (n_total, n_classes) raw logits
        "y_hidden_full":  tensor of shape (n_total, hidden_dim) penultimate activations
    }

For Fig 1 reconstruction plots you need trainloss/logitcorrs/mles.
For Figs 2-4 you need PNTK + final model + audit_loader.
For IPS UMAP plots (Figs 5-7) + Table 1 you need IPS + y_*_full arrays.
"""
from __future__ import annotations

import time
from copy import deepcopy
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from .audit import AuditAccumulator
from .ips import IPSAccumulator


def train_with_audit(
    model: nn.Module,
    train_loader,
    test_loader,
    audit_loader,
    *,
    epochs: int = 50,
    train_batches: int = 200,
    batch_size: int = 256,
    lr: float = 1.5e-5,
    device: torch.device | str = "cpu",
    do_audit: bool = True,
    do_ips: bool = False,
    ips_update_every: int = 200,
    ips_proj_dim: int | None = None,
    nahead: int = 50,
    target_batches: int = 1,
    audit_batch_size: int | None = None,
    verbose: bool = True,
    step_callback = None,
    epoch_callback = None,
) -> dict[str, Any]:
    """Train + optionally accumulate audit / IPS.

    Order per step:

    1. Snapshot pre-step train gradients into the audit's rolling store
       (before optimizer.step, so we know the pre-step model state gave rise
       to the step being taken).
    2. optimizer.step()
    3. Compute post-step audit gradients (NTKtest).
    4. Trapezoidal PNTK update.
    5. IPS update (only when ``(step+1) % ips_update_every == 0``).
    6. ``step_callback`` invoked (if provided) — see hook signature below.

    End-of-epoch: ``epoch_callback`` invoked before eval.

    Hooks (Group 2 / §6.3 support):
      step_callback(*, audit, ips, optimizer, epoch, idx, update, lr)
      epoch_callback(*, audit, ips, model, epoch, epochs, optimizer, train_batches, lr)

    Callbacks read the accumulator's current-step state (``NTKtrain_store``,
    ``NTKtest``, ``PNTK``) to record per-layer, per-time-slice, or SVD
    decompositions without duplicating the training loop.

    Returns the artifact dict described in the module docstring.
    """
    device = torch.device(device)
    model = model.to(device)
    numparams = int(sum(p.numel() for p in model.parameters() if not p.is_meta))

    # -- setup accumulators --
    # If audit_batch_size is passed explicitly, use it; otherwise default to
    # the training batch_size (paper's convention). Also compute
    # n_audit_samples from whichever we're using.
    audit_bs = audit_batch_size if audit_batch_size is not None else batch_size
    n_train_samples = train_batches * batch_size
    n_audit_samples = target_batches * audit_bs

    audit = None
    ips = None
    if do_audit:
        audit = AuditAccumulator(
            model=model,
            numparams=numparams,
            ndat_train=n_train_samples,
            n_audit=n_audit_samples,
            batch_size=batch_size,
            device=device,
            nahead=nahead,
            audit_batch_size=audit_bs,
        )
    if do_ips:
        ips = IPSAccumulator(
            model=model,
            numparams=numparams,
            n_train_samples=n_train_samples,
            n_audit_samples=n_audit_samples,
            batch_size=batch_size,
            device=device,
            update_every=ips_update_every,
            proj_dim=ips_proj_dim,
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = CrossEntropyLoss()

    num_updates = train_batches * epochs
    trainloss = np.zeros(num_updates)
    testacc = np.zeros(epochs)
    logitcorrs = np.zeros(epochs)
    mles = np.zeros(epochs)
    model_init_state = deepcopy(model.state_dict())

    # Prime the audit with initial gradients on the audit set
    y_init, y_target = (
        audit.prime_test_gradients(audit_loader, target_batches)
        if audit is not None
        else (
            _compute_audit_outputs(model, audit_loader, batch_size, target_batches),
            _extract_audit_labels(audit_loader, batch_size, target_batches),
        )
    )
    loss_init = torch.zeros(n_audit_samples)
    for i in range(len(y_init)):
        loss_init[i] = loss_fn(y_init[i], y_target[i].long())

    eta_scratch = torch.zeros(numparams, device=device)  # for IPS

    update = 0
    t0 = time.time()
    for _epoch in range(epochs):
        for idx, (train_x, train_label) in enumerate(train_loader):
            if idx >= train_batches:
                break
            optimizer.zero_grad()
            predict_y = model(train_x.to(device).float())
            _error = loss_fn(predict_y, train_label.to(device).long())
            _error.backward(retain_graph=True)
            optimizer.step()

            if audit is not None:
                audit.update(
                    train_x=train_x, train_label=train_label, idx=idx,
                    audit_loader=audit_loader, optimizer=optimizer,
                    epoch=_epoch, train_batches=train_batches, lr_use=lr,
                )
            if ips is not None:
                ips.maybe_update(
                    step=update, train_loader=train_loader, audit_loader=audit_loader,
                    optimizer=optimizer, epoch=_epoch, train_batches=train_batches,
                    eta_scratch=eta_scratch,
                )

            # User callback invoked AFTER audit + IPS updates so it sees the
            # accumulator's current step state. Signature:
            #   step_callback(*, audit, ips, optimizer, epoch, idx, update, lr)
            if step_callback is not None:
                step_callback(
                    audit=audit, ips=ips, optimizer=optimizer,
                    epoch=_epoch, idx=idx, update=update, lr=lr,
                )

            trainloss[update] = _error.item()
            update += 1

        # End-of-epoch callback (before eval to give callbacks the freshest
        # accumulator state before any measurement).
        if epoch_callback is not None:
            epoch_callback(
                audit=audit, ips=ips, model=model,
                epoch=_epoch, epochs=epochs,
                optimizer=optimizer, train_batches=train_batches, lr=lr,
            )

        # -- end-of-epoch eval --
        correct = 0
        total = 0
        for test_x, test_label in test_loader:
            with torch.no_grad():
                pred = model(test_x.to(device).float())
            pred_y = np.argmax(pred.data.cpu().numpy(), axis=-1)
            correct += int(np.sum(pred_y == test_label.data.numpy()))
            total += test_label.shape[0]
        testacc[_epoch] = correct / total

        # Audit metrics per epoch (need loss_final at this checkpoint)
        y_final = torch.zeros(n_audit_samples, 10)
        loss_final = torch.zeros(n_audit_samples)
        with torch.no_grad():
            for idy, (test_x, _tl) in enumerate(audit_loader):
                if idy < target_batches:
                    preds = model(test_x.to(device).float())
                    n_this = preds.shape[0]
                    s = idy * audit_bs
                    e = s + n_this
                    y_final[s:e, :] = preds.cpu()
            for i in range(len(y_final)):
                loss_final[i] = loss_fn(y_final[i], y_target[i].long())

        if audit is not None:
            mre, lc = audit.loss_reconstruction(loss_init, loss_final)
            logitcorrs[_epoch] = lc
            mles[_epoch] = mre
            if verbose:
                print(
                    f"Epoch {_epoch:3d}: acc={testacc[_epoch]:.4f}  MRE={mre:.4g}  Corr={lc:.6f}"
                )
        else:
            if verbose:
                print(f"Epoch {_epoch:3d}: acc={testacc[_epoch]:.4f}")

    if verbose:
        print(f"Total train time: {time.time() - t0:.1f}s")

    out = {
        "trainloss": trainloss,
        "testacc": testacc,
        "logitcorrs": logitcorrs,
        "mles": mles,
        "loss_init": loss_init,
        "loss_final": loss_final,
        "y_target": y_target,
        "model_init_state": model_init_state,
        "model_final_state": deepcopy(model.state_dict()),
        "PNTK": audit.PNTK if audit is not None else None,
        "IPS": ips.finalize() if ips is not None else None,
    }
    if ips is not None:
        out.update(
            collect_full_eval(
                model=model, train_loader=train_loader, test_loader=test_loader,
                train_batches=train_batches, target_batches=target_batches,
                batch_size=batch_size, audit_batch_size=audit_bs, device=device,
            )
        )
    return out


def collect_full_eval(
    model: nn.Module,
    train_loader,
    test_loader,
    *,
    train_batches: int,
    target_batches: int,
    batch_size: int,
    device: torch.device,
    audit_batch_size: int | None = None,
) -> dict[str, torch.Tensor]:
    """Collect per-example labels / predictions / losses / logits / hidden.

    Union of first ``train_batches`` train batches and first
    ``target_batches`` test batches, aligned with the IPS matrix.

    ``audit_batch_size`` defaults to ``batch_size``; pass it explicitly when
    the audit / test loader uses a different batch size (e.g. adversarial
    audit sets that pack N successful examples into a single batch).

    Depends on the model's ``forward(x, doHidden=True)`` returning
    ``(hidden, logits)``. The paper's LNModel and SVHN CNN both support this.
    """
    from torch.nn import CrossEntropyLoss
    loss_fn = CrossEntropyLoss()
    audit_bs = audit_batch_size if audit_batch_size is not None else batch_size
    n_train_samples = train_batches * batch_size
    n_audit_samples = target_batches * audit_bs
    n_total = n_train_samples + n_audit_samples
    # infer hidden_dim from a probe pass
    with torch.no_grad():
        probe_x, _ = next(iter(train_loader))
        probe_h, _probe_y = model(probe_x[:1].to(device), doHidden=True)
        hidden_dim = probe_h.shape[-1]
        n_classes = _probe_y.shape[-1]

    y_target_full = torch.zeros(n_total)
    y_pred_full = torch.zeros(n_total)
    y_loss_full = torch.zeros(n_total)
    y_logits_full = torch.zeros(n_total, n_classes)
    y_hidden_full = torch.zeros(n_total, hidden_dim)

    with torch.no_grad():
        for idx, (x, label) in enumerate(train_loader):
            if idx >= train_batches:
                break
            h, y_logit = model(x.to(device).float(), doHidden=True)
            h = h.cpu(); y_logit = y_logit.cpu()
            s = idx * batch_size; e = s + batch_size
            y_target_full[s:e] = label
            y_pred_full[s:e] = torch.argmax(y_logit, 1)
            y_logits_full[s:e] = y_logit
            y_hidden_full[s:e] = h
            for i in range(batch_size):
                y_loss_full[s + i] = loss_fn(y_logit[i], label[i].long())
        for idy, (x, label) in enumerate(test_loader):
            if idy >= target_batches:
                break
            h, y_logit = model(x.to(device).float(), doHidden=True)
            h = h.cpu(); y_logit = y_logit.cpu()
            n_this = int(label.shape[0])
            s = n_train_samples + idy * audit_bs; e = s + n_this
            y_target_full[s:e] = label
            y_pred_full[s:e] = torch.argmax(y_logit, 1)
            y_logits_full[s:e] = y_logit
            y_hidden_full[s:e] = h
            for i in range(n_this):
                y_loss_full[s + i] = loss_fn(y_logit[i], label[i].long())

    return {
        "y_target_full": y_target_full,
        "y_pred_full": y_pred_full,
        "y_loss_full": y_loss_full,
        "y_logits_full": y_logits_full,
        "y_hidden_full": y_hidden_full,
    }


def _compute_audit_outputs(model, audit_loader, batch_size: int, target_batches: int):
    Dout = 10
    n_audit = target_batches * batch_size
    y = torch.zeros(n_audit, Dout)
    with torch.no_grad():
        for idy, (test_x, _label) in enumerate(audit_loader):
            if idy < target_batches:
                y[idy * batch_size:(idy + 1) * batch_size] = model(test_x)
    return y


def _extract_audit_labels(audit_loader, batch_size: int, target_batches: int):
    n_audit = target_batches * batch_size
    y_target = torch.zeros(n_audit)
    for idy, (_test_x, test_label) in enumerate(audit_loader):
        if idy < target_batches:
            y_target[idy * batch_size:(idy + 1) * batch_size] = test_label
    return y_target
