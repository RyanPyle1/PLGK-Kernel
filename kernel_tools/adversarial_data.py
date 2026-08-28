# Copyright (C) 2026 Ryan Pyle
# SPDX-License-Identifier: MIT
"""Adversarial audit-set generation (Section 6.3).

Given a trained model + a batch of test images/labels, produces:

  - ``adv_data`` (256, 1, 28, 28) — Task A audit set of clean/adv/random
    quadruplets for 16 successful test examples × 4 attacks
    (FGSM, PGD-L∞, BIM-L∞, DeepFool-L∞). Layout per slot of 16:
       [0]  clean            [1]  adv attack-0    [2] random1     [3] random2
       [4]  clean            [5]  adv attack-1    [6] random1     [7] random2
       ... same pattern for attack-2 (BIM) and attack-3 (DeepFool)

    The random perturbations are ``eps``-matched to the adversarial (the
    smallest eps at which that attack succeeded on that image).

  - ``adv_data2`` (256, 1, 28, 28) — Task B interpolation-ray audit set:
    128 points on the linear ray from ``adv_data[0]`` (clean) to
    ``adv_data[1]`` (adv), followed by 128 points on the ray from
    ``adv_data[0]`` to ``adv_data[2]`` (a matched-eps random perturbation).
    Used for the Fig 11 perturbation-path analysis.

Uses the ``foolbox`` package.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


DEFAULT_EPSILONS = [0.0, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24]


def build_adversarial_audit_sets(
    model: nn.Module,
    test_x: torch.Tensor,
    test_label: torch.Tensor,
    *,
    n_successful: int = 16,
    epsilons: list[float] | None = None,
    bounds: Tuple[float, float] = (0.0, 1.0),
    device: torch.device | str = "cpu",
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Build Task A + Task B adversarial audit sets.

    Returns a dict with keys:
      ``adv_data``       (n_successful*16, 1, 28, 28)
      ``adv_label``      (n_successful*16,)
      ``adv_data2``      (256, 1, 28, 28) — assumes n_successful=16
      ``adv_label2``     (256,)  all set to adv_label[0]
      ``success_inds``   list[int] — original indices in ``test_x`` used
      ``epsilons_used``  (n_successful, 4)  — per-attack per-example eps

    ``test_x``/``test_label``: first N (e.g. 256) test samples. We scan
    linearly and stop as soon as ``n_successful`` examples have (a) been
    correctly classified at eps=0, and (b) been successfully attacked by
    ALL 4 attacks at max epsilon. Raises RuntimeError if the input batch
    doesn't contain enough qualifying examples.
    """
    import foolbox as fb

    if epsilons is None:
        epsilons = DEFAULT_EPSILONS
    epsilons = list(epsilons)

    device = torch.device(device)
    model = model.to(device).eval()

    fmodel = fb.PyTorchModel(model, bounds=bounds)
    attacks = [
        fb.attacks.FGSM(),
        fb.attacks.LinfPGD(),
        fb.attacks.LinfBasicIterativeAttack(),
        fb.attacks.LinfDeepFoolAttack(),
    ]

    x = test_x.to(device); y = test_label.to(device)

    # Run each attack across all epsilons on the whole batch
    per_attack_advs: list[list[torch.Tensor]] = []
    per_attack_success: list[torch.Tensor] = []
    for atk in attacks:
        _raw, advs, success = atk(fmodel, x, y, epsilons=epsilons)
        per_attack_advs.append(advs)
        per_attack_success.append(success)

    # A "successful index" is one where:
    #   - eps=0 (index 0) success == 0 (i.e. clean classification is correct;
    #     foolbox reports success=1 when the attack fools the model, so at
    #     eps=0 success==0 means the clean prediction is already correct)
    #   - AT LEAST max eps success == 1 for ALL 4 attacks
    n_in = int(x.shape[0])
    success_inds: list[int] = []
    for i in range(n_in):
        clean_ok = all(per_attack_success[a][0, i].item() == 0 for a in range(4))
        max_flips = all(per_attack_success[a][-1, i].item() == 1 for a in range(4))
        if clean_ok and max_flips:
            success_inds.append(i)
            if len(success_inds) == n_successful:
                break
    if len(success_inds) < n_successful:
        raise RuntimeError(
            f"Only {len(success_inds)}/{n_successful} test examples satisfy the paper's success "
            "criterion (clean-correct + all-4-attacks flip at max eps). Pass a larger test batch."
        )

    # Assemble adv_data: 16 slots per successful example, 4 attacks each,
    # each attack contributing [clean, adv, rand1, rand2].
    rng = torch.Generator(device="cpu").manual_seed(seed)
    N = n_successful * 16
    adv_data = torch.zeros((N,) + tuple(test_x.shape[1:]))
    adv_label = torch.zeros(N, dtype=test_label.dtype)
    eps_used = torch.zeros(n_successful, 4)

    for i, ti in enumerate(success_inds):
        clean_img = test_x[ti].detach().cpu()
        for a in range(4):
            first_a = int(torch.where(per_attack_success[a][:, ti] > 0)[0][0].item())
            eps_a = epsilons[first_a]
            eps_used[i, a] = eps_a
            slot = 16 * i + 4 * a
            # [clean, adv, rand1, rand2] each eps-matched
            adv_img = per_attack_advs[a][first_a][ti].detach().cpu()
            rand1 = clean_img + (2 * torch.rand(clean_img.shape, generator=rng) - 1) * eps_a
            rand2 = clean_img + (2 * torch.rand(clean_img.shape, generator=rng) - 1) * eps_a
            adv_data[slot] = clean_img
            adv_data[slot + 1] = adv_img
            adv_data[slot + 2] = rand1
            adv_data[slot + 3] = rand2
        adv_label[16 * i:16 * (i + 1)] = test_label[ti]

    # Task B: interpolation rays on the FIRST example only.
    # N//2 points on the adv ray + N//2 on the random ray, both anchored at
    # adv_data[0]. Paper uses N=256 (128+128); we scale to len(adv_data) so
    # a smaller ``n_successful`` (e.g. for smoke tests) still produces a
    # coherent bundle.
    N = adv_data.shape[0]
    half = N // 2
    adv_data2 = torch.zeros_like(adv_data)
    adv_label2 = torch.full_like(adv_label, int(adv_label[0]))
    if half > 0:
        denom = max(half - 1, 1)
        for i in range(half):
            adv_data2[i] = adv_data[0] + (i / denom) * (adv_data[1] - adv_data[0])
            adv_data2[half + i] = adv_data[0] + (i / denom) * (adv_data[2] - adv_data[0])

    return {
        "adv_data": adv_data,
        "adv_label": adv_label,
        "adv_data2": adv_data2,
        "adv_label2": adv_label2,
        "success_inds": success_inds,
        "epsilons_used": eps_used,
    }
