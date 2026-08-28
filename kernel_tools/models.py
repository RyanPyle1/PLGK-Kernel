# Copyright (C) 2026 Ryan Pyle
# SPDX-License-Identifier: MIT
"""Model architectures used in the paper's MNIST + SVHN experiments.

The MNIST architecture (``LNModel``) is the LeNet-5 variant used throughout
Sections 5, 6, and Appendix A of the paper. Parameter count: 44,426 (see
Appendix A.3).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class LNModel(nn.Module):
    """LeNet-5 (44,426 params).

    Layer breakdown (matches Appendix A.3):
      conv1: 1->6 channels, 5x5    -> 6 + 150 params
      conv2: 6->16 channels, 5x5   -> 16 + 2400 params
      fc1:   256 -> 120            -> 120 + 30720 params
      fc2:   120 -> 84             -> 84 + 10080 params
      fc3:   84  -> 10             -> 10 + 840 params
      total: 44,426 params
    """

    def __init__(self, device: torch.device | str = "cpu") -> None:
        super().__init__()
        self.device = torch.device(device)
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(256, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)
        self.to(self.device)
        self.numparams, self.num_nonmeta, self.param_sizes = count_params(self)

    def forward(self, x: torch.Tensor, doHidden: bool = False):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.fc2(y)
        h = self.relu4(y)
        y = self.fc3(h)
        if doHidden:
            return h, y
        return y


class SVHNModel(nn.Module):
    """Small CNN for SVHN (Appendix A.16), ~110k params — ~2.5x LeNet.

    Non-BatchNorm variant used in the paper. GroupNorm is used instead so
    that per-sample gradients via ``torch.func.vmap(grad)`` remain valid
    (BN's running-stats state breaks functional evaluation).

    Layout (channels default (32, 64, 128)):
      Block 1: Conv3x3(3->32) + GN + ReLU + Conv3x3(32->32) + GN + ReLU + MaxPool(2)
      Block 2: Conv3x3(32->64) + GN + ReLU + Conv3x3(64->64) + GN + ReLU + MaxPool(2)
      Block 3: Conv3x3(64->128) + GN + ReLU
      Head:    GlobalAvgPool -> Linear(128 -> 10)
    """

    def __init__(
        self,
        device: torch.device | str = "cpu",
        channels: tuple[int, int, int] = (32, 64, 128),
        num_classes: int = 10,
        groups: int = 8,
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        c1, c2, c3 = channels

        self.conv1 = nn.Conv2d(3, c1, kernel_size=3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(groups, c1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(c1, c1, kernel_size=3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(groups, c1)
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(c1, c2, kernel_size=3, padding=1, bias=False)
        self.gn3 = nn.GroupNorm(groups, c2)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(c2, c2, kernel_size=3, padding=1, bias=False)
        self.gn4 = nn.GroupNorm(groups, c2)
        self.relu4 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)

        self.conv5 = nn.Conv2d(c2, c3, kernel_size=3, padding=1, bias=False)
        self.gn5 = nn.GroupNorm(groups, c3)
        self.relu5 = nn.ReLU()

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(c3, num_classes)

        self.to(self.device)
        self.numparams, self.num_nonmeta, self.param_sizes = count_params(self)

    def forward(self, x: torch.Tensor, doHidden: bool = False):
        y = self.pool1(self.relu2(self.gn2(self.conv2(self.relu1(self.gn1(self.conv1(x)))))))
        y = self.pool2(self.relu4(self.gn4(self.conv4(self.relu3(self.gn3(self.conv3(y)))))))
        y = self.relu5(self.gn5(self.conv5(y)))
        y = self.gap(y)
        h = y.view(y.shape[0], -1)
        y = self.fc(h)
        if doHidden:
            return h, y
        return y


def count_params(model: nn.Module) -> tuple[int, int, np.ndarray]:
    """Return (total_params, num_nonmeta_tensors, per-tensor sizes).

    Skips meta tensors, which torch/functorch sometimes creates alongside real
    tensors and would double-count if included.
    """
    total = 0
    n_nonmeta = 0
    sizes: list[int] = []
    for p in model.parameters():
        if not p.is_meta:
            n = int(p.flatten().size(0))
            total += n
            n_nonmeta += 1
            sizes.append(n)
    return total, n_nonmeta, np.asarray(sizes, dtype=np.int64)
