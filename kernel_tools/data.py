# Copyright (C) 2026 Ryan Pyle
# SPDX-License-Identifier: MIT
"""Dataset loaders used across the paper's MNIST + SVHN experiments."""
from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader, Subset
from torchvision.datasets import SVHN, mnist
from torchvision.transforms import Compose, Normalize, ToTensor


SVHN_MEAN = (0.4377, 0.4438, 0.4728)
SVHN_STD = (0.1980, 0.2010, 0.1970)


def mnist_loaders(
    data_root: str | Path = "data/MNIST",
    batch_size: int = 256,
    train_batches: int | None = 200,
    audit_batches: int = 1,
    num_workers: int = 0,
    pin_memory: bool = False,
):
    """Return (train_loader, test_loader, audit_loader).

    Matches the paper's Appendix A.3 protocol: batch size 256, 200 train
    batches used per epoch, 1 test batch (256 samples) used as the audit set.

    ``audit_loader`` is a ``Subset(test_dataset)`` view keeping the first
    ``audit_batches * batch_size`` test samples in fixed order (order matters
    for reproducibility of per-audit-point figures).
    """
    train_root = Path(data_root) / "train"
    test_root = Path(data_root) / "test"

    train_ds = mnist.MNIST(root=str(train_root), train=True, download=True, transform=ToTensor())
    test_ds = mnist.MNIST(root=str(test_root), train=False, download=True, transform=ToTensor())

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    audit_size = audit_batches * batch_size
    audit_indices = range(min(audit_size, len(test_ds)))
    audit_loader = DataLoader(
        Subset(test_ds, list(audit_indices)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return train_loader, test_loader, audit_loader


def svhn_loaders(
    data_root: str | Path = "data/SVHN",
    batch_size: int = 256,
    train_batches: int | None = 200,
    audit_batches: int = 1,
    num_workers: int = 0,
    pin_memory: bool = False,
):
    """Return (train_loader, test_loader, audit_loader) for SVHN.

    Uses ``torchvision.datasets.SVHN`` with the paper's normalization
    (mean/std computed from the SVHN train split). Same audit-set contract as
    ``mnist_loaders``: the first ``audit_batches * batch_size`` test-split
    samples in fixed order.

    Paper defaults (Appendix A.16): batch size 256, 200 train batches per
    epoch, 1 audit batch.
    """
    root = Path(data_root)
    transform = Compose([ToTensor(), Normalize(SVHN_MEAN, SVHN_STD)])

    train_ds = SVHN(root=str(root), split="train", download=True, transform=transform)
    test_ds = SVHN(root=str(root), split="test", download=True, transform=transform)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    audit_size = audit_batches * batch_size
    audit_indices = range(min(audit_size, len(test_ds)))
    audit_loader = DataLoader(
        Subset(test_ds, list(audit_indices)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return train_loader, test_loader, audit_loader
