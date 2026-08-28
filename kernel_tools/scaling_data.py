# Copyright (C) 2026 Ryan Pyle
# SPDX-License-Identifier: MIT
"""Dataset loaders for Appendix A.18 scaling experiments.

  * ``cifar10_loaders`` — torchvision auto-download (~170 MB).
  * ``imagenet_loaders`` — bring-your-own ImageFolder rooted at
    ``<root>/train`` and ``<root>/val``. Not downloadable via torchvision.

Both mirror the ``mnist_loaders`` / ``svhn_loaders`` interface: return
``(train_loader, test_loader, audit_loader)`` where the audit loader is
the first ``audit_batches * batch_size`` test samples in fixed order.
"""
from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10, ImageFolder


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def cifar10_loaders(
    data_root: str | Path = "data/CIFAR10",
    batch_size: int = 128,
    train_batches: int | None = None,
    audit_batches: int = 1,
    num_workers: int = 0,
    pin_memory: bool = False,
    augment: bool = False,
):
    """CIFAR-10 loaders. ``augment=False`` for audit reproducibility.

    Paper's Appendix A.18 uses no augmentation for the audit runs (so that
    the per-training-sample gradient identity holds across the audit
    pass). Set ``augment=True`` only for baseline sanity-check training.
    """
    root = Path(data_root)
    normalize = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    if augment:
        train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_tf = transforms.Compose([transforms.ToTensor(), normalize])
    test_tf = transforms.Compose([transforms.ToTensor(), normalize])

    train_ds = CIFAR10(root=str(root), train=True, download=True, transform=train_tf)
    test_ds = CIFAR10(root=str(root), train=False, download=True, transform=test_tf)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    audit_size = audit_batches * batch_size
    audit_indices = list(range(min(audit_size, len(test_ds))))
    audit_loader = DataLoader(
        Subset(test_ds, audit_indices),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        drop_last=False,
    )
    return train_loader, test_loader, audit_loader


def imagenet_loaders(
    data_root: str | Path,
    batch_size: int = 256,
    train_batches: int | None = None,
    audit_batches: int = 1,
    num_workers: int = 4,
    pin_memory: bool = True,
    image_size: int = 224,
):
    """ImageNet loaders. ``data_root`` must contain ``train/`` and ``val/``
    ImageFolder-formatted subdirectories.

    ImageNet is NOT auto-downloaded — obtain via ImageNet.org and preprocess.
    Paper A.18 protocol: no augmentation on train pass (audit reproducibility).
    """
    root = Path(data_root)
    train_dir = root / "train"
    val_dir = root / "val"
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            f"ImageNet not found under {root}. Expected {train_dir} and {val_dir}."
        )
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    train_tf = transforms.Compose([
        transforms.Resize(int(image_size * 1.15)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])
    val_tf = transforms.Compose([
        transforms.Resize(int(image_size * 1.15)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])

    train_ds = ImageFolder(str(train_dir), transform=train_tf)
    val_ds = ImageFolder(str(val_dir), transform=val_tf)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        val_ds, batch_size=batch_size,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    audit_size = audit_batches * batch_size
    audit_indices = list(range(min(audit_size, len(val_ds))))
    audit_loader = DataLoader(
        Subset(val_ds, audit_indices),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        drop_last=False,
    )
    return train_loader, test_loader, audit_loader
