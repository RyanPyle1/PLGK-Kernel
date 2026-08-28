# Copyright (C) 2026 Ryan Pyle
# SPDX-License-Identifier: MIT
"""Scale-up model architectures for Appendix A.18 (CIFAR + ImageNet).

Two models power the scaling table:
  * ResNet18GN  — ResNet-18 with BatchNorm replaced by GroupNorm.
    ~11.2M params (CIFAR-10 head) / ~11.7M (ImageNet head).
  * ViTTiny4    — ViT-Tiny with patch size 4 (for CIFAR-10).
    ~2.70M params. Standard ViT-Tiny (hidden=192, heads=3) with 6 layers.

Both use ONLY GroupNorm / LayerNorm — never BatchNorm. This is required for
audit / IPS: BN's running-stats break per-sample gradient evaluation via
``torch.func.vmap(grad)`` (each sample would see the same buffer state).
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------------------------
# ResNet-18-GN
# -------------------------------------------------------------------

def _gn(num_channels: int, groups: int = 8) -> nn.GroupNorm:
    """GroupNorm sized so groups divide num_channels; fall back to 1 group."""
    g = groups
    while num_channels % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(g, num_channels)


class BasicBlockGN(nn.Module):
    """ResNet basic block with GroupNorm instead of BatchNorm.

    Optional activation swap: ``activation`` is 'relu' (default, non-smooth)
    or 'silu' (smooth C^inf). SiLU used to isolate ReLU-boundary as the
    trajectory-bifurcation cause vs norm-layer sqrt(var+eps).
    """
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1,
                  activation: str = "relu") -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                                padding=1, bias=False)
        self.gn1 = _gn(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                                padding=1, bias=False)
        self.gn2 = _gn(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                _gn(self.expansion * planes),
            )
        assert activation in ("relu", "silu")
        self._act_fn = F.relu if activation == "relu" else F.silu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._act_fn(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out = out + self.shortcut(x)
        return self._act_fn(out)


class ResNet18GN(nn.Module):
    """ResNet-18 with GroupNorm (BN-free for audit / IPS compatibility).

    Two input-stem variants:
      * ``variant='cifar'``   — 3x3 first conv, no maxpool (standard CIFAR
        ResNet). ~11.17M params with 10-class head.
      * ``variant='imagenet'``— 7x7 first conv + 3x3 maxpool (standard
        ImageNet ResNet). ~11.68M params with 1000-class head.
    """

    def __init__(
        self,
        device: torch.device | str = "cpu",
        num_classes: int = 10,
        variant: str = "cifar",
        activation: str = "relu",
    ) -> None:
        super().__init__()
        assert variant in ("cifar", "imagenet"), variant
        assert activation in ("relu", "silu")
        self.device = torch.device(device)
        self.variant = variant
        self.activation = activation
        act_mod = nn.ReLU if activation == "relu" else nn.SiLU

        self.in_planes = 64
        if variant == "cifar":
            self.stem = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
                _gn(64),
                act_mod(inplace=True) if activation == "relu" else act_mod(),
            )
        else:
            self.stem = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                _gn(64),
                act_mod(inplace=True) if activation == "relu" else act_mod(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )

        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

        self.to(self.device)
        self.numparams, self.num_nonmeta, self.param_sizes = _count(self)

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlockGN(self.in_planes, planes, s, activation=self.activation))
            self.in_planes = planes * BasicBlockGN.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, doHidden: bool = False):
        y = self.stem(x)
        y = self.layer1(y); y = self.layer2(y)
        y = self.layer3(y); y = self.layer4(y)
        y = self.avgpool(y)
        h = y.view(y.shape[0], -1)
        y = self.fc(h)
        if doHidden:
            return h, y
        return y


# -------------------------------------------------------------------
# ResNet-18-LN (LayerNorm variant — trajectory-smoother than GN)
# -------------------------------------------------------------------

class _LayerNormConv(nn.Module):
    """LayerNorm applied to a (B, C, H, W) tensor by normalizing over
    (C, H, W) per-sample. Equivalent to nn.LayerNorm(normalized_shape=(C,H,W))
    but works when H/W are runtime-known.

    We use per-channel affine params (scale + shift), which is standard for
    LN on conv feature maps.
    """
    def __init__(self, num_channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize over (C, H, W) per-sample
        # mean/var over dims (1,2,3), keepdim=True for broadcast
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        var = x.var(dim=(1, 2, 3), unbiased=False, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        # Per-channel affine
        return x_norm * self.weight[None, :, None, None] + self.bias[None, :, None, None]


class BasicBlockLN(nn.Module):
    """ResNet basic block with LayerNorm-over-(C,H,W) instead of BatchNorm/GroupNorm."""
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                                padding=1, bias=False)
        self.ln1 = _LayerNormConv(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                                padding=1, bias=False)
        self.ln2 = _LayerNormConv(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                _LayerNormConv(self.expansion * planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.ln1(self.conv1(x)))
        out = self.ln2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class ResNet18LN(nn.Module):
    """ResNet-18 with LayerNorm (over C,H,W per-sample). BN-free like ResNet18GN.

    Normalization spec differs from ResNet18GN only in the normalization dim:
      GN(groups=8): normalize over C/8 * H * W per sample per group
      LN:           normalize over  C * H * W  per sample (one group)

    LN's larger normalization pool means variance is more stable — hypothesis:
    trajectory bifurcation from `1/sqrt(var+eps)` non-smoothness is much rarer
    under LN than GN, so the trajectory-exact FD gate should pass at eps_0
    up to 1e-3 like ViT-Tiny.

    Variant options identical to ResNet18GN.
    """

    def __init__(
        self,
        device: torch.device | str = "cpu",
        num_classes: int = 10,
        variant: str = "cifar",
    ) -> None:
        super().__init__()
        assert variant in ("cifar", "imagenet"), variant
        self.device = torch.device(device)
        self.variant = variant

        self.in_planes = 64
        if variant == "cifar":
            self.stem = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
                _LayerNormConv(64),
                nn.ReLU(inplace=True),
            )
        else:
            self.stem = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                _LayerNormConv(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )

        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

        self.to(self.device)
        self.numparams, self.num_nonmeta, self.param_sizes = _count(self)

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlockLN(self.in_planes, planes, s))
            self.in_planes = planes * BasicBlockLN.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, doHidden: bool = False):
        y = self.stem(x)
        y = self.layer1(y); y = self.layer2(y)
        y = self.layer3(y); y = self.layer4(y)
        y = self.avgpool(y)
        h = y.view(y.shape[0], -1)
        y = self.fc(h)
        if doHidden:
            return h, y
        return y


# -------------------------------------------------------------------
# ViT-Tiny / patch 4 (~2.70M params on CIFAR-10)
# -------------------------------------------------------------------

class _PatchEmbed(nn.Module):
    def __init__(self, image_size: int, patch_size: int, in_ch: int, dim: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch_size, stride=patch_size)
        self.num_patches = (image_size // patch_size) ** 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)  # (B, N_patches, dim)


class _MHA(nn.Module):
    def __init__(self, dim: int, heads: int, attn_drop: float, proj_drop: float) -> None:
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(F.softmax(attn, dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.proj_drop(self.proj(out))


class _MLP(nn.Module):
    def __init__(self, dim: int, hidden: int, drop: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(F.gelu(self.fc1(x)))))


class _Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float, drop: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = _MHA(dim, heads, drop, drop)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = _MLP(dim, int(dim * mlp_ratio), drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class ViTTiny4(nn.Module):
    """ViT-Tiny with patch size 4 (CIFAR-10). ~2.7M params.

    Default: hidden=192, layers=6, heads=3, mlp_ratio=4.
    Standard ViT-Tiny has layers=12; we use 6 to hit the paper's 2.70M
    param count (12 layers would be ~5.5M).
    """

    def __init__(
        self,
        device: torch.device | str = "cpu",
        num_classes: int = 10,
        image_size: int = 32,
        patch_size: int = 4,
        dim: int = 192,
        depth: int = 6,
        heads: int = 3,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.patch_embed = _PatchEmbed(image_size, patch_size, in_ch=3, dim=dim)
        n_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.drop = nn.Dropout(drop)
        self.blocks = nn.ModuleList([
            _Block(dim, heads, mlp_ratio, drop) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

        self.to(self.device)
        self.numparams, self.num_nonmeta, self.param_sizes = _count(self)

    def forward(self, x: torch.Tensor, doHidden: bool = False):
        B = x.shape[0]
        x = self.patch_embed(x)                              # (B, N, D)
        cls = self.cls_token.expand(B, -1, -1)               # (B, 1, D)
        x = torch.cat((cls, x), dim=1) + self.pos_embed      # (B, N+1, D)
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        h = x[:, 0]                                          # CLS token
        y = self.head(h)
        if doHidden:
            return h, y
        return y


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _count(model: nn.Module) -> tuple[int, int, np.ndarray]:
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
