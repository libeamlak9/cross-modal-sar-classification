from __future__ import annotations

from typing import List, Optional

import torch
from torch import nn


class MAF(nn.Module):
    def __init__(self, in_channels: List[int], out_dim: int = 1024) -> None:
        super().__init__()
        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, 256, kernel_size=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
            )
            for ch in in_channels
        ])
        self.fuse = nn.Sequential(
            nn.Linear(256 * len(in_channels), out_dim, bias=False),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        pooled = []
        for f, proj in zip(feats, self.proj):
            pooled.append(proj(f).flatten(1))
        concat = torch.cat(pooled, dim=1)
        return self.fuse(concat)


class ICD(nn.Module):
    def __init__(self, feat_dim: int, num_classes: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, feat_dim, bias=False),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(feat_dim, num_classes, bias=False)

    def forward(self, z_e: torch.Tensor, base_logits: torch.Tensor) -> torch.Tensor:
        resid = self.fc(self.proj(z_e))
        return base_logits + resid


class AlphaGenerator(nn.Module):
    def __init__(self, feat_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 4, bias=False),
            nn.BatchNorm1d(feat_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim // 4, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, z_e: torch.Tensor) -> torch.Tensor:
        return self.net(z_e)


class LDCHead(nn.Module):
    def __init__(self, in_channels: List[int], feat_dim: int, num_classes: int, use_maf_head: bool = False) -> None:
        super().__init__()
        self.maf = MAF(in_channels=in_channels, out_dim=feat_dim)
        self.icd = ICD(feat_dim=feat_dim, num_classes=num_classes)
        self.alpha_gen = AlphaGenerator(feat_dim=feat_dim)
        self.use_maf_head = use_maf_head
        self.maf_head = None
        if use_maf_head:
            self.maf_head = nn.Sequential(
                nn.Linear(feat_dim, feat_dim, bias=False),
                nn.BatchNorm1d(feat_dim),
                nn.ReLU(inplace=True),
                nn.Linear(feat_dim, num_classes, bias=False),
            )

    def forward(self, feats: List[torch.Tensor], base_logits: torch.Tensor,
                disable_icd: bool = False):
        z_e = self.maf(feats)
        s_icd = base_logits if disable_icd else self.icd(z_e, base_logits)
        alpha = self.alpha_gen(z_e)
        s_alf = alpha * s_icd + (1.0 - alpha) * base_logits
        s_maf = self.maf_head(z_e) if self.maf_head is not None else None
        return z_e, s_icd, s_alf, s_maf

