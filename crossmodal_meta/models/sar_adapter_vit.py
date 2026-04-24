from __future__ import annotations

from typing import Iterable

import torch
from torch import nn


class ViTAdapterBlock(nn.Module):
    def __init__(self, dim: int, bottleneck_ratio: int = 16) -> None:
        super().__init__()
        hidden = max(dim // bottleneck_ratio, 8)
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, hidden, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.up = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != self.down.weight.dtype:
            x = x.to(self.down.weight.dtype)
        return x + self.up(self.relu(self.down(self.norm(x))))


class SARViTAdapters(nn.Module):
    def __init__(self, dim: int, num_blocks: int, enabled_blocks: Iterable[int], bottleneck_ratio: int) -> None:
        super().__init__()
        self.enabled_blocks = set(enabled_blocks)
        self.adapters = nn.ModuleDict()
        for idx in range(num_blocks):
            if idx in self.enabled_blocks:
                self.adapters[f"block_{idx}"] = ViTAdapterBlock(dim, bottleneck_ratio=bottleneck_ratio)

    def apply(self, block_idx: int, x: torch.Tensor) -> torch.Tensor:
        key = f"block_{block_idx}"
        if key in self.adapters:
            return self.adapters[key](x)
        return x
