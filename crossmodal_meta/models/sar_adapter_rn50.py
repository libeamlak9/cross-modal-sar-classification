from __future__ import annotations

from typing import Dict, Iterable

import torch
from torch import nn


class BottleneckAdapter(nn.Module):
    def __init__(self, channels: int, bottleneck_ratio: int = 16) -> None:
        super().__init__()
        hidden = max(channels // bottleneck_ratio, 8)
        self.norm = nn.GroupNorm(1, channels)
        self.down = nn.Conv2d(channels, hidden, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.up = nn.Conv2d(hidden, channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != self.norm.weight.dtype:
            x = x.to(self.norm.weight.dtype)
        return x + self.up(self.relu(self.down(self.norm(x))))


class SARBlockAdapters(nn.Module):
    def __init__(self, channels: Dict[str, int], blocks: Dict[str, int],
                 enabled_layers: Iterable[str], bottleneck_ratio: int) -> None:
        super().__init__()
        self.enabled_layers = set(enabled_layers)
        self.adapters = nn.ModuleDict()
        for layer_name in self.enabled_layers:
            for block_idx in range(blocks[layer_name]):
                key = f"{layer_name}_{block_idx}"
                self.adapters[key] = BottleneckAdapter(channels[layer_name], bottleneck_ratio=bottleneck_ratio)

    def apply(self, layer_name: str, block_idx: int, x: torch.Tensor) -> torch.Tensor:
        key = f"{layer_name}_{block_idx}"
        if key in self.adapters:
            return self.adapters[key](x)
        return x


class SARBlockAdaptersV2(nn.Module):
    """Support per-layer bottleneck ratios."""
    def __init__(self, channels: Dict[str, int], blocks: Dict[str, int],
                 enabled_layers: Iterable[str], 
                 bottleneck_ratios: Dict[str, int]) -> None:
        super().__init__()
        self.enabled_layers = set(enabled_layers)
        self.adapters = nn.ModuleDict()
        for layer_name in self.enabled_layers:
            for block_idx in range(blocks[layer_name]):
                key = f"{layer_name}_{block_idx}"
                bottleneck = bottleneck_ratios.get(layer_name, 16)
                self.adapters[key] = BottleneckAdapter(channels[layer_name], bottleneck_ratio=bottleneck)

    def apply(self, layer_name: str, block_idx: int, x: torch.Tensor) -> torch.Tensor:
        key = f"{layer_name}_{block_idx}"
        if key in self.adapters:
            return self.adapters[key](x)
        return x

