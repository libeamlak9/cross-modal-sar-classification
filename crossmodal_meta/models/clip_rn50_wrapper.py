from __future__ import annotations

from typing import Iterable, List, Tuple

import torch
from torch import nn

import clip_ldc as clip


class _DummyConfig:
    def __init__(self) -> None:
        self.fuse_type = 2
from .sar_adapter_rn50 import SARBlockAdapters, SARBlockAdaptersV2
from .lora import apply_visual_lora, apply_text_lora


class CLIPRN50Wrapper(nn.Module):
    def __init__(
        self,
        adapter_layers: Iterable[str],
        adapter_bottleneck: int,
        cache_dir: str,
        visual_lora: Iterable[str] | None = None,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.0,
        text_lora_r: int = 0,  # 0 = disabled
        unfreeze_text_encoder: bool = False,  # Unfreeze entire text encoder
        adapter_bottleneck_per_layer: dict | None = None,  # Per-layer bottleneck (e.g., {"layer2": 8, "layer3": 8, "layer4": 16})
    ) -> None:
        super().__init__()
        self.clip_model, _ = clip.load("RN50", download_root=cache_dir, config=_DummyConfig())
        self.clip_model.float()
        self.visual = self.clip_model.visual
        self.adapter_layers = list(adapter_layers)
        self.visual_lora = list(visual_lora) if visual_lora is not None else []
        self.lora_r = int(lora_r)
        self.lora_alpha = int(lora_alpha)
        self.lora_dropout = float(lora_dropout)
        self.text_lora_r = int(text_lora_r)
        self.text_lora_enabled = text_lora_r > 0
        self.unfreeze_text_encoder = bool(unfreeze_text_encoder)
        channels = {
            "layer1": 256,
            "layer2": 512,
            "layer3": 1024,
            "layer4": 2048,
        }
        blocks = {
            "layer1": len(self.visual.layer1),
            "layer2": len(self.visual.layer2),
            "layer3": len(self.visual.layer3),
            "layer4": len(self.visual.layer4),
        }
        
        # Use per-layer bottleneck if provided, otherwise use single bottleneck
        if adapter_bottleneck_per_layer is not None:
            self.sar_adapter = SARBlockAdaptersV2(
                channels=channels,
                blocks=blocks,
                enabled_layers=self.adapter_layers,
                bottleneck_ratios=adapter_bottleneck_per_layer,
            )
        else:
            self.sar_adapter = SARBlockAdapters(
                channels=channels,
                blocks=blocks,
                enabled_layers=self.adapter_layers,
                bottleneck_ratio=adapter_bottleneck,
            )

        if self.visual_lora:
            apply_visual_lora(
                self.visual,
                layers=self.visual_lora,
                r=self.lora_r,
                alpha=self.lora_alpha,
                dropout=self.lora_dropout,
            )

        # Apply LoRA to text transformer
        if self.text_lora_enabled:
            apply_text_lora(
                self.clip_model.transformer,
                r=self.text_lora_r,
                alpha=self.lora_alpha,
                dropout=self.lora_dropout,
                num_layers=12,  # RN50 has 12 transformer layers
            )

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def _forward_stem(self, x: torch.Tensor) -> torch.Tensor:
        for conv, bn in [(self.visual.conv1, self.visual.bn1),
                         (self.visual.conv2, self.visual.bn2),
                         (self.visual.conv3, self.visual.bn3)]:
            x = self.visual.relu(bn(conv(x)))
        x = self.visual.avgpool(x)
        return x

    def _forward_layer(self, layer, x: torch.Tensor, layer_name: str, use_adapter: bool) -> torch.Tensor:
        for idx, block in enumerate(layer):
            x = block(x)
            if use_adapter:
                x = self.sar_adapter.apply(layer_name, idx, x)
        return x

    def _forward_features(self, images: torch.Tensor, use_adapter: bool) -> Tuple[List[torch.Tensor], torch.Tensor]:
        x = images.type(self.dtype)
        x = self._forward_stem(x)
        x1 = self._forward_layer(self.visual.layer1, x, "layer1", use_adapter)
        x2 = self._forward_layer(self.visual.layer2, x1, "layer2", use_adapter)
        x3 = self._forward_layer(self.visual.layer3, x2, "layer3", use_adapter)
        x4 = self._forward_layer(self.visual.layer4, x3, "layer4", use_adapter)
        x5 = self.visual.attnpool(x4)
        return [x1, x2, x3, x4], x5

    def forward_optical(self, images: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        feats, emb = self._forward_features(images, use_adapter=False)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb, feats

    def forward_sar(self, images: torch.Tensor, disable_adapter: bool = False) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        feats, emb = self._forward_features(images, use_adapter=not disable_adapter)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb, feats

    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        # If text LoRA is enabled or text encoder is unfrozen, don't use no_grad so gradients can flow
        if self.text_lora_enabled or self.unfreeze_text_encoder:
            text_feats = self.clip_model.encode_text(tokens)
        else:
            with torch.no_grad():
                text_feats = self.clip_model.encode_text(tokens)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        return text_feats

