from __future__ import annotations

from typing import Iterable, List, Tuple

import torch
from torch import nn

import clip_ldc as clip
from .sar_adapter_vit import SARViTAdapters
from .clip_rn50_wrapper import CLIPRN50Wrapper


class _DummyConfig:
    def __init__(self) -> None:
        self.fuse_type = 2


class CLIPBackboneWrapper(nn.Module):
    def __init__(
        self,
        backbone: str,
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
        self.backbone = backbone
        self.adapter_layers = list(adapter_layers)
        self.adapter_bottleneck = adapter_bottleneck
        self.adapter_bottleneck_per_layer = adapter_bottleneck_per_layer
        self.cache_dir = cache_dir
        self.visual_lora = list(visual_lora) if visual_lora is not None else []
        self.lora_r = int(lora_r)
        self.lora_alpha = int(lora_alpha)
        self.lora_dropout = float(lora_dropout)
        self.text_lora_r = int(text_lora_r)
        self.unfreeze_text_encoder = bool(unfreeze_text_encoder)

        if self.backbone == "rn50":
            self.rn50 = CLIPRN50Wrapper(
                adapter_layers=self.adapter_layers,
                adapter_bottleneck=self.adapter_bottleneck,
                cache_dir=self.cache_dir,
                visual_lora=self.visual_lora,
                lora_r=self.lora_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                text_lora_r=self.text_lora_r,
                unfreeze_text_encoder=self.unfreeze_text_encoder,
                adapter_bottleneck_per_layer=self.adapter_bottleneck_per_layer,
            )
            self.clip_model = self.rn50.clip_model
            self.sar_adapter = self.rn50.sar_adapter
            return

        if self.backbone != "vit_b32":
            raise ValueError(f"Unknown backbone: {self.backbone}")

        self.clip_model, _ = clip.load("ViT-B/32", download_root=self.cache_dir, config=_DummyConfig())
        self.clip_model.float()
        self.visual = self.clip_model.visual

        self.embed_dim = self.visual.proj.shape[0]
        self.patch_size = self.visual.conv1.kernel_size[0]
        self.grid = self.visual.input_resolution // self.patch_size

        num_blocks = len(self.visual.transformer.resblocks)
        if not self.adapter_layers:
            self.adapter_layers = [f"block{i}" for i in range(num_blocks - 4, num_blocks)]
        enabled_blocks = self._parse_vit_blocks(self.adapter_layers)
        self.sar_adapter = SARViTAdapters(
            dim=self.visual.ln_pre.normalized_shape[0],
            num_blocks=num_blocks,
            enabled_blocks=enabled_blocks,
            bottleneck_ratio=self.adapter_bottleneck,
        )

    @staticmethod
    def _parse_vit_blocks(layers: Iterable[str]) -> List[int]:
        blocks: List[int] = []
        for layer in layers:
            if not layer.startswith("block"):
                raise ValueError("ViT adapter layers must be like 'block8,block9,block10,block11'.")
            blocks.append(int(layer.replace("block", "")))
        return blocks

    @property
    def dtype(self):
        if self.backbone == "rn50":
            return self.rn50.dtype
        return self.visual.conv1.weight.dtype

    def _vit_patch_embed(self, images: torch.Tensor) -> torch.Tensor:
        x = self.visual.conv1(images)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        cls_token = self.visual.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
        )
        x = torch.cat([cls_token, x], dim=1)  # [*, grid ** 2 + 1, width]
        x = x + self.visual.positional_embedding.to(x.dtype)
        x = self.visual.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        return x

    def _vit_tokens_to_map(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = x[:, 1:, :]  # drop class token
        x = x.reshape(x.shape[0], self.grid, self.grid, x.shape[-1])
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

    def _forward_vit_features(self, images: torch.Tensor, use_adapter: bool) -> Tuple[List[torch.Tensor], torch.Tensor]:
        x = images.type(self.dtype)
        x = self._vit_patch_embed(x)

        features: List[torch.Tensor] = []
        block_indices = list(range(len(self.visual.transformer.resblocks)))
        last_blocks = set(block_indices[-4:])

        for idx, block in enumerate(self.visual.transformer.resblocks):
            x = block(x)
            if use_adapter:
                x = self.sar_adapter.apply(idx, x)
            if idx in last_blocks:
                features.append(self._vit_tokens_to_map(x))

        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.visual.ln_post(x[:, 0, :])
        emb = x @ self.visual.proj
        return features, emb

    def forward_optical(self, images: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        if self.backbone == "rn50":
            return self.rn50.forward_optical(images)
        feats, emb = self._forward_vit_features(images, use_adapter=False)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb, feats

    def forward_sar(self, images: torch.Tensor, disable_adapter: bool = False) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        if self.backbone == "rn50":
            return self.rn50.forward_sar(images, disable_adapter=disable_adapter)
        feats, emb = self._forward_vit_features(images, use_adapter=not disable_adapter)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb, feats

    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        # Enable gradients if text LoRA is enabled OR if text encoder is unfrozen
        if self.text_lora_r > 0 or self.unfreeze_text_encoder:
            # Text LoRA is enabled or text encoder unfrozen - need gradients for training
            text_feats = self.clip_model.encode_text(tokens)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        else:
            with torch.no_grad():
                text_feats = self.clip_model.encode_text(tokens)
                text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        return text_feats
