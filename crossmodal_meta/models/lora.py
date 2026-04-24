from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class LoRAConv2d(nn.Module):
    def __init__(self, conv: nn.Conv2d, r: int, alpha: int, dropout: float = 0.0) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank r must be > 0")
        if conv.groups != 1:
            raise ValueError("LoRAConv2d only supports groups=1")
        self.conv = conv
        for p in self.conv.parameters():
            p.requires_grad = False

        k_h, k_w = self.conv.kernel_size
        in_dim = self.conv.in_channels * k_h * k_w
        self.lora_A = nn.Parameter(torch.empty(r, in_dim))
        self.lora_B = nn.Parameter(torch.zeros(self.conv.out_channels, r))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B)

        self.r = int(r)
        self.alpha = int(alpha)
        self.scale = float(alpha) / float(r)  # Remove 0.5 factor
        self.dropout = nn.Dropout(p=float(dropout)) if dropout > 0 else nn.Identity()
        self.enable_lora = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.conv(x)
        if not self.enable_lora:
            return base
        x_lora = self.dropout(x)
        if x_lora.dtype != self.lora_A.dtype:
            x_lora = x_lora.to(self.lora_A.dtype)
        k_h, k_w = self.conv.kernel_size
        weight = torch.matmul(self.lora_B, self.lora_A)
        weight = weight.view(self.conv.out_channels, self.conv.in_channels, k_h, k_w)
        update = F.conv2d(
            x_lora,
            weight,
            bias=None,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=1,
        )
        if base.dtype != update.dtype:
            update = update.to(base.dtype)
        return base + self.scale * update


class LoRALinear(nn.Module):
    """LoRA for linear (dense) layers - used for text transformer.
    
    This class wraps a linear layer and adds LoRA as a parallel path.
    It properly exposes the weight attribute for PyTorch compatibility.
    """
    
    def __init__(self, linear: nn.Linear, r: int, alpha: int, dropout: float = 0.0) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank r must be > 0")
        self.linear = linear
        # Freeze original linear weights
        for p in self.linear.parameters():
            p.requires_grad = False
        
        self.r = int(r)
        self.alpha = int(alpha)
        self.scale = float(alpha) / float(r)
        self.dropout = nn.Dropout(p=float(dropout)) if dropout > 0 else nn.Identity()
        
        # LoRA matrices - IMPORTANT: A uses in_features, B uses out_features
        # This matches the standard LoRA paper: output = W + (B @ A) * scale
        self.lora_A = nn.Parameter(torch.empty(r, linear.in_features))  # (r, in_dim)
        self.lora_B = nn.Parameter(torch.zeros(linear.out_features, r))  # (out_dim, r)
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B)
        
        self.enable_lora = True
    
    @property
    def weight(self) -> nn.Parameter:
        """Expose the original linear's weight for PyTorch compatibility."""
        return self.linear.weight
    
    @property
    def bias(self) -> nn.Parameter | None:
        """Expose the original linear's bias for PyTorch compatibility."""
        return self.linear.bias
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.linear(x)
        if not self.enable_lora:
            return base
        # Apply LoRA: output = base + (x @ B @ A) * scale
        # F.linear expects weight of shape (out_features, in_features)
        # So we need: lora_B @ lora_A = (out_dim, r) @ (r, in_dim) = (out_dim, in_dim)
        x_lora = self.dropout(x)
        if x_lora.dtype != self.lora_A.dtype:
            x_lora = x_lora.to(self.lora_A.dtype)
        lora_weight = torch.matmul(self.lora_B, self.lora_A)  # (out_dim, in_dim)
        lora_out = F.linear(x_lora, lora_weight, None) * self.scale
        if base.dtype != lora_out.dtype:
            lora_out = lora_out.to(base.dtype)
        return base + lora_out


def _find_visual_layer(visual: nn.Module, layer_name: str) -> nn.Module | None:
    if hasattr(visual, layer_name):
        return getattr(visual, layer_name)
    for name, module in visual.named_modules():
        if name.endswith(layer_name) and isinstance(module, (nn.Sequential, nn.ModuleList)):
            return module
    return None


def apply_visual_lora(
    visual: nn.Module,
    layers: Iterable[str],
    r: int,
    alpha: int,
    dropout: float,
) -> List[str]:
    applied: List[str] = []
    for layer_name in layers:
        layer = _find_visual_layer(visual, layer_name)
        if layer is None:
            continue
        for block_idx, block in enumerate(layer):
            for conv_name in ("conv1", "conv2", "conv3"):
                if not hasattr(block, conv_name):
                    continue
                conv = getattr(block, conv_name)
                if isinstance(conv, LoRAConv2d):
                    continue
                if not isinstance(conv, nn.Conv2d):
                    continue
                setattr(block, conv_name, LoRAConv2d(conv, r=r, alpha=alpha, dropout=dropout))
                applied.append(f"{layer_name}.{block_idx}.{conv_name}")
    return applied


def apply_text_lora(
    transformer: nn.Module,
    r: int,
    alpha: int,
    dropout: float,
    num_layers: int = 12,
) -> List[str]:
    """Apply LoRA to text transformer layers.
    
    Args:
        transformer: CLIP transformer module
        r: LoRA rank
        alpha: LoRA alpha scaling
        dropout: Dropout probability
        num_layers: Number of transformer layers
    
    Returns:
        List of applied layer names
    """
    applied: List[str] = []
    
    # Find the resblocks (transformer layers)
    if hasattr(transformer, 'resblocks'):
        resblocks = transformer.resblocks
        for idx, block in enumerate(resblocks):
            if idx >= num_layers:
                break
            # Apply LoRA to attn.out_proj (output projection)
            if hasattr(block, 'attn') and hasattr(block.attn, 'out_proj'):
                out_proj = block.attn.out_proj
                if not isinstance(out_proj, LoRALinear):
                    block.attn.out_proj = LoRALinear(out_proj, r=r, alpha=alpha, dropout=dropout)
                    applied.append(f"resblocks.{idx}.attn.out_proj")
            # Apply LoRA to mlp.c_fc and mlp.c_proj
            if hasattr(block, 'mlp'):
                mlp = block.mlp
                if hasattr(mlp, 'c_fc') and not isinstance(mlp.c_fc, LoRALinear):
                    mlp.c_fc = LoRALinear(mlp.c_fc, r=r, alpha=alpha, dropout=dropout)
                    applied.append(f"resblocks.{idx}.mlp.c_fc")
                if hasattr(mlp, 'c_proj') and not isinstance(mlp.c_proj, LoRALinear):
                    mlp.c_proj = LoRALinear(mlp.c_proj, r=r, alpha=alpha, dropout=dropout)
                    applied.append(f"resblocks.{idx}.mlp.c_proj")
    
    return applied


def iter_lora_parameters(module: nn.Module) -> Iterable[Tuple[str, nn.Parameter]]:
    for name, mod in module.named_modules():
        if isinstance(mod, (LoRAConv2d, LoRALinear)):
            yield f"{name}.lora_A", mod.lora_A
            yield f"{name}.lora_B", mod.lora_B


def iter_lora_modules(module: nn.Module) -> Iterable[LoRAConv2d]:
    for mod in module.modules():
        if isinstance(mod, (LoRAConv2d, LoRALinear)):
            yield mod


def set_lora_enabled(module: nn.Module, enabled: bool) -> None:
    for mod in iter_lora_modules(module):
        mod.enable_lora = bool(enabled)


def lora_state_dict(module: nn.Module) -> Dict[str, torch.Tensor]:
    return {name: param.detach().cpu() for name, param in iter_lora_parameters(module)}


class LoRAParamStore(nn.Module):
    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        for name, param in iter_lora_parameters(source):
            self.register_parameter(name.replace(".", "_"), param)

    def named_lora_parameters(self) -> Iterable[Tuple[str, nn.Parameter]]:
        for name, param in self.named_parameters():
            yield name, param

def load_lora_state_dict(module: nn.Module, state: Dict[str, torch.Tensor], strict: bool = True) -> None:
    missing: List[str] = []
    for name, param in iter_lora_parameters(module):
        if name not in state:
            missing.append(name)
            continue
        param.data.copy_(state[name].to(param.device, dtype=param.dtype))
    if strict and missing:
        raise RuntimeError(f"Missing LoRA keys: {missing[:10]}")
