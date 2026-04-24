"""Multi-Level Domain Discriminator for Cross-Domain Meta-Learning.

Extracts domain predictions from multiple ResNet layers (Layer 2, 3, 4)
to enforce adversarial invariance at multiple semantic levels.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class GradientReversalLayer(torch.autograd.Function):
    """Gradient Reversal Layer for domain adversarial training."""
    
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class DomainDiscriminatorHead(nn.Module):
    """Single domain discriminator head for a specific feature level."""
    
    def __init__(self, feat_dim: int, num_domains: int, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim // 2, num_domains)
        )
    
    def forward(self, x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        """Forward pass with gradient reversal.
        
        Args:
            x: Input features [B, feat_dim]
            alpha: Gradient reversal coefficient (higher = stronger reversal)
        
        Returns:
            Domain logits [B, num_domains]
        """
        x = GradientReversalLayer.apply(x, alpha)
        return self.net(x)


class MultiLevelDiscriminator(nn.Module):
    """Multi-level domain discriminator operating on ResNet layers 2, 3, and 4.
    
    Each layer has its own discriminator head:
    - Layer 2: 512-dim features (mid-level, texture/shape)
    - Layer 3: 1024-dim features (high-level, object parts)
    - Layer 4: 2048-dim features (semantic, object-level)
    
    This enforces domain invariance at multiple levels of abstraction.
    """
    
    def __init__(
        self,
        layer2_dim: int = 512,
        layer3_dim: int = 1024,
        layer4_dim: int = 2048,
        num_domains: int = 32,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.num_domains = num_domains
        
        # Three separate heads for each layer
        self.head_layer2 = DomainDiscriminatorHead(layer2_dim, num_domains, hidden_dim)
        self.head_layer3 = DomainDiscriminatorHead(layer3_dim, num_domains, hidden_dim)
        self.head_layer4 = DomainDiscriminatorHead(layer4_dim, num_domains, hidden_dim)
        
        # Global average pooling for spatial features
        self.gap = nn.AdaptiveAvgPool2d(1)
    
    def _extract_features(self, feats: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract pooled features from layer 2, 3, 4.
        
        Args:
            feats: List of feature maps from ResNet [x1, x2, x3, x4]
                   x1: [B, 256, H, W] - layer1 (not used)
                   x2: [B, 512, H, W] - layer2
                   x3: [B, 1024, H, W] - layer3
                   x4: [B, 2048, H, W] - layer4
        
        Returns:
            Tuple of (f2, f3, f4) where each is [B, dim] after GAP
        """
        # feats[0] = x1 (layer1, 256-dim) - not used
        # feats[1] = x2 (layer2, 512-dim)
        # feats[2] = x3 (layer3, 1024-dim)
        # feats[3] = x4 (layer4, 2048-dim)
        
        f2 = self.gap(feats[1]).flatten(1)  # [B, 512]
        f3 = self.gap(feats[2]).flatten(1)  # [B, 1024]
        f4 = self.gap(feats[3]).flatten(1)  # [B, 2048]
        
        return f2, f3, f4
    
    def forward(
        self,
        feats: list[torch.Tensor],
        alpha: float = 1.0
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through all three discriminator heads.
        
        Args:
            feats: List of feature maps [x1, x2, x3, x4] from ResNet
            alpha: Gradient reversal coefficient
        
        Returns:
            Tuple of (logits_l2, logits_l3, logits_l4) each [B, num_domains]
        """
        f2, f3, f4 = self._extract_features(feats)
        
        logits_l2 = self.head_layer2(f2, alpha)
        logits_l3 = self.head_layer3(f3, alpha)
        logits_l4 = self.head_layer4(f4, alpha)
        
        return logits_l2, logits_l3, logits_l4
    
    def compute_loss(
        self,
        feats: list[torch.Tensor],
        domain_labels: torch.Tensor,
        alpha: float = 1.0
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute domain classification loss for all three levels.
        
        Args:
            feats: List of feature maps [x1, x2, x3, x4]
            domain_labels: Domain labels [B] with values in [0, num_domains-1]
            alpha: Gradient reversal coefficient
        
        Returns:
            Tuple of (total_loss, loss_dict) where loss_dict contains
            individual losses for debugging
        """
        logits_l2, logits_l3, logits_l4 = self.forward(feats, alpha)
        
        loss_l2 = F.cross_entropy(logits_l2, domain_labels)
        loss_l3 = F.cross_entropy(logits_l3, domain_labels)
        loss_l4 = F.cross_entropy(logits_l4, domain_labels)
        
        # Average loss across all three levels
        total_loss = (loss_l2 + loss_l3 + loss_l4) / 3.0
        
        loss_dict = {
            'domain_loss_total': total_loss.item(),
            'domain_loss_l2': loss_l2.item(),
            'domain_loss_l3': loss_l3.item(),
            'domain_loss_l4': loss_l4.item(),
        }
        
        return total_loss, loss_dict
