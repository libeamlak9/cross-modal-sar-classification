"""Stage 2: Cross-Modal Meta-Learning with Multi-Level Domain Adversarial Training.

Source of Truth for So2Sat LCZ42 cross-modal few-shot learning.
Features:
- Multi-Level Domain Discriminator (Layer 2, 3, 4)
- Multimodal Prototype Fusion (3 Optical + 2 SAR support)
- Zero-Loss Guardrail for GRL health monitoring
- Evaluation on Eastern halves (testing.h5)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

def _ensure_valid_omp_threads(default: int = 1) -> None:
    omp_value = os.environ.get("OMP_NUM_THREADS")
    if omp_value is None:
        return
    try:
        if int(omp_value) <= 0:
            raise ValueError
    except ValueError:
        os.environ["OMP_NUM_THREADS"] = str(default)

_ensure_valid_omp_threads()

import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from crossmodal_meta.data.crossdomain_multimodal_sampler_v2 import CrossDomainMultimodalSampler
from crossmodal_meta.models.clip_backbone_wrapper import CLIPBackboneWrapper
from crossmodal_meta.models.ldc_head import LDCHead
from crossmodal_meta.models.lora import iter_lora_parameters
from crossmodal_meta.models.text_prompts import DEFAULT_TEMPLATES_SAR, get_classnames
from crossmodal_meta.models.multilevel_discriminator import MultiLevelDiscriminator
from crossmodal_meta.utils.logging import SimpleLogger
from crossmodal_meta.utils.metrics import episodic_accuracy, mean_confidence_interval
from crossmodal_meta.utils.seed import set_seed
import clip_ldc as clip


# =============================================================================
# Zero-Loss Guardrail
# =============================================================================

class ZeroLossGuardrail:
    """Monitors domain losses to detect broken GRL backprop paths."""
    
    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance
        self.violation_count = 0
    
    def check(self, loss_dict: dict[str, float], epoch: int, episode: int) -> bool:
        """Check if any domain loss is effectively zero.
        
        Returns:
            True if healthy (no zero losses), False if broken.
        """
        zero_losses = []
        for key, value in loss_dict.items():
            if 'domain_loss' in key and abs(value) < self.tolerance:
                zero_losses.append((key, value))
        
        if zero_losses:
            self.violation_count += 1
            print(f"\n🚨 ZERO-LOSS GUARDRAIL TRIGGERED! Epoch {epoch}, Episode {episode}")
            for key, value in zero_losses:
                print(f"   {key} = {value:.6f} (BELOW TOLERANCE {self.tolerance})")
            print(f"   This indicates broken GRL backprop path!")
            return False
        return True


class ModalityAttentionWeights(nn.Module):
    """Learnable per-class attention weights for modality fusion.
    
    For each class, learns weights for optical vs SAR modality contributions.
    """
    
    def __init__(self, n_way: int, init_value: float = 1.0):
        super().__init__()
        # Initialize with equal weights (logits, will be softmaxed)
        self.logits = nn.Parameter(
            torch.full((n_way, 2), init_value)
        )
    
    def forward(self) -> torch.Tensor:
        """Returns normalized weights [n_way, 2] via softmax."""
        return F.softmax(self.logits, dim=-1)


# =============================================================================
# Utility Functions
# =============================================================================

def build_text_features(
    model: CLIPBackboneWrapper,
    dataset_root: Path,
    class_ids: list,
    device: torch.device,
) -> torch.Tensor:
    """Build normalized text features for episode classes."""
    classnames = get_classnames(dataset_root, class_ids)
    prompts = [t.format(cn) for cn in classnames for t in DEFAULT_TEMPLATES_SAR]
    tokens = clip.tokenize(prompts).to(device)
    text_feats = model.encode_text(tokens)
    text_feats = text_feats.view(len(classnames), len(DEFAULT_TEMPLATES_SAR), -1).mean(dim=1)
    return F.normalize(text_feats, dim=-1)


def compute_multimodal_prototypes(
    support_emb: torch.Tensor,
    support_labels: torch.Tensor,
    n_way: int,
) -> torch.Tensor:
    """Compute prototypes from multimodal support (3 Optical + 2 SAR = 5 total).
    
    Args:
        support_emb: Support embeddings [n_way * k_shot, feat_dim]
        support_labels: Support labels [n_way * k_shot]
        n_way: Number of classes
    
    Returns:
        Prototypes [n_way, feat_dim] - mean of all 5 support embeddings per class
    """
    prototypes = []
    for cls in range(n_way):
        cls_emb = support_emb[support_labels == cls]
        # Mean of all support samples for this class (multimodal fusion)
        proto = F.normalize(cls_emb.mean(dim=0), dim=-1)
        prototypes.append(proto)
    return torch.stack(prototypes, dim=0)


def compute_multimodal_prototypes_correct(
    model: CLIPBackboneWrapper,
    support: torch.Tensor,
    support_labels: torch.Tensor,
    support_modalities: torch.Tensor,
    n_way: int,
    device: torch.device,
    modality_weights: torch.Tensor = None,
) -> torch.Tensor:
    """Compute prototypes with CORRECT modality-specific encoding and optional attention weighting.
    
    This fixes the bug where SAR support samples were incorrectly passed 
    through the optical encoder.
    
    Args:
        model: CLIP backbone wrapper
        support: Support images [n_way * k_shot, C, H, W]
        support_labels: Support labels [n_way * k_shot]
        support_modalities: Modality indicators [n_way * k_shot] (0=optical, 1=sar)
        n_way: Number of classes
        device: torch device
        modality_weights: Optional learnable weights [n_way, 2] for (optical, SAR) per class
    
    Returns:
        Prototypes [n_way, feat_dim] - properly fused from both modalities
    """
    # Create masks for each modality
    optical_mask = support_modalities == 0  # Optical samples
    sar_mask = support_modalities == 1      # SAR samples
    
    # Encode optical samples with optical encoder (frozen)
    optical_support = support[optical_mask]
    with torch.no_grad():
        optical_emb, _ = model.forward_optical(optical_support)
        optical_emb = F.normalize(optical_emb, dim=-1)
    
    # Encode SAR samples with SAR encoder + adapter
    sar_support = support[sar_mask]
    with torch.no_grad():
        sar_emb, _ = model.forward_sar(sar_support, disable_adapter=False)
        sar_emb = F.normalize(sar_emb, dim=-1)
    
    # Get labels for each modality
    optical_labels = support_labels[optical_mask]
    sar_labels = support_labels[sar_mask]
    
    # Compute prototypes by fusing both modalities per class
    prototypes = []
    for cls in range(n_way):
        opt_cls = optical_emb[optical_labels == cls]
        sar_cls = sar_emb[sar_labels == cls]
        
        # Fuse: concatenate and mean
        if len(opt_cls) > 0 and len(sar_cls) > 0:
            if modality_weights is not None:
                # Attention-weighted fusion
                w_opt, w_sar = modality_weights[cls]
                combined = torch.cat([w_opt * opt_cls, w_sar * sar_cls], dim=0)
            else:
                combined = torch.cat([opt_cls, sar_cls], dim=0)
            proto = F.normalize(combined.mean(dim=0), dim=-1)
        elif len(opt_cls) > 0:
            proto = F.normalize(opt_cls.mean(dim=0), dim=-1)
        else:
            proto = F.normalize(sar_cls.mean(dim=0), dim=-1)
        
        prototypes.append(proto)
    
    return torch.stack(prototypes, dim=0)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Stage 2: Multi-Level Domain Adversarial Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data arguments
    parser.add_argument("--data_root", type=Path, required=True,
                        help="Path to So2Sat_LCZ42_v4 HDF5 directory")
    parser.add_argument("--split", type=str, default="train",
                        help="Training split name")
    parser.add_argument("--sen2_rgb_idx", type=str, default="3,2,1",
                        help="Sentinel-2 RGB band indices")
    parser.add_argument("--sen1_map", type=str, default="vv_vh_diff",
                        choices=["repeat", "vv_vh_diff"],
                        help="SAR channel mapping strategy")

    # Episode configuration
    parser.add_argument("--n_way", type=int, default=5,
                        help="Number of classes per episode")
    parser.add_argument("--backbone", type=str, default="rn50", choices=["rn50", "vit_b32"],
                        help="CLIP backbone: rn50 or vit_b32")
    parser.add_argument("--k_shot", type=int, default=5,
                        help="Number of support samples per class (3 optical + 2 SAR)")
    parser.add_argument("--q_query", type=int, default=15,
                        help="Number of query samples per class (SAR only)")
    parser.add_argument("--support_optical_count", type=int, default=3,
                        help="Number of optical samples in support set")
    parser.add_argument("--episodes_per_epoch", type=int, default=500,
                        help="Number of episodes per training epoch")
    parser.add_argument("--val_episodes", type=int, default=200,
                        help="Number of episodes for validation")

    # Model architecture
    parser.add_argument("--adapter_bottleneck", type=int, default=64,
                        help="Bottleneck dimension for SAR adapter")
    parser.add_argument("--adapter_layers", type=str, default="layer4",
                        help="RN50 layers to apply adapters (comma-separated)")
    # Ablation study flags
    parser.add_argument("--no_bottleneck_adapter", action="store_true",
                        help="Disable bottleneck adapters (for ablation)")
    parser.add_argument("--no_visual_lora", action="store_true",
                        help="Disable visual LoRA (for ablation)")
    parser.add_argument("--no_text_guidance", action="store_true",
                        help="Disable text-guided classification (for ablation)")
    parser.add_argument("--no_domain_adv", action="store_true",
                        help="Disable multi-level domain adversarial training (for ablation)")
    parser.add_argument("--no_ldc_head", action="store_true",
                        help="Disable LDC head, use base logits only (for ablation)")
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha scaling")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                        help="LoRA dropout rate")
    parser.add_argument("--text_lora_r", type=int, default=0,
                        help="Text encoder LoRA rank (0=disabled)")
    parser.add_argument("--unfreeze_text_encoder", action="store_true",
                        help="Unfreeze text encoder for fine-tuning (incompatible with text_lora_r)")
    parser.add_argument("--visual_lora", type=str, default="layer3,layer4",
                        help="Visual encoder layers for LoRA (comma-separated, or 'none')")
    parser.add_argument("--use_maf_head", action="store_true",
                        help="Enable MAF head for direct multi-level feature fusion classification")
    parser.add_argument("--enable_modality_weights", action="store_true",
                        help="Enable learnable per-class modality attention weights for prototype fusion")
    parser.add_argument("--inner_steps", type=int, default=0,
                        help="Number of inner loop steps for query adaptation (0=disabled, 1-5=MAML-style)")
    parser.add_argument("--inner_lr", type=float, default=0.1,
                        help="Learning rate for inner loop query adaptation")

    # Multi-level domain discriminator
    parser.add_argument("--num_domains", type=int, default=32,
                        help="Number of domains for discriminator (32 valid training cities)")
    parser.add_argument("--domain_adv_weight", type=float, default=0.1,
                        help="Weight for domain adversarial loss")
    parser.add_argument("--enable_adaptive_weights", action="store_true",
                        help="Enable adaptive loss weighting based on train-val gap")
    parser.add_argument("--overfit_threshold", type=float, default=0.10,
                        help="Train-val gap threshold for overfitting detection (0.10 = 10%)")
    parser.add_argument("--weight_adjustment", type=float, default=0.20,
                        help="Percentage to adjust weights when overfitting detected (0.20 = 20%)")
    parser.add_argument("--disc_hidden_dim", type=int, default=512,
                        help="Hidden dimension for discriminator heads")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=15,
                        help="Number of training epochs (target: >40% by epoch 15)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay")
    parser.add_argument("--text_weight", type=float, default=0.5,
                        help="Weight for text-based logits")

    # Checkpointing
    parser.add_argument("--checkpoint_dir", type=Path, default=Path("./checkpoints_stage2"),
                        help="Directory for saving checkpoints")
    parser.add_argument("--resume", type=Path, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--val_interval", type=int, default=1,
                        help="Validation interval (epochs)")
    parser.add_argument("--early_stop_patience", type=int, default=0,
                        help="Early stopping patience (0 to disable)")

    # Misc
    parser.add_argument("--seed", type=int, default=2024,
                        help="Random seed")

    return parser.parse_args()


def evaluate(
    model: CLIPBackboneWrapper,
    ldchead: LDCHead,
    text_weight: float,
    sampler: CrossDomainMultimodalSampler,
    device: torch.device,
    n_way: int,
    data_root: Path,
    num_episodes: int,
) -> tuple[float, float]:
    """Evaluate model on episodic tasks (Eastern halves).
    
    NOTE: Keep backbone in train mode for batch statistics (same as eval_stage2.py).
    """
    # Keep backbone in train mode for batch statistics
    model.train()
    if hasattr(model, 'sar_adapter'):
        model.sar_adapter.eval()
    ldchead.eval()

    acc_list = []

    with torch.no_grad():
        for episode in tqdm(sampler, total=num_episodes, desc="Eval", leave=False):
            support = episode.support_images.to(device)
            query = episode.query_images.to(device)
            y_query = episode.query_labels.to(device)

            # Compute multimodal support prototypes (FIXED: use correct encoders per modality)
            prototypes = compute_multimodal_prototypes_correct(
                model=model,
                support=support,
                support_labels=episode.support_labels,
                support_modalities=episode.support_modalities,
                n_way=n_way,
                device=device,
            )

            # Query forward pass (SAR with adapter)
            query_emb, feats = model.forward_sar(query, disable_adapter=False)

            # Compute logits
            proto_logits = query_emb @ prototypes.t()
            proto_logits = proto_logits * model.clip_model.logit_scale.exp()

            text_feats = build_text_features(model, data_root, episode.class_ids, device)
            text_logits = query_emb @ text_feats.t()
            text_logits = text_logits * model.clip_model.logit_scale.exp()

            base_logits = proto_logits + text_weight * text_logits

            # LDC head refinement
            _, _, s_alf, _ = ldchead(feats, base_logits, disable_icd=False)

            acc_list.append(episodic_accuracy(s_alf, y_query))

    return mean_confidence_interval(acc_list)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = SimpleLogger()

    logger.write("=" * 70)
    logger.write("Stage 2: Multi-Level Domain Adversarial Training")
    logger.write("=" * 70)
    logger.write(f"Device: {device}")
    logger.write(f"Data root: {args.data_root}")
    logger.write(f"Episodes/epoch: {args.episodes_per_epoch}")
    logger.write(f"Domains: {args.num_domains}")
    logger.write(f"Domain adv weight: {args.domain_adv_weight}")
    # Ablation study settings
    logger.write(f"Ablation - Bottleneck Adapter: {not args.no_bottleneck_adapter}")
    logger.write(f"Ablation - Visual LoRA: {not args.no_visual_lora}")
    logger.write(f"Ablation - Text Guidance: {not args.no_text_guidance}")
    logger.write(f"Ablation - Domain Adv: {not args.no_domain_adv}")
    logger.write(f"Ablation - LDC Head: {not args.no_ldc_head}")
    if args.no_text_guidance:
        logger.write("Text guidance disabled (ablation) - using prototype-only logits")

    # Parse layer configurations
    adapter_layers = [s.strip() for s in args.adapter_layers.split(",") if s.strip()]
    visual_lora = [] if args.visual_lora == "none" else [s.strip() for s in args.visual_lora.split(",") if s.strip()]

    # Build samplers using Cross-Domain Multimodal Sampler
    logger.write(f"Support: {args.support_optical_count} Optical + {args.k_shot - args.support_optical_count} SAR")
    logger.write(f"Query: SAR only (strict)")
    
    sampler = CrossDomainMultimodalSampler(
        data_root=args.data_root,
        split=args.split,
        n_way=args.n_way,
        k_shot=args.k_shot,
        q_query=args.q_query,
        support_optical_count=args.support_optical_count,
        episodes_per_epoch=args.episodes_per_epoch,
        seed=args.seed,
    )

    # Validation on Western halves (validation.h5)
    logger.write("Validation: Western halves (validation.h5)")
    val_sampler = CrossDomainMultimodalSampler(
        data_root=args.data_root,
        split="valid",  # Western halves
        n_way=args.n_way,
        k_shot=args.k_shot,
        q_query=args.q_query,
        support_optical_count=args.support_optical_count,
        episodes_per_epoch=args.val_episodes,
        seed=args.seed,
    )

    # Build model
    # For ablation: conditionally enable adapter and LoRA
    enable_bottleneck_adapter = not args.no_bottleneck_adapter
    enable_visual_lora = not args.no_visual_lora
    
    if enable_bottleneck_adapter:
        model = CLIPBackboneWrapper(
            backbone=args.backbone,
            adapter_layers=adapter_layers,
            adapter_bottleneck=args.adapter_bottleneck,
            cache_dir=str(Path("./model/clip")),
            visual_lora=visual_lora if enable_visual_lora else [],
            lora_r=args.lora_r if enable_visual_lora else 0,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            text_lora_r=args.text_lora_r,
            unfreeze_text_encoder=args.unfreeze_text_encoder,
        )
    else:
        # No adapter - use raw CLIP features
        model = CLIPBackboneWrapper(
            backbone=args.backbone,
            adapter_layers=[],  # No adapter
            adapter_bottleneck=0,
            cache_dir=str(Path("./model/clip")),
            visual_lora=visual_lora if enable_visual_lora else [],
            lora_r=args.lora_r if enable_visual_lora else 0,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            text_lora_r=args.text_lora_r,
            unfreeze_text_encoder=args.unfreeze_text_encoder,
        )
    model.to(device)

    # Set feature dimensions based on backbone
    if args.backbone == "vit_b32":
        ldc_in_channels = [768, 768, 768, 768]
        domain_l2_dim = 768
        domain_l3_dim = 768
        domain_l4_dim = 768
    else:
        ldc_in_channels = [256, 512, 1024, 2048]
        domain_l2_dim = 512
        domain_l3_dim = 1024
        domain_l4_dim = 2048

    ldchead = LDCHead(
        in_channels=ldc_in_channels,
        feat_dim=1024,
        num_classes=args.n_way,
        use_maf_head=args.use_maf_head
    )
    ldchead.to(device)

    # Multi-level domain discriminator (Layer 2, 3, 4)
    domain_disc = MultiLevelDiscriminator(
        layer2_dim=domain_l2_dim,
        layer3_dim=domain_l3_dim,
        layer4_dim=domain_l4_dim,
        num_domains=args.num_domains,
        hidden_dim=args.disc_hidden_dim,
    )
    domain_disc.to(device)
    logger.write(f"MultiLevelDiscriminator: 3 heads x {args.num_domains} domains")

    # Freeze CLIP backbone, train adapters (respecting ablation flags)
    enable_bottleneck_adapter = not args.no_bottleneck_adapter
    enable_visual_lora = not args.no_visual_lora
    enable_domain_adv = not args.no_domain_adv
    use_ldc_head = not args.no_ldc_head
    
    for p in model.clip_model.parameters():
        p.requires_grad = False
    
    # Bottleneck adapter: only train if enabled
    if enable_bottleneck_adapter and hasattr(model, 'sar_adapter'):
        for p in model.sar_adapter.parameters():
            p.requires_grad = True
        logger.write(f"Training bottleneck adapter: {args.adapter_layers}")
    else:
        if hasattr(model, 'sar_adapter'):
            for p in model.sar_adapter.parameters():
                p.requires_grad = False
        logger.write("Bottleneck adapter disabled (ablation)")
    
    # Visual LoRA: only train if enabled
    if enable_visual_lora:
        for _, p in iter_lora_parameters(model.clip_model.visual):
            p.requires_grad = True
        logger.write(f"Training visual LoRA: {visual_lora}, r={args.lora_r}")
    else:
        logger.write("Visual LoRA disabled (ablation)")
    
    # Enable text LoRA gradients if enabled
    if args.text_lora_r > 0:
        for _, p in iter_lora_parameters(model.clip_model.transformer):
            p.requires_grad = True
        logger.write(f"Text LoRA enabled: rank={args.text_lora_r}")
    # Unfreeze text encoder if flag is set
    if args.unfreeze_text_encoder:
        for p in model.clip_model.transformer.parameters():
            p.requires_grad = True
        logger.write("Text encoder unfrozen for fine-tuning")
    
    # LDC Head: only train if enabled
    if use_ldc_head:
        for p in ldchead.parameters():
            p.requires_grad = True
        logger.write("Training LDC head")
    else:
        for p in ldchead.parameters():
            p.requires_grad = False
        logger.write("LDC head disabled (ablation) - using base logits")
    
    # Domain discriminator: only train if enabled
    if enable_domain_adv:
        for p in domain_disc.parameters():
            p.requires_grad = True
        logger.write(f"Training domain discriminator: {args.num_domains} domains")
    else:
        for p in domain_disc.parameters():
            p.requires_grad = False
        logger.write("Domain adversarial training disabled (ablation)")

    # Optional: Modality attention weights for prototype fusion
    modality_weights = None
    if args.enable_modality_weights:
        modality_weights = ModalityAttentionWeights(n_way=args.n_way)
        modality_weights.to(device)
        logger.write(f"ModalityAttentionWeights: {args.n_way} classes x 2 modalities")

    # Collect trainable parameters (respecting ablation flags)
    train_params = []
    
    # Bottleneck adapter
    if enable_bottleneck_adapter and hasattr(model, 'sar_adapter'):
        train_params += list(model.sar_adapter.parameters())
    
    # LDC Head
    if use_ldc_head:
        train_params += list(ldchead.parameters())
    
    # Domain discriminator
    if enable_domain_adv:
        train_params += list(domain_disc.parameters())
    
    # Visual LoRA
    if enable_visual_lora:
        train_params += [p for _, p in iter_lora_parameters(model.clip_model.visual)]
    
    # Text LoRA parameters if enabled
    if args.text_lora_r > 0:
        train_params += [p for _, p in iter_lora_parameters(model.clip_model.transformer)]
    
    # Unfrozetext encoder parameters if enabled
    if args.unfreeze_text_encoder:
        for p in model.clip_model.transformer.parameters():
            if p.requires_grad:
                train_params.append(p)
    
    if modality_weights is not None:
        train_params += list(modality_weights.parameters())
    
    train_params = [p for p in train_params if p.requires_grad]

    # Handle edge case: no trainable parameters (all ablations enabled)
    if len(train_params) == 0:
        logger.write("WARNING: No trainable parameters! All components disabled.")
        logger.write("Using dummy parameter for optimizer.")
        # Create a dummy parameter to allow training to proceed
        dummy_param = nn.Parameter(torch.zeros(1), requires_grad=True)
        train_params = [dummy_param]

    optimizer = torch.optim.AdamW(train_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Zero-loss guardrail
    guardrail = ZeroLossGuardrail(tolerance=1e-6)

    trainable_count = sum(p.numel() for p in train_params)
    logger.write(f"Trainable parameters: {trainable_count:,}")

    # Setup checkpointing
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.checkpoint_dir / "train_log.csv"
    if not csv_path.exists():
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "train_loss", "train_acc", "valid_acc", "valid_ci",
                "domain_loss_total", "domain_loss_l2", "domain_loss_l3", "domain_loss_l4", "lr",
                "domain_adv_weight"
            ])

    # Track best val and best epoch
    start_epoch = 0
    best_val_acc = 0.0
    best_epoch = -1
    epochs_no_improve = 0
    
    # Adaptive loss weighting state
    current_domain_adv_weight = args.domain_adv_weight
    current_task_weight = 1.0  # Reference task weight
    
    # Logging for adaptive weights
    adaptive_log_path = args.checkpoint_dir / "adaptive_weights_log.csv"
    if args.enable_adaptive_weights and not adaptive_log_path.exists():
        with adaptive_log_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_acc", "val_acc", "gap", "domain_weight", "action"])

    if args.resume is not None and args.resume.exists():
        logger.write(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.sar_adapter.load_state_dict(ckpt["model_state"])
        ldchead.load_state_dict(ckpt["ldc_state"])
        domain_disc.load_state_dict(ckpt["domain_disc_state"])
        if "lora_state" in ckpt:
            for name, param in iter_lora_parameters(model.clip_model.visual):
                if name in ckpt["lora_state"]:
                    param.data.copy_(ckpt["lora_state"][name])
        if "text_lora_state" in ckpt and args.text_lora_r > 0:
            for name, param in iter_lora_parameters(model.clip_model.transformer):
                if name in ckpt["text_lora_state"]:
                    param.data.copy_(ckpt["text_lora_state"][name])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_acc = ckpt.get("best_val_acc", 0.0)
        best_epoch = ckpt.get("best_epoch", -1)
        if "current_domain_adv_weight" in ckpt:
            current_domain_adv_weight = ckpt["current_domain_adv_weight"]
            logger.write(f"   Resumed adaptive weight: {current_domain_adv_weight:.4f}")

    # Training loop
    logger.write("=" * 70)
    logger.write("Training started - Zero-Loss Guardrail ACTIVE")
    logger.write("Target: Validation Accuracy > 40% within 15 epochs")
    logger.write("=" * 70)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        ldchead.train()
        domain_disc.train()

        episode_acc = []
        total_loss = 0.0
        total_domain_loss = 0.0
        domain_loss_l2_acc = 0.0
        domain_loss_l3_acc = 0.0
        domain_loss_l4_acc = 0.0

        # GRL alpha: Active from Epoch 0 (no warm-up)
        alpha = 1.0  # Fixed alpha for consistent gradient reversal

        pbar = tqdm(range(args.episodes_per_epoch), desc=f"Epoch {epoch}", leave=False)
        for episode_idx in pbar:
            episode = sampler.sample_episode()
            support = episode.support_images.to(device)
            query = episode.query_images.to(device)
            y_query = episode.query_labels.to(device)

            # Build text features
            text_feats = build_text_features(model, args.data_root, episode.class_ids, device)

            # Compute multimodal support prototypes (FIXED: use correct encoders per modality)
            modality_weights_for_proto = modality_weights() if modality_weights is not None else None
            prototypes = compute_multimodal_prototypes_correct(
                model=model,
                support=support,
                support_labels=episode.support_labels,
                support_modalities=episode.support_modalities,
                n_way=args.n_way,
                device=device,
                modality_weights=modality_weights_for_proto,
            )

            # Query forward pass (SAR with adapter - respects ablation flag)
            disable_adapter = args.no_bottleneck_adapter
            query_emb, feats = model.forward_sar(query, disable_adapter=disable_adapter)

            # Compute logits
            proto_logits = query_emb @ prototypes.t()
            proto_logits = proto_logits * model.clip_model.logit_scale.exp()

            # Text-guided classification (ablation: can disable)
            if not args.no_text_guidance:
                text_logits = query_emb @ text_feats.t()
                text_logits = text_logits * model.clip_model.logit_scale.exp()
                base_logits = proto_logits + args.text_weight * text_logits
            else:
                base_logits = proto_logits

            # Query Adaptation (MAML-style inner loop)
            # Do a few gradient steps on query to adapt to the support set
            if args.inner_steps > 0:
                # Clone query embeddings for adaptation
                query_emb_adapted = query_emb.detach().clone()
                query_emb_adapted.requires_grad = True
                
                # Create a temporary optimizer for inner loop
                inner_optimizer = torch.optim.SGD([query_emb_adapted], lr=args.inner_lr)
                
                # Use query labels for inner loop (the ground truth we want to predict)
                y_query_for_inner = y_query
                
                for _ in range(args.inner_steps):
                    inner_optimizer.zero_grad()
                    # Compute logits with adapted query against prototypes
                    inner_logits = query_emb_adapted @ prototypes.t()
                    inner_logits = inner_logits * model.clip_model.logit_scale.exp()
                    # Use query labels as targets
                    inner_loss = F.cross_entropy(inner_logits, y_query_for_inner)
                    inner_loss.backward()
                    inner_optimizer.step()
                
                # Use adapted query embeddings for final classification
                query_emb_final = query_emb_adapted
                
                # Recompute logits with adapted query
                proto_logits = query_emb_final @ prototypes.t()
                proto_logits = proto_logits * model.clip_model.logit_scale.exp()
                
                # Text-guided classification in inner loop (ablation: can disable)
                if not args.no_text_guidance:
                    text_logits = query_emb_final @ text_feats.t()
                    text_logits = text_logits * model.clip_model.logit_scale.exp()
                    base_logits = proto_logits + args.text_weight * text_logits
                else:
                    base_logits = proto_logits
            else:
                query_emb_final = query_emb

            # LDC head (ablation: can disable to use base logits only)
            if not args.no_ldc_head:
                z_e, s_icd, s_alf, _ = ldchead(feats, base_logits, disable_icd=False)
                # Task loss with ICD regularization
                loss_task = F.cross_entropy(s_alf, y_query) + 0.5 * F.cross_entropy(s_icd, y_query)
            else:
                # Use base logits directly (no LDC refinement)
                s_alf = base_logits  # Use base logits as predictions
                loss_task = F.cross_entropy(s_alf, y_query)

            # Multi-level domain adversarial loss (ablation: can disable)
            loss_domain = torch.tensor(0.0, device=device)
            loss_dict = {
                'domain_loss_total': 0.0,
                'domain_loss_l2': 0.0,
                'domain_loss_l3': 0.0,
                'domain_loss_l4': 0.0,
            }
            
            # Only compute domain loss if enabled and domain_id is valid
            if not args.no_domain_adv and episode.query_domain_id is not None and episode.query_domain_id < args.num_domains:
                domain_label = torch.full(
                    (query_emb.size(0),), episode.query_domain_id,
                    dtype=torch.long, device=device
                )
                loss_domain, loss_dict = domain_disc.compute_loss(feats, domain_label, alpha=alpha)
                
                # Zero-Loss Guardrail Check
                if not guardrail.check(loss_dict, epoch, episode_idx):
                    logger.write(f"\n🚨 STOPPING: Zero-loss detected at Epoch {epoch}, Episode {episode_idx}")
                    logger.write(f"   GRL backprop path is broken - check nn.Module connections!")
                    raise RuntimeError("Zero-Loss Guardrail triggered - GRL backprop broken")

            # Total loss (use adaptive weight if enabled)
            domain_weight = current_domain_adv_weight if args.enable_adaptive_weights else args.domain_adv_weight
            loss = loss_task + domain_weight * loss_domain

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(train_params, max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_domain_loss += loss_domain.item()
            domain_loss_l2_acc += loss_dict['domain_loss_l2']
            domain_loss_l3_acc += loss_dict['domain_loss_l3']
            domain_loss_l4_acc += loss_dict['domain_loss_l4']
            episode_acc.append(episodic_accuracy(s_alf, y_query))

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.3f}',
                'task': f'{loss_task.item():.3f}',
                'dom': f'{loss_domain.item():.3f}',
                'w': f'{domain_weight:.3f}' if args.enable_adaptive_weights else f'{domain_weight:.2f}',
            })

        scheduler.step()

        # Epoch statistics
        mean_acc, ci95 = mean_confidence_interval(episode_acc)
        train_loss = total_loss / len(episode_acc)
        avg_domain_loss = total_domain_loss / len(episode_acc)
        avg_l2 = domain_loss_l2_acc / len(episode_acc)
        avg_l3 = domain_loss_l3_acc / len(episode_acc)
        avg_l4 = domain_loss_l4_acc / len(episode_acc)

        log_msg = f"Epoch {epoch}: loss={train_loss:.4f} acc={mean_acc*100:.2f}±{ci95*100:.2f}"
        log_msg += f" | domain=[L2:{avg_l2:.3f} L3:{avg_l3:.3f} L4:{avg_l4:.3f}]"
        logger.write(log_msg)

        # Validation
        if epoch % args.val_interval == 0 or epoch == args.epochs - 1:
            val_acc, val_ci95 = evaluate(
                model, ldchead, args.text_weight, val_sampler, device,
                args.n_way, args.data_root, args.val_episodes
            )
            logger.write(f"Epoch {epoch}: VALIDATION acc={val_acc*100:.2f}±{val_ci95*100:.2f} best={best_val_acc*100:.2f}@{best_epoch}")

            # Check if target reached
            if val_acc > 0.65 and best_val_acc <= 0.65:
                logger.write(f"🎯 TARGET REACHED! Validation accuracy > 65% at epoch {epoch}!")
            
            # Adaptive Loss Weighting: Adjust domain_adv_weight based on train-val gap
            if args.enable_adaptive_weights:
                train_val_gap = mean_acc - val_acc
                
                if train_val_gap > args.overfit_threshold:
                    # Overfitting: increase domain adversarial weight, reduce overfitting
                    old_weight = current_domain_adv_weight
                    current_domain_adv_weight = current_domain_adv_weight * (1.0 + args.weight_adjustment)
                    current_domain_adv_weight = min(current_domain_adv_weight, 2.0)  # Cap at 2.0
                    action = f"INCREASE domain_weight {old_weight:.4f} -> {current_domain_adv_weight:.4f}"
                    logger.write(f"   📈 ADAPTIVE: {action} (gap={train_val_gap*100:.1f}% > {args.overfit_threshold*100:.0f}%)")
                elif train_val_gap < args.overfit_threshold - 0.05:
                    # Good generalization: can reduce domain adversarial weight
                    old_weight = current_domain_adv_weight
                    current_domain_adv_weight = current_domain_adv_weight * (1.0 - args.weight_adjustment)
                    current_domain_adv_weight = max(current_domain_adv_weight, 0.01)  # Floor at 0.01
                    action = f"DECREASE domain_weight {old_weight:.4f} -> {current_domain_adv_weight:.4f}"
                    logger.write(f"   📉 ADAPTIVE: {action} (gap={train_val_gap*100:.1f}% < {args.overfit_threshold*100:.0f}%)")
                else:
                    action = "MAINTAIN weights"
                
                # Log adaptive weights
                with adaptive_log_path.open("a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        epoch,
                        f"{mean_acc:.6f}",
                        f"{val_acc:.6f}",
                        f"{train_val_gap:.6f}",
                        f"{current_domain_adv_weight:.6f}",
                        action
                    ])

            # Log to CSV
            with csv_path.open("a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch,
                    f"{train_loss:.6f}",
                    f"{mean_acc:.6f}",
                    f"{val_acc:.6f}",
                    f"{val_ci95:.6f}",
                    f"{avg_domain_loss:.6f}",
                    f"{avg_l2:.6f}",
                    f"{avg_l3:.6f}",
                    f"{avg_l4:.6f}",
                    f"{scheduler.get_last_lr()[0]:.6e}",
                    f"{current_domain_adv_weight:.6f}",
                ])

            # Save checkpoint
            ckpt = {
                "epoch": epoch,
                "model_state": model.sar_adapter.state_dict() if enable_bottleneck_adapter else {},
                "ldc_state": ldchead.state_dict() if use_ldc_head else {},
                "domain_disc_state": domain_disc.state_dict() if enable_domain_adv else {},
                "lora_state": {n: p.detach().cpu() for n, p in iter_lora_parameters(model.clip_model.visual)} if enable_visual_lora else {},
                "text_lora_state": {n: p.detach().cpu() for n, p in iter_lora_parameters(model.clip_model.transformer)} if args.text_lora_r > 0 else {},
                "text_encoder_state": {n: p.detach().cpu() for n, p in model.clip_model.transformer.named_parameters() if p.requires_grad} if args.unfreeze_text_encoder else {},
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "best_val_acc": best_val_acc,
                "best_epoch": best_epoch,
                "current_domain_adv_weight": current_domain_adv_weight,
                # Save ablation flags for reproducibility
                "ablation_flags": {
                    "no_bottleneck_adapter": args.no_bottleneck_adapter,
                    "no_visual_lora": args.no_visual_lora,
                    "no_text_guidance": args.no_text_guidance,
                    "no_domain_adv": args.no_domain_adv,
                    "no_ldc_head": args.no_ldc_head,
                },
            }

            torch.save(ckpt, args.checkpoint_dir / "last.pt")

            # Best model tracking
            if val_acc > best_val_acc + 0.001:
                best_val_acc = val_acc
                best_epoch = epoch
                epochs_no_improve = 0
                ckpt["best_val_acc"] = best_val_acc
                ckpt["best_epoch"] = best_epoch
                torch.save(ckpt, args.checkpoint_dir / "best.pt")
                logger.write(f"*** New best: {val_acc*100:.2f}% ***")
            else:
                epochs_no_improve += 1

            # Early stopping
            if args.early_stop_patience > 0 and epochs_no_improve >= args.early_stop_patience:
                logger.write(f"Early stopping at epoch {epoch}. Best: {best_val_acc*100:.2f}%@{best_epoch}")
                break
        else:
            # Log training only
            with csv_path.open("a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch,
                    f"{train_loss:.6f}",
                    f"{mean_acc:.6f}",
                    "N/A", "N/A",
                    f"{avg_domain_loss:.6f}",
                    f"{avg_l2:.6f}",
                    f"{avg_l3:.6f}",
                    f"{avg_l4:.6f}",
                    f"{scheduler.get_last_lr()[0]:.6e}",
                    f"{current_domain_adv_weight:.6f}",
                ])

    logger.write("=" * 70)
    logger.write(f"Training complete. Best: {best_val_acc*100:.2f}%@{best_epoch}")
    if best_val_acc > 0.65:
        logger.write("🎯 TARGET ACHIEVED: Validation accuracy > 65%!")
    else:
        logger.write("⚠️ Target not reached. Consider adjusting hyperparameters.")
    logger.write("=" * 70)


if __name__ == "__main__":
    main()
