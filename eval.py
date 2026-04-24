"""Stage 2 Single-Domain Evaluation: Support and Query from Same Domain.

Evaluates on test/validation splits using the cross-domain multimodal sampler
but with support and query from the SAME domain (single-domain setting).
Support set: 3 Optical + 2 SAR (multimodal fusion via mean)
Query set: SAR-only (same domain as support)

Key Features:
- Same-domain evaluation (support domain = query domain)
- LDC Head for logit refinement
- Bottleneck adapters for SAR adaptation
- Text-guided zero-shot classification
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

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

import json
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

# Try to import matplotlib for visualization
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available, skipping visualization")

from crossmodal_meta.data.crossdomain_multimodal_sampler_v2 import CrossDomainMultimodalSampler
from crossmodal_meta.models.clip_backbone_wrapper import CLIPBackboneWrapper
from crossmodal_meta.models.ldc_head import LDCHead
from crossmodal_meta.models.lora import iter_lora_parameters
from crossmodal_meta.models.text_prompts import DEFAULT_TEMPLATES_SAR, get_classnames
from crossmodal_meta.utils.logging import SimpleLogger
from crossmodal_meta.utils.metrics import episodic_accuracy, mean_confidence_interval
from crossmodal_meta.utils.seed import set_seed
import clip_ldc as clip


# =============================================================================
# LCZ Class Names Mapping
# =============================================================================

CLASS_NAME_MAP = {
    0: "Compact High-rise",
    1: "Compact Mid-rise",
    2: "Compact Low-rise",
    3: "Open High-rise",
    4: "Open Mid-rise",
    5: "Open Low-rise",
    6: "Lightweight Low-rise",
    7: "Large Low-rise",
    8: "Sparsely Built",
    9: "Heavy Industry",
    10: "Dense Trees",
    11: "Scattered Trees",
    12: "Bush and Scrub",
    13: "Low Plants",
    14: "Rock or Paved",
    15: "Bare Soil/Sand",
    16: "Water",
}

# Short names for confusion matrix
SHORT_NAMES = {
    0: "Compact High-rise",
    1: "Compact Mid-rise",
    2: "Compact Low-rise",
    3: "Open High-rise",
    4: "Open Mid-rise",
    5: "Open Low-rise",
    6: "Lightweight Low-rise",
    7: "Large Low-rise",
    8: "Sparsely Built",
    9: "Heavy Industry",
    10: "Dense Trees",
    11: "Scattered Trees",
    12: "Bush and Scrub",
    13: "Low Plants",
    14: "Rock or Paved",
    15: "Bare Soil/Sand",
    16: "Water",
}


def get_short_classname(class_id: int) -> str:
    """Get short class name for visualization."""
    return SHORT_NAMES.get(class_id, f"Class{class_id}")


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


def compute_multimodal_prototypes_correct(
    model: CLIPBackboneWrapper,
    support: torch.Tensor,
    support_labels: torch.Tensor,
    support_modalities: torch.Tensor,
    n_way: int,
    device: torch.device,
    disable_adapter: bool = False,
) -> torch.Tensor:
    """Compute prototypes with CORRECT modality-specific encoding.
    
    This fixes the bug where SAR support samples were incorrectly passed 
    through the optical encoder.
    
    Args:
        model: CLIP backbone wrapper
        support: Support images [n_way * k_shot, C, H, W]
        support_labels: Support labels [n_way * k_shot]
        support_modalities: Modality indicators [n_way * k_shot] (0=optical, 1=sar)
        n_way: Number of classes
        device: torch device
        disable_adapter: Whether to disable the SAR adapter
    
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
        sar_emb, _ = model.forward_sar(sar_support, disable_adapter=disable_adapter)
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
        description="Stage 2: Single-Domain Evaluation (Support=Query Domain)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data arguments
    parser.add_argument("--data_root", type=Path, required=True,
                        help="Path to So2Sat_LCZ42_v4 HDF5 directory")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "valid", "test"],
                        help="Evaluation split (train=Training cities, valid=Western halves, test=Eastern halves)")
    parser.add_argument("--sen2_rgb_idx", type=str, default="3,2,1",
                        help="Sentinel-2 RGB band indices")
    parser.add_argument("--sen1_map", type=str, default="vv_vh_diff",
                        choices=["repeat", "vv_vh_diff"],
                        help="SAR channel mapping strategy")

    # Episode configuration
    parser.add_argument("--n_way", type=int, default=5,
                        help="Number of classes per episode")
    parser.add_argument("--k_shot", type=int, default=5,
                        help="Number of support samples per class")
    parser.add_argument("--q_query", type=int, default=15,
                        help="Number of query samples per class (SAR only)")
    parser.add_argument("--support_optical_count", type=int, default=3,
                        help="Number of optical samples in support set (remaining are SAR)")
    parser.add_argument("--episodes", type=int, default=600,
                        help="Number of evaluation episodes")

    # Model architecture (must match training)
    parser.add_argument("--adapter_bottleneck", type=int, default=64,
                        help="Bottleneck dimension for SAR adapter")
    parser.add_argument("--adapter_layers", type=str, default="layer4",
                        help="RN50 layers with adapters (comma-separated)")
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha scaling")
    parser.add_argument("--visual_lora", type=str, default="layer3,layer4",
                        help="Visual encoder layers for LoRA (comma-separated, or 'none')")
    parser.add_argument("--use_maf_head", action="store_true",
                        help="Enable MAF head for direct multi-level feature fusion classification")
    parser.add_argument("--no_ldc_head", action="store_true",
                        help="Disable LDC head, use base logits only (for ablation)")
    parser.add_argument("--unfreeze_text_encoder", action="store_true",
                        help="Unfreeze text encoder for fine-tuning")
    parser.add_argument("--text_lora_r", type=int, default=0,
                        help="Text encoder LoRA rank (0=disabled)")
    parser.add_argument("--backbone", type=str, default="rn50", choices=["rn50", "vit_b32"],
                        help="CLIP backbone: rn50 or vit_b32")
    parser.add_argument("--no_bottleneck_adapter", action="store_true",
                        help="Disable bottleneck adapters (for ablation)")
    parser.add_argument("--no_visual_lora", action="store_true",
                        help="Disable visual LoRA (for ablation)")
    parser.add_argument("--no_text_guidance", action="store_true",
                        help="Disable text-guided classification (for ablation)")

    # Inference settings
    parser.add_argument("--text_weight", type=float, default=0.5,
                        help="Weight for text-based logits")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Path to model checkpoint")

    # Baseline modes
    parser.add_argument("--baseline", type=str, default=None,
                        choices=["proto_only", "text_only", "no_adapter"],
                        help="Baseline evaluation mode")

    # Output
    parser.add_argument("--output_json", type=Path, default=None,
                        help="Path to save detailed results as JSON")
    parser.add_argument("--save_cm", action="store_true",
                        help="Save confusion matrix visualization")

    # Misc
    parser.add_argument("--seed", type=int, default=2024,
                        help="Random seed")

    return parser.parse_args()


def sample_same_domain_episode(sampler: CrossDomainMultimodalSampler) -> object:
    """Sample an episode where support and query come from the SAME domain.
    
    This is a modified version of sample_episode() that uses the same domain
    for both support and query sets.
    """
    # Select a SINGLE domain for both support and query
    domain = int(sampler.rng.choice(sampler.valid_domains, size=1)[0])
    
    # Get available classes in this domain
    available_classes = list(sampler.domain_class_indices[domain].keys())
    
    if len(available_classes) < sampler.n_way:
        raise ValueError(
            f"Not enough classes in domain {domain}. "
            f"Found {len(available_classes)}, need {sampler.n_way}"
        )
    
    # Sample N classes
    selected_classes = sampler.rng.choice(available_classes, size=sampler.n_way, replace=False)
    
    # Sample support set (multimodal from the SAME domain)
    support_images = []
    support_labels = []
    support_modalities = []
    
    for new_label, cls in enumerate(selected_classes):
        cls_data = sampler.domain_class_indices[domain][cls]
        
        # Sample optical
        optical_indices = sampler.rng.choice(
            cls_data['optical'],
            size=sampler.support_optical_count,
            replace=len(cls_data['optical']) < sampler.support_optical_count
        )
        for idx in optical_indices:
            support_images.append(sampler._load_optical(idx))
            support_labels.append(new_label)
            support_modalities.append(0)
        
        # Sample SAR
        sar_indices = sampler.rng.choice(
            cls_data['sar'],
            size=sampler.support_sar_count,
            replace=len(cls_data['sar']) < sampler.support_sar_count
        )
        for idx in sar_indices:
            support_images.append(sampler._load_sar(idx))
            support_labels.append(new_label)
            support_modalities.append(1)
    
    # Sample query set (SAR only from the SAME domain)
    query_images = []
    query_labels = []
    
    for new_label, cls in enumerate(selected_classes):
        cls_data = sampler.domain_class_indices[domain][cls]
        
        query_indices = sampler.rng.choice(
            cls_data['sar'],
            size=sampler.q_query,
            replace=len(cls_data['sar']) < sampler.q_query
        )
        for idx in query_indices:
            query_images.append(sampler._load_sar(idx))
            query_labels.append(new_label)
    
    # Convert to tensors
    support_images = torch.stack(support_images, dim=0)
    support_labels = torch.tensor(support_labels, dtype=torch.long)
    support_modalities = torch.tensor(support_modalities, dtype=torch.long)
    query_images = torch.stack(query_images, dim=0)
    query_labels = torch.tensor(query_labels, dtype=torch.long)
    
    # Import the dataclass
    from crossmodal_meta.data.crossdomain_multimodal_sampler_v2 import MultimodalEpisodeBatch
    
    return MultimodalEpisodeBatch(
        class_ids=selected_classes.tolist(),
        support_images=support_images,
        support_labels=support_labels,
        support_modalities=support_modalities,
        query_images=query_images,
        query_labels=query_labels,
        query_domain_id=sampler.domain_id_to_idx[domain],
        support_domain_id=sampler.domain_id_to_idx[domain],  # SAME domain
        n_optical_in_support=sampler.support_optical_count,
    )


def evaluate_episodes_samedomain(
    model: CLIPBackboneWrapper,
    ldchead: LDCHead,
    sampler: CrossDomainMultimodalSampler,
    device: torch.device,
    args: argparse.Namespace,
    logger: SimpleLogger,
) -> dict:
    """Run episodic evaluation with SAME domain for support and query."""
    # Keep backbone in train mode for batch statistics (same fix as eval_stage2)
    model.train()
    if hasattr(model, 'sar_adapter') and not args.no_bottleneck_adapter:
        model.sar_adapter.eval()
    
    use_ldc_head = not args.no_ldc_head
    if use_ldc_head:
        ldchead.eval()

    acc_list = []
    all_predictions = []
    all_labels = []
    episode_details = []

    # Ablation flags
    disable_adapter = args.baseline == "no_adapter" or args.no_bottleneck_adapter

    with torch.no_grad():
        for ep_idx in tqdm(range(args.episodes), desc="Evaluating (Same-Domain)"):
            # Use same-domain sampling
            episode = sample_same_domain_episode(sampler)
            
            support = episode.support_images.to(device)
            query = episode.query_images.to(device)
            y_query = episode.query_labels.to(device)

            # Compute multimodal support prototypes (FIXED: use correct encoders per modality)
            prototypes = compute_multimodal_prototypes_correct(
                model=model,
                support=support,
                support_labels=episode.support_labels,
                support_modalities=episode.support_modalities,
                n_way=args.n_way,
                device=device,
                disable_adapter=disable_adapter,
            )

            # Query forward pass (SAR with adapter)
            query_emb, feats = model.forward_sar(query, disable_adapter=disable_adapter)

            # Compute logits based on mode
            proto_logits = query_emb @ prototypes.t()
            proto_logits = proto_logits * model.clip_model.logit_scale.exp()

            text_feats = build_text_features(model, args.data_root, episode.class_ids, device)
            text_logits = query_emb @ text_feats.t()
            text_logits = text_logits * model.clip_model.logit_scale.exp()

            # Baseline modes
            if args.baseline == "proto_only":
                base_logits = proto_logits
            elif args.baseline == "text_only":
                base_logits = text_logits
            else:
                # Default: combine proto + text (unless text guidance disabled)
                if not args.no_text_guidance:
                    base_logits = proto_logits + args.text_weight * text_logits
                else:
                    base_logits = proto_logits

            # LDC head refinement (or use base logits directly)
            if use_ldc_head:
                _, _, s_alf, _ = ldchead(feats, base_logits, disable_icd=False)
            else:
                s_alf = base_logits

            # Collect predictions for confusion matrix
            preds = s_alf.argmax(dim=-1)
            all_predictions.extend(preds.cpu().tolist())
            all_labels.extend(y_query.cpu().tolist())

            # Compute accuracy
            acc = episodic_accuracy(s_alf, y_query)
            acc_list.append(acc)

            # Track domain info
            episode_details.append({
                "episode_idx": ep_idx,
                "class_ids": [int(c) for c in episode.class_ids],
                "domain": int(episode.support_domain_id),  # Same for support and query
                "accuracy": float(acc),
            })

    # Compute overall statistics
    mean_acc, ci95 = mean_confidence_interval(acc_list)

    # Compute min and max accuracy
    min_acc = float(np.min(acc_list))
    max_acc = float(np.max(acc_list))
    std_acc = float(np.std(acc_list, ddof=1))  # Sample standard deviation

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_predictions, labels=list(range(args.n_way)))

    # Compute per-class accuracy
    per_class_acc = []
    for i in range(args.n_way):
        if cm[i].sum() > 0:
            acc = cm[i, i] / cm[i].sum()
        else:
            acc = 0.0
        per_class_acc.append(float(acc))

    # Compute F1 scores
    f1_macro = f1_score(all_labels, all_predictions, average='macro', labels=list(range(args.n_way)), zero_division=0)
    f1_weighted = f1_score(all_labels, all_predictions, average='weighted', labels=list(range(args.n_way)), zero_division=0)

    # Compute Precision and Recall
    precision_macro = precision_score(all_labels, all_predictions, average='macro', labels=list(range(args.n_way)), zero_division=0)
    precision_weighted = precision_score(all_labels, all_predictions, average='weighted', labels=list(range(args.n_way)), zero_division=0)
    recall_macro = recall_score(all_labels, all_predictions, average='macro', labels=list(range(args.n_way)), zero_division=0)
    recall_weighted = recall_score(all_labels, all_predictions, average='weighted', labels=list(range(args.n_way)), zero_division=0)

    results = {
        "overall": {
            "mean_accuracy": float(mean_acc),
            "ci95": float(ci95),
            "std_accuracy": std_acc,
            "min_accuracy": min_acc,
            "max_accuracy": max_acc,
            "f1_macro": float(f1_macro),
            "f1_weighted": float(f1_weighted),
            "precision_macro": float(precision_macro),
            "precision_weighted": float(precision_weighted),
            "recall_macro": float(recall_macro),
            "recall_weighted": float(recall_weighted),
            "n_episodes": len(acc_list),
        },
        "confusion_matrix": cm.tolist(),
        "per_class_accuracy": per_class_acc,
        "config": {
            "split": args.split,
            "n_way": args.n_way,
            "k_shot": args.k_shot,
            "q_query": args.q_query,
            "support_composition": f"{args.support_optical_count} optical + {args.k_shot - args.support_optical_count} SAR",
            "text_weight": args.text_weight,
            "baseline": args.baseline,
            "checkpoint": str(args.checkpoint),
            "evaluation_mode": "same_domain",  # Key difference
        },
        "episodes": episode_details,
    }

    return results


def visualize_confusion_matrix(
    cm: np.ndarray,
    class_ids: list,
    output_path: Path,
    short_names: dict = SHORT_NAMES,
) -> None:
    """Visualize and save confusion matrix."""
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib not available, skipping confusion matrix visualization")
        return
    
    # Get short names for this episode's classes
    labels = [short_names.get(cid, f"C{cid}") for cid in class_ids]
    
    # Create figure
    plt.figure(figsize=(16, 14))
    
    # Plot heatmap with larger annotations
    hm = sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={'label': 'Count'},
        annot_kws={'size': 24}
    )
    
    # Increase colorbar tick fontsize
    cbar = hm.collections[0].colorbar
    cbar.ax.tick_params(labelsize=20)
    
    plt.xlabel('Predicted', fontsize=28)
    plt.ylabel('True', fontsize=28)
    plt.xticks(fontsize=20, rotation=45, ha='right')
    plt.yticks(fontsize=20, rotation=0)
    plt.tight_layout()
    
    # Save
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved to: {output_path}")


def plot_per_class_accuracy(
    per_class_acc: list,
    class_ids: list,
    output_path: Path,
    short_names: dict = SHORT_NAMES,
) -> None:
    """Plot per-class accuracy bar chart."""
    if not MATPLOTLIB_AVAILABLE:
        return
    
    # Get short names for this episode's classes
    labels = [short_names.get(cid, f"C{cid}") for cid in class_ids]
    
    plt.figure(figsize=(10, 6))
    
    x = np.arange(len(labels))
    bars = plt.bar(x, [a * 100 for a in per_class_acc], color='steelblue', edgecolor='navy')
    
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.xticks(x, labels, rotation=45, ha='right')
    plt.ylim(0, 100)
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, acc in zip(bars, per_class_acc):
        plt.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 1,
            f'{acc*100:.1f}%',
            ha='center',
            va='bottom',
            fontsize=9
        )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Per-class accuracy plot saved to: {output_path}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = SimpleLogger()

    logger.write("=" * 70)
    logger.write("Stage 2: Single-Domain Evaluation (Support=Query Domain)")
    logger.write("=" * 70)
    logger.write(f"Device: {device}")
    split_desc = {
        "train": "Training cities (32 valid domains)",
        "valid": "Validation cities (Western halves)",
        "test": "Test cities (Eastern halves)"
    }
    logger.write(f"Split: {args.split} ({split_desc[args.split]})")
    logger.write(f"Checkpoint: {args.checkpoint}")
    logger.write(f"Episodes: {args.episodes}")
    logger.write(f"Configuration: {args.n_way}-way {args.k_shot}-shot {args.q_query}-query")
    logger.write(f"Support: {args.support_optical_count} Optical + {args.k_shot - args.support_optical_count} SAR")
    logger.write(f"Query: SAR only (SAME DOMAIN as support)")
    logger.write("⚠️  NOTE: This is SINGLE-DOMAIN evaluation (easier than cross-domain)")
    if args.baseline:
        logger.write(f"Baseline mode: {args.baseline}")

    # Parse layer configurations
    adapter_layers = [s.strip() for s in args.adapter_layers.split(",") if s.strip()]
    visual_lora = [] if args.visual_lora == "none" else [s.strip() for s in args.visual_lora.split(",") if s.strip()]

    # Build cross-domain multimodal sampler
    logger.write("Building Cross-Domain Multimodal Sampler...")
    sampler = CrossDomainMultimodalSampler(
        data_root=args.data_root,
        split=args.split,
        n_way=args.n_way,
        k_shot=args.k_shot,
        q_query=args.q_query,
        support_optical_count=args.support_optical_count,
        episodes_per_epoch=args.episodes,
        seed=args.seed,
    )
    
    logger.write(f"Valid domains: {len(sampler.valid_domains)}")

    # Build model (respecting ablation flags)
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
            lora_dropout=0.0,
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
            lora_dropout=0.0,
            text_lora_r=args.text_lora_r,
            unfreeze_text_encoder=args.unfreeze_text_encoder,
        )
    model.to(device)

    # Set feature dimensions based on backbone
    if args.backbone == "vit_b32":
        ldc_in_channels = [768, 768, 768, 768]
    else:
        ldc_in_channels = [256, 512, 1024, 2048]

    use_ldc_head = not args.no_ldc_head
    ldchead = LDCHead(
        in_channels=ldc_in_channels,
        feat_dim=1024,
        num_classes=args.n_way,
        use_maf_head=args.use_maf_head if use_ldc_head else False
    )
    ldchead.to(device)

    # Log model settings
    logger.write(f"Backbone: {args.backbone}")
    logger.write(f"Bottleneck Adapter: {enable_bottleneck_adapter}")
    logger.write(f"Visual LoRA: {enable_visual_lora}")
    logger.write(f"LDC Head: {use_ldc_head}")
    logger.write(f"MAF Head: {args.use_maf_head}")
    logger.write(f"Text Encoder: {'unfrozen' if args.unfreeze_text_encoder else 'frozen'}")
    logger.write(f"Text LoRA: {args.text_lora_r}")

    # Load checkpoint
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    logger.write(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    # Load adapter weights only if adapter exists in model
    if hasattr(model, 'sar_adapter') and enable_bottleneck_adapter:
        if "model_state" in ckpt and len(ckpt["model_state"]) > 0:
            model.sar_adapter.load_state_dict(ckpt["model_state"], strict=False)
            logger.write("Loaded adapter weights")
        else:
            logger.write("No adapter weights in checkpoint (training may have used --no_bottleneck_adapter)")
    else:
        logger.write("Skipping adapter weights (disabled for ablation)")
    
    # Load LDC head weights
    if use_ldc_head and "ldc_state" in ckpt:
        ldchead.load_state_dict(ckpt["ldc_state"])
        logger.write("Loaded LDC head weights")
    else:
        logger.write("No LDC head weights in checkpoint")
    
    if "lora_state" in ckpt and len(ckpt["lora_state"]) > 0:
        for name, param in iter_lora_parameters(model.clip_model.visual):
            if name in ckpt["lora_state"]:
                param.data.copy_(ckpt["lora_state"][name])
        logger.write(f"Loaded LoRA weights: {len(ckpt['lora_state'])} parameters")
    else:
        logger.write("No LoRA weights in checkpoint")
    
    if "text_lora_state" in ckpt and args.text_lora_r > 0:
        for name, param in iter_lora_parameters(model.clip_model.transformer):
            if name in ckpt["text_lora_state"]:
                param.data.copy_(ckpt["text_lora_state"][name])
        logger.write(f"Loaded Text LoRA weights: {len(ckpt['text_lora_state'])} parameters")

    # Load unfrozen text encoder weights if applicable
    if "text_encoder_state" in ckpt and args.unfreeze_text_encoder:
        text_encoder_loaded = 0
        for name, param in model.clip_model.transformer.named_parameters():
            if name in ckpt["text_encoder_state"]:
                param.data.copy_(ckpt["text_encoder_state"][name])
                text_encoder_loaded += 1
        logger.write(f"Loaded unfrozen text encoder weights: {text_encoder_loaded} parameters")

    # Log checkpoint info
    if "epoch" in ckpt:
        logger.write(f"Checkpoint epoch: {ckpt['epoch']}")
    if "best_val_acc" in ckpt:
        logger.write(f"Checkpoint best val acc (cross-domain): {ckpt['best_val_acc']*100:.2f}%")

    # Run evaluation
    logger.write("-" * 70)
    results = evaluate_episodes_samedomain(model, ldchead, sampler, device, args, logger)

    # Print results
    logger.write("-" * 70)
    logger.write("RESULTS - Single-Domain Evaluation")
    logger.write("-" * 70)
    logger.write(f"Overall: {results['overall']['mean_accuracy']*100:.2f}% ± {results['overall']['ci95']*100:.2f}%")
    logger.write(f"Std Dev: {results['overall']['std_accuracy']*100:.2f}%")
    logger.write(f"Min Accuracy: {results['overall']['min_accuracy']*100:.2f}%")
    logger.write(f"Max Accuracy: {results['overall']['max_accuracy']*100:.2f}%")
    logger.write(f"F1 Score (Macro): {results['overall']['f1_macro']*100:.2f}%")
    logger.write(f"F1 Score (Weighted): {results['overall']['f1_weighted']*100:.2f}%")
    logger.write(f"Precision (Macro): {results['overall']['precision_macro']*100:.2f}%")
    logger.write(f"Precision (Weighted): {results['overall']['precision_weighted']*100:.2f}%")
    logger.write(f"Recall (Macro): {results['overall']['recall_macro']*100:.2f}%")
    logger.write(f"Recall (Weighted): {results['overall']['recall_weighted']*100:.2f}%")
    logger.write(f"Support Composition: {results['config']['support_composition']}")
    logger.write(f"Evaluation Mode: {results['config']['evaluation_mode']}")
    
    # Print per-class accuracy
    logger.write("")
    logger.write("Per-Class Accuracy:")
    for i, acc in enumerate(results['per_class_accuracy']):
        class_id = results['episodes'][0]['class_ids'][i] if i < len(results['episodes'][0]['class_ids']) else i
        class_name = CLASS_NAME_MAP.get(class_id, f"Class {class_id}")
        logger.write(f"  {class_name}: {acc*100:.1f}%")

    # Save confusion matrix visualization
    if args.save_cm and MATPLOTLIB_AVAILABLE:
        # Aggregate confusion matrix across all episodes
        cm = np.array(results['confusion_matrix'])
        
        # Use first episode's class IDs for labeling (they may vary)
        first_episode_class_ids = results['episodes'][0]['class_ids'] if results['episodes'] else list(range(args.n_way))
        
        # Save overall confusion matrix
        cm_output_path = Path("confusion_matrix_samedomain.png")
        visualize_confusion_matrix(cm, first_episode_class_ids, cm_output_path)
        
        # Save per-class accuracy plot
        acc_output_path = Path("per_class_accuracy_samedomain.png")
        plot_per_class_accuracy(
            results['per_class_accuracy'],
            first_episode_class_ids,
            acc_output_path
        )

    # Save detailed results
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w") as f:
            # Remove episode details for cleaner output (can be large)
            output_results = {k: v for k, v in results.items() if k != "episodes"}
            json.dump(output_results, f, indent=2)
        logger.write(f"\nResults saved to: {args.output_json}")

    logger.write("")
    logger.write("💡 Compare with cross-domain eval using:")
    logger.write(f"   python eval_stage2.py --split {args.split} --checkpoint {args.checkpoint}")

    logger.write("=" * 70)


if __name__ == "__main__":
    main()
