#!/usr/bin/env python3
"""
Example: Training and evaluating the Cross-Modal Meta-Learning model.

This script demonstrates the complete workflow from training to evaluation
for cross-modal few-shot learning on So2Sat LCZ42 dataset.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def example_basic_training():
    """Example basic training configuration."""
    print("=" * 70)
    print("Example 1: Basic Training")
    print("=" * 70)
    
    command = """
python train.py \\
    --data_root ../So2Sat_LCZ42_v4 \\
    --epochs 100
    """
    print(command)
    print()


def example_full_training():
    """Example full training with all features."""
    print("=" * 70)
    print("Example 2: Full Training (with Domain Adversarial)")
    print("=" * 70)
    
    command = """
python train.py \\
    --data_root ../So2Sat_LCZ42_v4 \\
    --split train \\
    --n_way 5 \\
    --k_shot 5 \\
    --q_query 15 \\
    --support_optical_count 3 \\
    --episodes_per_epoch 500 \\
    --val_episodes 200 \\
    --epochs 100 \\
    --text_weight 0.5 \\
    --checkpoint_dir ../checkpoints_latest \\
    --use_maf_head \\
    --unfreeze_text_encoder \\
    --adapter_bottleneck 64 \\
    --adapter_layers layer4 \\
    --lora_r 16 \\
    --lora_alpha 32 \\
    --visual_lora layer3,layer4 \\
    --domain_adv_weight 0.1 \\
    --lr 0.0001
    """
    print(command)
    print()


def example_standard_evaluation():
    """Example standard evaluation."""
    print("=" * 70)
    print("Example 3: Standard Evaluation")
    print("=" * 70)
    
    command = """
python eval.py \\
    --data_root ../So2Sat_LCZ42_v4 \\
    --split test \\
    --checkpoint ../checkpoints_latest/best.pt \\
    --use_maf_head \\
    --unfreeze_text_encoder \\
    --episodes 100
    """
    print(command)
    print()


def example_geo_evaluation():
    """Example geographic (per-city) evaluation."""
    print("=" * 70)
    print("Example 4: Per-City Geographic Evaluation")
    print("=" * 70)
    
    command = """
python eval.py \\
    --data_root ../So2Sat_LCZ42_v4 \\
    --split test \\
    --checkpoint ../checkpoints_latest/best.pt \\
    --use_maf_head \\
    --unfreeze_text_encoder \\
    --geo_eval \\
    --episodes 100 \\
    --output_json results.json
    """
    print(command)
    print()


def example_baseline_comparison():
    """Example baseline comparisons for ablation study."""
    print("=" * 70)
    print("Example 5: Baseline Comparisons (Ablation Study)")
    print("=" * 70)
    
    print("# Prototype-only (no text guidance)")
    command1 = """
python eval.py \\
    --data_root ../So2Sat_LCZ42_v4 \\
    --checkpoint ../checkpoints_latest/best.pt \\
    --baseline proto_only
    """
    print(command1)
    print()
    
    print("# Text-only (no prototype)")
    command2 = """
python eval.py \\
    --data_root ../So2Sat_LCZ42_v4 \\
    --checkpoint ../checkpoints_latest/best.pt \\
    --baseline text_only
    """
    print(command2)
    print()
    
    print("# No adapter (raw CLIP features)")
    command3 = """
python eval.py \\
    --data_root ../So2Sat_LCZ42_v4 \\
    --checkpoint ../checkpoints_latest/best.pt \\
    --baseline no_adapter
    """
    print(command3)
    print()


def example_config_training():
    """Example training with configuration file."""
    print("=" * 70)
    print("Example 6: Training with Config File")
    print("=" * 70)
    
    command = """
python train.py --config configs/default_config.yaml
    """
    print(command)
    print()


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("Cross-Modal Meta-Learning for SAR Classification")
    print("Example Usage Scripts")
    print("=" * 70 + "\n")
    
    example_basic_training()
    example_full_training()
    example_standard_evaluation()
    example_geo_evaluation()
    example_baseline_comparison()
    example_config_training()
    
    print("=" * 70)
    print("For more information, see docs/USAGE.md")
    print("=" * 70)


if __name__ == '__main__':
    main()
