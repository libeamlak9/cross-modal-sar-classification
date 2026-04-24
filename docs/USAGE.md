# Usage Guide

## Table of Contents

1. [Training](#training)
2. [Evaluation](#evaluation)
3. [Configuration](#configuration)
4. [Model Architecture](#model-architecture)
5. [Troubleshooting](#troubleshooting)

## Training

### Basic Training

Train a model with default settings:

```bash
python train.py --data_root ../So2Sat_LCZ42_v4 --epochs 100
```

### Full Training (Recommended)

Complete training with all features enabled:

```bash
python train.py \
    --data_root ../So2Sat_LCZ42_v4 \
    --split train \
    --n_way 5 \
    --k_shot 5 \
    --q_query 15 \
    --support_optical_count 3 \
    --episodes_per_epoch 500 \
    --val_episodes 200 \
    --epochs 100 \
    --text_weight 0.5 \
    --checkpoint_dir ../checkpoints_latest \
    --use_maf_head \
    --unfreeze_text_encoder \
    --adapter_bottleneck 64 \
    --adapter_layers layer4 \
    --lora_r 16 \
    --lora_alpha 32 \
    --visual_lora layer3,layer4 \
    --domain_adv_weight 0.1 \
    --lr 0.0001
```

### Training with Configuration File

```bash
python train.py --config configs/default_config.yaml
```

### Training Options Reference

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--data_root` | Path to So2Sat LCZ42 v4 HDF5 directory | Required |
| `--split` | Dataset split to use | train |
| `--n_way` | Number of classes per episode | 5 |
| `--k_shot` | Number of support samples per class | 5 |
| `--q_query` | Number of query samples per class | 15 |
| `--support_optical_count` | Optical samples in multimodal support | 3 |
| `--episodes_per_epoch` | Training episodes per epoch | 500 |
| `--val_episodes` | Validation episodes per epoch | 200 |
| `--epochs` | Total training epochs | 100 |
| `--text_weight` | Weight for text-based logits | 0.5 |
| `--checkpoint_dir` | Directory to save checkpoints | ../checkpoints_latest |
| `--use_maf_head` | Enable MAF head | False |
| `--unfreeze_text_encoder` | Unfreeze CLIP text encoder | False |
| `--adapter_bottleneck` | SAR adapter bottleneck dimension | 64 |
| `--adapter_layers` | Layers to apply SAR adapter | layer4 |
| `--lora_r` | LoRA rank | 16 |
| `--lora_alpha` | LoRA alpha scaling | 32 |
| `--visual_lora` | Visual encoder layers for LoRA | layer3,layer4 |
| `--domain_adv_weight` | Domain adversarial loss weight | 0.0 (disabled) |
| `--lr` | Learning rate | 0.0001 |

## Evaluation

### Standard Evaluation

Evaluate a trained model on the test set:

```bash
python eval.py \
    --data_root ../So2Sat_LCZ42_v4 \
    --split test \
    --checkpoint ../checkpoints_latest/best.pt \
    --use_maf_head \
    --unfreeze_text_encoder \
    --episodes 100
```

### Per-City Evaluation

Evaluate performance across different cities (geographic evaluation):

```bash
python eval.py \
    --data_root ../So2Sat_LCZ42_v4 \
    --split test \
    --checkpoint ../checkpoints_latest/best.pt \
    --use_maf_head \
    --unfreeze_text_encoder \
    --geo_eval \
    --episodes 100 \
    --output_json results.json
```

### Baseline Comparisons

Compare against ablated versions:

```bash
# Prototype-only (no text guidance)
python eval.py \
    --data_root ../So2Sat_LCZ42_v4 \
    --checkpoint ../checkpoints_latest/best.pt \
    --baseline proto_only

# Text-only (no prototype)
python eval.py \
    --data_root ../So2Sat_LCZ42_v4 \
    --checkpoint ../checkpoints_latest/best.pt \
    --baseline text_only

# No adapter (raw CLIP features)
python eval.py \
    --data_root ../So2Sat_LCZ42_v4 \
    --checkpoint ../checkpoints_latest/best.pt \
    --baseline no_adapter
```

### Evaluation Options Reference

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--data_root` | Path to So2Sat LCZ42 v4 HDF5 directory | Required |
| `--split` | Dataset split to use | test |
| `--checkpoint` | Path to model checkpoint | Required |
| `--episodes` | Number of evaluation episodes | 100 |
| `--use_maf_head` | Enable MAF head (must match training) | False |
| `--unfreeze_text_encoder` | Unfreeze text encoder (must match training) | False |
| `--geo_eval` | Enable per-city evaluation | False |
| `--output_json` | Path to save JSON results | None |
| `--baseline` | Baseline mode for ablation | None |

## Configuration

### Using Config Files

Create a custom configuration file:

```yaml
# my_config.yaml
data:
  data_root: '../So2Sat_LCZ42_v4'
  n_way: 5
  k_shot: 5

model:
  adapter_bottleneck: 64
  lora_r: 16
  use_maf_head: true

training:
  epochs: 50
  learning_rate: 0.0001
```

Run with config:

```bash
python train.py --config my_config.yaml
```

### Configuration Hierarchy

1. Default config (`configs/default_config.yaml`)
2. Custom config file (if provided via `--config`)
3. Command-line arguments (highest priority)

## Model Architecture

### CLIP Backbone

- **Image Encoder**: ResNet-50 based CLIP visual encoder
- **Text Encoder**: CLIP transformer-based text encoder
- **Pretrained Weights**: Loaded from OpenAI CLIP

### Adaptation Components

1. **SAR Adapter**: Bottleneck adapter for SAR modality alignment
   - Applied to specified ResNet layers
   - Default: 64-dim bottleneck on layer4

2. **LoRA (Low-Rank Adaptation)**: Parameter-efficient fine-tuning
   - Visual encoder: Applied to layer3 and layer4
   - Rank: 16, Alpha: 32

3. **Domain Discriminator**: Multi-level adversarial alignment
   - Operates on layer2, layer3, layer4 features
   - Gradient Reversal Layer (GRL) for domain confusion

### Classification Head

**LDC (Learnable Dynamic Classifier)**:
- **MAF**: Multi-Attention Fusion for feature refinement
- **ICD**: Inter-Class Discrimination for better separation
- **ALF**: Adaptive Loss Fusion for balanced training

## Troubleshooting

### Common Issues

**Issue**: CUDA out of memory
```bash
# Solution: Reduce batch size or query samples
python train.py --data_root ../So2Sat_LCZ42_v4 --q_query 10
```

**Issue**: Dataset not found
```bash
# Solution: Check data_root path
# Ensure So2Sat_LCZ42_v4 contains training.h5, validation.h5, testing.h5
ls ../So2Sat_LCZ42_v4/
```

**Issue**: Checkpoint loading error
```bash
# Solution: Ensure --use_maf_head and --unfreeze_text_encoder match training
python eval.py \
    --checkpoint ../checkpoints_latest/best.pt \
    --use_maf_head \
    --unfreeze_text_encoder
```

**Issue**: Zero domain loss (GRL broken)
```bash
# The training script includes a Zero-Loss Guardrail that monitors this
# If triggered, check:
# 1. Domain adversarial weight is not zero
# 2. GRL lambda is properly configured
# 3. Discriminator is receiving gradients
```

**Issue**: Slow training
```bash
# Solution: Reduce episodes per epoch or use fewer validation episodes
python train.py \
    --episodes_per_epoch 200 \
    --val_episodes 50
```

### Getting Help

- Check existing issues on GitHub
- Review the configuration examples
- Ensure all dependencies are installed correctly
- Verify dataset format matches expected HDF5 structure
