# Cross-Modal Meta-Learning for SAR Classification

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)

Cross-modal few-shot learning using CLIP RN50 for Optical→SAR domain transfer on So2Sat LCZ42 dataset. This repository implements a meta-learning framework that leverages both optical (Sentinel-2) and SAR (Sentinel-1) modalities for land cover classification.

## Key Features

- **Cross-Modal Learning**: Transfers knowledge from optical to SAR domain using CLIP
- **Multi-Level Domain Adversarial Training**: Aligns features across domains at multiple levels
- **CLIP-Based Architecture**: Utilizes frozen CLIP encoders with learnable adapters
- **LoRA Fine-Tuning**: Parameter-efficient fine-tuning with Low-Rank Adaptation
- **Multimodal Prototype Fusion**: Combines optical and SAR support samples
- **LDC Head**: Learnable Dynamic Classifier for logit refinement (MAF + ICD + ALF)
- **Zero-Loss Guardrail**: Monitors domain losses to detect broken GRL backprop paths

## Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended)
- 16GB+ RAM recommended

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/cross-modal-sar-classification.git
cd cross-modal-sar-classification/Code

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Dataset Setup

### So2Sat LCZ42 Dataset

1. Download the So2Sat LCZ42 v4 dataset from [source](https://github.com/zhu-xlab/So2Sat-LCZ42)
2. Organize the data as follows:
```
So2Sat_LCZ42_v4/
├── training.h5
├── validation.h5
└── testing.h5
```

3. The dataset contains:
   - **Sentinel-2 RGB**: indices [3, 2, 1] (B4, B3, B2) - 10m resolution
   - **Sentinel-1 SAR**: VV/VH channels - 10m resolution
   - **17 LCZ Classes**: Local Climate Zone labels

4. Place the dataset in the project root or specify path via `--data_root`.

## Quick Start

### Training

```bash
# Basic episodic training
python train.py --data_root ../So2Sat_LCZ42_v4 --epochs 100

# Full training with domain adversarial and all features
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

### Evaluation

```bash
# Standard evaluation
python eval.py \
    --data_root ../So2Sat_LCZ42_v4 \
    --split test \
    --checkpoint ../checkpoints_latest/best.pt \
    --use_maf_head \
    --unfreeze_text_encoder \
    --episodes 100

# Per-city evaluation (West-half test set)
python eval.py \
    --data_root ../So2Sat_LCZ42_v4 \
    --split test \
    --checkpoint ../checkpoints_latest/best.pt \
    --use_maf_head \
    --unfreeze_text_encoder \
    --geo_eval \
    --episodes 100 \
    --output_json results.json

# Baseline comparisons
python eval.py --data_root ../So2Sat_LCZ42_v4 --checkpoint ../checkpoints_latest/best.pt --baseline proto_only
python eval.py --data_root ../So2Sat_LCZ42_v4 --checkpoint ../checkpoints_latest/best.pt --baseline text_only
python eval.py --data_root ../So2Sat_LCZ42_v4 --checkpoint ../checkpoints_latest/best.pt --baseline no_adapter
```

## Configuration

The framework supports YAML configuration files. See `configs/default_config.yaml` for all available options:

```yaml
# Example configuration
data:
  data_root: '../So2Sat_LCZ42_v4'
  n_way: 5
  k_shot: 5
  q_query: 15

model:
  adapter_bottleneck: 64
  lora_r: 16
  lora_alpha: 32
  text_weight: 0.5

training:
  epochs: 100
  episodes_per_epoch: 500
  learning_rate: 0.0001
  domain_adv_weight: 0.1
```

## Project Structure

```
.
├── configs/                  # Configuration files
│   └── default_config.yaml
├── crossmodal_meta/          # Source code
│   ├── data/                 # Dataset and episode sampling
│   │   └── crossdomain_multimodal_sampler_v2.py
│   ├── models/               # Model architectures
│   │   ├── clip_backbone_wrapper.py
│   │   ├── clip_rn50_wrapper.py
│   │   ├── ldc_head.py
│   │   ├── lora.py
│   │   ├── multilevel_discriminator.py
│   │   ├── sar_adapter_rn50.py
│   │   └── text_prompts.py
│   └── utils/                # Utilities
│       ├── logging.py
│       ├── metrics.py
│       └── seed.py
├── clip_ldc/                 # CLIP implementation
│   ├── clip.py
│   ├── model.py
│   └── simple_tokenizer.py
├── docs/                     # Documentation
│   └── USAGE.md
├── examples/                 # Example usage
│   └── example_usage.py
├── train.py                  # Training entry point
├── eval.py                   # Evaluation entry point
├── requirements.txt
├── README.md
└── LICENSE
```

## Method Overview

- **Backbone**: CLIP RN50 image encoder (frozen with LoRA adapters)
- **Text Encoder**: CLIP text encoder (optionally unfrozen)
- **Support Set**: Multimodal (3 Optical + 2 SAR patches)
- **Query Set**: SAR-only patches
- **Adaptation**: SAR adapter + LoRA on visual encoder layers 3 & 4
- **Classification**: Prototype + text logits → LDC head (MAF + ICD + ALF)
- **Domain Alignment**: Multi-level domain discriminator with GRL

## Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_root` | required | Path to So2Sat_LCZ42_v4 HDF5 directory |
| `--n_way` | 5 | Classes per episode |
| `--k_shot` | 5 | Support samples per class |
| `--q_query` | 15 | Query samples per class |
| `--support_optical_count` | 3 | Number of optical samples in support |
| `--adapter_bottleneck` | 64 | SAR adapter bottleneck dimension |
| `--lora_r` | 16 | LoRA rank |
| `--lora_alpha` | 32 | LoRA alpha scaling |
| `--text_weight` | 0.5 | Weight for text-based logits |
| `--domain_adv_weight` | 0.1 | Weight for domain adversarial loss |
| `--use_maf_head` | False | Enable MAF (Multi-Attention Fusion) head |
| `--unfreeze_text_encoder` | False | Unfreeze CLIP text encoder |

## Results

Our model achieves the following performance on So2Sat LCZ42:

| Setting | Accuracy | Notes |
|---------|----------|-------|
| 5-way 5-shot | XX.X% | With domain adversarial |
| 5-way 5-shot | XX.X% | Without domain adversarial |

*Note: Update with actual results after training*

## LCZ Classes

The model classifies 17 Local Climate Zone classes:

| ID | Class Name | ID | Class Name |
|----|------------|----|------------|
| 0 | Compact High-rise | 9 | Heavy Industry |
| 1 | Compact Mid-rise | 10 | Dense Trees |
| 2 | Compact Low-rise | 11 | Scattered Trees |
| 3 | Open High-rise | 12 | Bush and Scrub |
| 4 | Open Mid-rise | 13 | Low Plants |
| 5 | Open Low-rise | 14 | Rock or Paved |
| 6 | Lightweight Low-rise | 15 | Bare Soil/Sand |
| 7 | Large Low-rise | 16 | Water |
| 8 | Sparsely Built | | |

## Citation

If you use this code in your research, please cite:

```bibtex
@article{weldemaryam2026crossmodal,
  title={Cross-Modal Meta-Learning for SAR Image Classification},
  author={Weldemaryam, Libeamlak Bekele},
  journal={},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- CLIP: [OpenAI CLIP](https://github.com/openai/CLIP)
- LDC: [Learnable Dynamic Classifier](https://github.com/LiShuo1001/LDC)
- So2Sat LCZ42 dataset by Zhu et al.
- Built with PyTorch

## Contact

For questions or issues, please open an issue on GitHub or contact legendariyy98@gmail.com
