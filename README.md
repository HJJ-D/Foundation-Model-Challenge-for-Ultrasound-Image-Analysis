# MTUS-Net: Multi-Task Ultrasound Image Analysis

A multi-task learning framework for the **Foundation Model Challenge for Ultrasound Image Analysis (FMC_UIA)**, supporting **27 subtasks** across **4 task types**: segmentation, classification, detection, and regression.

## Table of Contents

- [Tasks Overview](#tasks-overview)
- [Project Structure](#project-structure)
- [Environment Setup](#environment-setup)
- [Quick Start](#quick-start)
- [Model Configuration](#model-configuration)
  - [Encoder Selection](#encoder-selection)
  - [Decoder (FPN) Configuration](#decoder-fpn-configuration)
  - [Head Configuration](#head-configuration)
- [Training Configuration](#training-configuration)
- [Inference & Docker](#inference--docker)
- [Troubleshooting](#troubleshooting)

---

## Tasks Overview

| Task Type | Count | Description | Output |
|-----------|-------|-------------|--------|
| **Segmentation** | 12 | Pixel-level tissue segmentation | Mask images |
| **Classification** | 9 | Image-level diagnosis | JSON |
| **Detection** | 3 | Lesion/structure localization | JSON (Bounding Boxes) |
| **Regression** | 3 | Anatomical keypoint regression | JSON (Coordinates) |

---

## Project Structure

```
code/
├── train.py                     # Training entry point
├── configs/
│   └── config.yaml              # Main configuration file
├── data/
│   ├── dataset.py               # MultiTaskDataset & sampler
│   └── new_dataloader.py
├── models/
│   ├── encoders.py              # Swin / ViT / ResNet / DINOv3 encoders
│   ├── decoders.py              # FPN decoders
│   ├── heads.py                 # Task-specific prediction heads
│   └── multitask_model.py       # Multi-task orchestration
├── losses/
│   └── loss_functions.py        # Task-specific losses
├── metrics/
│   └── __init__.py              # Evaluation metrics
└── utils/
    ├── common.py                # Utilities
    ├── logger.py                # Training logger
    └── plot_training.py         # Visualization

dataset/                         # Dataset directory
├── csv_files/                   # Task index CSVs
├── Segmentation/
├── Classification/
├── Detection/
└── Regression/
```

---

## Environment Setup

```bash
conda create -n ultrasound python=3.8
conda activate ultrasound

pip install torch torchvision
pip install albumentations segmentation-models-pytorch timm pandas tqdm omegaconf
```

---

## Quick Start

### 1. Prepare Data

Place the dataset under `dataset/`. Ensure `csv_files/` index files match actual image paths.

### 2. Edit Config

```bash
# Edit code/configs/config.yaml, set your data path
data:
  root_path: "/path/to/dataset/"
```

### 3. Train

```bash
cd code
python train.py
```

Outputs (logs, best model) are saved to `outputs/{experiment_name}/`.

---

## Model Configuration

The model follows a **shared encoder → FPN decoder → task-specific heads** architecture. All settings are in `code/configs/config.yaml` under the `model:` section.

### Encoder Selection

Configure the backbone encoder under `model.encoder`:

```yaml
model:
  encoder:
    name: swin_b              # Encoder name (see table below)
    pretrained: imagenet       # Pretrained weights (imagenet or null)
    freeze_encoder: false      # Whether to freeze encoder weights
```

#### Supported Encoders

| Name | Model | Parameters | Notes |
|------|-------|-----------|-------|
| `swin_t` | Swin Transformer Tiny | 28M | Fast, good baseline |
| `swin_s` | Swin Transformer Small | 50M | Balanced |
| `swin_b` | Swin Transformer Base | 88M | **Recommended**, best accuracy |
| `swin_l` | Swin Transformer Large | 197M | Largest Swin variant |
| `vit_t` | ViT Tiny (patch16) | 6M | Lightweight ViT |
| `vit_s` | ViT Small (patch16) | 22M | Small ViT |
| `vit_b` | ViT Base (patch16) | 86M | Standard ViT |
| `vit_l` | ViT Large (patch16) | 304M | Large ViT |
| `resnet50` | ResNet-50 | 25M | Classic CNN, fast training |
| `efficientnet-b4` | EfficientNet-B4 | 19M | Efficient CNN |
| `timm:xxx` | Any timm model | varies | Prefix with `timm:` for any timm model |

**Swin Transformer** models natively output hierarchical multi-scale features (stride 4/8/16/32), which work well with the FPN decoder.

**ViT** models output single-scale tokens. An internal adapter automatically creates multi-scale feature maps. You can optionally set `adapter_channels` to control the adapter output dimension:

```yaml
model:
  encoder:
    name: vit_b
    pretrained: imagenet
    adapter_channels: 256    # Optional: adapter output channels for ViT
```

**SMP encoders** (e.g., `resnet50`, `efficientnet-b4`) are loaded via `segmentation_models_pytorch` and produce multi-scale features natively.

**Any timm model** can be used by prefixing with `timm:`:

```yaml
model:
  encoder:
    name: "timm:convnext_base"
    pretrained: imagenet
```

---

### Decoder (FPN) Configuration

The framework uses **Feature Pyramid Network (FPN)** decoders to fuse multi-scale encoder features. Each task type can have a **separate** or **shared** FPN decoder.

```yaml
model:
  decoder:
    type: fpn
    pyramid_channels: 256          # FPN internal channel dimension
    segmentation_channels: 128     # FPN output channel dimension
    dropout: 0.2
    merge_policy: cat              # "cat" or "add"

    # Whether each task type has its own FPN decoder (true) or shares with segmentation (false)
    separate_detection_fpn: true
    separate_classification_fpn: true
    separate_regression_fpn: true

    # Whether classification/regression use FPN features or raw encoder features
    use_fpn_for_classification: false
    use_fpn_for_regression: false
```

| Setting | Description | Default |
|---------|-------------|---------|
| `pyramid_channels` | Number of channels inside FPN layers | 256 |
| `segmentation_channels` | Final FPN output channels | 128 |
| `merge_policy` | How to merge FPN levels: `cat` (concat) or `add` | `cat` |
| `separate_*_fpn` | Use a dedicated FPN for that task type | `true` |
| `use_fpn_for_classification` | Route classification through FPN (otherwise uses encoder features directly) | `false` |
| `use_fpn_for_regression` | Route regression through FPN (otherwise uses encoder features directly) | `false` |

> **Tip**: For classification and regression, setting `use_fpn_for_*: false` often performs better since these tasks benefit from high-level global features rather than multi-scale spatial features.

---

### Head Configuration

Each task type has a dedicated prediction head. Configure under `model.heads`:

#### Segmentation Head

```yaml
model:
  heads:
    segmentation:
      type: default          # "default" | "unet_like"
      upsampling: 4          # Upsample factor to match input resolution
      mid_channels: 128      # Hidden channels in conv blocks
      num_blocks: 2          # Number of conv-norm-act blocks (for default type)
      # Deep supervision (optional)
      use_deep_supervision: false
      num_aux_outputs: 3
      aux_loss_weights: [0.5, 0.3, 0.2]
```

| Type | Description |
|------|-------------|
| `default` | Conv blocks → 1×1 conv → upsample. Standard and efficient. |
| `unet_like` | Progressive 2× upsampling with conv blocks. Better boundary quality, slightly slower. |

**Deep supervision** adds auxiliary segmentation outputs at intermediate levels. Helps convergence but increases memory usage.

#### Classification Head

```yaml
model:
  heads:
    classification:
      type: default          # "default" | "baseline"
      mid_channels: 256      # Hidden MLP dimension (default type only)
      dropout: 0.3
```

| Type | Description |
|------|-------------|
| `default` | GAP → optional MLP → linear classifier. Supports `mid_channels` for MLP hidden dim. |
| `baseline` | Simple GAP → linear. Minimal, fast. |

#### Detection Head

```yaml
model:
  heads:
    detection:
      type: centernet        # "centernet" | "default" | "baseline"
      mid_channels: 128
```

| Type | Description |
|------|-------------|
| `centernet` | **Recommended.** Anchor-free CenterNet: heatmap + size + offset heads. |
| `default` | Anchor-based detection with channel attention (SE-like). |
| `baseline` | Simple 2-layer conv → output. Minimal. |

#### Regression Head

```yaml
model:
  heads:
    regression:
      type: default          # "default" | "baseline"
      hidden_dims: [256, 128]
      use_tanh: true         # Tanh activation to keep output in [0, 1]
      dropout: 0.3
```

| Type | Description |
|------|-------------|
| `default` | GAP → multi-layer MLP → tanh → output. Supports custom `hidden_dims`. |
| `baseline` | GAP → single linear layer. Minimal. |

---

## Training Configuration

### Optimizer & Scheduler

```yaml
training:
  num_epochs: 50
  optimizer:
    type: AdamW
    learning_rate: 0.0001
    weight_decay: 0.0001
    use_grouped_lr: true             # Use different LR for encoder vs heads
    encoder_lr_multiplier: 0.1       # Encoder LR = base_lr × 0.1
    head_lr_multiplier: 1.0          # Head LR = base_lr × 1.0

  scheduler:
    type: CosineAnnealingLR          # CosineAnnealingLR | StepLR | ReduceLROnPlateau
    T_max: 50
    eta_min: 1e-6

  gradient_clip: 1.0
  accumulation_steps: 1
```

### Loss Configuration

```yaml
training:
  loss_weights:
    segmentation: 1.0
    classification: 1.0
    detection: 1.0
    regression: 1.0

  loss_configs:
    segmentation:
      type: DiceLoss
      mode: multiclass

    classification:
      type: CrossEntropyLoss

    detection:
      type: Centernet
      heatmap_alpha: 2.0
      heatmap_gamma: 4.0
      size_weight: 1.0
      offset_weight: 1.0

    regression:
      type: MSELoss
```

### Data Augmentation

```yaml
data:
  image_size: 224
  batch_size: 20
  augmentation:
    train:
      random_brightness_contrast: 0.2
      gauss_noise: 0.1
      horizontal_flip: 0.0      # Use with caution for medical images
      vertical_flip: 0.0
    normalize:
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
```

### Validation

```yaml
validation:
  enabled: true
  freq: 1                          # Validate every N epochs
  save_best_model: true
  metric_for_best: mean_score      # Metric used to select best model
```

### Device

```yaml
device:
  use_cuda: true
  multi_gpu: false
  device_ids: [0]
  mixed_precision: false            # Enable AMP for faster training
```

### Single-Task Training

To train on a single task only:

```yaml
training:
  single_task:
    enabled: true
    task_id: "T2A_fetal_abdomen"
    task_name: segmentation
```

---

## Inference & Docker

### Docker Deployment

```bash
# Build
docker build -t my-submission:latest docker/

# Test locally
docker run --gpus all --rm \
  -v /path/to/data:/input/:ro \
  -v /path/to/output:/output \
  my-submission:latest

# Debug inside container
docker run --gpus all --rm \
  -v /path/to/data:/input/:ro \
  -v /path/to/output:/output \
  -it my-submission:latest /bin/bash
```

### Output Format

- **Segmentation**: mask images saved to `{output_dir}/` matching CSV `mask_path`
- **Classification / Detection / Regression**: JSON files

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| CUDA OOM | Reduce `data.batch_size`, or enable `device.mixed_precision: true` |
| Loss not decreasing | Check learning rate; try `encoder_lr_multiplier: 0.01`; adjust `loss_weights` |
| Poor segmentation | Enable deep supervision; switch to `unet_like` head; increase `mid_channels` |
| ViT training slow | Freeze encoder first (`freeze_encoder: true`), finetune later |

---

## Acknowledgments

- [timm](https://github.com/huggingface/pytorch-image-models) - PyTorch Image Models
- [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch) - Segmentation architectures
- [albumentations](https://github.com/albumentations-team/albumentations) - Data augmentation

