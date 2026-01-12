# Foundation Model Challenge for Ultrasound Image Analysis (FMC_UIA)

🏥 Multi-Task Learning Framework for Medical Ultrasound Image Analysis

This repository provides a **modular, configuration-driven training pipeline** for the Foundation Model Challenge for Ultrasound Image Analysis. The codebase has been refactored with clean architecture, supporting **27 tasks across 4 types**: segmentation, classification, detection, and regression.

## 📋 Table of Contents

- [Competition Tasks](#-competition-tasks)
- [Features](#-features)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Configuration](#-configuration)
- [Training](#-training)
- [Inference](#-inference)
- [Docker Deployment](#-docker-deployment)
- [Advanced Usage](#-advanced-usage)

---

## 🎯 Competition Tasks

This challenge includes **4 types of medical image analysis tasks** with a total of **27 subtasks**:

| Task Type | Count | Description | Output Format |
|-----------|-------|-------------|---------------|
| **Segmentation** | 12 | Pixel-level tissue classification | Mask Images |
| **Classification** | 9 | Image-level diagnosis | JSON |
| **Detection** | 3 | Lesion/structure localization | JSON (Bounding Boxes) |
| **Regression** | 3 | Anatomical keypoint localization | JSON (Coordinates) |

---

## ✨ Features

- 🧩 **Modular Architecture**: Clean separation of concerns with dedicated modules for data, models, losses, and utilities
- ⚙️ **Configuration-Driven**: All hyperparameters managed via YAML configuration files
- 🎨 **Flexible Model Design**: Support for multiple encoders (Swin Transformer, ResNet, etc.) and decoder architectures
- 📊 **Advanced Training**: Multi-task learning with task-specific loss weighting and deep supervision
- 🔍 **Comprehensive Logging**: Detailed training logs with metrics tracking and visualization
- 🚀 **Production Ready**: Docker support for seamless deployment

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create virtual environment
conda create -n ultrasound python=3.8
conda activate ultrasound

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

Place your data in the following structure:

```
data/
└── train/
    ├── csv_files/              # Task CSV index files
    │   ├── task1.csv
    │   ├── task2.csv
    │   └── ...
    ├── Classification/         # Classification images
    ├── Segmentation/          # Segmentation images and masks
    │   ├── Two/               # Binary segmentation
    │   ├── Three/             # 3-class segmentation
    │   ├── Four/              # 4-class segmentation
    │   └── Five/              # 5-class segmentation
    ├── Detection/             # Detection images
    └── Regression/            # Regression images
```

### 3. Configure Training

Edit `code/configs/config.yaml` to set your data path and hyperparameters:

```yaml
data:
  root_path: "/path/to/your/data/train"
  batch_size: 4
  image_size: 224

model:
  encoder:
    name: "swin_b"
    pretrained: "imagenet"
```

### 4. Train Model

```bash
cd code
python train.py
```

### 5. Monitor Training

Training logs and checkpoints will be saved to `outputs/swin_b_multitask_baseline/`:
- `training.log`: Detailed training logs
- `best_model.pth`: Best performing model
- `checkpoint_epoch_*.pth`: Periodic checkpoints

---

## 📁 Project Structure

```
Foundation-Model-Challenge-for-Ultrasound-Image-Analysis/
│
├── code/                          # Main codebase (modular architecture)
│   ├── configs/                   # Configuration management
│   │   ├── __init__.py           # Config loader
│   │   └── config.yaml           # Main configuration file
│   │
│   ├── data/                      # Data loading and processing
│   │   ├── __init__.py
│   │   └── dataset.py            # Multi-task dataset and samplers
│   │
│   ├── models/                    # Model architectures
│   │   ├── __init__.py           # Model factory
│   │   ├── encoders.py           # Feature extractors (Swin, ResNet, etc.)
│   │   ├── decoders.py           # FPN and other decoders
│   │   ├── heads.py              # Task-specific heads
│   │   └── multitask_model.py    # Main multi-task model
│   │
│   ├── losses/                    # Loss functions
│   │   ├── __init__.py           # Loss factory
│   │   └── loss_functions.py     # Task-specific losses
│   │
│   ├── metrics/                   # Evaluation metrics
│   │   └── __init__.py
│   │
│   ├── utils/                     # Utilities
│   │   ├── __init__.py
│   │   ├── common.py             # Common utilities
│   │   ├── logger.py             # Training logger
│   │   └── plot_training.py      # Visualization tools
│   │
│   └── train.py                   # Main training script
│
├── docker/                        # Docker deployment
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── model.py                  # Inference script
│   ├── build.sh
│   └── run_test.sh
│
├── data/                          # Data directory (not tracked)
└── README.md                      # This file
```

---

## ⚙️ Configuration

### Configuration File Structure

The `code/configs/config.yaml` file controls all aspects of training:

#### Data Configuration
```yaml
data:
  root_path: "/path/to/data/train"
  val_split: 0.2
  batch_size: 4
  image_size: 224
```

#### Model Architecture
```yaml
model:
  encoder:
    name: "swin_b"              # swin_t, swin_s, swin_b, resnet50, etc.
    pretrained: "imagenet"
    freeze_encoder: false
  
  decoder:
    type: "fpn"
    pyramid_channels: 256
    separate_detection_fpn: true
```

#### Training Settings
```yaml
training:
  num_epochs: 50
  optimizer:
    type: "AdamW"
    learning_rate: 1.0e-4
    weight_decay: 1.0e-4
  
  scheduler:
    type: "CosineAnnealingLR"
    T_max: 50
```

#### Loss Weights
```yaml
loss:
  task_weights:
    segmentation: 1.0
    classification: 1.0
    detection: 1.0
    Regression: 1.0
```

For full configuration options, see [code/configs/config.yaml](code/configs/config.yaml).

---

## 🎓 Training

### Basic Training

```bash
cd code
python train.py
```

### Command Line Arguments

```bash
# Use custom config file
python train.py --config configs/custom_config.yaml

# Override config parameters
python train.py --data.batch_size 8 --training.num_epochs 100

# Resume from checkpoint
python train.py --resume outputs/swin_b_multitask_baseline/checkpoint_epoch_30.pth
```

### Training Output

During training, the following outputs are generated:

- **Checkpoints**: `outputs/{experiment_name}/best_model.pth`
- **Logs**: `outputs/{experiment_name}/training.log`
- **Metrics**: Displayed in console and saved to log file

Example training log:
```
Epoch 10/50
Train Loss: 0.2456 | Seg: 0.1234 | Cls: 0.0456 | Det: 0.0567 | Reg: 0.0199
Val Loss: 0.2123 | Seg: 0.1056 | Cls: 0.0398 | Det: 0.0489 | Reg: 0.0180
✓ New best model saved!
```

---

## 🔮 Inference

### Local Inference

```bash
cd docker
python model.py
```

### Modify Inference Paths

Edit `docker/model.py`:

```python
if __name__ == '__main__':
    data_root = '/path/to/test/data'
    output_dir = 'predictions/'
    batch_size = 8
    
    model = Model()
    model.predict(data_root, output_dir, batch_size=batch_size)
```

### Output Structure

```
predictions/
├── classification_predictions.json     # Classification results
├── detection_predictions.json          # Detection bboxes
├── regression_predictions.json         # Keypoint coordinates
└── Segmentation/                       # Segmentation masks
    ├── Two/
    ├── Three/
    ├── Four/
    └── Five/
```

**Important**: Segmentation masks must preserve the original directory structure from the CSV `mask_path` field.

---

## 🐳 Docker Deployment

### Build Docker Image

```bash
cd docker
chmod +x build.sh
./build.sh
```

### Test Docker Image

```bash
chmod +x run_test.sh
./run_test.sh
```

### Required Files for Docker

```
docker/
├── Dockerfile
├── requirements.txt
├── model.py                  # Inference script
├── model_factory.py          # Model architecture
├── best_model.pth            # Trained weights
├── build.sh
└── run_test.sh
```

### Docker Submission

After successful testing, submit your Docker image following the competition guidelines.

---

## 🔧 Advanced Usage

### Custom Model Architecture

To use a different encoder:

```yaml
model:
  encoder:
    name: "resnet50"  # or any timm/smp supported model
    pretrained: "imagenet"
```

### Task-Specific Loss Tuning

Adjust task weights in config:

```yaml
loss:
  task_weights:
    segmentation: 2.0      # Increase segmentation importance
    classification: 1.0
    detection: 0.5
    Regression: 1.5
```

### Data Augmentation

Customize augmentation in config:

```yaml
data:
  augmentation:
    train:
      random_brightness_contrast: 0.3
      gauss_noise: 0.15
      horizontal_flip: 0.5  # Note: Use with caution for medical images
```

### Learning Rate Scheduling

Multiple scheduler options available:

```yaml
training:
  scheduler:
    # Option 1: Cosine Annealing (Recommended)
    type: "CosineAnnealingLR"
    T_max: 50
    eta_min: 1.0e-6
    
    # Option 2: Reduce on Plateau
    # type: "ReduceLROnPlateau"
    # mode: "max"
    # factor: 0.5
    # patience: 5
    
    # Option 3: Step LR
    # type: "StepLR"
    # step_size: 20
    # gamma: 0.1
```

### Deep Supervision for Segmentation

Enable deep supervision to improve segmentation performance:

```yaml
model:
  heads:
    segmentation:
      use_deep_supervision: true
      num_aux_outputs: 3
      aux_loss_weights: [0.5, 0.3, 0.2]
```

---

## 📊 Model Performance Tips

### 1. Encoder Selection

Different encoders have different trade-offs:

| Encoder | Parameters | Speed | Performance |
|---------|-----------|-------|-------------|
| `swin_t` | 28M | Fast | Good |
| `swin_s` | 50M | Medium | Better |
| `swin_b` | 88M | Slow | Best |
| `resnet50` | 25M | Fast | Good |

### 2. Batch Size and Learning Rate

General rule of thumb:
- If you increase batch size by 2x, increase learning rate by 2x
- Recommended: batch_size=4, lr=1e-4

### 3. Training Time Estimates

On a single GPU (RTX 3090):
- ~2 hours per epoch with `swin_b`, batch_size=4
- ~1 hour per epoch with `swin_t`, batch_size=8

---

## 🐛 Troubleshooting

### CUDA Out of Memory

```yaml
# Reduce batch size
data:
  batch_size: 2

# Or use gradient accumulation (coming soon)
```

### Training Loss Not Decreasing

1. Check learning rate (too high/low)
2. Verify data loading (visualize a batch)
3. Adjust task loss weights
4. Try different encoder initialization

### Validation Performance Poor

1. Reduce overfitting:
   - Increase dropout
   - Add more augmentation
   - Reduce model size
2. Check for data leakage
3. Ensure proper train/val split

---

## 📚 Key Modules Explained

### Dataset (`code/data/dataset.py`)

- **MultiTaskDataset**: Loads data from CSV files for all tasks
- **MultiTaskUniformSampler**: Ensures balanced sampling across tasks

### Model (`code/models/`)

- **Encoders**: Feature extraction backbones (Swin, ResNet, etc.)
- **Decoders**: FPN for multi-scale feature aggregation
- **Heads**: Task-specific prediction heads
- **MultiTaskModel**: Orchestrates encoder + decoder + heads

### Losses (`code/losses/loss_functions.py`)

Task-specific loss functions with automatic weighting:
- Segmentation: Dice Loss + Cross Entropy
- Classification: Cross Entropy with class weighting
- Detection: IoU Loss + Confidence Loss
- Regression: Smooth L1 Loss (normalized coordinates)

### Logger (`code/utils/logger.py`)

Comprehensive logging system:
- Training/validation metrics per epoch
- Best model tracking
- Loss decomposition by task

---

## 🤝 Contributing

This is a competition repository. For questions or issues:

1. Check existing documentation
2. Review the code comments
3. Contact competition organizers

---

## 📄 License

This code is provided for the Foundation Model Challenge for Ultrasound Image Analysis competition.

---

## 🙏 Acknowledgments

- **timm**: PyTorch Image Models
- **segmentation_models.pytorch**: Segmentation architectures
- **albumentations**: Data augmentation library

---

## 📞 Support

For technical questions about this codebase:
- Open an issue in the repository
- Contact: [Competition Platform]

For competition-related questions:
- Visit the [Competition Website]
- Check the [Forum/Discussion Board]

---

**Good luck with your submission! 🚀**

**Format**: JSON file

**Path**: `{output_dir}/regression_predictions.json`

**Content**:
```json
[
  {
    "image_path": "relative/path/image_001.jpg",
    "task_id": "FUGC",
    "predicted_points_normalized": [0.3, 0.4, 0.6, 0.7],
    "predicted_points_pixels": [150, 200, 300, 350]
  },
  ...
]
```

**Description**:
- `predicted_points_normalized`: [x1, y1, x2, y2, ...] (normalized coordinates)
- `predicted_points_pixels`: [x1, y1, x2, y2, ...] (pixel coordinates)

### 2. Docker Environment Requirements

- **Input mount point**: `/input/` (read-only)
- **Output mount point**: `/output/` (writable)
- **Memory limit**: Recommended not to exceed 16GB



## ❓ FAQ

### Q1: How to modify model architecture?

**A**: You can freely modify the model structure in `model_factory.py`, but make sure:
- Input: RGB images
- Output: Format that meets each task's requirements

### Q2: What if I run out of GPU memory during training?

**A**: Reduce the batch size:
```python
# train.py
BATCH_SIZE = 4  # Change from 8 to 4
```

### Q3: Docker build is very slow?

**A**: 
- First build needs to download base image (~6GB), which takes time
- Subsequent builds will use cache and be faster
- Ensure stable network connection

### Q4: How to verify if output format is correct?

**A**: 
1. Run Docker on validation set
2. Upload output to Codabench platform
3. Platform will automatically validate format and return evaluation results
4. If format is incorrect, there will be clear error messages

### Q5: Can I use my own pre-trained model?

**A**: Yes!
- Model weight files are included in the Docker image

### Q6: Must the inference output path be strictly followed?

**A**: **Yes!** Especially for segmentation task mask paths, they must be completely consistent with the `mask_path` field in CSV (removing the leading `../`). Otherwise, the evaluation platform cannot find the files.

### Q7: How to debug Docker internal issues?

**A**: Enter container for debugging:
```bash
docker run --gpus all --rm \
  -v /path/to/data:/input/:ro \
  -v /path/to/output:/output \
  -it my-submission:latest /bin/bash

# Run manually inside container
python model.py
```


## 📄 License

This baseline code is for competition use only.

---

## 🎉 Good Luck with the Competition!

Remember the key steps:
1. ✅ Train model
2. ✅ Test inference locally
3. ✅ Build Docker
4. ✅ **Test Docker on validation set**
5. ✅ **Upload predictions to Codabench for validation**
6. ✅ Submit Docker image

**Passing the Codabench evaluation on the validation set ensures your final submission is correct!**

---

## 📊 Training Logs and Analysis (NEW!)

### Automatic Training Logs

The training script now automatically records all metrics for easy analysis and paper writing:

- ✅ **Training losses** (per task, with mean/std/min/max)
- ✅ **Validation metrics** (Accuracy, F1, Dice, IoU, MAE)
- ✅ **Learning rate schedule**
- ✅ **Training time per epoch**
- ✅ **Automatic plotting** of training curves
- ✅ **Timestamp labeled** for each experiment

### Quick Usage

**Train (logs saved automatically):**
```bash
python code/train.py --config code/configs/config.yaml
```

**Quick analysis:**
```bash
python quick_example.py outputs/experiment_20260101_123456
```

**Generate all plots:**
```bash
python code/utils/plot_training.py outputs/experiment_20260101_123456
```

### Output Files

```
outputs/experiment_20260101_123456/
├── training_summary.csv       # ⭐ Best for plotting
├── train_losses.csv           # Detailed training losses
├── val_metrics.csv            # Detailed validation metrics
├── training_history.json      # Complete history
├── best_model.pth             # Best model weights
└── training_curves.png        # Auto-generated plots
```

### Documentation

- **[QUICKSTART_LOGS.md](QUICKSTART_LOGS.md)** - Quick start guide (2 min read)
- **[TRAINING_LOGS_GUIDE.md](TRAINING_LOGS_GUIDE.md)** - Complete guide with examples
- **[LOGS_OUTPUT_EXAMPLE.md](LOGS_OUTPUT_EXAMPLE.md)** - Output format examples
- **[SUMMARY.md](SUMMARY.md)** - Summary of all changes

### Example: Plot Accuracy

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('outputs/experiment/training_summary.csv')
plt.plot(df['epoch'], df['avg_accuracy'], marker='o')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.savefig('accuracy.png', dpi=300)  # High-res for papers
```

---

Good luck! 🚀

