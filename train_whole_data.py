import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from collections import defaultdict
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch.losses as smp_losses
import numpy as np
import random

# Import local modules
from dataset import MultiTaskDataset, MultiTaskUniformSampler
from model_factory import MultiTaskModelFactory, TASK_CONFIGURATIONS
from utils import (
    multi_task_collate_fn, 
    evaluate, 
    DetectionLoss, 
    set_seed
)

# Training configuration
LEARNING_RATE = 1e-4
BATCH_SIZE = 4  # Reduced for Swin-B due to higher memory requirements
NUM_EPOCHS = 50 
DATA_ROOT_PATH = '/proj/uppmax2025-2-369/Cgrain/ult/data/train'
ENCODER = 'swin_b'  # Swin-Base for better performance on complex multi-task learning
ENCODER_WEIGHTS = 'imagenet'
RANDOM_SEED = 42
MODEL_SAVE_PATH = 'best_model.pth' 
VAL_SPLIT = 0.0  # Disabled: using external validation set provided by competition
PRINT_FREQ = 50  # Print training status every N batches (set to 0 to disable batch-level printing)

def main():
    set_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device used: {device}")

    # Data loading and splitting
    # Training transforms with augmentation
    train_transforms = A.Compose([
        A.Resize(224, 224),  # Swin-Base expects 224x224 input
        A.RandomBrightnessContrast(p=0.2),
        A.GaussNoise(p=0.1), 
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], clip=True, min_visibility=0.1))
    
    # Validation transforms without augmentation
    val_transforms = A.Compose([
        A.Resize(224, 224),  # Swin-Base expects 224x224 input
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], clip=True, min_visibility=0.1))

    # Create full dataset for training (no validation split)
    train_dataset = MultiTaskDataset(data_root=DATA_ROOT_PATH, transforms=train_transforms)
    dataset_size = len(train_dataset)
    
    print(f"Dataset: {dataset_size} training samples (using all data, validation provided separately)")
    
    train_sampler = MultiTaskUniformSampler(train_dataset, batch_size=BATCH_SIZE)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_sampler=train_sampler, 
        num_workers=0,  # Set to 0 for Windows to avoid multiprocessing issues
        pin_memory=True,
        collate_fn=multi_task_collate_fn
    )
    
    # Model and loss setup
    model = MultiTaskModelFactory(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS, task_configs=TASK_CONFIGURATIONS).to(device)
    
    loss_functions = {
        'segmentation': smp_losses.DiceLoss(mode='multiclass'), 
        'classification': nn.CrossEntropyLoss(),
        'Regression': nn.MSELoss(), 
        'detection': DetectionLoss()
    }
    task_id_to_name = {cfg['task_id']: cfg['task_name'] for cfg in TASK_CONFIGURATIONS}

    # Optimization setup
    print("\n--- Setting parameter groups ---")
    param_groups = [
        {'params': model.encoder.parameters(), 'lr': LEARNING_RATE * 1},
    ]
    print(f"  - Shared Encoder                 -> LR: {LEARNING_RATE * 1}")
    
    for task_id, head in model.heads.items():
        lr_multiplier = 10.0
        current_lr = LEARNING_RATE * lr_multiplier
        param_groups.append({'params': head.parameters(), 'lr': current_lr})
        print(f"  - Task Head '{task_id:<25}' -> LR: {current_lr}")

    optimizer = optim.AdamW(param_groups)
    
    # Cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    print("\n--- Cosine Annealing Scheduler configured ---")

    print("\n" + "="*50 + "\n--- Start Training ---")
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_train_losses = defaultdict(list)
        # Disable tqdm progress bar to reduce output
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]", disable=True)
        
        batch_idx = 0
        for batch in loop:
            batch_idx += 1
            images = batch['image'].to(device)
            task_ids = batch['task_id']
            # Manually stack labels list to tensor
            labels = torch.stack(batch['label']).to(device)

            # All samples in batch belong to the same task due to sampler
            current_task_id = task_ids[0]
            task_name = task_id_to_name[current_task_id]

            outputs = model(images, task_id=current_task_id)
            
            # Grid-based detection logic
            if task_name == 'detection':
                _, _, h, w = outputs.shape
                
                # Calculate center of GT box (normalized)
                gt_center_x = (labels[:, 0] + labels[:, 2]) / 2.0
                gt_center_y = (labels[:, 1] + labels[:, 3]) / 2.0

                # Map to grid coordinates
                coord_h = torch.clamp((gt_center_y * h).long(), 0, h - 1)
                coord_w = torch.clamp((gt_center_x * w).long(), 0, w - 1)

                # Extract prediction from the specific grid cell
                final_outputs = torch.zeros((images.shape[0], 5), device=device)
                for i in range(images.shape[0]):
                    final_outputs[i] = outputs[i, :, coord_h[i], coord_w[i]]
            else:
                final_outputs = outputs
            
            loss = loss_functions[task_name](final_outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_train_losses[current_task_id].append(loss.item())
            
            # Print training status every PRINT_FREQ batches
            if PRINT_FREQ > 0 and batch_idx % PRINT_FREQ == 0:
                print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Batch {batch_idx}/{len(train_loader)} | Task: {current_task_id} | Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        # Train reporting
        print("\n--- Epoch {} Average Train Loss Report ---".format(epoch + 1))
        sorted_task_ids = sorted(epoch_train_losses.keys())
        for task_id in sorted_task_ids:
            avg_loss = np.mean(epoch_train_losses[task_id])
            print(f"  - Task '{task_id:<25}': {avg_loss:.4f}")
        print("-" * 40)

        # Save model checkpoint each epoch
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"-> Model checkpoint saved to: {MODEL_SAVE_PATH}\n")
        
        # Update scheduler
        scheduler.step()

    print(f"\n--- Training Finished ---\nFinal model saved at: {MODEL_SAVE_PATH}")

if __name__ == '__main__':
    main()