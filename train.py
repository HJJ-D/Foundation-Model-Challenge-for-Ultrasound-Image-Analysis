import torch
import torch.nn as nn
import torch.nn.functional as F
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
ENCODER = 'swin_l'  # Swin-Base for better performance on complex multi-task learning
ENCODER_WEIGHTS = 'imagenet'
IMG_SIZE = 224  # Input image size (224 to match Swin ImageNet pretrained weights)
RANDOM_SEED = 42
MODEL_SAVE_PATH = 'best_model.pth' 
VAL_SPLIT = 0.2
PRINT_FREQ = 50  # Print training status every N batches (set to 0 to disable batch-level printing)

# Deep Supervision configuration
USE_DEEP_SUPERVISION = True  # Enable deep supervision for segmentation tasks
NUM_AUX_OUTPUTS = 3  # Number of auxiliary outputs
AUX_LOSS_WEIGHTS = [0.5, 0.3, 0.2]  # Weights for auxiliary losses (should sum to ~1.0)

# Detection-specific configurations
USE_SEPARATE_DETECTION_FPN = True  # Use independent FPN decoder for detection
DETECTION_CLS_WEIGHT = 2.0  # Classification loss weight (increased)
DETECTION_BOX_WEIGHT = 1.0  # Bbox regression loss weight

def main():
    set_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device used: {device}")

    # Data loading and splitting
    # Training transforms with augmentation
    train_transforms = A.Compose([
        A.Resize(224, 224),  # Match Swin pretrained size
        A.RandomBrightnessContrast(p=0.2),
        A.GaussNoise(p=0.1), 
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], clip=True, min_visibility=0.1))
    
    # Validation transforms without augmentation
    val_transforms = A.Compose([
        A.Resize(224, 224),  # Match Swin pretrained size
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], clip=True, min_visibility=0.1))

    # Create full dataset to get indices
    temp_dataset = MultiTaskDataset(data_root=DATA_ROOT_PATH, transforms=train_transforms)
    dataset_size = len(temp_dataset)
    val_size = int(dataset_size * VAL_SPLIT)
    train_size = dataset_size - val_size
    
    # Split indices
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    indices = list(range(dataset_size))
    train_indices, val_indices = torch.utils.data.random_split(indices, [train_size, val_size], generator=generator)
    
    # Create separate datasets with different transforms
    train_dataset = MultiTaskDataset(data_root=DATA_ROOT_PATH, transforms=train_transforms)
    val_dataset = MultiTaskDataset(data_root=DATA_ROOT_PATH, transforms=val_transforms)
    
    # Create subsets
    train_subset = torch.utils.data.Subset(train_dataset, train_indices.indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices.indices)
    
    print(f"Dataset split: {train_size} training samples, {val_size} validation samples")
    
    # Fix dataframe reference for subset
    train_subset.dataframe = train_dataset.dataframe.iloc[train_indices.indices].reset_index(drop=True)
    
    train_sampler = MultiTaskUniformSampler(train_subset, batch_size=BATCH_SIZE)
    train_loader = torch.utils.data.DataLoader(
        train_subset, 
        batch_sampler=train_sampler, 
        num_workers=0,  # Set to 0 for Windows to avoid multiprocessing issues
        pin_memory=True,
        collate_fn=multi_task_collate_fn
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_subset, 
        batch_size=8,
        shuffle=False, 
        num_workers=0,  # Set to 0 for Windows to avoid multiprocessing issues
        pin_memory=True,
        collate_fn=multi_task_collate_fn
    )
    
    # Model and loss setup
    model = MultiTaskModelFactory(
        encoder_name=ENCODER, 
        encoder_weights=ENCODER_WEIGHTS, 
        task_configs=TASK_CONFIGURATIONS,
        use_deep_supervision=USE_DEEP_SUPERVISION,
        num_aux_outputs=NUM_AUX_OUTPUTS,
        use_separate_detection_fpn=USE_SEPARATE_DETECTION_FPN,
        img_size=IMG_SIZE  # Pass image size to avoid Swin mismatch
    ).to(device)
    
    loss_functions = {
        'segmentation': smp_losses.DiceLoss(mode='multiclass'), 
        'classification': nn.CrossEntropyLoss(),
        'Regression': nn.MSELoss(), 
        'detection': DetectionLoss(
            classification_weight=DETECTION_CLS_WEIGHT,
            box_regression_weight=DETECTION_BOX_WEIGHT
        )
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

    best_val_score = -float('inf')
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
            
            # Handle deep supervision outputs for segmentation tasks
            if task_name == 'segmentation' and USE_DEEP_SUPERVISION:
                main_out, aux_outs = outputs
                
                # Compute main loss
                loss = loss_functions[task_name](main_out, labels)
                
                # Compute auxiliary losses
                aux_loss = 0.0
                target_size = labels.shape[-2:]  # Get H, W from labels
                for i, aux_out in enumerate(aux_outs):
                    # Upsample auxiliary output to match label size
                    aux_out_upsampled = F.interpolate(aux_out, size=target_size, mode='bilinear', align_corners=False)
                    aux_loss_i = loss_functions[task_name](aux_out_upsampled, labels)
                    aux_loss += AUX_LOSS_WEIGHTS[i] * aux_loss_i
                
                # Total loss = main loss + weighted auxiliary losses
                loss = loss + aux_loss
            else:
                # For non-segmentation tasks or when deep supervision is disabled
                if task_name == 'detection':
                    # Single grid training (stable and proven to work)
                    B, C, H, W = outputs.shape  # [B, 5, H, W]
                    
                    # Extract prediction at GT center grid
                    gt_center_x = (labels[:, 0] + labels[:, 2]) / 2.0
                    gt_center_y = (labels[:, 1] + labels[:, 3]) / 2.0
                    coord_h = torch.clamp((gt_center_y * H).long(), 0, H - 1)
                    coord_w = torch.clamp((gt_center_x * W).long(), 0, W - 1)
                    
                    # Extract predictions at center positions
                    final_outputs = torch.zeros((B, 5), device=device)
                    for i in range(B):
                        final_outputs[i] = outputs[i, :, coord_h[i], coord_w[i]]
                    
                    # Create target: [bbox(4), objectness(1)] to match model output format
                    targets = torch.cat([
                        labels,  # [B, 4] bbox coordinates (already in [0,1])
                        torch.ones((B, 1), device=device)  # objectness=1 for all samples
                    ], dim=1)  # [B, 5]
                    
                    loss = loss_functions[task_name](final_outputs, targets)
                else:
                    loss = loss_functions[task_name](outputs, labels)
            
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

        # Validation
        val_results_df = evaluate(model, val_loader, device)
        
        score_cols = [col for col in val_results_df.columns if 'MAE' not in col and isinstance(val_results_df[col].iloc[0], (int, float))]
        avg_val_score = 0
        if not val_results_df.empty and score_cols:
            avg_val_score = val_results_df[score_cols].mean().mean()

        print("\n--- Epoch {} Validation Report ---".format(epoch + 1))
        if not val_results_df.empty:
            print(val_results_df.to_string(index=False))
        print(f"--- Average Val Score (Higher is better): {avg_val_score:.4f} ---")

        if avg_val_score > best_val_score:
            best_val_score = avg_val_score
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"-> New best model saved! Score improved to: {best_val_score:.4f}\n")
        
        # Update scheduler
        scheduler.step()

    print(f"\n--- Training Finished ---\nBest model saved at: {MODEL_SAVE_PATH}")

if __name__ == '__main__':
    main()