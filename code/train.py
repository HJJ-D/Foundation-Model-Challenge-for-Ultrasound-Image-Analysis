"""
Multi-Task Ultrasound Image Analysis - Modular Training Script

This script provides a clean, configuration-driven training pipeline for multi-task learning.
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
from collections import defaultdict
from tqdm import tqdm
import numpy as np

# Import modular components
from configs import load_config
from models import build_model
from losses import build_all_losses
from data import MultiTaskDataset, MultiTaskUniformSampler
from utils import set_seed, multi_task_collate_fn
from utils.logger import TrainingLogger
from metrics import evaluate


def build_dataloaders(config):
    """Build train and validation dataloaders."""
    # Training transforms
    aug_cfg = config.get_augmentation_config('train')
    train_transforms = A.Compose([
        A.Resize(config.image_size, config.image_size),
        A.RandomBrightnessContrast(p=aug_cfg.get('random_brightness_contrast', 0.2)),
        A.GaussNoise(p=aug_cfg.get('gauss_noise', 0.1)),
        A.Normalize(
            mean=config.get('data.augmentation.normalize.mean'),
            std=config.get('data.augmentation.normalize.std')
        ),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], clip=True, min_visibility=0.1))
    
    # Validation transforms
    val_transforms = A.Compose([
        A.Resize(config.image_size, config.image_size),
        A.Normalize(
            mean=config.get('data.augmentation.normalize.mean'),
            std=config.get('data.augmentation.normalize.std')
        ),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], clip=True, min_visibility=0.1))
    
    # Create datasets
    train_dataset = MultiTaskDataset(data_root=config.data_root, transforms=train_transforms)
    val_dataset = MultiTaskDataset(data_root=config.data_root, transforms=val_transforms)
    
    # Build task_configs dynamically from dataset (like original code)
    print("\nBuilding task configurations from dataset...")
    task_configs = []
    task_config_map = {}
    
    for _, row in train_dataset.dataframe.iterrows():
        task_id = row['task_id']
        if task_id not in task_config_map:
            task_config = {
                'task_id': task_id,
                'task_name': row['task_name'],
                'num_classes': int(row['num_classes'])
            }
            task_config_map[task_id] = task_config
            task_configs.append(task_config)
    
    # Update config with dynamic task_configs
    config.config['tasks'] = task_configs
    
    print(f"Detected {len(task_configs)} tasks:")
    for cfg in sorted(task_configs, key=lambda x: x['task_id']):
        print(f"  - {cfg['task_id']}: {cfg['task_name']}, num_classes={cfg['num_classes']}")
    
    dataset_size = len(train_dataset)
    val_size = int(dataset_size * config.val_split)
    train_size = dataset_size - val_size
    
    # Create samplers (no subset, use full datasets with sampler)
    train_sampler = MultiTaskUniformSampler(
        dataset=train_dataset,
        batch_size=config.batch_size,
        steps_per_epoch=config.get('training.steps_per_epoch')
    )
    
    # Create dataloaders
    # Note: MultiTaskUniformSampler returns batches, so use as batch_sampler
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=config.num_workers,
        pin_memory=config.get('data.pin_memory', True),
        collate_fn=multi_task_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.get('data.pin_memory', True),
        collate_fn=multi_task_collate_fn
    )
    
    print(f"✓ Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    return train_loader, val_loader


def build_optimizer(model, config, loss_weighter=None):
    """Build optimizer with optional grouped learning rates and adaptive loss parameters."""
    use_grouped_lr = config.get('training.optimizer.use_grouped_lr', True)
    lr = config.learning_rate
    weight_decay = config.weight_decay
    
    if use_grouped_lr:
        encoder_params, head_params = model.get_trainable_parameters()
        encoder_lr_mult = config.get('training.optimizer.encoder_lr_multiplier', 0.1)
        head_lr_mult = config.get('training.optimizer.head_lr_multiplier', 1.0)
        
        param_groups = [
            {'params': encoder_params, 'lr': lr * encoder_lr_mult},
            {'params': head_params, 'lr': lr * head_lr_mult}
        ]
        print(f"✓ Using grouped LR: encoder={lr * encoder_lr_mult:.2e}, heads={lr * head_lr_mult:.2e}")
    else:
        param_groups = model.parameters()
    
    # Add adaptive loss parameters if using adaptive weighting
    from losses.loss_functions import AdaptiveLossWeighter
    if isinstance(loss_weighter, AdaptiveLossWeighter):
        adaptive_lr = config.get('training.adaptive_loss.learning_rate', lr)
        param_groups.append({
            'params': loss_weighter.parameters(),
            'lr': adaptive_lr
        })
        print(f"✓ Added adaptive loss parameters (lr={adaptive_lr:.2e})")
    
    optimizer_type = config.get('training.optimizer.type', 'AdamW')
    
    if optimizer_type == 'AdamW':
        optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'Adam':
        optimizer = torch.optim.Adam(param_groups, lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'SGD':
        momentum = config.get('training.optimizer.momentum', 0.9)
        optimizer = torch.optim.SGD(param_groups, lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    print(f"✓ Optimizer: {optimizer_type}")
    
    return optimizer


def build_scheduler(optimizer, config):
    """Build learning rate scheduler."""
    scheduler_type = config.get('training.scheduler.type', 'CosineAnnealingLR')
    
    if scheduler_type == 'CosineAnnealingLR':
        T_max = config.get('training.scheduler.T_max', config.num_epochs)
        eta_min = config.get('training.scheduler.eta_min', 1e-6)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=T_max, eta_min=eta_min
        )
    elif scheduler_type == 'ReduceLROnPlateau':
        mode = config.get('training.scheduler.mode', 'max')
        factor = config.get('training.scheduler.factor', 0.5)
        patience = config.get('training.scheduler.patience', 5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=mode, factor=factor, patience=patience
        )
    elif scheduler_type == 'StepLR':
        step_size = config.get('training.scheduler.step_size', 20)
        gamma = config.get('training.scheduler.gamma', 0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )
    elif scheduler_type == 'None' or scheduler_type is None:
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    if scheduler:
        print(f"✓ Scheduler: {scheduler_type}")
    
    return scheduler


def train_epoch(model, train_loader, loss_functions, loss_weights, optimizer, device, config, current_epoch=0):
    """Train for one epoch."""
    model.train()
    epoch_losses = defaultdict(list)
    epoch_task_weights = defaultdict(list)  # Track adaptive weights
    
    task_id_to_name = {cfg['task_id']: cfg['task_name'] for cfg in config.get_task_configs()}
    use_deep_supervision = config.get('model.heads.segmentation.use_deep_supervision', False)
    aux_loss_weights = config.get('model.heads.segmentation.aux_loss_weights', [0.5, 0.3, 0.2])
    
    # Check if using adaptive loss
    from losses.loss_functions import AdaptiveLossWeighter
    use_adaptive_loss = isinstance(loss_weights, AdaptiveLossWeighter)
    if use_adaptive_loss:
        loss_weights.train()  # Ensure it's in training mode
        
        # Warmup: freeze adaptive weights for first N epochs
        warmup_epochs = config.get('training.adaptive_loss.warmup_epochs', 0)
        freeze_adaptive = current_epoch < warmup_epochs
        if freeze_adaptive and current_epoch == 0:
            print(f"  [Adaptive Loss Warmup] Freezing adaptive weights for first {warmup_epochs} epochs")
        elif not freeze_adaptive and current_epoch == warmup_epochs:
            print(f"  [Adaptive Loss Warmup] Unfreezing adaptive weights from epoch {current_epoch+1}")
    
    print_freq = config.get('training.print_freq', 50)
    
    for batch_idx, batch in enumerate(train_loader):
        images = batch['image'].to(device)
        labels_list = batch['label']  # Keep as list, different tasks have different shapes
        task_ids = batch['task_id']
        
        current_task_id = task_ids[0]
        task_name = task_id_to_name[current_task_id]
        
        # Stack labels based on task type (different tasks have different shapes)
        if task_name == 'segmentation':
            # Segmentation: [H, W] -> stack to [B, H, W]
            labels = torch.stack(labels_list).to(device)
        elif task_name == 'classification':
            # Classification: scalar -> stack to [B]
            labels = torch.stack(labels_list).to(device)
        elif task_name in ['detection', 'Regression']:
            # Detection/Regression: [N] -> stack to [B, N]
            labels = torch.stack(labels_list).to(device)
        else:
            labels = torch.stack(labels_list).to(device)
        
        # Forward pass
        outputs = model(images, task_id=current_task_id)
        
        # Compute loss
        if task_name == 'segmentation' and use_deep_supervision and isinstance(outputs, tuple):
            main_out, aux_outs = outputs
            target_size = labels.shape[-2:]
            
            # Main loss
            loss = loss_functions[task_name](main_out, labels)
            
            # Auxiliary losses
            aux_loss = 0.0
            for i, aux_out in enumerate(aux_outs):
                aux_out_upsampled = torch.nn.functional.interpolate(
                    aux_out, size=target_size, mode='bilinear', align_corners=False
                )
                aux_loss += aux_loss_weights[i] * loss_functions[task_name](aux_out_upsampled, labels)
            
            loss = loss + aux_loss
        
        elif task_name == 'detection':
            # Extract prediction at GT center grid
            B, C, H, W = outputs.shape
            gt_center_x = (labels[:, 0] + labels[:, 2]) / 2.0
            gt_center_y = (labels[:, 1] + labels[:, 3]) / 2.0
            coord_h = torch.clamp((gt_center_y * H).long(), 0, H - 1)
            coord_w = torch.clamp((gt_center_x * W).long(), 0, W - 1)
            
            final_outputs = torch.zeros((B, 5), device=device)
            for i in range(B):
                final_outputs[i] = outputs[i, :, coord_h[i], coord_w[i]]
            
            # Create target: [bbox(4), objectness(1)]
            targets = torch.cat([labels, torch.ones((B, 1), device=device)], dim=1)
            
            loss = loss_functions[task_name](final_outputs, targets)
        
        else:
            loss = loss_functions[task_name](outputs, labels)
        
        # Apply task weight (adaptive or fixed)
        if use_adaptive_loss:
            # Use adaptive weighting
            losses_dict = {task_name: loss}
            total_loss, weighted_losses, task_weights = loss_weights(losses_dict)
            epoch_task_weights[task_name].append(task_weights[task_name])
        else:
            # Use fixed weight
            task_weight = loss_weights.get(task_name, 1.0)
            total_loss = loss * task_weight
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        if config.get('training.gradient_clip', 0) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.get('training.gradient_clip'))
        
        # During warmup, don't update adaptive loss parameters
        if use_adaptive_loss and freeze_adaptive:
            # Zero out gradients for adaptive loss parameters
            for param in loss_weights.parameters():
                if param.grad is not None:
                    param.grad.zero_()
        
        optimizer.step()
        
        epoch_losses[current_task_id].append(total_loss.item())
        
        # Print progress
        if print_freq > 0 and (batch_idx + 1) % print_freq == 0:
            avg_loss = np.mean(epoch_losses[current_task_id])
            if use_adaptive_loss and task_name in epoch_task_weights and len(epoch_task_weights[task_name]) > 0:
                avg_weight = np.mean(epoch_task_weights[task_name])
                print(f"  Batch [{batch_idx+1}/{len(train_loader)}] | Task: {current_task_id} | Loss: {avg_loss:.4f} | Weight: {avg_weight:.4f}")
            else:
                print(f"  Batch [{batch_idx+1}/{len(train_loader)}] | Task: {current_task_id} | Loss: {avg_loss:.4f}")
    
    # Return both losses and weights
    if use_adaptive_loss:
        return epoch_losses, epoch_task_weights
    else:
        return epoch_losses


def main(config_path=None):
    """Main training function."""
    import time
    
    # Load configuration
    config = load_config(config_path)
    
    # Set seed
    set_seed(config.seed)
    
    print(f"\n{'='*80}")
    print(f"Multi-Task Ultrasound Image Analysis Training")
    print(f"Experiment: {config.exp_name}")
    print(f"{'='*80}\n")
    
    # Initialize training logger
    logger = TrainingLogger(
        log_dir=config.output_dir,
        experiment_name=config.exp_name
    )
    
    # Save configuration
    logger.save_config(config.config)
    
    # Build dataloaders FIRST (this will populate task_configs from dataset)
    train_loader, val_loader = build_dataloaders(config)
    
    # Build model (using the task_configs from dataset)
    model = build_model(config).to(config.device)
    
    # Build losses
    loss_functions, loss_weights = build_all_losses(config)
    
    # Build optimizer and scheduler (pass loss_weights for adaptive loss support)
    optimizer = build_optimizer(model, config, loss_weighter=loss_weights)
    scheduler = build_scheduler(optimizer, config)
    
    print(f"\n{'='*80}")
    print("Starting Training...")
    print(f"{'='*80}\n")
    
    # Initialize best validation score for model saving
    best_val_score = -float('inf')
    best_epoch = 0
    best_model_path = logger.get_experiment_dir() / 'best_model.pth'
    
    # Training loop
    for epoch in range(config.num_epochs):
        epoch_start_time = time.time()
        
        print(f"\nEpoch [{epoch+1}/{config.num_epochs}]")
        print("-" * 80)
        
        # Train
        train_result = train_epoch(
            model, train_loader, loss_functions, loss_weights,
            optimizer, config.device, config, current_epoch=epoch
        )
        
        # Handle return value (might include weights for adaptive loss)
        from losses.loss_functions import AdaptiveLossWeighter
        if isinstance(loss_weights, AdaptiveLossWeighter):
            epoch_losses, epoch_task_weights = train_result
        else:
            epoch_losses = train_result
            epoch_task_weights = None
        
        epoch_train_time = time.time() - epoch_start_time
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1} Train Loss Summary:")
        for task_id, losses in sorted(epoch_losses.items()):
            avg_loss = np.mean(losses)
            print(f"  {task_id:<30}: {avg_loss:.4f}")
        
        # Print adaptive loss weights and uncertainties if using adaptive loss
        from losses.loss_functions import AdaptiveLossWeighter
        if isinstance(loss_weights, AdaptiveLossWeighter):
            print(f"\nAdaptive Loss Weights and Uncertainties:")
            task_weights = loss_weights.get_task_weights()
            sigmas = loss_weights.get_sigmas()
            for task_name in sorted(task_weights.keys()):
                print(f"  {task_name:<20}: weight={task_weights[task_name]:.4f}, sigma={sigmas[task_name]:.4f}")
        
        # Validation
        print(f"\nRunning validation...")
        val_results_df = evaluate(model, val_loader, config.device, config.get_task_configs())
        
        # Calculate average validation score (higher is better)
        # Exclude MAE columns as lower is better for MAE
        score_cols = [col for col in val_results_df.columns 
                     if 'MAE' not in col and isinstance(val_results_df[col].iloc[0] if len(val_results_df) > 0 else 0, (int, float, np.number))]
        avg_val_score = 0
        if not val_results_df.empty and score_cols:
            avg_val_score = val_results_df[score_cols].mean().mean()
        
        print(f"\n--- Epoch {epoch+1} Validation Report ---")
        if not val_results_df.empty:
            print(val_results_df.to_string(index=False))
        print(f"--- Average Validation Score (Higher is better): {avg_val_score:.4f} ---")
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log epoch metrics
        epoch_total_time = time.time() - epoch_start_time
        
        # Prepare adaptive weights for logging
        adaptive_weights = None
        from losses.loss_functions import AdaptiveLossWeighter
        if isinstance(loss_weights, AdaptiveLossWeighter):
            adaptive_weights = {
                'weights': loss_weights.get_task_weights(),
                'sigmas': loss_weights.get_sigmas()
            }
        
        logger.log_epoch(
            epoch=epoch + 1,
            train_losses=epoch_losses,
            val_results_df=val_results_df,
            learning_rate=current_lr,
            epoch_time=epoch_total_time,
            adaptive_weights=adaptive_weights
        )
        
        # Save best model
        if avg_val_score > best_val_score:
            best_val_score = avg_val_score
            best_epoch = epoch + 1
            torch.save(model.state_dict(), best_model_path)
            print(f"✓ New best model saved! Score improved to: {best_val_score:.4f}")
            print(f"  Model saved at: {best_model_path}\n")
        
        # Step scheduler
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                # Use validation score for ReduceLROnPlateau
                scheduler.step(avg_val_score)
            else:
                scheduler.step()
            
            print(f"  Learning Rate: {current_lr:.2e}")
        
        print(f"  Epoch Time: {epoch_total_time:.2f}s")
        
        # Save checkpoint
        if config.get('experiment.save_checkpoints', True):
            if (epoch + 1) % config.get('experiment.checkpoint_freq', 5) == 0:
                checkpoint_path = logger.get_experiment_dir() / f"checkpoint_epoch_{epoch+1}.pth"
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': config.config,
                    'best_val_score': best_val_score
                }
                
                # Save adaptive loss parameters if using adaptive loss
                from losses.loss_functions import AdaptiveLossWeighter
                if isinstance(loss_weights, AdaptiveLossWeighter):
                    checkpoint['adaptive_loss_state_dict'] = loss_weights.state_dict()
                
                torch.save(checkpoint, checkpoint_path)
                print(f"✓ Saved checkpoint: {checkpoint_path}")
    
    # Save final summary
    logger.save_final_summary(best_epoch=best_epoch, best_score=best_val_score)
    
    print(f"\n{'='*80}")
    print("Training Complete!")
    print(f"Best validation score: {best_val_score:.4f} (Epoch {best_epoch})")
    print(f"Best model saved at: {best_model_path}")
    print(f"Training logs saved at: {logger.get_experiment_dir()}")
    print(f"{'='*80}\n")
    
    # Generate training curves plot
    try:
        from utils.logger import plot_training_curves
        plot_training_curves(logger.get_experiment_dir())
        print("Training curves plot generated successfully.")
    except Exception as e:
        print(f"Could not generate training curves plot: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train multi-task ultrasound model')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    args = parser.parse_args()
    
    main(args.config)
