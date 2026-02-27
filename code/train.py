"""
Multi-Task Ultrasound Image Analysis - Modular Training Script

This script provides a clean, configuration-driven training pipeline for multi-task learning.
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import pandas as pd

# Import modular components
from configs import load_config
from models import build_model
from losses import build_all_losses
from data import MultiTaskDataset, MultiTaskUniformSampler
from utils import set_seed, multi_task_collate_fn
from utils.common import gaussian_radius, draw_gaussian
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
    
    # Create full dataset first
    full_dataset = MultiTaskDataset(data_root=config.data_root, transforms=None)
    
    # Build task_configs dynamically from dataset (like original code)
    print("\nBuilding task configurations from dataset...")
    task_configs = []
    task_config_map = {}
    
    for _, row in full_dataset.dataframe.iterrows():
        task_id = row['task_id']
        if task_id not in task_config_map:
            task_config = {
                'task_id': task_id,
                'task_name': row['task_name'],
                'num_classes': int(row['num_classes'])
            }
            task_config_map[task_id] = task_config
            task_configs.append(task_config)
    
    # Optional single-task training filter
    single_task_cfg = config.get('training.single_task', {}) or {}
    if single_task_cfg.get('enabled', False):
        single_task_id = single_task_cfg.get('task_id')
        single_task_name = single_task_cfg.get('task_name')
        if single_task_id and single_task_name:
            raise ValueError("Set only one of training.single_task.task_id or task_name, not both.")
        if not single_task_id and not single_task_name:
            raise ValueError("training.single_task.task_id or task_name must be set when single-task mode is enabled.")
        if single_task_id:
            if single_task_id not in task_config_map:
                available = ", ".join(sorted(task_config_map.keys()))
                raise ValueError(f"Unknown task_id '{single_task_id}'. Available task_ids: {available}")
            task_configs = [task_config_map[single_task_id]]
            full_dataset.dataframe = full_dataset.dataframe[
                full_dataset.dataframe['task_id'] == single_task_id
            ].reset_index(drop=True)
            print(f"Single-task mode enabled by task_id: {single_task_id}")
        else:
            matching = [
                cfg for cfg in task_configs
                if str(cfg.get('task_name', '')).lower() == str(single_task_name).lower()
            ]
            if not matching:
                available_names = sorted({cfg['task_name'] for cfg in task_configs})
                raise ValueError(f"Unknown task_name '{single_task_name}'. Available task_names: {available_names}")
            task_configs = matching
            full_dataset.dataframe = full_dataset.dataframe[
                full_dataset.dataframe['task_name'].str.lower() == str(single_task_name).lower()
            ].reset_index(drop=True)
            print(f"Single-task mode enabled by task_name: {single_task_name}")

    config.set_task_configs_from_dataset(task_configs)
    print("Using dataset-derived task configurations for model/task-prompt (config tasks are overwritten at runtime).")

    print(f"Detected {len(task_configs)} tasks:")
    for cfg in sorted(task_configs, key=lambda x: x['task_id']):
        print(f"  - {cfg['task_id']}: {cfg['task_name']}, num_classes={cfg['num_classes']}")
    
   # split dataset into train and val while preserving per-task ratios
    dataset_size = len(full_dataset)
    rng = np.random.RandomState(config.seed)  # use seed for reproducibility
    df = full_dataset.dataframe
    
    train_indices = []
    val_indices = []
    for task_id, group in df.groupby('task_id'):
        group_indices = group.index.to_numpy()
        rng.shuffle(group_indices)
        group_val_size = int(len(group_indices) * config.val_split)
        val_indices.extend(group_indices[:group_val_size].tolist())
        train_indices.extend(group_indices[group_val_size:].tolist())
    
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    
    train_size = len(train_indices)
    val_size = len(val_indices)
    
    print(f"\n✓ Dataset split (seed={config.seed}):")
    print(f"  - Total samples: {dataset_size}")
    print(f"  - Train samples: {train_size} ({100*(train_size/dataset_size):.1f}%)")
    print(f"  - Val samples: {val_size} ({100*(val_size/dataset_size):.1f}%)")
    
    # create subsets
    train_dataset = MultiTaskDataset(data_root=config.data_root, transforms=train_transforms)
    train_dataset.dataframe = full_dataset.dataframe.iloc[train_indices].reset_index(drop=True)
    
    val_dataset = MultiTaskDataset(data_root=config.data_root, transforms=val_transforms)
    val_dataset.dataframe = full_dataset.dataframe.iloc[val_indices].reset_index(drop=True)
    
    # Create samplers
    train_sampler = MultiTaskUniformSampler(
        dataset=train_dataset,
        batch_size=config.batch_size,
        steps_per_epoch=config.get('training.steps_per_epoch'),
        seed=config.seed
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
    
    return train_loader, val_loader


def build_optimizer(model, config, loss_weighter=None):
    """Build optimizer with optional grouped learning rates and adaptive loss parameters."""
    use_grouped_lr = config.get('training.optimizer.use_grouped_lr', True)
    lr = float(config.learning_rate)
    weight_decay = float(config.weight_decay)
    
    if use_grouped_lr:
        encoder_params, head_params = model.get_trainable_parameters()
        encoder_lr_mult = float(config.get('training.optimizer.encoder_lr_multiplier', 0.1))
        head_lr_mult = float(config.get('training.optimizer.head_lr_multiplier', 1.0))
        
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
        adaptive_lr = float(config.get('training.adaptive_loss.learning_rate', lr))
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
        momentum = float(config.get('training.optimizer.momentum', 0.9))
        optimizer = torch.optim.SGD(param_groups, lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    print(f"✓ Optimizer: {optimizer_type}")
    
    return optimizer


def build_scheduler(optimizer, config):
    """Build learning rate scheduler."""
    scheduler_type = config.get('training.scheduler.type', 'CosineAnnealingLR')
    
    if scheduler_type == 'CosineAnnealingLR':
        T_max = int(config.get('training.scheduler.T_max', config.num_epochs))
        eta_min = float(config.get('training.scheduler.eta_min', 1e-6))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=T_max, eta_min=eta_min
        )
    elif scheduler_type == 'ReduceLROnPlateau':
        mode = config.get('training.scheduler.mode', 'max')
        factor = float(config.get('training.scheduler.factor', 0.5))
        patience = int(config.get('training.scheduler.patience', 5))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=mode, factor=factor, patience=patience
        )
    elif scheduler_type == 'StepLR':
        step_size = int(config.get('training.scheduler.step_size', 20))
        gamma = float(config.get('training.scheduler.gamma', 0.1))
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
    moe_task_stats = {}
    moe_group_stats = {}
    
    task_id_to_name = {cfg['task_id']: cfg['task_name'] for cfg in config.get_task_configs()}
    use_deep_supervision = config.get('model.heads.segmentation.use_deep_supervision', False)
    aux_loss_weights = [float(w) for w in config.get('model.heads.segmentation.aux_loss_weights', [0.5, 0.3, 0.2])]
    moe_balance_weight = float(config.get('model.moe.balance_loss_weight', 0.0))
    
    # Check if using adaptive loss
    from losses.loss_functions import AdaptiveLossWeighter
    use_adaptive_loss = isinstance(loss_weights, AdaptiveLossWeighter)
    if use_adaptive_loss:
        loss_weights.train()  # Ensure it's in training mode
        
        # Warmup: freeze adaptive weights for first N epochs
        warmup_epochs = int(config.get('training.adaptive_loss.warmup_epochs', 0))
        freeze_adaptive = current_epoch < warmup_epochs
        if freeze_adaptive and current_epoch == 0:
            print(f"  [Adaptive Loss Warmup] Freezing adaptive weights for first {warmup_epochs} epochs")
        elif not freeze_adaptive and current_epoch == warmup_epochs:
            print(f"  [Adaptive Loss Warmup] Unfreezing adaptive weights from epoch {current_epoch+1}")
    
    print_freq = int(config.get('training.print_freq', 50))

    def _update_moe_stats(stats_dict, key, task_name, importance, load, aux_val):
        if key not in stats_dict:
            stats_dict[key] = {
                'task_name': task_name,
                'importance_sum': importance.copy(),
                'load_sum': load.copy(),
                'count': 1,
                'aux_sum': float(aux_val) if aux_val is not None else 0.0,
                'aux_count': 1 if aux_val is not None else 0,
            }
            return
        entry = stats_dict[key]
        entry['importance_sum'] += importance
        entry['load_sum'] += load
        entry['count'] += 1
        if aux_val is not None:
            entry['aux_sum'] += float(aux_val)
            entry['aux_count'] += 1
    
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

        # MoE stats aggregation (per task_id and task_name)
        if hasattr(model, "get_moe_stats"):
            moe_stats = model.get_moe_stats()
            if moe_stats:
                importance = torch.stack([s["importance"] for s in moe_stats]).mean(dim=0)
                load = torch.stack([s["load"] for s in moe_stats]).mean(dim=0)
                importance = importance.detach().cpu().numpy()
                load = load.detach().cpu().numpy()
                aux_val = None
                if hasattr(model, "get_moe_aux_loss"):
                    aux_val = float(model.get_moe_aux_loss().detach().cpu())
                _update_moe_stats(moe_task_stats, current_task_id, task_name, importance, load, aux_val)
                _update_moe_stats(moe_group_stats, task_name, task_name, importance, load, aux_val)
        
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
            if isinstance(outputs, dict) and 'heatmap' in outputs:
                # CenterNet-style targets
                heatmap = outputs['heatmap']
                size = outputs['size']
                offset = outputs['offset']
                B, _, H, W = heatmap.shape

                target_heatmap = torch.zeros_like(heatmap)
                target_size = torch.zeros_like(size)
                target_offset = torch.zeros_like(offset)
                target_mask = torch.zeros((B, 1, H, W), device=device)

                valid_mask = (labels >= 0).all(dim=1)
                for i in range(B):
                    if not valid_mask[i]:
                        continue
                    x1, y1, x2, y2 = labels[i]
                    cx = (x1 + x2) * 0.5
                    cy = (y1 + y2) * 0.5
                    gw = torch.clamp((cx * W).long(), 0, W - 1)
                    gh = torch.clamp((cy * H).long(), 0, H - 1)
                    target_size[i, 0, gh, gw] = (x2 - x1) * W   # box_w in feature cells
                    target_size[i, 1, gh, gw] = (y2 - y1) * H   # box_h in feature cells

                    target_offset[i, 0, gh, gw] = cx * W - gw.float()
                    target_offset[i, 1, gh, gw] = cy * H - gh.float()
                    target_mask[i, 0, gh, gw] = 1.0
                    box_h = (y2 - y1) * H
                    box_w = (x2 - x1) * W
                    radius = int(max(1, gaussian_radius((box_h.item(), box_w.item()))))
                    draw_gaussian(target_heatmap[i, 0], (int(gw.item()), int(gh.item())), radius)

                targets = {
                    'heatmap': target_heatmap,
                    'size': target_size,
                    'offset': target_offset,
                    'mask': target_mask
                }
                loss = loss_functions[task_name](outputs, targets)
            else:
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
                valid_mask = (labels >= 0).all(dim=1)
                labels_clean = labels.clone()
                labels_clean[~valid_mask] = 0.0
                targets = torch.cat([labels_clean, valid_mask.float().unsqueeze(1)], dim=1)
                
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

        # Add MoE load-balancing loss (Switch-style)
        if moe_balance_weight > 0 and hasattr(model, "get_moe_aux_loss"):
            moe_aux_loss = model.get_moe_aux_loss()
            total_loss = total_loss + moe_balance_weight * moe_aux_loss
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        gradient_clip = float(config.get('training.gradient_clip', 0))
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        
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

            # MoE stats (importance/load) summary
            if hasattr(model, "get_moe_stats"):
                moe_stats = model.get_moe_stats()
                if moe_stats:
                    importance = torch.stack([s["importance"] for s in moe_stats]).mean(dim=0)
                    load = torch.stack([s["load"] for s in moe_stats]).mean(dim=0)
                    importance = importance.detach().cpu().numpy()
                    load = load.detach().cpu().numpy()
                    imp_str = ", ".join([f"{v:.2f}" for v in importance])
                    load_str = ", ".join([f"{v:.2f}" for v in load])
                    moe_aux = model.get_moe_aux_loss() if hasattr(model, "get_moe_aux_loss") else None
                    if moe_aux is not None:
                        moe_aux_val = float(moe_aux.detach().cpu())
                        print(f"    MoE avg importance: [{imp_str}] | avg load: [{load_str}] | aux: {moe_aux_val:.4f}")
                    else:
                        print(f"    MoE avg importance: [{imp_str}] | avg load: [{load_str}]")
    
    # Return both losses and weights
    def _finalize_moe_stats(stats_dict):
        output = {}
        for key, entry in stats_dict.items():
            count = entry['count']
            if count == 0:
                continue
            importance_mean = (entry['importance_sum'] / count).tolist()
            load_mean = (entry['load_sum'] / count).tolist()
            out_entry = {
                'task_name': entry['task_name'],
                'importance': importance_mean,
                'load': load_mean,
            }
            if entry['aux_count'] > 0:
                out_entry['aux_loss'] = float(entry['aux_sum'] / entry['aux_count'])
            output[key] = out_entry
        return output

    moe_stats_output = None
    if moe_task_stats or moe_group_stats:
        moe_stats_output = {
            'by_task_id': _finalize_moe_stats(moe_task_stats),
            'by_task_name': _finalize_moe_stats(moe_group_stats),
        }
    if use_adaptive_loss:
        return epoch_losses, epoch_task_weights, moe_stats_output
    else:
        return epoch_losses, moe_stats_output


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
    if isinstance(loss_weights, nn.Module):
        loss_weights = loss_weights.to(config.device)
    
    # Build optimizer and scheduler (pass loss_weights for adaptive loss support)
    optimizer = build_optimizer(model, config, loss_weighter=loss_weights)
    scheduler = build_scheduler(optimizer, config)
    
    print(f"\n{'='*80}")
    print("Starting Training...")
    print(f"{'='*80}\n")
    
    # Initialize best validation score for model saving
    best_val_score = -float('inf')
    best_epoch = 0
    best_model_eval_on_train = None
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
            epoch_losses, epoch_task_weights, moe_stats = train_result
        else:
            epoch_losses, moe_stats = train_result
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
        
        # Calculate average validation score (higher is better).
        # Per-task scoring:
        #   classification : (Accuracy + F1-Score) / 2  (avoids double-counting)
        #   segmentation   : Dice
        #   detection      : IoU
        #   Regression     : (upper_bound - MAE) / (upper_bound - lower_bound), clipped to [0, 1]
        MAE_UPPER_BOUND = 100.0  # pixels — adjust if typical MAE range differs
        MAE_LOWER_BOUND = 0.0
        task_scores = []
        if not val_results_df.empty:
            for _, row in val_results_df.iterrows():
                task_name = row['Task Name']
                if task_name == 'classification':
                    acc = row.get('Accuracy', np.nan)
                    f1  = row.get('F1-Score', np.nan)
                    vals = [v for v in [acc, f1] if pd.notna(v)]
                    if vals:
                        task_scores.append(float(np.mean(vals)))
                elif task_name == 'segmentation':
                    dice = row.get('Dice', np.nan)
                    if pd.notna(dice):
                        task_scores.append(float(dice))
                elif task_name == 'detection':
                    iou = row.get('IoU', np.nan)
                    if pd.notna(iou):
                        task_scores.append(float(iou))
                elif task_name == 'Regression':
                    mae = row.get('MAE (pixels)', np.nan)
                    if pd.notna(mae):
                        normalized = (MAE_UPPER_BOUND - mae) / (MAE_UPPER_BOUND - MAE_LOWER_BOUND)
                        task_scores.append(float(np.clip(normalized, 0.0, 1.0)))
        avg_val_score = float(np.mean(task_scores)) if task_scores else 0.0
        
        print(f"\n--- Epoch {epoch+1} Validation Report ---")
        if not val_results_df.empty:
            print(val_results_df.to_string(index=False))
        print(f"--- Average Validation Score (Higher is better): {avg_val_score:.4f} ---")
        
        # Print validation summary (similar format to train summary)
        print(f"\nEpoch {epoch+1} Val Summary:")
        if not val_results_df.empty:
            for _, row in val_results_df.iterrows():
                task_id = row['Task ID']
                task_name = row['Task Name']
                # Collect key metrics for summary
                metrics_str_parts = []
                if 'Accuracy' in row and pd.notna(row['Accuracy']):
                    metrics_str_parts.append(f"Acc={row['Accuracy']:.4f}")
                if 'F1-Score' in row and pd.notna(row['F1-Score']):
                    metrics_str_parts.append(f"F1={row['F1-Score']:.4f}")
                if 'Dice' in row and pd.notna(row['Dice']):
                    metrics_str_parts.append(f"Dice={row['Dice']:.4f}")
                if 'IoU' in row and pd.notna(row['IoU']):
                    metrics_str_parts.append(f"IoU={row['IoU']:.4f}")
                if 'MAE (pixels)' in row and pd.notna(row['MAE (pixels)']):
                    metrics_str_parts.append(f"MAE={row['MAE (pixels)']:.4f}")
                metrics_str = ", ".join(metrics_str_parts) if metrics_str_parts else "N/A"
                print(f"  {task_id:<30}: {metrics_str}")
        
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
            adaptive_weights=adaptive_weights,
            moe_stats=moe_stats,
        )
        
        # Save best model and best model summary
        if avg_val_score > best_val_score:
            best_val_score = avg_val_score
            best_epoch = epoch + 1
            torch.save(model.state_dict(), best_model_path)
            # Defer best-model training evaluation to the end of training
                     
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

    # Evaluate best model on training set at the end
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, map_location=config.device))
        train_eval_df = evaluate(model, train_loader, config.device, config.get_task_configs())

        # Calculate average scores for each task group
        best_model_eval_on_train = {}
        task_groups = {
           "classification": ["Accuracy", "F1-Score"],  # Include both Accuracy and F1-Score for classification
           "segmentation": ["Dice"],
           "detection": ["IoU"],
           "regression": ["MAE (pixels)"]
        }

        for group_name, metrics in task_groups.items():
           group_scores = {metric: [] for metric in metrics}  # Separate scores for each metric
           for _, row in train_eval_df.iterrows():
                 for metric in metrics:
                    if metric in row and pd.notna(row[metric]):
                       group_scores[metric].append(row[metric])

           # Calculate mean for each metric
           group_means = {
               metric: np.mean(scores) if scores else None
               for metric, scores in group_scores.items()
           }

           # Store results
           if group_name == "classification":
               # Special handling for classification to store multiple metrics
               best_model_eval_on_train[group_name] = {
                   "Accuracy": group_means.get("Accuracy"),
                   "F1-Score": group_means.get("F1-Score")
               }
           else:
               # For other groups, store the first available metric
               best_model_eval_on_train[group_name] = next((v for v in group_means.values() if v is not None), None)

    # Save best model summary at the end of training
    logger._save_best_model_summary_txt(best_model_eval_on_train)
    
    print(f"\n{'='*80}")
    print("Training Complete!")
    print(f"Best validation score: {best_val_score:.4f} (Epoch {best_epoch})")
    print(f"Best model saved at: {best_model_path}")
    print(f"Training logs saved at: {logger.get_experiment_dir()}")
    print(f"{'='*80}\n")
    
    # Generate training curves plot
    try:
        from utils.logger import plot_training_curves, plot_comprehensive_training_curves
        plot_training_curves(logger.get_experiment_dir())
        plot_comprehensive_training_curves(logger.get_experiment_dir())
        print("Training curves plot generated successfully.")
    except Exception as e:
        print(f"Could not generate training curves plot: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train multi-task ultrasound model')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    args = parser.parse_args()
    
    main(args.config)
