"""Loss functions for multi-task learning."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import segmentation_models_pytorch.losses as smp_losses


class DetectionLoss(nn.Module):
    """
    Detection loss for single-grid training.
    BCE for objectness + Smooth L1 for bbox regression.
    """
    
    def __init__(self, classification_weight=2.0, box_regression_weight=1.0):
        super().__init__()
        self.cls_w = classification_weight
        self.box_w = box_regression_weight
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: [B, 5] - [bbox(4), objectness_logit(1)]
            targets: [B, 5] - [bbox(4), objectness_label(1)]
        
        Returns:
            loss: Combined detection loss
        """
        pred_bbox = predictions[:, :4]  # [B, 4]
        pred_objectness = predictions[:, 4]  # [B]
        
        target_bbox = targets[:, :4]  # [B, 4]
        target_objectness = targets[:, 4]  # [B]
        
        # BCE for objectness classification
        cls_loss = F.binary_cross_entropy_with_logits(
            pred_objectness,
            target_objectness
        )
        
        # Smooth L1 for bbox regression (only for positive samples)
        pos_mask = target_objectness > 0.5
        if pos_mask.any():
            box_loss = F.smooth_l1_loss(
                pred_bbox[pos_mask],
                target_bbox[pos_mask]
            )
        else:
            box_loss = pred_bbox.new_tensor(0.0)
        
        total_loss = self.cls_w * cls_loss + self.box_w * box_loss
        return total_loss


class CenterNetLoss(nn.Module):
    """CenterNet-style loss for heatmap + size + offset."""

    def __init__(self, heatmap_alpha=2.0, heatmap_gamma=4.0, size_weight=1.0, offset_weight=1.0):
        super().__init__()
        self.heatmap_loss = FocalLoss(alpha=heatmap_alpha, gamma=heatmap_gamma)
        self.size_weight = size_weight
        self.offset_weight = offset_weight

    def forward(self, predictions, targets):
        """
        Args:
            predictions: dict with heatmap [B,1,H,W], size [B,2,H,W], offset [B,2,H,W]
            targets: dict with heatmap, size, offset, mask [B,1,H,W]
        """
        pred_heatmap = predictions['heatmap']
        pred_size = predictions['size']
        pred_offset = predictions['offset']

        tgt_heatmap = targets['heatmap']
        tgt_size = targets['size']
        tgt_offset = targets['offset']
        mask = targets['mask']

        heatmap_loss = self.heatmap_loss(pred_heatmap, tgt_heatmap)

        if mask.sum() > 0:
            size_loss = F.l1_loss(pred_size * mask, tgt_size * mask, reduction='sum') / (mask.sum() + 1e-6)
            offset_loss = F.l1_loss(pred_offset * mask, tgt_offset * mask, reduction='sum') / (mask.sum() + 1e-6)
        else:
            size_loss = pred_size.new_tensor(0.0)
            offset_loss = pred_offset.new_tensor(0.0)

        return heatmap_loss + self.size_weight * size_loss + self.offset_weight * offset_loss


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance."""
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss


def build_loss_function(task_name, loss_config):
    """
    Build loss function from configuration.
    
    Args:
        task_name: Name of the task
        loss_config: Loss configuration dict for this task
    
    Returns:
        loss_fn: Loss function
    """
    loss_type = loss_config.get('type', '')
    
    if task_name == 'segmentation':
        if loss_type == 'DiceLoss':
            mode = loss_config.get('mode', 'multiclass')
            loss_fn = smp_losses.DiceLoss(mode=mode)
        elif loss_type == 'CrossEntropyLoss':
            loss_fn = nn.CrossEntropyLoss()
        else:
            loss_fn = smp_losses.DiceLoss(mode='multiclass')
    
    elif task_name == 'classification':
        loss_fn = nn.CrossEntropyLoss()
    
    elif task_name == 'detection':
        loss_type = loss_config.get('type', 'CenterNet')
        if loss_type.lower() == 'centernet':
            loss_fn = CenterNetLoss(
                heatmap_alpha=loss_config.get('heatmap_alpha', 2.0),
                heatmap_gamma=loss_config.get('heatmap_gamma', 4.0),
                size_weight=loss_config.get('size_weight', 1.0),
                offset_weight=loss_config.get('offset_weight', 1.0)
            )
        else:
            cls_weight = loss_config.get('classification_weight', 2.0)
            box_weight = loss_config.get('box_regression_weight', 1.0)
            loss_fn = DetectionLoss(
                classification_weight=cls_weight,
                box_regression_weight=box_weight
            )
    
    elif task_name == 'Regression':
        if loss_type == 'L1Loss':
            loss_fn = nn.L1Loss()
        elif loss_type == 'SmoothL1Loss':
            loss_fn = nn.SmoothL1Loss()
        else:
            loss_fn = nn.MSELoss()
    
    else:
        raise ValueError(f"Unknown task name: {task_name}")
    
    return loss_fn


class AdaptiveLossWeighter(nn.Module):
    """
    Adaptive loss weighting for multi-task learning using uncertainty estimation.
    
    Based on "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics"
    (Kendall et al., CVPR 2018)
    
    The loss weight for each task is computed as:
        weight = 1 / (2 * sigma^2)
    And a regularization term log(sigma) is added to prevent sigma from going to zero.
    
    Args:
        task_names: List of task names
        init_log_vars: Initial values for log(sigma^2). Default: 0.0 (sigma=1.0)
    """
    
    def __init__(self, task_names, init_log_vars=0.0):
        super().__init__()
        self.task_names = task_names
        
        # Learnable log variance parameters (one per task)
        # Using log(sigma^2) for numerical stability
        if isinstance(init_log_vars, (int, float)):
            init_log_vars = [init_log_vars] * len(task_names)
        
        self.log_vars = nn.ParameterDict({
            task_name: nn.Parameter(torch.tensor(init_val, dtype=torch.float32))
            for task_name, init_val in zip(task_names, init_log_vars)
        })
    
    def forward(self, losses_dict):
        """
        Compute weighted loss with uncertainty.
        
            weighted_losses: Dictionary of weighted losses for logging
            task_weights: Dictionary of computed weights for logging
        """
        total_loss = 0.0
        weighted_losses = {}
        task_weights = {}
        
        for task_name, loss in losses_dict.items():
            if task_name not in self.log_vars:
                # If task not in log_vars (shouldn't happen), use weight=1.0
                if loss.ndim != 0:
                    loss = loss.mean()
                weighted_loss = loss
                task_weights[task_name] = 1.0
            else:
                log_var = self._stable_log_var(self.log_vars[task_name])
                
                # Compute precision (inverse variance): 1 / (2 * sigma^2) = exp(-log_var) / 2
                precision = torch.exp(-log_var)
                
                # Weighted loss: (1 / (2 * sigma^2)) * loss + log(sigma)
                # = (exp(-log_var) / 2) * loss + 0.5 * log_var
                if loss.ndim != 0:
                    loss = loss.mean()
                weighted_loss = 0.5 * precision * loss + 0.5 * log_var

                task_weights[task_name] = (0.5 * precision).item()
            
            weighted_losses[task_name] = weighted_loss
            total_loss = total_loss + weighted_loss
        
        return total_loss, weighted_losses, task_weights
    
    def get_task_weights(self):
        """Get current task weights as a dictionary."""
        weights = {}
        for task_name, log_var in self.log_vars.items():
            log_var = self._stable_log_var(log_var)
            precision = torch.exp(-log_var)
            weights[task_name] = (0.5 * precision).item()
        return weights
    
    def get_sigmas(self):
        """Get current sigma values (uncertainty) as a dictionary."""
        sigmas = {}
        for task_name, log_var in self.log_vars.items():
            log_var = self._stable_log_var(log_var)
            sigma = torch.exp(0.5 * log_var)
            sigmas[task_name] = sigma.item()
        return sigmas

    @staticmethod
    def _stable_log_var(log_var):
        # Smoothly bound log_var to avoid zero gradients at hard clamp limits.
        # Range: [-3, 3] -> sigma in [0.22, 4.48].
        return 3.0 * torch.tanh(log_var / 3.0)


def build_all_losses(config):
    """
    Build all loss functions from configuration.
    
    Args:
        config: Configuration object
    
    Returns:
        loss_functions: Dictionary mapping task_name to loss function
        loss_weights: Dictionary mapping task_name to loss weight (or AdaptiveLossWeighter)
    """
    loss_functions = {}
    
    # Get unique task names
    task_names = set()
    for task_cfg in config.get_task_configs():
        task_names.add(task_cfg['task_name'])
    
    # Build loss for each task type
    for task_name in task_names:
        loss_config = config.get_loss_config(task_name)
        loss_fn = build_loss_function(task_name, loss_config)
        loss_functions[task_name] = loss_fn
    
    # Check if using adaptive loss weighting
    use_adaptive_loss = config.get('training.adaptive_loss.enabled', False)
    
    if use_adaptive_loss:
        # Check for per-task initialization
        init_log_vars_per_task = config.get('training.adaptive_loss.init_log_vars_per_task', None)
        
        if init_log_vars_per_task:
            # Use per-task initialization
            init_log_vars = [init_log_vars_per_task.get(task_name, 0.0) for task_name in task_names]
            print(f"✓ Built {len(loss_functions)} loss functions with Adaptive Loss Weighting (per-task init):")
            for task_name, init_val in zip(task_names, init_log_vars):
                sigma = np.exp(0.5 * init_val)
                weight = 0.5 * np.exp(-init_val)
                print(f"  - {task_name}: init_log_var={init_val:.2f}, sigma={sigma:.2f}, weight={weight:.2f}")
        else:
            # Use same initialization for all tasks
            init_log_vars = config.get('training.adaptive_loss.init_log_vars', 0.0)
            sigma = np.exp(0.5 * init_log_vars)
            weight = 0.5 * np.exp(-init_log_vars)
            print(f"✓ Built {len(loss_functions)} loss functions with Adaptive Loss Weighting:")
            print(f"  Initial: log_var={init_log_vars:.2f}, sigma={sigma:.2f}, weight={weight:.2f}")
            for task_name, loss_fn in loss_functions.items():
                print(f"  - {task_name}: {loss_fn.__class__.__name__}")
        
        loss_weighter = AdaptiveLossWeighter(list(task_names), init_log_vars)
        
        return loss_functions, loss_weighter
    else:
        loss_weights = config.get('training.loss_weights', {})
        
        print(f"✓ Built {len(loss_functions)} loss functions:")
        for task_name, loss_fn in loss_functions.items():
            weight = loss_weights.get(task_name, 1.0)
            print(f"  - {task_name}: {loss_fn.__class__.__name__} (weight={weight})")
        
        return loss_functions, loss_weights
