"""Loss functions for multi-task learning."""

import torch
import torch.nn as nn
import torch.nn.functional as F
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
        
        # Smooth L1 for bbox regression
        box_loss = F.smooth_l1_loss(
            pred_bbox,
            target_bbox
        )
        
        total_loss = self.cls_w * cls_loss + self.box_w * box_loss
        return total_loss


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


def build_all_losses(config):
    """
    Build all loss functions from configuration.
    
    Args:
        config: Configuration object
    
    Returns:
        loss_functions: Dictionary mapping task_name to loss function
        loss_weights: Dictionary mapping task_name to loss weight
    """
    loss_functions = {}
    loss_weights = config.get('training.loss_weights', {})
    
    # Get unique task names
    task_names = set()
    for task_cfg in config.get_task_configs():
        task_names.add(task_cfg['task_name'])
    
    # Build loss for each task type
    for task_name in task_names:
        loss_config = config.get_loss_config(task_name)
        loss_fn = build_loss_function(task_name, loss_config)
        loss_functions[task_name] = loss_fn
    
    print(f"✓ Built {len(loss_functions)} loss functions:")
    for task_name, loss_fn in loss_functions.items():
        weight = loss_weights.get(task_name, 1.0)
        print(f"  - {task_name}: {loss_fn.__class__.__name__} (weight={weight})")
    
    return loss_functions, loss_weights
