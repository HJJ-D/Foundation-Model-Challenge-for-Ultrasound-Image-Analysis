import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from model_factory import TASK_CONFIGURATIONS  # Needed for task name mapping

# Focal Loss for handling class imbalance in detection
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Focal Loss for binary classification
        Args:
            alpha: Weighting factor in range (0,1) to balance positive/negative examples
            gamma: Exponent of the modulating factor (1 - p_t)^gamma
            reduction: 'mean' or 'sum'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # pt = p if target=1, 1-p if target=0
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss

def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def multi_task_collate_fn(batch):
    """
    Custom collate function to handle different label shapes in multi-task learning.
    Images are stacked; labels and task_ids remain as lists.
    """
    images = [item['image'] for item in batch]
    labels = [item['label'] for item in batch]
    task_ids = [item['task_id'] for item in batch]
    
    # Stack images as they have consistent dimensions
    images = torch.stack(images, 0)
    
    return {'image': images, 'label': labels, 'task_id': task_ids}

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
        Both bbox are already in [0,1] range. Objectness in predictions is logit.
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

# --- Metric Calculations ---

def calculate_accuracy(y_true, y_pred_logits):
    y_pred = torch.argmax(y_pred_logits, dim=1).cpu().numpy()
    y_true = y_true.cpu().numpy()
    return accuracy_score(y_true, y_pred)

def calculate_f1_score(y_true, y_pred_logits):
    y_pred = torch.argmax(y_pred_logits, dim=1).cpu().numpy()
    y_true = y_true.cpu().numpy()
    return f1_score(y_true, y_pred, average='macro', zero_division=0)

def calculate_dice_coefficient(y_true, y_pred_logits):
    y_pred_mask = torch.argmax(y_pred_logits, dim=1)
    num_classes = y_pred_logits.shape[1]
    y_true_one_hot = F.one_hot(y_true, num_classes=num_classes).permute(0, 3, 1, 2)
    y_pred_one_hot = F.one_hot(y_pred_mask, num_classes=num_classes).permute(0, 3, 1, 2)
    intersection = torch.sum(y_true_one_hot[:, 1:] * y_pred_one_hot[:, 1:])
    union = torch.sum(y_true_one_hot[:, 1:]) + torch.sum(y_pred_one_hot[:, 1:])
    dice = (2. * intersection + 1e-6) / (union + 1e-6)
    return dice.item()

def calculate_mae(y_true, y_pred, image_size=(224, 224)):
    h, w = image_size
    y_true_px = y_true.cpu().numpy().copy()
    y_pred_px = y_pred.cpu().numpy().copy()
    y_true_px[:, 0::2] *= w; y_true_px[:, 1::2] *= h
    y_pred_px[:, 0::2] *= w; y_pred_px[:, 1::2] *= h
    return np.mean(np.abs(y_true_px - y_pred_px))

def calculate_iou(y_true, y_pred):
    y_true = y_true.cpu().numpy(); y_pred = y_pred.cpu().numpy()
    batch_ious = []
    for i in range(y_true.shape[0]):
        box_true, box_pred = y_true[i], y_pred[i]
        xA = max(box_true[0], box_pred[0]); yA = max(box_true[1], box_pred[1])
        xB = min(box_true[2], box_pred[2]); yB = min(box_true[3], box_pred[3])
        inter_area = max(0, xB - xA) * max(0, yB - yA)
        box_true_area = (box_true[2] - box_true[0]) * (box_true[3] - box_true[1])
        box_pred_area = (box_pred[2] - box_pred[0]) * (box_pred[3] - box_pred[1])
        union_area = box_true_area + box_pred_area - inter_area
        iou = inter_area / (union_area + 1e-6)
        batch_ious.append(iou)
    return np.mean(batch_ious)

def evaluate(model, val_loader, device):
    """
    Evaluation loop supporting multi-task batches.
    """
    model.eval()
    task_metrics = defaultdict(lambda: defaultdict(list))
    task_id_to_name = {cfg['task_id']: cfg['task_name'] for cfg in TASK_CONFIGURATIONS}
    
    with torch.no_grad():
        loop = tqdm(val_loader, desc="[Validation]")
        for batch in loop:
            images = batch['image'].to(device)
            labels = batch['label']
            task_ids = batch['task_id']

            unique_tasks_in_batch = set(task_ids)

            for task_id in unique_tasks_in_batch:
                task_indices = [i for i, t_id in enumerate(task_ids) if t_id == task_id]
                task_images = images[task_indices]
                
                # Extract and stack labels for the current task
                task_labels_list = [labels[i] for i in task_indices]
                task_labels = torch.stack(task_labels_list, 0)
                
                outputs = model(task_images, task_id=task_id)
                task_name = task_id_to_name[task_id]
                
                # Handle deep supervision outputs (only use main output for evaluation)
                if task_name == 'segmentation' and isinstance(outputs, tuple):
                    outputs = outputs[0]  # Use only the main output
                
                if task_name == 'classification':
                    task_metrics[task_id]['Accuracy'].append(calculate_accuracy(task_labels, outputs))
                    task_metrics[task_id]['F1-Score'].append(calculate_f1_score(task_labels, outputs))
                
                elif task_name == 'segmentation':
                    task_metrics[task_id]['Dice'].append(calculate_dice_coefficient(task_labels.to(device), outputs))
                
                elif task_name == 'Regression':
                    task_metrics[task_id]['MAE (pixels)'].append(calculate_mae(task_labels, outputs))
                
                elif task_name == 'detection':
                    # Logic to extract best bounding box from grid prediction
                    batch_size, _, h, w = outputs.shape
                    scores = outputs[:, 4, :, :].view(batch_size, -1) 
                    _, best_indices = torch.max(scores, dim=1)
                    
                    best_h = best_indices // w
                    best_w = best_indices % w
                    
                    final_boxes = torch.zeros((batch_size, 4), device=device)
                    for i in range(batch_size):
                        final_boxes[i] = outputs[i, :4, best_h[i], best_w[i]]
                    
                    task_metrics[task_id]['IoU'].append(calculate_iou(task_labels, final_boxes))

    results = []
    sorted_task_ids = sorted(list(task_id_to_name.keys()))
    for task_id in sorted_task_ids:
        if task_id in task_metrics:
            task_name = task_id_to_name[task_id]
            result_row = {'Task ID': task_id, 'Task Name': task_name}
            for metric_name, values in task_metrics[task_id].items():
                result_row[metric_name] = np.mean(values)
            results.append(result_row)
    return pd.DataFrame(results)