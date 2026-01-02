"""
Metrics and evaluation utilities.
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score


def calculate_accuracy(y_true, y_pred_logits):
    """Calculate classification accuracy."""
    y_pred = torch.argmax(y_pred_logits, dim=1).cpu().numpy()
    y_true = y_true.cpu().numpy()
    return accuracy_score(y_true, y_pred)


def calculate_f1_score(y_true, y_pred_logits):
    """Calculate F1 score."""
    y_pred = torch.argmax(y_pred_logits, dim=1).cpu().numpy()
    y_true = y_true.cpu().numpy()
    return f1_score(y_true, y_pred, average='macro', zero_division=0)


def calculate_dice_coefficient(y_true, y_pred_logits):
    """Calculate Dice coefficient for segmentation."""
    y_pred_mask = torch.argmax(y_pred_logits, dim=1)
    num_classes = y_pred_logits.shape[1]
    y_true_one_hot = F.one_hot(y_true, num_classes=num_classes).permute(0, 3, 1, 2)
    y_pred_one_hot = F.one_hot(y_pred_mask, num_classes=num_classes).permute(0, 3, 1, 2)
    intersection = torch.sum(y_true_one_hot[:, 1:] * y_pred_one_hot[:, 1:])
    union = torch.sum(y_true_one_hot[:, 1:]) + torch.sum(y_pred_one_hot[:, 1:])
    dice = (2. * intersection + 1e-6) / (union + 1e-6)
    return dice.item()


def calculate_mae(y_true, y_pred, image_size=(224, 224)):
    """Calculate mean absolute error for regression (in pixels)."""
    h, w = image_size
    y_true_px = y_true.cpu().numpy().copy()
    y_pred_px = y_pred.cpu().numpy().copy()
    y_true_px[:, 0::2] *= w
    y_true_px[:, 1::2] *= h
    y_pred_px[:, 0::2] *= w
    y_pred_px[:, 1::2] *= h
    return np.mean(np.abs(y_true_px - y_pred_px))


def calculate_iou(y_true, y_pred):
    """Calculate IoU for detection."""
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    batch_ious = []
    for i in range(y_true.shape[0]):
        box_true, box_pred = y_true[i], y_pred[i]
        xA = max(box_true[0], box_pred[0])
        yA = max(box_true[1], box_pred[1])
        xB = min(box_true[2], box_pred[2])
        yB = min(box_true[3], box_pred[3])
        inter_area = max(0, xB - xA) * max(0, yB - yA)
        box_true_area = (box_true[2] - box_true[0]) * (box_true[3] - box_true[1])
        box_pred_area = (box_pred[2] - box_pred[0]) * (box_pred[3] - box_pred[1])
        union_area = box_true_area + box_pred_area - inter_area
        iou = inter_area / (union_area + 1e-6)
        batch_ious.append(iou)
    return np.mean(batch_ious)


def evaluate(model, val_loader, device, task_configs):
    """
    Evaluation loop supporting multi-task batches.
    
    Args:
        model: The multi-task model
        val_loader: Validation data loader
        device: torch device
        task_configs: List of task configuration dictionaries
        
    Returns:
        pd.DataFrame: Results dataframe with metrics for each task
    """
    model.eval()
    task_metrics = defaultdict(lambda: defaultdict(list))
    task_id_to_name = {cfg['task_id']: cfg['task_name'] for cfg in task_configs}
    
    with torch.no_grad():
        loop = tqdm(val_loader, desc="[Validation]", leave=False)
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

    # Build results dataframe
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


__all__ = [
    'calculate_accuracy',
    'calculate_f1_score',
    'calculate_dice_coefficient',
    'calculate_mae',
    'calculate_iou',
    'evaluate'
]
