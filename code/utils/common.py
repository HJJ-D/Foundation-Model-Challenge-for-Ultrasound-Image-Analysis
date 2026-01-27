"""Utility functions."""

import random
import numpy as np
import torch


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


def count_parameters(model):
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_lr(optimizer):
    """Get current learning rate from optimizer."""
    for param_group in optimizer.param_groups:
        return param_group['lr']


def gaussian_radius(det_size, min_overlap=0.7):
    """Compute Gaussian radius for CenterNet-style heatmaps."""
    height, width = det_size

    a1 = 1
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(max(0.0, b1 ** 2 - 4 * a1 * c1))
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(max(0.0, b2 ** 2 - 4 * a2 * c2))
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(max(0.0, b3 ** 2 - 4 * a3 * c3))
    r3 = (b3 + sq3) / 2

    return min(r1, r2, r3)


def draw_gaussian(heatmap, center, radius, k=1.0):
    """Draw a 2D Gaussian on the heatmap at the given center."""
    if radius <= 0:
        x, y = center
        if 0 <= x < heatmap.shape[1] and 0 <= y < heatmap.shape[0]:
            heatmap[y, x] = max(heatmap[y, x], k)
        return heatmap

    diameter = 2 * radius + 1
    sigma = diameter / 6.0

    y = torch.arange(0, diameter, device=heatmap.device, dtype=heatmap.dtype)
    x = torch.arange(0, diameter, device=heatmap.device, dtype=heatmap.dtype)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    gaussian = torch.exp(-((xx - radius) ** 2 + (yy - radius) ** 2) / (2 * sigma ** 2))

    x_center, y_center = center
    height, width = heatmap.shape
    left = min(x_center, radius)
    right = min(width - x_center - 1, radius)
    top = min(y_center, radius)
    bottom = min(height - y_center - 1, radius)

    masked_heatmap = heatmap[y_center - top:y_center + bottom + 1, x_center - left:x_center + right + 1]
    masked_gaussian = gaussian[radius - top:radius + bottom + 1, radius - left:radius + right + 1]
    torch.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap
