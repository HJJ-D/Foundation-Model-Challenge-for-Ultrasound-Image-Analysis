"""
Post-training visualization script for multi-task ultrasound model.

Usage:
    # Visualize a specific task_id
    python visualize.py --output_dir outputs/experiment_name --task_id breast_2cls

    # Visualize all tasks of a specific type
    python visualize.py --output_dir outputs/experiment_name --task_type segmentation

    # Visualize all tasks
    python visualize.py --output_dir outputs/experiment_name

    # Custom number of random samples and dataset path
    python visualize.py --output_dir outputs/experiment_name --task_id FUGC --num_samples 8 --data_root /path/to/dataset
"""

import argparse
import os
import sys
import json
import random
import glob
import math

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from collections import defaultdict
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

# Project imports
from configs import load_config
from models import build_model
from data import MultiTaskDataset
from utils import set_seed, multi_task_collate_fn
from metrics import (
    calculate_accuracy,
    calculate_f1_score,
    calculate_dice_coefficient,
    calculate_mae,
    calculate_iou,
)


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def denormalize_image(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """Convert a normalized tensor [C, H, W] back to a displayable numpy image [H, W, 3] in [0, 1]."""
    img = tensor.cpu().float()
    for c in range(3):
        img[c] = img[c] * std[c] + mean[c]
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return img


def compute_sample_score(task_name, label, output, image_size):
    """Return a scalar score for a single sample (higher = better)."""
    if task_name == 'classification':
        pred_cls = torch.argmax(output).item()
        gt_cls = label.item()
        return 1.0 if pred_cls == gt_cls else 0.0

    elif task_name == 'segmentation':
        # Dice for a single sample — add batch dim
        return calculate_dice_coefficient(label.unsqueeze(0), output.unsqueeze(0))

    elif task_name == 'detection':
        gt = label.unsqueeze(0)
        if isinstance(output, dict) and 'heatmap' in output:
            pred_box = decode_centernet_single(output)
        else:
            pred_box = output[:4].unsqueeze(0)
        valid = (gt >= 0).all(dim=1)
        if valid.any():
            return calculate_iou(gt[valid], pred_box[valid])
        return 0.0

    elif task_name == 'Regression':
        h, w = image_size
        gt_px = label.cpu().numpy().copy()
        pred_px = output.cpu().numpy().copy()
        gt_px[0::2] *= w
        gt_px[1::2] *= h
        pred_px[0::2] *= w
        pred_px[1::2] *= h
        mae = float(np.mean(np.abs(gt_px - pred_px)))
        # Convert to "higher is better" (100 - MAE, clamped)
        return max(0.0, 100.0 - mae)

    return 0.0


def decode_centernet_single(output_dict):
    """Decode CenterNet dict output for a single sample (already batched with B=1)."""
    heatmap = output_dict['heatmap']  # [1, 1, H, W]
    size = output_dict['size']
    offset = output_dict['offset']
    _, _, h, w = heatmap.shape
    scores = heatmap.view(1, -1)
    _, best_idx = torch.max(scores, dim=1)
    best_h = best_idx // w
    best_w = best_idx % w
    off_x = offset[0, 0, best_h[0], best_w[0]]
    off_y = offset[0, 1, best_h[0], best_w[0]]
    cx = (best_w[0].float() + off_x) / w
    cy = (best_h[0].float() + off_y) / h
    box_w = size[0, 0, best_h[0], best_w[0]] / w
    box_h = size[0, 1, best_h[0], best_w[0]] / h
    x1 = cx - box_w * 0.5
    y1 = cy - box_h * 0.5
    x2 = cx + box_w * 0.5
    y2 = cy + box_h * 0.5
    return torch.stack([x1, y1, x2, y2]).clamp(0, 1).unsqueeze(0)


def decode_centernet_batch(output_dict, device):
    """Decode CenterNet dict output for a batch."""
    heatmap = output_dict['heatmap']
    size = output_dict['size']
    offset = output_dict['offset']
    B, _, h, w = heatmap.shape
    scores = heatmap.view(B, -1)
    _, best_indices = torch.max(scores, dim=1)
    best_h = best_indices // w
    best_w = best_indices % w
    boxes = torch.zeros((B, 4), device=device)
    for i in range(B):
        off_x = offset[i, 0, best_h[i], best_w[i]]
        off_y = offset[i, 1, best_h[i], best_w[i]]
        cx = (best_w[i].float() + off_x) / w
        cy = (best_h[i].float() + off_y) / h
        bw = size[i, 0, best_h[i], best_w[i]] / w
        bh = size[i, 1, best_h[i], best_w[i]] / h
        boxes[i] = torch.stack([cx - bw * 0.5, cy - bh * 0.5, cx + bw * 0.5, cy + bh * 0.5]).clamp(0, 1)
    return boxes


# ──────────────────────────────────────────────────────────────────────
# Per-task-type visualisation
# ──────────────────────────────────────────────────────────────────────

def vis_classification(ax, img, label, output, task_id, score):
    """Visualize a classification sample."""
    ax.imshow(img)
    pred_cls = torch.argmax(output).item()
    gt_cls = label.item()
    probs = F.softmax(output, dim=0)
    conf = probs[pred_cls].item()
    color = 'green' if pred_cls == gt_cls else 'red'
    ax.set_title(
        f"{task_id}\nGT={gt_cls}  Pred={pred_cls} ({conf:.2f})\nScore={score:.2f}",
        fontsize=9, color=color,
    )
    ax.axis('off')


def vis_segmentation(ax, img, label, output, task_id, score):
    """Visualize a segmentation sample: image with overlay."""
    pred_mask = torch.argmax(output, dim=0).cpu().numpy()
    gt_mask = label.cpu().numpy()
    h, w = gt_mask.shape

    # Create overlay
    overlay = img.copy()
    # Green channel for GT, Red for pred, Yellow for overlap
    gt_binary = (gt_mask > 0).astype(np.float32)
    pred_binary = (pred_mask > 0).astype(np.float32)
    overlap = gt_binary * pred_binary
    gt_only = gt_binary * (1 - overlap)
    pred_only = pred_binary * (1 - overlap)

    overlay[:, :, 1] = np.clip(overlay[:, :, 1] + gt_only * 0.4, 0, 1)   # green = GT only
    overlay[:, :, 0] = np.clip(overlay[:, :, 0] + pred_only * 0.4, 0, 1)  # red = pred only
    overlay[:, :, 0] = np.clip(overlay[:, :, 0] + overlap * 0.4, 0, 1)    # yellow = overlap
    overlay[:, :, 1] = np.clip(overlay[:, :, 1] + overlap * 0.4, 0, 1)

    ax.imshow(overlay)
    ax.set_title(f"{task_id}\nDice={score:.4f}", fontsize=9)
    ax.axis('off')


def vis_detection(ax, img, label, output, task_id, score, image_size):
    """Visualize a detection sample with GT and predicted bounding box."""
    h, w = image_size
    ax.imshow(img)

    # GT box (green)
    gt = label.cpu().numpy()
    if (gt >= 0).all():
        gx1, gy1, gx2, gy2 = gt[0] * w, gt[1] * h, gt[2] * w, gt[3] * h
        rect_gt = Rectangle((gx1, gy1), gx2 - gx1, gy2 - gy1,
                             linewidth=2, edgecolor='green', facecolor='none', linestyle='--')
        ax.add_patch(rect_gt)

    # Pred box (red)
    if isinstance(output, dict) and 'heatmap' in output:
        pred_box = decode_centernet_single(
            {k: v.unsqueeze(0) if v.dim() == 3 else v for k, v in output.items()}
        ).squeeze(0)
    else:
        pred_box = output[:4]
    pred_box = pred_box.cpu().numpy()
    px1, py1, px2, py2 = pred_box[0] * w, pred_box[1] * h, pred_box[2] * w, pred_box[3] * h
    rect_pred = Rectangle((px1, py1), px2 - px1, py2 - py1,
                           linewidth=2, edgecolor='red', facecolor='none')
    ax.add_patch(rect_pred)

    ax.set_title(f"{task_id}\nIoU={score:.4f}", fontsize=9)
    ax.legend(['GT (green)', 'Pred (red)'], fontsize=7, loc='lower right')
    ax.axis('off')


def vis_regression(ax, img, label, output, task_id, score, image_size):
    """Visualize regression keypoints on image."""
    h, w = image_size
    ax.imshow(img)
    gt_coords = label.cpu().numpy()
    pred_coords = output.cpu().numpy()

    for i in range(0, len(gt_coords), 2):
        gx, gy = gt_coords[i] * w, gt_coords[i + 1] * h
        px, py = pred_coords[i] * w, pred_coords[i + 1] * h
        ax.plot(gx, gy, 'go', markersize=8, label='GT' if i == 0 else None)
        ax.plot(px, py, 'rx', markersize=8, markeredgewidth=2, label='Pred' if i == 0 else None)
        ax.plot([gx, px], [gy, py], 'y--', linewidth=1)

    mae_px = 100.0 - score  # reverse the conversion
    ax.set_title(f"{task_id}\nMAE={mae_px:.2f}px", fontsize=9)
    ax.legend(fontsize=7, loc='lower right')
    ax.axis('off')


# ──────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────

def build_val_dataset(config, data_root):
    """Replicate the train/val split from train.py and return the val subset."""
    full_dataset = MultiTaskDataset(data_root=data_root, transforms=None)

    # Build task configs from dataset
    task_config_map = {}
    task_configs = []
    for _, row in full_dataset.dataframe.iterrows():
        tid = row['task_id']
        if tid not in task_config_map:
            tc = {'task_id': tid, 'task_name': row['task_name'], 'num_classes': int(row['num_classes'])}
            task_config_map[tid] = tc
            task_configs.append(tc)

    # Apply single-task filter if configured
    single_task_cfg = config.get('training.single_task', {}) or {}
    if single_task_cfg.get('enabled', False):
        single_task_id = single_task_cfg.get('task_id')
        single_task_name = single_task_cfg.get('task_name')
        if single_task_id:
            task_configs = [task_config_map[single_task_id]]
            full_dataset.dataframe = full_dataset.dataframe[
                full_dataset.dataframe['task_id'] == single_task_id
            ].reset_index(drop=True)
        elif single_task_name:
            task_configs = [c for c in task_configs if c['task_name'].lower() == single_task_name.lower()]
            full_dataset.dataframe = full_dataset.dataframe[
                full_dataset.dataframe['task_name'].str.lower() == single_task_name.lower()
            ].reset_index(drop=True)

    config.set_task_configs_from_dataset(task_configs)

    # Stratified split — same logic as train.py
    rng = np.random.RandomState(config.seed)
    df = full_dataset.dataframe
    val_indices = []
    for _, group in df.groupby('task_id'):
        group_idx = group.index.to_numpy()
        rng.shuffle(group_idx)
        val_size = int(len(group_idx) * config.val_split)
        val_indices.extend(group_idx[:val_size].tolist())
    rng.shuffle(val_indices)

    # Validation transforms (no augmentation, just resize + normalize)
    norm_mean = config.get('data.augmentation.normalize.mean', [0.485, 0.456, 0.406])
    norm_std = config.get('data.augmentation.normalize.std', [0.229, 0.224, 0.225])
    val_transforms = A.Compose(
        [
            A.Resize(config.image_size, config.image_size),
            A.Normalize(mean=norm_mean, std=norm_std),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], clip=True, min_visibility=0.1),
        keypoint_params=A.KeypointParams(format='xy', remove_invisible=False),
    )

    val_dataset = MultiTaskDataset(data_root=data_root, transforms=val_transforms)
    val_dataset.dataframe = full_dataset.dataframe.iloc[val_indices].reset_index(drop=True)

    return val_dataset, task_configs, norm_mean, norm_std


def filter_indices_by_task(dataset, task_id=None, task_type=None):
    """Return indices in *dataset* that match the requested filter."""
    df = dataset.dataframe
    mask = pd.Series([True] * len(df))
    if task_id is not None:
        mask &= df['task_id'] == task_id
    if task_type is not None:
        mask &= df['task_name'].str.lower() == task_type.lower()
    return df.index[mask].tolist()


def run_inference_on_indices(model, dataset, indices, device, task_configs):
    """Run model inference on given indices and return per-sample results."""
    task_id_to_name = {c['task_id']: c['task_name'] for c in task_configs}
    image_size = (dataset.transforms.transforms[0].height, dataset.transforms.transforms[0].width)

    results = []  # list of dicts: {idx, task_id, task_name, image_tensor, label, output, score}
    model.eval()
    with torch.no_grad():
        for idx in tqdm(indices, desc="Running inference"):
            sample = dataset[idx]
            img_tensor = sample['image'].unsqueeze(0).to(device)
            label = sample['label']
            tid = sample['task_id']
            tname = task_id_to_name[tid]

            output = model(img_tensor, task_id=tid)

            # Handle deep-supervision / DSNT tuple outputs
            if tname == 'segmentation' and isinstance(output, tuple):
                output = output[0]
            if tname == 'Regression' and isinstance(output, tuple):
                output = output[0]

            # For detection CenterNet, keep dict for score computation then convert
            if tname == 'detection' and isinstance(output, dict):
                pred_box = decode_centernet_batch(output, device).squeeze(0)
                raw_output = pred_box  # [4]
            else:
                raw_output = output.squeeze(0)  # remove batch dim

            score = compute_sample_score(tname, label.to(device), raw_output, image_size)

            results.append({
                'idx': idx,
                'task_id': tid,
                'task_name': tname,
                'image_tensor': sample['image'],
                'label': label,
                'output': raw_output.cpu(),
                'score': score,
            })
    return results


def visualize_results(results, norm_mean, norm_std, save_path, image_size, title="Visualization"):
    """Create and save a figure with the given results."""
    n = len(results)
    if n == 0:
        print("  No results to visualize.")
        return
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = np.array(axes).flatten()

    for i, r in enumerate(results):
        ax = axes[i]
        img = denormalize_image(r['image_tensor'], mean=norm_mean, std=norm_std)
        tname = r['task_name']
        if tname == 'classification':
            vis_classification(ax, img, r['label'], r['output'], r['task_id'], r['score'])
        elif tname == 'segmentation':
            vis_segmentation(ax, img, r['label'], r['output'], r['task_id'], r['score'])
        elif tname == 'detection':
            vis_detection(ax, img, r['label'], r['output'], r['task_id'], r['score'], image_size)
        elif tname == 'Regression':
            vis_regression(ax, img, r['label'], r['output'], r['task_id'], r['score'], image_size)

    # Hide unused axes
    for j in range(n, len(axes)):
        axes[j].axis('off')

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def visualize_task(model, dataset, task_id, task_configs, device, norm_mean, norm_std,
                   num_samples, save_dir, image_size):
    """Visualize a single task_id: random samples + best/worst."""
    indices = filter_indices_by_task(dataset, task_id=task_id)
    if not indices:
        print(f"  [SKIP] No validation samples for task_id={task_id}")
        return

    print(f"\n--- Task: {task_id} ({len(indices)} val samples) ---")

    # Run inference on ALL samples to find best/worst
    all_results = run_inference_on_indices(model, dataset, indices, device, task_configs)

    if not all_results:
        print(f"  [SKIP] No valid results for task_id={task_id}")
        return

    # Sort by score
    all_results.sort(key=lambda r: r['score'])
    worst = all_results[0]
    best = all_results[-1]

    # Random samples (exclude best/worst to avoid duplicates)
    remaining = [r for r in all_results if r['idx'] != best['idx'] and r['idx'] != worst['idx']]
    actual_num = min(num_samples, len(remaining))
    random_samples = random.sample(remaining, actual_num) if actual_num > 0 else []

    # Save random samples
    if random_samples:
        path = os.path.join(save_dir, f"{task_id}_random.png")
        visualize_results(random_samples, norm_mean, norm_std, path, image_size,
                          title=f"{task_id} — Random Samples")

    # Save best + worst
    bw_results = [
        {**best, '_tag': 'BEST'},
        {**worst, '_tag': 'WORST'},
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for i, (r, tag) in enumerate([(best, 'BEST'), (worst, 'WORST')]):
        ax = axes[i]
        img = denormalize_image(r['image_tensor'], mean=norm_mean, std=norm_std)
        tname = r['task_name']
        if tname == 'classification':
            vis_classification(ax, img, r['label'], r['output'], r['task_id'], r['score'])
        elif tname == 'segmentation':
            vis_segmentation(ax, img, r['label'], r['output'], r['task_id'], r['score'])
        elif tname == 'detection':
            vis_detection(ax, img, r['label'], r['output'], r['task_id'], r['score'], image_size)
        elif tname == 'Regression':
            vis_regression(ax, img, r['label'], r['output'], r['task_id'], r['score'], image_size)
        # Prepend tag to existing title
        existing = ax.get_title()
        ax.set_title(f"[{tag}]\n{existing}", fontsize=9, fontweight='bold')

    fig.suptitle(f"{task_id} — Best vs Worst", fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(save_dir, f"{task_id}_best_worst.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")

    # Print summary
    scores = [r['score'] for r in all_results]
    print(f"  Score range: [{min(scores):.4f}, {max(scores):.4f}], mean={np.mean(scores):.4f}")


def visualize_dataset_overview(data_root: str, save_dir: str):
    """
    Generate one composite figure per task type with GT annotations.
    All task_ids included, at most 3 per row. No model needed.
    Saves: overview_segmentation.png, overview_classification.png, etc.
    """
    csv_path = os.path.join(data_root, 'csv_files')
    all_csv_files = glob.glob(os.path.join(csv_path, '*.csv'))
    if not all_csv_files:
        print(f"ERROR: No CSV files found in {csv_path}")
        return

    df = pd.concat([pd.read_csv(f) for f in all_csv_files], ignore_index=True)

    task_configs = {}
    for _, row in df.iterrows():
        tid = row['task_id']
        if tid not in task_configs:
            task_configs[tid] = {
                'task_type': row['task_name'].lower(),
                'num_classes': int(row['num_classes']),
            }

    type_to_ids = defaultdict(list)
    for tid, cfg in task_configs.items():
        type_to_ids[cfg['task_type']].append(tid)

    os.makedirs(save_dir, exist_ok=True)

    type_display = [
        ('segmentation',   'Segmentation'),
        ('classification', 'Classification'),
        ('detection',      'Detection'),
        ('regression',     'Regression'),
    ]

    for type_key, display_name in type_display:
        if type_key not in type_to_ids:
            continue

        task_ids = sorted(type_to_ids[type_key])
        n = len(task_ids)
        cols = min(3, n)
        rows = math.ceil(n / cols)

        fig, axes = plt.subplots(rows, cols, squeeze=False,
                                 figsize=(cols * 4.5, rows * 4.5 + 0.8))
        fig.suptitle(display_name, fontsize=22, fontweight='bold', y=1.01)

        for i, task_id in enumerate(task_ids):
            ax = axes[i // cols][i % cols]
            task_data = df[df['task_id'] == task_id]
            if len(task_data) == 0:
                ax.axis('off')
                continue

            sample = task_data.sample(1, random_state=42).iloc[0]
            img_path = os.path.normpath(os.path.join(csv_path, sample['image_path']))
            image = cv2.imread(img_path)
            if image is None:
                ax.text(0.5, 0.5, 'Image not found', ha='center', va='center',
                        transform=ax.transAxes, fontsize=13)
                ax.axis('off')
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            orig_h, orig_w = image.shape[:2]
            cell_px = 256
            scale_x = cell_px / orig_w
            scale_y = cell_px / orig_h
            image = cv2.resize(image, (cell_px, cell_px), interpolation=cv2.INTER_LINEAR)

            ax.imshow(image)
            num_classes = task_configs[task_id]['num_classes']

            if type_key == 'segmentation':
                if pd.notna(sample.get('mask_path')):
                    mask_path = os.path.normpath(os.path.join(csv_path, sample['mask_path']))
                    gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    if gt_mask is not None:
                        gt_mask = cv2.resize(gt_mask, (cell_px, cell_px),
                                             interpolation=cv2.INTER_NEAREST)
                        overlay = np.ma.masked_where(gt_mask == 0, gt_mask)
                        ax.imshow(overlay, alpha=0.55, cmap='jet',
                                  vmin=0, vmax=max(num_classes - 1, 1))

            elif type_key == 'classification':
                if pd.notna(sample.get('mask')):
                    ax.text(0.04, 0.96, f'Class: {int(sample["mask"])}',
                            transform=ax.transAxes, fontsize=13, color='white',
                            fontweight='bold', va='top',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='navy', alpha=0.8))

            elif type_key == 'detection':
                bbox_cols = ['x_min', 'y_min', 'x_max', 'y_max']
                if all(c in sample.index and pd.notna(sample[c]) for c in bbox_cols):
                    x0, y0, x1, y1 = [float(sample[c]) for c in bbox_cols]
                    x0, x1 = x0 * scale_x, x1 * scale_x
                    y0, y1 = y0 * scale_y, y1 * scale_y
                    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                                           linewidth=3, edgecolor='lime', facecolor='none'))

            elif type_key == 'regression':
                gt_points = []
                for pi in range(1, num_classes + 1):
                    col = f'point_{pi}_xy'
                    if col in sample.index and pd.notna(sample[col]):
                        try:
                            gt_points.extend(json.loads(sample[col]))
                        except Exception:
                            gt_points.extend([0, 0])
                    else:
                        gt_points.extend([0, 0])
                if gt_points:
                    pts = np.array(gt_points, dtype=float).reshape(-1, 2)
                    pts[:, 0] *= scale_x
                    pts[:, 1] *= scale_y
                    ax.scatter(pts[:, 0], pts[:, 1], c='lime', s=100,
                               marker='o', edgecolors='white', linewidths=2, zorder=5)
                    for j, (x, y) in enumerate(pts):
                        ax.text(x, y - 8, str(j + 1), color='white', fontsize=11,
                                ha='center', fontweight='bold',
                                bbox=dict(boxstyle='round,pad=0.2', facecolor='green', alpha=0.8))

            ax.set_title(task_id.replace('_', ' '), fontsize=13, fontweight='bold', pad=4)
            ax.axis('off')

        for i in range(n, rows * cols):
            axes[i // cols][i % cols].axis('off')

        plt.tight_layout()
        save_path = os.path.join(save_dir, f'overview_{type_key}.png')
        plt.savefig(save_path, dpi=200, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close()
        print(f"Saved: {save_path}")


# ──────────────────────────────────────────────────────────────────────
# Model comparison on identical samples
# ──────────────────────────────────────────────────────────────────────

def find_model_checkpoint(model_dir):
    """Return path to a model checkpoint in model_dir, or None."""
    for name in ['best_model.pth', 'best_model.pt', 'model.pth', 'model.pt']:
        p = os.path.join(model_dir, name)
        if os.path.isfile(p):
            return p
    return None


def pick_fixed_sample_per_task(val_dataset, seed):
    """Deterministically pick exactly one dataset-position per task_id.

    Sorting by (task_id, image_path) + a local RandomState guarantees the
    selection is identical across independent runs regardless of global RNG.
    Returns: dict {task_id: dataset_position}.
    """
    df = val_dataset.dataframe.copy().reset_index(drop=True)
    df['_pos'] = np.arange(len(df))
    df_sorted = df.sort_values(['task_id', 'image_path']).reset_index(drop=True)
    rng = np.random.RandomState(seed)
    picks = {}
    for tid in sorted(df_sorted['task_id'].unique()):
        group = df_sorted[df_sorted['task_id'] == tid]
        choice = int(rng.randint(0, len(group)))
        picks[tid] = int(group.iloc[choice]['_pos'])
    return picks


def load_trained_model(model_dir, data_root, task_configs, device):
    """Load a trained model from its directory using its own config, but force
    the task_configs to match a shared reference set so head shapes align."""
    config_path = os.path.join(model_dir, 'config.yaml')
    if not os.path.isfile(config_path):
        print(f"  [SKIP] config.yaml not found in {model_dir}")
        return None
    ckpt_path = find_model_checkpoint(model_dir)
    if ckpt_path is None:
        print(f"  [SKIP] no checkpoint found in {model_dir}")
        return None

    cfg = load_config(config_path)
    cfg.config['data']['root_path'] = data_root
    cfg.data_root = data_root
    cfg.config.setdefault('training', {}).setdefault('single_task', {})['enabled'] = False
    cfg.set_task_configs_from_dataset(task_configs)

    model = build_model(cfg).to(device)
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [WARN] {len(missing)} missing keys when loading {ckpt_path}")
    if unexpected:
        print(f"  [WARN] {len(unexpected)} unexpected keys when loading {ckpt_path}")
    model.eval()
    print(f"  Loaded checkpoint: {ckpt_path}")
    return model


def _render_panel(ax, result, norm_mean, norm_std, image_size):
    """Dispatch to the correct vis_* function based on task_name."""
    img = denormalize_image(result['image_tensor'], mean=norm_mean, std=norm_std)
    tname = result['task_name']
    if tname == 'classification':
        vis_classification(ax, img, result['label'], result['output'],
                           result['task_id'], result['score'])
    elif tname == 'segmentation':
        vis_segmentation(ax, img, result['label'], result['output'],
                         result['task_id'], result['score'])
    elif tname == 'detection':
        vis_detection(ax, img, result['label'], result['output'],
                      result['task_id'], result['score'], image_size)
    elif tname == 'Regression':
        vis_regression(ax, img, result['label'], result['output'],
                       result['task_id'], result['score'], image_size)


def _row_best_col(model_labels, results_by_label, task_id):
    """Return index of the model that scored highest for this task, or -1 if none."""
    scores = []
    for lbl in model_labels:
        r = results_by_label.get(lbl, {}).get(task_id)
        scores.append(r['score'] if r is not None else float('-inf'))
    if not scores or max(scores) == float('-inf'):
        return -1
    return int(np.argmax(scores))


def visualize_comparison_per_type(all_results, task_configs, save_dir,
                                   norm_mean, norm_std, image_size, model_labels):
    """One figure per task type. Rows = task_ids, columns = models. Marks the
    best-performing model per row with a ★ in its title."""
    type_to_ids = defaultdict(list)
    for tc in task_configs:
        type_to_ids[tc['task_name'].lower()].append(tc['task_id'])

    for type_key, task_ids in type_to_ids.items():
        task_ids = sorted(task_ids)
        n_tasks = len(task_ids)
        n_models = len(model_labels)
        if n_tasks == 0 or n_models == 0:
            continue

        cell = 3.8
        fig, axes = plt.subplots(n_tasks, n_models, squeeze=False,
                                 figsize=(cell * n_models, cell * n_tasks))

        for col, label in enumerate(model_labels):
            axes[0][col].annotate(label,
                                  xy=(0.5, 1.18), xycoords='axes fraction',
                                  ha='center', va='bottom',
                                  fontsize=14, fontweight='bold')

        for row, tid in enumerate(task_ids):
            best_col = _row_best_col(model_labels, all_results, tid) if n_models > 1 else -1
            for col, label in enumerate(model_labels):
                ax = axes[row][col]
                r = all_results.get(label, {}).get(tid)
                if r is None:
                    ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                            transform=ax.transAxes)
                    ax.axis('off')
                    continue
                _render_panel(ax, r, norm_mean, norm_std, image_size)
                if col == best_col:
                    existing = ax.get_title()
                    ax.set_title(f"★ {existing}", fontsize=9, fontweight='bold',
                                 color='darkgreen')

        fig.suptitle(f"Model Comparison — {type_key.capitalize()}",
                     fontsize=18, fontweight='bold', y=1.0)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        save_path = os.path.join(save_dir, f'compare_{type_key}.png')
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {save_path}")


def visualize_comparison_per_task(all_results, task_configs, save_dir,
                                   norm_mean, norm_std, image_size, model_labels):
    """One small figure per task (1 × N_models). Convenient for paper insets."""
    per_task_dir = os.path.join(save_dir, 'per_task')
    os.makedirs(per_task_dir, exist_ok=True)
    task_ids = sorted({tc['task_id'] for tc in task_configs})
    for tid in task_ids:
        if not any(tid in all_results.get(lbl, {}) for lbl in model_labels):
            continue
        n_models = len(model_labels)
        cell = 4.0
        fig, axes = plt.subplots(1, n_models, squeeze=False,
                                 figsize=(cell * n_models, cell))
        best_col = _row_best_col(model_labels, all_results, tid) if n_models > 1 else -1
        for col, label in enumerate(model_labels):
            ax = axes[0][col]
            r = all_results.get(label, {}).get(tid)
            if r is None:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                        transform=ax.transAxes)
                ax.axis('off')
                continue
            _render_panel(ax, r, norm_mean, norm_std, image_size)
            existing = ax.get_title()
            tag = f"[{label}]"
            if col == best_col:
                tag = f"★ {tag}"
            ax.set_title(f"{tag}\n{existing}", fontsize=9, fontweight='bold',
                         color='darkgreen' if col == best_col else 'black')
        fig.suptitle(tid, fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        save_path = os.path.join(per_task_dir, f'{tid}.png')
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    print(f"  Per-task figures saved to: {per_task_dir}")


def compare_models_on_same_samples(model_dirs, model_labels, data_root, save_dir,
                                    compare_seed, device):
    """Run multiple checkpoints on a deterministic 1-per-task validation subset
    and render side-by-side comparison figures.

    The picked images are reproducible across independent runs because:
      (1) the val split is rebuilt from the first model's config seed, and
      (2) `compare_seed` drives a local RandomState applied to a sort-stable
          (task_id, image_path) ordering of the val set.
    """
    if not model_dirs:
        print("ERROR: No model directories provided for comparison.")
        return

    first_cfg_path = os.path.join(model_dirs[0], 'config.yaml')
    if not os.path.isfile(first_cfg_path):
        print(f"ERROR: config.yaml not found in {model_dirs[0]}")
        return
    base_config = load_config(first_cfg_path)
    base_config.config['data']['root_path'] = data_root
    base_config.data_root = data_root
    # Compare needs all task heads; disable single-task filter from the ref config
    base_config.config.setdefault('training', {}).setdefault('single_task', {})['enabled'] = False

    set_seed(base_config.seed)
    val_dataset, task_configs, norm_mean, norm_std = build_val_dataset(base_config, data_root)
    print(f"Reference val set: {len(val_dataset)} samples across {len(task_configs)} tasks")

    picks = pick_fixed_sample_per_task(val_dataset, seed=compare_seed)
    print(f"Picked {len(picks)} samples (1 per task) with compare_seed={compare_seed}")
    sample_indices = list(picks.values())

    all_results = {}
    for mdir, label in zip(model_dirs, model_labels):
        print(f"\n=== Model: {label}  ({mdir}) ===")
        model = load_trained_model(mdir, data_root, task_configs, device)
        if model is None:
            continue
        results = run_inference_on_indices(model, val_dataset, sample_indices,
                                           device, task_configs)
        all_results[label] = {r['task_id']: r for r in results}
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not all_results:
        print("No models loaded successfully; nothing to visualize.")
        return

    os.makedirs(save_dir, exist_ok=True)

    # Manifest pins the exact images that were used — makes the run auditable
    manifest = {}
    for tid, pos in picks.items():
        row = val_dataset.dataframe.iloc[pos]
        manifest[tid] = {
            'position': int(pos),
            'image_path': str(row['image_path']),
            'task_name': str(row['task_name']),
        }
    with open(os.path.join(save_dir, 'compare_manifest.json'), 'w', encoding='utf-8') as f:
        json.dump({'compare_seed': compare_seed,
                   'models': dict(zip(model_labels, model_dirs)),
                   'samples': manifest},
                  f, indent=2, ensure_ascii=False)

    image_size = (base_config.image_size, base_config.image_size)
    valid_labels = [lbl for lbl in model_labels if lbl in all_results]
    print("\nRendering comparison figures...")
    visualize_comparison_per_type(all_results, task_configs, save_dir,
                                  norm_mean, norm_std, image_size, valid_labels)
    visualize_comparison_per_task(all_results, task_configs, save_dir,
                                  norm_mean, norm_std, image_size, valid_labels)
    print(f"\nDone! Comparison results saved to: {save_dir}")


# ──────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Visualize multi-task model predictions')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Path to the training output folder containing best_model.pth and config.yaml')
    parser.add_argument('--task_id', type=str, default=None,
                        help='Specific task_id to visualize (e.g., breast_2cls)')
    parser.add_argument('--task_type', type=str, default=None,
                        help='Task type to visualize: segmentation, classification, detection, Regression')
    parser.add_argument('--num_samples', type=int, default=4,
                        help='Number of random samples to visualize per task (default: 4)')
    parser.add_argument('--data_root', type=str,
                        default=r'/proj/berzbiomedicalimagingkth/users/x_hujun/Foundation-Model-Challenge-for-Ultrasound-Image-Analysis/dataset/',
                        help='Path to the dataset root directory')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (default: auto-detect)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for sampling (default: use config seed)')
    parser.add_argument('--overview_only', action='store_true',
                        help='Generate dataset overview with GT annotations only (no model needed)')
    parser.add_argument('--save_dir', type=str, default='visualizations/overview',
                        help='Output directory for --overview_only / --compare_models figures')
    parser.add_argument('--compare_models', type=str, nargs='+', default=None,
                        help='Compare multiple models on the SAME deterministically-selected '
                             'test images (one per task_id, 27 in total). Pass the output dirs '
                             'space-separated, e.g. --compare_models outputs/expA outputs/expB')
    parser.add_argument('--compare_labels', type=str, nargs='+', default=None,
                        help='Optional display labels for --compare_models (must match count). '
                             'Defaults to folder basenames.')
    parser.add_argument('--compare_seed', type=int, default=42,
                        help='Seed for deterministic sample picking in compare mode. Same seed '
                             '+ same val split = same 27 images across independent runs.')
    args = parser.parse_args()

    # ── Model comparison mode (multiple checkpoints, identical samples) ──
    if args.compare_models:
        labels = (args.compare_labels
                  or [os.path.basename(os.path.normpath(d)) for d in args.compare_models])
        if len(labels) != len(args.compare_models):
            print(f"ERROR: --compare_labels count ({len(labels)}) must match "
                  f"--compare_models count ({len(args.compare_models)})")
            sys.exit(1)
        if args.device:
            device = torch.device(args.device)
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {device}")

        # Use a comparison-specific default save dir when user kept the overview default
        save_dir = args.save_dir
        if save_dir == 'visualizations/overview':
            save_dir = 'visualizations/comparison'

        compare_models_on_same_samples(
            model_dirs=args.compare_models,
            model_labels=labels,
            data_root=args.data_root,
            save_dir=save_dir,
            compare_seed=args.compare_seed,
            device=device,
        )
        return

    # ── GT-only dataset overview mode ────────────────────────────────
    if args.overview_only or args.output_dir is None:
        print("Generating dataset overview (GT annotations only, no model required)...")
        visualize_dataset_overview(args.data_root, args.save_dir)
        print(f"\nDone! Overview figures saved to: {args.save_dir}")
        return

    # ── Locate model and config ──────────────────────────────────────
    output_dir = args.output_dir
    config_path = os.path.join(output_dir, 'config.yaml')
    model_path = os.path.join(output_dir, 'best_model.pth')

    if not os.path.isfile(config_path):
        print(f"ERROR: config.yaml not found in {output_dir}")
        sys.exit(1)
    if not os.path.isfile(model_path):
        # Try common alternative names
        for alt in ['best_model.pt', 'model.pth', 'model.pt']:
            alt_path = os.path.join(output_dir, alt)
            if os.path.isfile(alt_path):
                model_path = alt_path
                break
        else:
            print(f"ERROR: No model checkpoint found in {output_dir}")
            sys.exit(1)

    print(f"Config:  {config_path}")
    print(f"Model:   {model_path}")

    # ── Load config ──────────────────────────────────────────────────
    config = load_config(config_path)

    # Override data root to local dataset path
    config.config['data']['root_path'] = args.data_root
    config.data_root = args.data_root

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device:  {device}")

    # Seed
    seed = args.seed if args.seed is not None else config.seed
    set_seed(seed)

    # ── Build validation dataset (same split as training) ────────────
    val_dataset, task_configs, norm_mean, norm_std = build_val_dataset(config, args.data_root)
    task_id_to_name = {c['task_id']: c['task_name'] for c in task_configs}

    print(f"\nValidation set: {len(val_dataset)} samples")
    print(f"Tasks: {len(task_configs)}")

    # ── Build and load model ─────────────────────────────────────────
    print("\nBuilding model...")
    model = build_model(config).to(device)
    state_dict = torch.load(model_path, map_location=device)
    # Handle state_dict wrapped in checkpoint dict
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print("Model loaded successfully.")

    # ── Determine which tasks to visualize ───────────────────────────
    if args.task_id:
        if args.task_id not in task_id_to_name:
            print(f"ERROR: Unknown task_id '{args.task_id}'. Available: {sorted(task_id_to_name.keys())}")
            sys.exit(1)
        target_task_ids = [args.task_id]
    elif args.task_type:
        target_task_ids = [
            tid for tid, tname in task_id_to_name.items()
            if tname.lower() == args.task_type.lower()
        ]
        if not target_task_ids:
            available_types = sorted(set(task_id_to_name.values()))
            print(f"ERROR: No tasks with type '{args.task_type}'. Available types: {available_types}")
            sys.exit(1)
    else:
        target_task_ids = sorted(task_id_to_name.keys())

    print(f"Visualizing {len(target_task_ids)} task(s): {target_task_ids}")

    # ── Create save directory ────────────────────────────────────────
    save_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(save_dir, exist_ok=True)

    image_size = (config.image_size, config.image_size)

    # ── Run visualization for each task ──────────────────────────────
    for tid in target_task_ids:
        visualize_task(
            model=model,
            dataset=val_dataset,
            task_id=tid,
            task_configs=task_configs,
            device=device,
            norm_mean=norm_mean,
            norm_std=norm_std,
            num_samples=args.num_samples,
            save_dir=save_dir,
            image_size=image_size,
        )

    print(f"\nDone! All visualizations saved to: {save_dir}")


if __name__ == '__main__':
    main()
