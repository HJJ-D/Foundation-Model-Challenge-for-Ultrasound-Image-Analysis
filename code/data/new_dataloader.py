import os
import cv2
import pandas as pd
import numpy as np
import torch
import glob
import random
import json
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm
from typing import Optional, Iterator, List
import albumentations as A

class MultiTaskDataset(Dataset):
    def __init__(self, data_root: str, transforms: Optional[A.Compose] = None, 
                 use_adaptive_norm: bool = False, bg_threshold: int = 'auto'):
        super().__init__()
        self.data_root = data_root
        self.transforms = transforms
        self.use_adaptive_norm = use_adaptive_norm  # Per-image normalization
        self.bg_threshold = bg_threshold  # Background detection: 'auto', 'adaptive', or number
        self.csv_path = os.path.join(self.data_root, 'csv_files')
        
        if not os.path.isdir(self.csv_path):
            raise FileNotFoundError(f"CSV path not found: {self.csv_path}")
            
        all_csv_files = glob.glob(os.path.join(self.csv_path, '*.csv'))
        if not all_csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.csv_path}")
            
        df_list = [pd.read_csv(csv_file) for csv_file in all_csv_files]
        self.dataframe = pd.concat(df_list, ignore_index=True).reset_index(drop=True)
        print(f"Data loaded. Total samples: {len(self.dataframe)}")
        
        # Normalization configuration check
        if self.use_adaptive_norm:
            print(f"✓ Using adaptive per-image normalization (bg_threshold={bg_threshold})")
            print("  WARNING: Ensure transforms does NOT contain A.Normalize, or set mean=[0,0,0], std=[1,1,1]")
            print("  to avoid double normalization!")
            self._check_normalize_conflict()
        else:
            print("✓ Using standard preprocessing (no per-image normalization)")
            print("  Recommended: Use A.Normalize in transforms with domain-specific stats")
    
    def _check_normalize_conflict(self):
        """Check if transforms contains A.Normalize that might conflict with adaptive norm"""
        if self.transforms is None:
            return
        
        # Recursively check for Normalize in transform pipeline
        def has_normalize(transform_list):
            for t in transform_list:
                if isinstance(t, A.Normalize):
                    # Check if it's NOT identity normalization
                    if not (t.mean == [0, 0, 0] and t.std == [1, 1, 1]):
                        return True
                # Check nested transforms (like Compose, OneOf, etc.)
                if hasattr(t, 'transforms'):
                    if has_normalize(t.transforms):
                        return True
            return False
        
        if hasattr(self.transforms, 'transforms'):
            if has_normalize(self.transforms.transforms):
                print("  ⚠️  CONFLICT DETECTED: Found A.Normalize with non-identity params in transforms!")
                print("      This will cause double normalization. Please remove it or use mean=[0,0,0], std=[1,1,1]")

    def __len__(self) -> int:
        return len(self.dataframe)
    
    def _detect_valid_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Detect valid region in ultrasound images (excluding black background)
        
        Args:
            image: Grayscale-replicated 3-channel image (H, W, 3), uint8 or float32
        
        Returns:
            mask: Boolean array, True indicates valid region
        """
        # Extract grayscale channel (all channels are identical after preprocessing)
        if image.dtype == np.uint8:
            gray = image[:, :, 0]
        else:
            # float32 case - convert to uint8 for thresholding
            gray = (image[:, :, 0] * 255).astype(np.uint8)
        
        # Determine threshold
        threshold = self.bg_threshold
        if threshold == 'auto':
            # Use Otsu automatic thresholding
            _, mask_rough = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
            if mask_rough.sum() > 0:
                otsu_thresh, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                threshold_val = max(otsu_thresh * 0.5, 10)
            else:
                threshold_val = 10
        elif threshold == 'adaptive':
            # Adaptive thresholding
            mask = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 51, -10
            )
            mask = mask > 0
            threshold_val = None
        else:
            threshold_val = threshold
        
        # Simple threshold segmentation
        if threshold_val is not None:
            mask = gray > threshold_val
        
        # Morphological operations: remove small noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Safety check
        if mask.sum() < mask.size * 0.1:
            mask = gray > 5
        
        return mask.astype(bool)
    
    def _adaptive_normalize(self, image: np.ndarray) -> np.ndarray:
        """
        Per-image standardization on valid region ONLY (background stays 0)
        
        Note: 
        - This method performs FULL normalization (mean=0, std=1 on valid region)
        - Do NOT use A.Normalize in transforms when this is enabled (to avoid double normalization)
        - If you must use A.Normalize, set mean=[0,0,0], std=[1,1,1] to make it identity
        - For ultrasound images, all 3 channels have identical values (grayscale replication)
        
        Args:
            image: RGB image (H, W, 3), uint8 [0, 255] - ultrasound images have identical channel values
        
        Returns:
            normalized image: float32, standardized in valid region, 0 in background
        """
        # Detect valid region
        valid_mask = self._detect_valid_mask(image)
        
        # Convert to float [0, 1]
        image_float = image.astype(np.float32) / 255.0
        
        # Calculate mean/std on valid region and standardize ONLY valid pixels
        # Background pixels remain 0
        if valid_mask.sum() > 0:
            for c in range(3):
                valid_pixels = image_float[:, :, c][valid_mask]
                if len(valid_pixels) > 0:
                    mean = valid_pixels.mean()
                    std = valid_pixels.std()
                    if std < 1e-6:
                        std = 1.0
                    # Standardize ONLY valid region: (x - mean) / std
                    image_float[:, :, c][valid_mask] = (valid_pixels - mean) / std
                    # Background stays 0 (implicit: pixels where valid_mask=False)
        
        return image_float

    def __getitem__(self, idx: int) -> dict:
        record = self.dataframe.iloc[idx]
        task_id = record['task_id']
        task_name = record['task_name']
        
        # Load image with Unicode path support
        image_abs_path = os.path.normpath(os.path.join(self.csv_path, record['image_path']))
        try:
            # Use numpy to read file (supports Unicode paths on Windows)
            image_stream = np.fromfile(image_abs_path, dtype=np.uint8)
            image = cv2.imdecode(image_stream, cv2.IMREAD_COLOR)
        except:
            image = None
        
        # Robustness check: retry next index if image load fails
        if image is None:
            print(f"Warning: Failed to load image: {image_abs_path}")
            return self.__getitem__((idx + 1) % len(self))
        
        # Ultrasound image processing strategy:
        # Convert ALL images to grayscale, including colormap-rendered ones
        # Rationale:
        #   - Most ultrasound is grayscale B-mode imaging
        #   - Some images may use colormap (jet/hot/turbo) for visualization
        #   - Colormap is just a rendering choice, not real color information
        #   - Converting to grayscale unifies the representation and removes
        #     colormap artifacts, allowing the model to learn intrinsic features
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Convert BGR to grayscale using standard luminance formula
            # This handles both pseudo-RGB and colormap-rendered images
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Replicate grayscale to 3 channels (required by pretrained models like Swin/DINOv2)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Apply adaptive per-image normalization if enabled
        # WARNING: Do NOT use A.Normalize in transforms when this is enabled
        if self.use_adaptive_norm:
            image = self._adaptive_normalize(image)  # Returns float32, already normalized
        
        # Save original image size BEFORE any transforms (needed for Regression only)
        original_height, original_width = image.shape[:2]

        # Load raw labels based on task
        label = None
        mask = None
        bboxes = []
        class_labels = []

        if task_name == 'segmentation':
            if pd.notna(record.get('mask_path')):
                mask_path = os.path.normpath(os.path.join(self.csv_path, record['mask_path']))
                try:
                    mask_stream = np.fromfile(mask_path, dtype=np.uint8)
                    mask = cv2.imdecode(mask_stream, cv2.IMREAD_GRAYSCALE)
                except:
                    mask = None
        
        elif task_name == 'classification':
            label = int(record['mask'])

        elif task_name == 'Regression':
            num_points = record['num_classes']
            coords = []
            for i in range(1, num_points + 1):
                col = f'point_{i}_xy'
                if col in record and pd.notna(record[col]):
                    coords.extend(json.loads(record[col]))
                else:
                    coords.extend([0, 0])
            label = np.array(coords, dtype=np.float32)

        elif task_name == 'detection':
            cols = ['x_min', 'y_min', 'x_max', 'y_max']
            if all(c in record and pd.notna(record[c]) for c in cols):
                box_coords = [float(record[c]) for c in cols]
                # Validate bbox: ensure x_max > x_min and y_max > y_min
                if box_coords[2] > box_coords[0] and box_coords[3] > box_coords[1]:
                    bboxes = [box_coords]
                    class_labels = [0]
                else:
                    # Invalid bbox, will be handled as missing detection
                    print(f"Warning: Invalid bbox at idx {idx}: {box_coords}, skipping...")

        # Apply augmentations
        if self.transforms:
            augmented = self.transforms(image=image, mask=mask, bboxes=bboxes, class_labels=class_labels)
            image = augmented['image']
            
            if task_name == 'segmentation':
                label = augmented.get('mask')
            elif task_name == 'detection':
                if augmented['bboxes']:
                    label = np.array(augmented['bboxes'][0][:4], dtype=np.float32)
                else:
                    label = np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32)

        # Format conversion & normalization
        final_label = None
        if isinstance(image, torch.Tensor):
            h, w = int(image.shape[1]), int(image.shape[2])  # CHW
        else:
            h, w = int(image.shape[0]), int(image.shape[1])  # HWC

        # Ensure label is numpy for processing
        if isinstance(label, torch.Tensor):
            label = label.cpu().numpy()

        if task_name == 'segmentation':
            if label is None:
                label = np.zeros((h, w), dtype=np.int64)
            final_label = torch.from_numpy(label).long()

        elif task_name == 'classification':
            final_label = torch.tensor(label).long()

        elif task_name in ['Regression', 'detection']:
            if not isinstance(label, np.ndarray):
                label = np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32)

            # Normalize coordinates to [0, 1]
            # CRITICAL: After albumentations transforms (especially Resize), 
            # bbox coordinates are in the TRANSFORMED image space, not original!
            # So we must normalize by the CURRENT image dimensions (h, w), not original dimensions.
            if task_name == 'detection' and np.all(label >= 0):
                label[[0, 2]] /= w  # Use current width (after transforms)
                label[[1, 3]] /= h  # Use current height (after transforms)
            elif task_name == 'Regression':
                # Regression uses original dimensions because points are not transformed by albumentations
                label[0::2] /= original_width
                label[1::2] /= original_height
            
            final_label = torch.from_numpy(label).float()
        
        return {'image': image, 'label': final_label, 'task_id': task_id}


class MultiTaskUniformSampler(Sampler[List[int]]):
    def __init__(self, dataset: MultiTaskDataset, batch_size: int, steps_per_epoch: Optional[int] = None, seed: Optional[int] = None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.indices_by_task = {}
        self.rng = random.Random(seed)

        # Group indices by task_id
        print("\n--- Initializing Sampler ---")
        for idx, task_id in enumerate(tqdm(dataset.dataframe['task_id'], desc="Grouping indices")):
            if task_id not in self.indices_by_task:
                self.indices_by_task[task_id] = []
            self.indices_by_task[task_id].append(idx)
            
        self.task_ids = list(self.indices_by_task.keys())
        
        # Initial shuffle
        for task_id in self.task_ids:
            self.rng.shuffle(self.indices_by_task[task_id])

        # Determine epoch length
        if steps_per_epoch is None:
            self.steps_per_epoch = len(self.dataset) // self.batch_size
        else:
            self.steps_per_epoch = steps_per_epoch

    def __iter__(self) -> Iterator[List[int]]:
        task_cursors = {task_id: 0 for task_id in self.task_ids}

        for _ in range(self.steps_per_epoch):
            # Randomly select a task
            task_id = self.rng.choice(self.task_ids)
            indices = self.indices_by_task[task_id]
            cursor = task_cursors[task_id]
            
            start_idx = cursor
            end_idx = start_idx + self.batch_size
            
            if end_idx > len(indices):
                # Wrap around
                batch_indices = indices[start_idx:]
                self.rng.shuffle(indices)
                remaining = self.batch_size - len(batch_indices)
                batch_indices.extend(indices[:remaining])
                task_cursors[task_id] = remaining
            else:
                batch_indices = indices[start_idx:end_idx]
                task_cursors[task_id] = end_idx
            
            yield batch_indices
            
    def __len__(self) -> int:
        return self.steps_per_epoch
