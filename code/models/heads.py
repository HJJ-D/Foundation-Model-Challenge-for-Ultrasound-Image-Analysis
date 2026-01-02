"""Task-specific head implementations."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


class SegmentationHead(nn.Module):
    """Standard segmentation head."""
    
    def __init__(self, in_channels, num_classes, kernel_size=1, upsampling=4):
        super().__init__()
        self.head = smp.base.SegmentationHead(
            in_channels=in_channels,
            out_channels=num_classes,
            kernel_size=kernel_size,
            upsampling=upsampling
        )
    
    def forward(self, x):
        return self.head(x)


class DeepSupervisionSegHead(nn.Module):
    """Segmentation head with deep supervision for multi-level auxiliary outputs."""
    
    def __init__(self, fpn_out_channels, num_classes, num_aux_outputs=3, upsampling=4):
        super().__init__()
        self.num_aux_outputs = num_aux_outputs
        
        # Main segmentation head with upsampling
        self.main_head = smp.base.SegmentationHead(
            in_channels=fpn_out_channels,
            out_channels=num_classes,
            kernel_size=1,
            upsampling=upsampling
        )
        
        # Auxiliary heads (no upsampling, will be upsampled during loss computation)
        self.aux_heads = nn.ModuleList([
            smp.base.SegmentationHead(
                in_channels=fpn_out_channels,
                out_channels=num_classes,
                kernel_size=1,
                upsampling=1  # No upsampling
            )
            for _ in range(num_aux_outputs)
        ])
    
    def forward(self, fpn_features):
        """
        Args:
            fpn_features: Output from FPN decoder [B, C, H, W]
        
        Returns:
            main_out: Main segmentation output [B, num_classes, H*4, W*4]
            aux_outs: List of auxiliary outputs, each [B, num_classes, H, W]
        """
        main_out = self.main_head(fpn_features)
        aux_outs = [aux_head(fpn_features) for aux_head in self.aux_heads]
        return main_out, aux_outs


class ClassificationHead(nn.Module):
    """Wrapper for SMP Classification Head."""
    
    def __init__(self, encoder_channels, num_classes, dropout=0.2):
        super().__init__()
        in_channels = encoder_channels[-1]
        self.head = smp.base.ClassificationHead(
            in_channels=in_channels,
            classes=num_classes,
            pooling="avg",
            dropout=dropout,
            activation=None,
        )
    
    def forward(self, features):
        """
        Args:
            features: List of encoder outputs, use the last one
        """
        # Use the last feature map from encoder
        return self.head(features[-1])


class DetectionHead(nn.Module):
    """Detection head designed for FPN outputs."""
    
    def __init__(self, fpn_out_channels, num_classes=1, mid_channels=128, num_anchors=1):
        super().__init__()
        num_outputs = num_anchors * (4 + num_classes)
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(fpn_out_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, num_outputs, kernel_size=1)
        )
    
    def forward(self, fpn_features):
        """
        Args:
            fpn_features: FPN decoder output [B, C, H, W]
        
        Returns:
            predictions_map: [B, 5, H, W] where 5 = [bbox(4), objectness(1)]
        """
        predictions_map = self.conv_block(fpn_features)
        
        # Apply sigmoid to bbox coordinates (first 4 channels)
        predictions_map[:, :4] = torch.sigmoid(predictions_map[:, :4])
        
        return predictions_map


class RegressionHead(nn.Module):
    """Custom head for regression tasks (coordinate prediction)."""
    
    def __init__(self, encoder_channels, num_points):
        """
        Args:
            encoder_channels: List of encoder output channels
            num_points: Number of points to predict (output will be num_points * 2 for x,y)
        """
        super().__init__()
        in_channels = encoder_channels[-1]
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        # Output dimension is num_points * 2 (x, y coordinates)
        self.linear = nn.Linear(in_channels, num_points * 2)

    def forward(self, features):
        """
        Args:
            features: List of encoder outputs, use the last one
        """
        x = self.pooling(features[-1])
        x = self.flatten(x)
        return self.linear(x)


def build_task_head(task_config, fpn_out_channels, encoder_channels, model_config):
    """
    Build task-specific head from configuration.
    
    Args:
        task_config: Single task configuration dict
        fpn_out_channels: FPN decoder output channels
        encoder_channels: Encoder output channels list
        model_config: Model configuration dict
    
    Returns:
        head: Task-specific head module
    """
    task_name = task_config['task_name']
    num_classes = task_config['num_classes']
    
    if task_name == 'segmentation':
        head_cfg = model_config.get('heads', {}).get('segmentation', {})
        use_deep_supervision = head_cfg.get('use_deep_supervision', False)
        
        if use_deep_supervision:
            num_aux = head_cfg.get('num_aux_outputs', 3)
            upsampling = head_cfg.get('upsampling', 4)
            head = DeepSupervisionSegHead(
                fpn_out_channels=fpn_out_channels,
                num_classes=num_classes,
                num_aux_outputs=num_aux,
                upsampling=upsampling
            )
        else:
            head = SegmentationHead(
                in_channels=fpn_out_channels,
                num_classes=num_classes,
                upsampling=head_cfg.get('upsampling', 4)
            )
    
    elif task_name == 'classification':
        head_cfg = model_config.get('heads', {}).get('classification', {})
        head = ClassificationHead(
            encoder_channels=encoder_channels,
            num_classes=num_classes,
            dropout=head_cfg.get('dropout', 0.2)
        )
    
    elif task_name == 'detection':
        head_cfg = model_config.get('heads', {}).get('detection', {})
        head = DetectionHead(
            fpn_out_channels=fpn_out_channels,
            num_classes=num_classes,
            mid_channels=head_cfg.get('mid_channels', 128),
            num_anchors=head_cfg.get('num_anchors', 1)
        )
    
    elif task_name == 'Regression':
        # num_classes is actually num_points for regression tasks
        num_points = num_classes
        head = RegressionHead(
            encoder_channels=encoder_channels,
            num_points=num_points
        )
    
    else:
        raise ValueError(f"Unknown task type: {task_name}")
    
    return head


def build_all_heads(task_configs, fpn_out_channels, encoder_channels, model_config):
    """
    Build all task heads from configuration.
    
    Args:
        task_configs: List of task configuration dicts
        fpn_out_channels: FPN decoder output channels
        encoder_channels: Encoder output channels list
        model_config: Model configuration dict
    
    Returns:
        heads: nn.ModuleDict of task heads
    """
    heads = nn.ModuleDict()
    
    for task_cfg in task_configs:
        task_id = task_cfg['task_id']
        head = build_task_head(task_cfg, fpn_out_channels, encoder_channels, model_config)
        heads[task_id] = head
    
    print(f"✓ Built {len(heads)} task-specific heads")
    
    return heads
