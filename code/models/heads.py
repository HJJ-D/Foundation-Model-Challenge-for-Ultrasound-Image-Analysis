"""Task-specific head implementations."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


def _gn_groups(channels):
    groups = min(32, channels)
    while channels % groups != 0:
        groups -= 1
    return groups


class SegmentationHead(nn.Module):
    """Standard segmentation head with a deeper pre-head stack."""

    def __init__(self, in_channels, num_classes, kernel_size=1, upsampling=4, mid_channels=None, num_layers=2):
        super().__init__()
        if mid_channels is None:
            mid_channels = in_channels

        layers = []
        cur_channels = in_channels
        for _ in range(num_layers):
            layers.append(nn.Conv2d(cur_channels, mid_channels, kernel_size=3, padding=1, bias=False))
            layers.append(nn.GroupNorm(_gn_groups(mid_channels), mid_channels))
            layers.append(nn.SiLU(inplace=True))
            cur_channels = mid_channels

        self.pre_head = nn.Identity() if not layers else nn.Sequential(*layers)
        self.head = smp.base.SegmentationHead(
            in_channels=cur_channels,
            out_channels=num_classes,
            kernel_size=kernel_size,
            upsampling=upsampling
        )

    def forward(self, x):
        x = self.pre_head(x)
        return self.head(x)


class UNetLikeSegHead(nn.Module):
    """UNet-like upsampling head to refine boundaries."""

    def __init__(self, in_channels, num_classes, mid_channels=None, upsampling=4, num_blocks=2):
        super().__init__()
        if mid_channels is None:
            mid_channels = in_channels

        blocks = []
        cur_channels = in_channels
        scale = upsampling
        while scale > 1:
            blocks.append(nn.Conv2d(cur_channels, mid_channels, kernel_size=3, padding=1, bias=False))
            blocks.append(nn.GroupNorm(_gn_groups(mid_channels), mid_channels))
            blocks.append(nn.SiLU(inplace=True))
            blocks.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
            cur_channels = mid_channels
            scale //= 2

        for _ in range(max(0, num_blocks - 1)):
            blocks.append(nn.Conv2d(cur_channels, mid_channels, kernel_size=3, padding=1, bias=False))
            blocks.append(nn.GroupNorm(_gn_groups(mid_channels), mid_channels))
            blocks.append(nn.SiLU(inplace=True))

        self.up_path = nn.Sequential(*blocks)
        self.out_conv = nn.Conv2d(cur_channels, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.up_path(x)
        return self.out_conv(x)


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
    """Wrapper for SMP Classification Head with optional MLP."""
    
    def __init__(self, encoder_channels, num_classes, dropout=0.2, mlp_hidden_dim=None, in_channels=None):
        super().__init__()
        if in_channels is None:
            in_channels = encoder_channels[-1]
        if mlp_hidden_dim:
            self.pre_fc = nn.Sequential(
                nn.Linear(in_channels, mlp_hidden_dim),
                nn.SiLU(inplace=True),
                nn.Dropout(p=dropout),
            )
            head_in = mlp_hidden_dim
        else:
            self.pre_fc = None
            head_in = in_channels
        self.head = smp.base.ClassificationHead(
            in_channels=head_in,
            classes=num_classes,
            pooling="avg",
            dropout=dropout,
            activation=None,
        )
    
    def forward(self, features):
        """
        Args:
            features: Encoder feature list or FPN feature map
        """
        if isinstance(features, (list, tuple)):
            x = features[-1]
        else:
            x = features
        if self.pre_fc is not None:
            x = F.adaptive_avg_pool2d(x, 1).flatten(1)
            x = self.pre_fc(x)
            return self.head(x.unsqueeze(-1).unsqueeze(-1))
        return self.head(x)


class DetectionHead(nn.Module):
    """Detection head designed for FPN outputs with lightweight attention."""
    
    def __init__(self, fpn_out_channels, num_classes=1, mid_channels=128, num_anchors=1):
        super().__init__()
        num_outputs = num_anchors * (4 + num_classes)
        
        # Initial projection
        self.input_conv = nn.Sequential(
            nn.Conv2d(fpn_out_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_gn_groups(mid_channels), mid_channels),
            nn.ReLU(),
        )
        
        # Feature refinement with residual connection
        self.refine_conv = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_gn_groups(mid_channels), mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_gn_groups(mid_channels), mid_channels),
        )
        
        # Lightweight channel attention (SE-like, but simpler)
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(mid_channels, mid_channels // 4),
            nn.ReLU(),
            nn.Linear(mid_channels // 4, mid_channels),
            nn.Sigmoid(),
        )
        
        # Output projection
        self.output_conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(mid_channels, num_outputs, kernel_size=1)
        )
    
    def forward(self, fpn_features, targets=None, use_giou=False):
        """
        Args:
            fpn_features: FPN decoder output [B, C, H, W]
            targets: (Optional) Ground truth boxes, shape [B, 4, H, W] or [B, 4, N]
            use_giou: (Optional) If True, return GIoU loss if targets is not None
        Returns:
            predictions_map: [B, 5, H, W] where 5 = [bbox(4), objectness(1)]
            (Optional) giou_loss: if use_giou and targets is not None
        """
        # Initial projection
        x = self.input_conv(fpn_features)
        # Feature refinement with residual
        residual = x
        x = self.refine_conv(x)
        # Apply channel attention
        attn = self.channel_attn(x).view(x.size(0), -1, 1, 1)
        x = x * attn
        # Residual connection
        x = x + residual
        # Output prediction
        predictions_map = self.output_conv(x)
        # Apply sigmoid to bbox coordinates (first 4 channels)
        predictions_map[:, :4] = torch.sigmoid(predictions_map[:, :4])
        if use_giou and targets is not None:
            giou_loss = self.giou_loss(predictions_map[:, :4], targets)
            return predictions_map, giou_loss
        return predictions_map

    @staticmethod
    def giou_loss(preds, targets, eps=1e-7):
        """
        Generalized IoU loss for bounding boxes.
        Args:
            preds: [B, 4, H, W] predicted boxes (x1, y1, x2, y2) normalized [0,1]
            targets: [B, 4, H, W] ground truth boxes (x1, y1, x2, y2) normalized [0,1]
        Returns:
            giou_loss: scalar
        """
        # Reshape to [N, 4]
        B, _, H, W = preds.shape
        preds = preds.permute(0, 2, 3, 1).reshape(-1, 4)
        targets = targets.permute(0, 2, 3, 1).reshape(-1, 4)

        # Convert (cx, cy, w, h) to (x1, y1, x2, y2) if needed
        # Here we assume input is already (x1, y1, x2, y2) normalized

        # Intersection
        x1 = torch.max(preds[:, 0], targets[:, 0])
        y1 = torch.max(preds[:, 1], targets[:, 1])
        x2 = torch.min(preds[:, 2], targets[:, 2])
        y2 = torch.min(preds[:, 3], targets[:, 3])
        inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

        # Areas
        area_pred = (preds[:, 2] - preds[:, 0]).clamp(min=0) * (preds[:, 3] - preds[:, 1]).clamp(min=0)
        area_gt = (targets[:, 2] - targets[:, 0]).clamp(min=0) * (targets[:, 3] - targets[:, 1]).clamp(min=0)

        # Union
        union = area_pred + area_gt - inter + eps
        iou = inter / union

        # Smallest enclosing box
        xc1 = torch.min(preds[:, 0], targets[:, 0])
        yc1 = torch.min(preds[:, 1], targets[:, 1])
        xc2 = torch.max(preds[:, 2], targets[:, 2])
        yc2 = torch.max(preds[:, 3], targets[:, 3])
        area_c = (xc2 - xc1).clamp(min=0) * (yc2 - yc1).clamp(min=0) + eps

        giou = iou - (area_c - union) / area_c
        loss = 1 - giou
        return loss.mean()


class CenterNetDetectionHead(nn.Module):
    """Anchor-free CenterNet-style detection head."""

    def __init__(self, fpn_out_channels, mid_channels=128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(fpn_out_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_gn_groups(mid_channels), mid_channels),
            nn.ReLU(inplace=True),
        )
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_gn_groups(mid_channels), mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 1, kernel_size=1)
        )
        self.size_head = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_gn_groups(mid_channels), mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 2, kernel_size=1),
        )
        self.offset_head = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_gn_groups(mid_channels), mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 2, kernel_size=1),
        )
        nn.init.constant_(self.heatmap_head[-1].bias, -2.19)

    def forward(self, fpn_features):
        x = self.stem(fpn_features)
        heatmap = self.heatmap_head(x)
        size = F.relu(self.size_head(x))
        offset = torch.sigmoid(self.offset_head(x))
        return {
            'heatmap': heatmap,
            'size': size,
            'offset': offset,
        }


class RegressionHead(nn.Module):
    """Custom head for regression tasks (coordinate prediction)."""
    
    def __init__(self, encoder_channels, num_points, hidden_dims=None, dropout=0.1, use_tanh=True, in_channels=None):
        """
        Args:
            encoder_channels: List of encoder output channels
            num_points: Number of points to predict (output will be num_points * 2 for x,y)
        """
        super().__init__()
        if in_channels is None:
            in_channels = encoder_channels[-1]
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        if hidden_dims is None:
            hidden_dims = [256, 128]
        dims = [in_channels] + list(hidden_dims) + [num_points * 2]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.SiLU(inplace=True))
                layers.append(nn.Dropout(p=dropout))
        self.mlp = nn.Sequential(*layers)
        self.use_tanh = use_tanh

    def forward(self, features):
        """
        Args:
            features: Encoder feature list or FPN feature map
        """
        if isinstance(features, (list, tuple)):
            x = features[-1]
        else:
            x = features
        x = self.pooling(x)
        x = self.flatten(x)
        x = self.mlp(x)
        if self.use_tanh:
            x = torch.tanh(x)
            x = (x + 1.0) * 0.5
        return x


def _get_heads_config(model_config):
    if not isinstance(model_config, dict):
        return {}
    if 'heads' in model_config:
        return model_config.get('heads', {}) or {}
    return model_config.get('model', {}).get('heads', {}) or {}


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
    heads_cfg = _get_heads_config(model_config)
    decoder_cfg = {}
    if isinstance(model_config, dict):
        decoder_cfg = model_config.get('decoder', {}) or {}
    use_fpn_for_cls = decoder_cfg.get('use_fpn_for_classification', True)
    use_fpn_for_reg = decoder_cfg.get('use_fpn_for_regression', True)
    
    if task_name == 'segmentation':
        head_cfg = heads_cfg.get('segmentation', {})
        use_deep_supervision = head_cfg.get('use_deep_supervision', False)
        
        if use_deep_supervision:
            num_aux = int(head_cfg.get('num_aux_outputs', 3))
            upsampling = int(head_cfg.get('upsampling', 4))
            head = DeepSupervisionSegHead(
                fpn_out_channels=fpn_out_channels,
                num_classes=num_classes,
                num_aux_outputs=num_aux,
                upsampling=upsampling
            )
        else:
            seg_type = head_cfg.get('type', 'standard')
            if seg_type == 'unet_like':
                mid_channels = head_cfg.get('mid_channels')
                head = UNetLikeSegHead(
                    in_channels=fpn_out_channels,
                    num_classes=num_classes,
                    upsampling=int(head_cfg.get('upsampling', 4)),
                    mid_channels=int(mid_channels) if mid_channels is not None else None,
                    num_blocks=int(head_cfg.get('num_blocks', 2))
                )
            else:
                mid_channels = head_cfg.get('mid_channels')
                head = SegmentationHead(
                    in_channels=fpn_out_channels,
                    num_classes=num_classes,
                    upsampling=int(head_cfg.get('upsampling', 4)),
                    mid_channels=int(mid_channels) if mid_channels is not None else None,
                    num_layers=int(head_cfg.get('num_layers', 2))
                )
    
    elif task_name == 'classification':
        head_cfg = heads_cfg.get('classification', {})
        mlp_hidden_dim = head_cfg.get('mlp_hidden_dim')
        head = ClassificationHead(
            encoder_channels=encoder_channels,
            num_classes=num_classes,
            dropout=float(head_cfg.get('dropout', 0.2)),
            mlp_hidden_dim=int(mlp_hidden_dim) if mlp_hidden_dim is not None else None,
            in_channels=fpn_out_channels if use_fpn_for_cls else None
        )
    
    elif task_name == 'detection':
        head_cfg = heads_cfg.get('detection', {})
        det_type = head_cfg.get('type', 'centernet')
        if det_type == 'centernet':
            head = CenterNetDetectionHead(
                fpn_out_channels=fpn_out_channels,
                mid_channels=int(head_cfg.get('mid_channels', 128))
            )
        else:
            head = DetectionHead(
                fpn_out_channels=fpn_out_channels,
                num_classes=num_classes,
                mid_channels=int(head_cfg.get('mid_channels', 128)),
                num_anchors=int(head_cfg.get('num_anchors', 1))
            )
    
    elif task_name == 'Regression':
        # num_classes is actually num_points for regression tasks
        num_points = num_classes
        head_cfg = heads_cfg.get('regression', {})
        hidden_dims = head_cfg.get('hidden_dims')
        head = RegressionHead(
            encoder_channels=encoder_channels,
            num_points=num_points,
            hidden_dims=[int(d) for d in hidden_dims] if hidden_dims is not None else None,
            dropout=float(head_cfg.get('dropout', 0.1)),
            use_tanh=head_cfg.get('use_tanh', True),
            in_channels=fpn_out_channels if use_fpn_for_reg else None
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
