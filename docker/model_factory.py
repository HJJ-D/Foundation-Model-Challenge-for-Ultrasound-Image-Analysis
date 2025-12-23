import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from typing import List, Dict
import timm

# Task configuration list
TASK_CONFIGURATIONS = [
    {'task_name': 'Regression', 'num_classes': 2, 'task_id': 'FUGC'},
    {'task_name': 'Regression', 'num_classes': 3, 'task_id': 'IUGC'},
    {'task_name': 'Regression', 'num_classes': 2, 'task_id': 'fetal_femur'},
    {'task_name': 'classification', 'num_classes': 2, 'task_id': 'breast_2cls'},
    {'task_name': 'classification', 'num_classes': 3, 'task_id': 'breast_3cls'},
    {'task_name': 'classification', 'num_classes': 8, 'task_id': 'fetal_head_pos_cls'},
    {'task_name': 'classification', 'num_classes': 6, 'task_id': 'fetal_plane_cls'},
    {'task_name': 'classification', 'num_classes': 8, 'task_id': 'fetal_sacral_pos_cls'},
    {'task_name': 'classification', 'num_classes': 2, 'task_id': 'liver_lesion_2cls'},
    {'task_name': 'classification', 'num_classes': 2, 'task_id': 'lung_2cls'},
    {'task_name': 'classification', 'num_classes': 3, 'task_id': 'lung_disease_3cls'},
    {'task_name': 'classification', 'num_classes': 6, 'task_id': 'organ_cls'},
    {'task_name': 'detection', 'num_classes': 1, 'task_id': 'spinal_cord_injury_loc'},
    {'task_name': 'detection', 'num_classes': 1, 'task_id': 'thyroid_nodule_det'},
    {'task_name': 'detection', 'num_classes': 1, 'task_id': 'uterine_fibroid_det'},
    {'task_name': 'segmentation', 'num_classes': 2, 'task_id': 'breast_lesion'},
    {'task_name': 'segmentation', 'num_classes': 4, 'task_id': 'cardiac_multi'},
    {'task_name': 'segmentation', 'num_classes': 2, 'task_id': 'carotid_artery'},
    {'task_name': 'segmentation', 'num_classes': 2, 'task_id': 'cervix'},
    {'task_name': 'segmentation', 'num_classes': 3, 'task_id': 'cervix_multi'},
    {'task_name': 'segmentation', 'num_classes': 5, 'task_id': 'fetal_abdomen_multi'},
    {'task_name': 'segmentation', 'num_classes': 2, 'task_id': 'fetal_head'},
    {'task_name': 'segmentation', 'num_classes': 2, 'task_id': 'fetal_heart'},
    {'task_name': 'segmentation', 'num_classes': 3, 'task_id': 'head_symphysis_multi'},
    {'task_name': 'segmentation', 'num_classes': 2, 'task_id': 'lung'},
    {'task_name': 'segmentation', 'num_classes': 2, 'task_id': 'ovary_tumor'},
    {'task_name': 'segmentation', 'num_classes': 2, 'task_id': 'thyroid_nodule'},
]

# ====================================================================
# --- Custom Swin Transformer Encoder ---
# ====================================================================

# Mapping of simplified names to full timm model names
SWIN_MODEL_MAPPING = {
    'swin_t': 'swin_tiny_patch4_window7_224',
    'swin_s': 'swin_small_patch4_window7_224',
    'swin_b': 'swin_base_patch4_window7_224',
    'swin_l': 'swin_large_patch4_window7_224',
    'swin_large_patch4_window12_384': 'swin_large_patch4_window12_384',  # Full name passthrough
}

class SwinTransformerEncoder(nn.Module):
    """Swin Transformer encoder wrapper compatible with segmentation_models_pytorch."""
    def __init__(self, model_name: str = 'swin_b', pretrained: bool = True):
        super().__init__()
        # Map simplified name to full timm model name
        full_model_name = SWIN_MODEL_MAPPING.get(model_name, model_name)
        
        self.model = timm.create_model(
            full_model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3)  # Swin has 4 stages
        )
        self._out_channels = self.model.feature_info.channels()
        self.output_stride = 32
        
    def forward(self, x):
        features = self.model(x)
        # Swin outputs features in (B, H, W, C) format, need to convert to (B, C, H, W)
        features = [feat.permute(0, 3, 1, 2).contiguous() for feat in features]
        return features
    
    @property
    def out_channels(self):
        # FPN expects encoder_channels in format: [input_channels, stage0, stage1, stage2, stage3]
        # For Swin with 4 stages: [3, 128, 256, 512, 1024]
        return [3] + list(self._out_channels)

# Task specific heads

class SmpClassificationHead(nn.Module):
    """Wrapper for SMP Classification Head."""
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.head = smp.base.ClassificationHead(
            in_channels=in_channels,
            classes=num_classes,
            pooling="avg",
            dropout=0.2,
            activation=None,
        )
        
    def forward(self, features: list):
        # Use the last feature map from encoder
        return self.head(features[-1])

class RegressionHead(nn.Module):
    """Improved regression head with spatial attention and deep MLP.
    
    Key improvements:
    1. Spatial attention to preserve location information
    2. Multi-layer MLP for better non-linear mapping
    3. Layer normalization for stable training
    4. Sigmoid output to constrain coordinates to [0,1]
    """
    def __init__(self, in_channels: int, num_points: int, use_spatial_attention: bool = True):
        super().__init__()
        self.num_points = num_points
        self.use_spatial_attention = use_spatial_attention
        
        if use_spatial_attention:
            # Spatial attention: generate attention map for each keypoint
            self.spatial_attention = nn.Sequential(
                nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
                nn.BatchNorm2d(in_channels // 4),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels // 4, num_points, kernel_size=1),
                nn.Sigmoid()  # Attention weights in [0,1]
            )
            
            # Feature dimension after attention-weighted pooling
            mlp_input_dim = in_channels * num_points
        else:
            # Fallback to global pooling
            self.pooling = nn.AdaptiveAvgPool2d(1)
            self.flatten = nn.Flatten()
            mlp_input_dim = in_channels
        
        # Deep MLP with residual-like structure
        hidden_dim = max(512, in_channels // 2)
        
        self.mlp = nn.Sequential(
            # First layer: dimensional transformation
            nn.Linear(mlp_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            # Second layer: feature refinement
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            # Output layer: coordinate prediction
            nn.Linear(hidden_dim, num_points * 2),
            nn.Sigmoid()  # Output in [0,1] for normalized coordinates
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small values for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, features: list):
        """
        Args:
            features: List of feature maps from encoder
        Returns:
            coords: [B, num_points*2] normalized coordinates in [0,1]
        """
        x = features[-1]  # Use the last (deepest) feature map, [B, C, H, W]
        
        if self.use_spatial_attention:
            # Generate attention map for each keypoint
            attn_maps = self.spatial_attention(x)  # [B, num_points, H, W]
            
            # Attention-weighted pooling for each keypoint
            B, C, H, W = x.shape
            weighted_features = []
            
            for i in range(self.num_points):
                # Get attention map for keypoint i
                attn = attn_maps[:, i:i+1, :, :]  # [B, 1, H, W]
                
                # Weighted sum: emphasize regions where keypoint likely exists
                weighted = (x * attn).sum(dim=[2, 3])  # [B, C]
                weighted_features.append(weighted)
            
            # Concatenate features for all keypoints
            x = torch.cat(weighted_features, dim=1)  # [B, C*num_points]
        else:
            # Fallback: global average pooling
            x = self.pooling(x)
            x = self.flatten(x)  # [B, C]
        
        # Multi-layer perception for coordinate regression
        coords = self.mlp(x)  # [B, num_points*2]
        
        return coords

class FPNGridDetectionHead(nn.Module):
    """Detection head designed for FPN outputs."""
    def __init__(self, fpn_out_channels: int, num_classes: int = 1, num_anchors: int = 1):
        super().__init__()
        mid_channels = 128
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

    def forward(self, fpn_features: torch.Tensor):
        # Input fpn_features is already a single fused tensor from FPN Decoder
        predictions_map = self.conv_block(fpn_features)
        
        # Apply sigmoid to bbox coordinates (first 4 channels)
        predictions_map[:, :4] = torch.sigmoid(predictions_map[:, :4])
        
        return predictions_map

# ====================================================================
# --- 2. Multi-Task Model Factory ---
# ====================================================================

class MultiTaskModelFactory(nn.Module):
    def __init__(self, encoder_name: str, encoder_weights: str, task_configs: List[Dict]):
        super().__init__()
        
        # Initialize shared encoder
        print(f"Initializing shared encoder: {encoder_name}")
        
        if encoder_name.startswith('swin_'):
            # Use custom Swin encoder
            pretrained = (encoder_weights == 'imagenet' or encoder_weights is not None)
            self.encoder = SwinTransformerEncoder(model_name=encoder_name, pretrained=pretrained)
            print(f"Using custom Swin Transformer: {encoder_name}")
            
            # Create FPN decoder for Swin
            from segmentation_models_pytorch.decoders.fpn.decoder import FPNDecoder
            encoder_channels = self.encoder.out_channels
            self.fpn_decoder = FPNDecoder(
                encoder_channels=encoder_channels,
                encoder_depth=5,  # 5 = input + 4 stages
                pyramid_channels=256,
                segmentation_channels=128,
                dropout=0.2,
                merge_policy="cat"
            )
        else:
            # Use standard smp encoder
            self.encoder = smp.encoders.get_encoder(
                name=encoder_name,
                in_channels=3,
                depth=5,
                weights=encoder_weights,
            )
            
            # Initialize shared FPN decoder
            temp_fpn_model = smp.FPN(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=3,
                classes=1, 
            )
            self.fpn_decoder = temp_fpn_model.decoder
        
        # Initialize task heads
        self.heads = nn.ModuleDict()
        
        print(f"Creating heads for {len(task_configs)} tasks...")
        for config in task_configs:
            task_id = config['task_id']
            task_name = config['task_name']
            num_classes = config['num_classes']
            
            head_module = None
            if task_name == 'segmentation':
                head_module = smp.base.SegmentationHead(
                    in_channels=self.fpn_decoder.out_channels, 
                    out_channels=num_classes, 
                    kernel_size=1,
                    upsampling=4 
                )

            elif task_name == 'classification':
                head_module = SmpClassificationHead(
                    in_channels=self.encoder.out_channels[-1],
                    num_classes=num_classes
                )

            elif task_name == 'Regression':
                num_points = config['num_classes']
                head_module = RegressionHead(
                    in_channels=self.encoder.out_channels[-1],
                    num_points=num_points
                )

            elif task_name == 'detection':
                head_module = FPNGridDetectionHead(
                    fpn_out_channels=self.fpn_decoder.out_channels,
                    num_classes=num_classes
                )

            if head_module:
                self.heads[task_id] = head_module
            else:
                print(f"Warning: Unknown task type '{task_name}' for {task_id}")

    def forward(self, x: torch.Tensor, task_id: str) -> torch.Tensor:
        features = self.encoder(x)
        
        if task_id not in self.heads:
            raise ValueError(f"Task ID '{task_id}' not found.")

        task_config = next((item for item in TASK_CONFIGURATIONS if item["task_id"] == task_id), None)
        task_name = task_config['task_name'] if task_config else None

        # Route features based on task type
        if task_name in ['segmentation', 'detection']:
            # Use FPN features for dense prediction tasks
            fpn_features = self.fpn_decoder(features)
            output = self.heads[task_id](fpn_features)
        else: 
            # Use encoder features directly for global prediction tasks
            output = self.heads[task_id](features)
            
        return output

# Example usage

if __name__ == '__main__':
    model = MultiTaskModelFactory(
        encoder_name='swin_b',
        encoder_weights='imagenet',
        task_configs=TASK_CONFIGURATIONS
    )

    print("\n--- Forward Pass Test ---")
    dummy_image_batch = torch.randn(2, 3, 224, 224)  # Swin-Base expects 224x224

    # Test specific tasks
    test_tasks = ['cardiac_multi', 'fetal_plane_cls', 'FUGC', 'thyroid_nodule_det']
    
    for t_id in test_tasks:
        try:
            out = model(dummy_image_batch, task_id=t_id)
            print(f"Task: {t_id:<25} | Output Shape: {out.shape}")
        except Exception as e:
            print(f"Task: {t_id:<25} | Error: {e}")
