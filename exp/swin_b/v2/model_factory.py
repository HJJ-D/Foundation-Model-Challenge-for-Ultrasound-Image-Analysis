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
    def __init__(self, model_name: str = 'swin_b', pretrained: bool = True, img_size: int = 256):
        super().__init__()
        # Map simplified name to full timm model name
        full_model_name = SWIN_MODEL_MAPPING.get(model_name, model_name)
        
        self.model = timm.create_model(
            full_model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),  # Swin has 4 stages
            img_size=img_size  # Specify input image size to avoid mismatch
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
    """Custom head for regression tasks."""
    def __init__(self, in_channels: int, num_points: int):
        super().__init__()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        # Output dimension is num_points * 2 (x, y)
        self.linear = nn.Linear(in_channels, num_points * 2)

    def forward(self, features: list):
        x = self.pooling(features[-1])
        x = self.flatten(x)
        return self.linear(x)

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

class DeepSupervisionSegHead(nn.Module):
    """Segmentation head with deep supervision for multi-level auxiliary outputs."""
    def __init__(self, fpn_out_channels: int, num_classes: int, num_aux_outputs: int = 3):
        super().__init__()
        self.num_aux_outputs = num_aux_outputs
        
        # Main segmentation head
        self.main_head = smp.base.SegmentationHead(
            in_channels=fpn_out_channels,
            out_channels=num_classes,
            kernel_size=1,
            upsampling=4
        )
        
        # Auxiliary heads - lightweight 1x1 conv for intermediate supervision
        self.aux_heads = nn.ModuleList([
            nn.Conv2d(fpn_out_channels, num_classes, kernel_size=1)
            for _ in range(num_aux_outputs)
        ])
    
    def forward(self, fpn_features: torch.Tensor):
        """
        Args:
            fpn_features: Output from FPN decoder [B, C, H, W]
        Returns:
            main_out: Main segmentation output [B, num_classes, H*4, W*4]
            aux_outs: List of auxiliary outputs, each [B, num_classes, H, W]
        """
        # Main output with upsampling
        main_out = self.main_head(fpn_features)
        
        # Auxiliary outputs (no upsampling, will be upsampled during loss computation)
        aux_outs = [aux_head(fpn_features) for aux_head in self.aux_heads]
        
        return main_out, aux_outs

# ====================================================================
# --- 2. Multi-Task Model Factory ---
# ====================================================================

class MultiTaskModelFactory(nn.Module):
    def __init__(self, encoder_name: str, encoder_weights: str, task_configs: List[Dict],
                 use_deep_supervision: bool = False, num_aux_outputs: int = 3,
                 use_separate_detection_fpn: bool = True, img_size: int = 256): 
        super().__init__()
        
        self.use_deep_supervision = use_deep_supervision
        self.num_aux_outputs = num_aux_outputs
        self.use_separate_detection_fpn = use_separate_detection_fpn
        
        
        if encoder_name.startswith('swin_'):
            # Use custom Swin encoder with specified image size
            pretrained = (encoder_weights == 'imagenet' or encoder_weights is not None)
            self.encoder = SwinTransformerEncoder(
                model_name=encoder_name, 
                pretrained=pretrained, 
                img_size=img_size
            )
            
            # Create FPN decoder for Swin
            from segmentation_models_pytorch.decoders.fpn.decoder import FPNDecoder
            encoder_channels = self.encoder.out_channels
            
            # Segmentation FPN decoder
            self.fpn_decoder_seg = FPNDecoder(
                encoder_channels=encoder_channels,
                encoder_depth=5,  # 5 = input + 4 stages
                pyramid_channels=256,
                segmentation_channels=128,
                dropout=0.2,
                merge_policy="cat"
            )
            
            # Detection FPN decoder (independent, to avoid conflict with segmentation)
            if use_separate_detection_fpn:
                self.fpn_decoder_det = FPNDecoder(
                    encoder_channels=encoder_channels,
                    encoder_depth=5,
                    pyramid_channels=256,
                    segmentation_channels=128,
                    dropout=0.2,
                    merge_policy="cat"
                )
            else:
                self.fpn_decoder_det = self.fpn_decoder_seg  # Share with segmentation
        else:
            # Use standard smp encoder
            self.encoder = smp.encoders.get_encoder(
                name=encoder_name,
                in_channels=3,
                depth=5,
                weights=encoder_weights,
            )
            
            # Initialize FPN decoders
            temp_fpn_model = smp.FPN(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=3,
                classes=1, 
            )
            self.fpn_decoder_seg = temp_fpn_model.decoder
            
            if use_separate_detection_fpn:
                temp_fpn_model_det = smp.FPN(
                    encoder_name=encoder_name,
                    encoder_weights=encoder_weights,
                    in_channels=3,
                    classes=1,
                )
                self.fpn_decoder_det = temp_fpn_model_det.decoder
            else:
                self.fpn_decoder_det = self.fpn_decoder_seg
        
        # Initialize task heads
        self.heads = nn.ModuleDict()
        for config in task_configs:
            task_id = config['task_id']
            task_name = config['task_name']
            num_classes = config['num_classes']
            
            head_module = None
            if task_name == 'segmentation':
                if self.use_deep_supervision:
                    head_module = DeepSupervisionSegHead(
                        fpn_out_channels=self.fpn_decoder_seg.out_channels,
                        num_classes=num_classes,
                        num_aux_outputs=self.num_aux_outputs
                    )
                else:
                    head_module = smp.base.SegmentationHead(
                        in_channels=self.fpn_decoder_seg.out_channels, 
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
                    fpn_out_channels=self.fpn_decoder_det.out_channels,
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
        if task_name == 'segmentation':
            # Use segmentation FPN
            fpn_features = self.fpn_decoder_seg(features)
            output = self.heads[task_id](fpn_features)
        elif task_name == 'detection':
            # Use detection FPN (separate or shared)
            fpn_features = self.fpn_decoder_det(features)
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
