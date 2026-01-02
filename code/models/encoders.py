"""Encoder implementations."""

import torch
import torch.nn as nn
import timm
import segmentation_models_pytorch as smp


# Swin Transformer model name mapping
SWIN_MODEL_MAPPING = {
    'swin_t': 'swin_tiny_patch4_window7_224',
    'swin_s': 'swin_small_patch4_window7_224',
    'swin_b': 'swin_base_patch4_window7_224',
}


class SwinTransformerEncoder(nn.Module):
    """Swin Transformer encoder wrapper compatible with FPN decoder."""
    
    def __init__(self, model_name: str = 'swin_b', pretrained: bool = True, img_size: int = 224):
        super().__init__()
        # Map simplified name to full timm model name
        full_model_name = SWIN_MODEL_MAPPING.get(model_name, model_name)
        
        self.model = timm.create_model(
            full_model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),  # Swin has 4 stages
            img_size=img_size  # Specify input image size
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
        """Return encoder output channels for FPN decoder."""
        # FPN expects encoder_channels in format: [input_channels, stage0, stage1, stage2, stage3]
        return [3] + list(self._out_channels)


def build_encoder(config):
    """
    Build encoder from configuration.
    
    Args:
        config: Configuration object
    
    Returns:
        encoder: PyTorch encoder module
    """
    encoder_name = config.get('model.encoder.name')
    encoder_weights = config.get('model.encoder.pretrained')
    img_size = config.get('data.image_size', 224)
    
    if encoder_name.startswith('swin_'):
        # Use custom Swin encoder
        pretrained = (encoder_weights == 'imagenet' or encoder_weights is not None)
        encoder = SwinTransformerEncoder(
            model_name=encoder_name,
            pretrained=pretrained,
            img_size=img_size
        )
        print(f"✓ Loaded Swin Transformer: {encoder_name} (img_size={img_size})")
    else:
        # Use standard segmentation_models_pytorch encoder
        encoder = smp.encoders.get_encoder(
            name=encoder_name,
            in_channels=3,
            depth=5,
            weights=encoder_weights
        )
        print(f"✓ Loaded encoder: {encoder_name}")
    
    return encoder
