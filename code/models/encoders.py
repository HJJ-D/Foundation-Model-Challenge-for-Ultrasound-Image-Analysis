"""Encoder implementations."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import segmentation_models_pytorch as smp


# Swin Transformer model name mapping
SWIN_MODEL_MAPPING = {
    'swin_t': 'swin_tiny_patch4_window7_224',
    'swin_s': 'swin_small_patch4_window7_224',
    'swin_b': 'swin_base_patch4_window7_224',
    'swin_l': 'swin_large_patch4_window7_224',
}

# ViT model name mapping (timm)
VIT_MODEL_MAPPING = {
    'vit_t': 'vit_tiny_patch16_224',
    'vit_s': 'vit_small_patch16_224',
    'vit_b': 'vit_base_patch16_224',
    'vit_l': 'vit_large_patch16_224',
}


class SwinTransformerEncoder(nn.Module):
    """Swin Transformer encoder wrapper compatible with FPN decoder."""
    is_timm_encoder = True
    
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


class TimmEncoder(nn.Module):
    """Generic timm encoder wrapper compatible with FPN decoder."""
    is_timm_encoder = True

    def __init__(self, model_name: str, pretrained: bool = True, img_size: int = 224, out_indices=None):
        super().__init__()
        full_model_name = VIT_MODEL_MAPPING.get(model_name, model_name)
        if out_indices is not None and not isinstance(out_indices, tuple):
            out_indices = tuple(out_indices)
        self.model = timm.create_model(
            full_model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices,
            img_size=img_size
        )
        self._target_stages = 4
        self._raw_channels = list(self.model.feature_info.channels())
        self._out_channels = self._normalize_channels(self._raw_channels)
        self.output_stride = self._infer_output_stride()

    def _infer_output_stride(self):
        if hasattr(self.model, "feature_info"):
            reductions = self.model.feature_info.reduction()
            if reductions:
                return reductions[min(len(reductions), self._target_stages) - 1]
        return 32

    def _normalize_channels(self, channels):
        if not channels:
            raise ValueError("timm encoder returned no feature channels")
        if len(channels) >= self._target_stages:
            return list(channels[:self._target_stages])
        return list(channels) + [channels[-1]] * (self._target_stages - len(channels))

    def _tokens_to_feature_map(self, tokens):
        # tokens: (B, N, C)
        grid_size = None
        if hasattr(self.model, "patch_embed") and hasattr(self.model.patch_embed, "grid_size"):
            grid_size = self.model.patch_embed.grid_size
        prefix_tokens = getattr(self.model, "num_prefix_tokens", 0)
        if grid_size is None:
            n_tokens = tokens.shape[1]
            usable = n_tokens - prefix_tokens
            side = int(math.sqrt(max(usable, 1)))
            if side * side != usable:
                # Fallback for unknown prefix tokens
                side = int(math.sqrt(max(n_tokens - 1, 1)))
                if side * side == n_tokens - 1:
                    prefix_tokens = 1
                else:
                    side = int(math.sqrt(max(n_tokens, 1)))
                    prefix_tokens = 0
            grid_size = (side, side)
        if prefix_tokens > 0:
            tokens = tokens[:, prefix_tokens:, :]
        return tokens.reshape(tokens.shape[0], grid_size[0], grid_size[1], tokens.shape[2])

    def _format_feature(self, feat, expected_channels):
        if feat.ndim == 3:
            feat = self._tokens_to_feature_map(feat)
        if feat.ndim == 4:
            if feat.shape[1] == expected_channels:
                return feat
            if feat.shape[-1] == expected_channels:
                return feat.permute(0, 3, 1, 2).contiguous()
        return feat

    def _normalize_features(self, features, input_hw):
        if len(features) > self._target_stages:
            features = features[:self._target_stages]
        while len(features) < self._target_stages:
            features.append(features[-1])

        strides = [4, 8, 16, 32]
        target_sizes = [
            (max(1, input_hw[0] // stride), max(1, input_hw[1] // stride))
            for stride in strides
        ]
        resized = []
        for feat, target in zip(features, target_sizes):
            if feat.shape[2] == target[0] and feat.shape[3] == target[1]:
                resized.append(feat)
                continue
            if feat.shape[2] >= target[0] and feat.shape[3] >= target[1]:
                resized.append(F.adaptive_avg_pool2d(feat, output_size=target))
            else:
                resized.append(F.interpolate(feat, size=target, mode="bilinear", align_corners=False))
        return resized

    def forward(self, x):
        features = self.model(x)
        expected = list(self.model.feature_info.channels())
        features = [
            self._format_feature(feat, expected[idx] if idx < len(expected) else expected[-1])
            for idx, feat in enumerate(features)
        ]
        features = self._normalize_features(features, x.shape[-2:])
        return features

    @property
    def out_channels(self):
        """Return encoder output channels for FPN decoder."""
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
    out_indices = config.get('model.encoder.out_indices', None)
    
    if encoder_name.startswith('swin_'):
        # Use custom Swin encoder
        pretrained = (encoder_weights == 'imagenet' or encoder_weights is not None)
        encoder = SwinTransformerEncoder(
            model_name=encoder_name,
            pretrained=pretrained,
            img_size=img_size
        )
        print(f"Loaded Swin Transformer: {encoder_name} (img_size={img_size})")
    else:
        timm_name = encoder_name
        use_timm = False
        if encoder_name.startswith('timm:'):
            timm_name = encoder_name.split(':', 1)[1]
            use_timm = True
        if use_timm:
            pretrained = (encoder_weights == 'imagenet' or encoder_weights is not None)
            encoder = TimmEncoder(
                model_name=timm_name,
                pretrained=pretrained,
                img_size=img_size,
                out_indices=out_indices
            )
            print(f"Loaded timm encoder: {timm_name} (img_size={img_size})")
        else:
            # Use standard segmentation_models_pytorch encoder, fallback to timm if not found
            try:
                encoder = smp.encoders.get_encoder(
                    name=encoder_name,
                    in_channels=3,
                    depth=5,
                    weights=encoder_weights
                )
                print(f"Loaded encoder: {encoder_name}")
            except Exception:
                pretrained = (encoder_weights == 'imagenet' or encoder_weights is not None)
                encoder = TimmEncoder(
                    model_name=timm_name,
                    pretrained=pretrained,
                    img_size=img_size,
                    out_indices=out_indices
                )
                print(f"Loaded timm encoder: {timm_name} (img_size={img_size})")
    
    return encoder
