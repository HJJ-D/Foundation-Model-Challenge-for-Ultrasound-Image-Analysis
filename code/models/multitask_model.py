"""Multi-task model architecture."""

import torch
import torch.nn as nn
from .encoders import build_encoder
from .decoders import build_decoders
from .heads import build_all_heads


class MultiTaskModel(nn.Module):
    """
    Multi-task learning model with shared encoder and task-specific heads.
    Supports segmentation, classification, detection, and regression tasks.
    """
    
    def __init__(self, config):
        """
        Initialize multi-task model from configuration.
        
        Args:
            config: Configuration object
        """
        super().__init__()
        
        self.config = config
        self.task_configs = config.get_task_configs()
        
        # Build encoder
        self.encoder = build_encoder(config)
        
        # Get encoder output channels
        if hasattr(self.encoder, 'out_channels'):
            encoder_channels = self.encoder.out_channels
        else:
            # For standard smp encoders
            encoder_channels = [3] + list(self.encoder.out_channels)
        
        # Build decoders
        decoders = build_decoders(self.encoder, config)
        self.fpn_decoder_seg = decoders['fpn_seg']
        self.fpn_decoder_det = decoders['fpn_det']
        
        # Get FPN output channels
        self.fpn_out_channels = self.fpn_decoder_seg.out_channels
        
        # Build task-specific heads
        model_config = config.config.get('model', {})
        self.heads = build_all_heads(
            task_configs=self.task_configs,
            fpn_out_channels=self.fpn_out_channels,
            encoder_channels=encoder_channels,
            model_config=model_config
        )
        
        # Create task_id to task_name mapping
        self.task_id_to_name = {
            cfg['task_id']: cfg['task_name'] 
            for cfg in self.task_configs
        }
        
        print(f"\n{'='*60}")
        print(f"Multi-Task Model Summary:")
        print(f"  - Encoder: {config.get('model.encoder.name')}")
        print(f"  - FPN output channels: {self.fpn_out_channels}")
        print(f"  - Total tasks: {len(self.heads)}")
        print(f"    · Segmentation: {sum(1 for cfg in self.task_configs if cfg['task_name'] == 'segmentation')}")
        print(f"    · Classification: {sum(1 for cfg in self.task_configs if cfg['task_name'] == 'classification')}")
        print(f"    · Detection: {sum(1 for cfg in self.task_configs if cfg['task_name'] == 'detection')}")
        print(f"    · Regression: {sum(1 for cfg in self.task_configs if cfg['task_name'] == 'Regression')}")
        print(f"{'='*60}\n")
    
    def forward(self, x, task_id):
        """
        Forward pass for a specific task.
        
        Args:
            x: Input images [B, 3, H, W]
            task_id: Task identifier string
        
        Returns:
            output: Task-specific predictions
        """
        if task_id not in self.heads:
            raise ValueError(f"Unknown task_id: {task_id}")
        
        # Get task name
        task_name = self.task_id_to_name[task_id]
        
        # Encode features
        features = self.encoder(x)
        
        # Route through appropriate decoder and head
        if task_name == 'segmentation':
            fpn_features = self.fpn_decoder_seg(features)
            output = self.heads[task_id](fpn_features)
        
        elif task_name == 'detection':
            fpn_features = self.fpn_decoder_det(features)
            output = self.heads[task_id](fpn_features)
        
        else:  # classification or regression
            # Use encoder features directly
            output = self.heads[task_id](features)
        
        return output
    
    def get_trainable_parameters(self):
        """
        Get trainable parameters grouped by encoder and heads.
        
        Returns:
            encoder_params: List of encoder parameters
            head_params: List of head parameters (including decoders)
        """
        encoder_params = list(self.encoder.parameters())
        
        head_params = []
        head_params += list(self.fpn_decoder_seg.parameters())
        if self.fpn_decoder_det is not self.fpn_decoder_seg:
            head_params += list(self.fpn_decoder_det.parameters())
        head_params += list(self.heads.parameters())
        
        return encoder_params, head_params
    
    def freeze_encoder(self):
        """Freeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        print("✓ Encoder frozen")
    
    def unfreeze_encoder(self):
        """Unfreeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        print("✓ Encoder unfrozen")


def build_model(config):
    """
    Build multi-task model from configuration.
    
    Args:
        config: Configuration object
    
    Returns:
        model: MultiTaskModel instance
    """
    model = MultiTaskModel(config)
    
    # Freeze encoder if specified
    if config.get('model.encoder.freeze_encoder', False):
        model.freeze_encoder()
    
    return model
