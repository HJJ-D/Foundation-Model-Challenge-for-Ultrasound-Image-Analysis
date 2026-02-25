"""Decoder implementations."""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.decoders.fpn.decoder import FPNDecoder


def build_fpn_decoder(encoder, config, decoder_type='seg'):
    """
    Build FPN decoder from configuration.
    
    Args:
        encoder: Encoder module
        config: Configuration object
        decoder_type: 'seg' for segmentation or 'det' for detection
    
    Returns:
        decoder: FPN decoder module
    """
    encoder_name = config.get('model.encoder.name')
    
    # Get FPN configuration
    pyramid_channels = int(config.get('model.decoder.pyramid_channels', 256))
    segmentation_channels = int(config.get('model.decoder.segmentation_channels', 128))
    dropout = float(config.get('model.decoder.dropout', 0.2))
    merge_policy = config.get('model.decoder.merge_policy', 'cat')
    
    # Get encoder output channels in FPN format: [input_channels, stage1, stage2, ...]
    # Custom wrapper encoders (Swin/ViT/DINOv3) already include input channel in out_channels
    # SMP encoders (ResNet/EfficientNet) don't include it, need to prepend [3]
    if hasattr(encoder, 'is_timm_encoder') and encoder.is_timm_encoder:
        encoder_channels = encoder.out_channels  # Already [3, c1, c2, c3, c4]
    else:
        # SMP encoders: [c1, c2, c3, c4, c5] -> [3, c1, c2, c3, c4, c5]
        encoder_channels = [3] + list(encoder.out_channels)
    
    # encoder_depth is the number of feature stages (excluding input)
    encoder_depth = len(encoder_channels) - 1
    
    # Create FPN decoder with encoder channel information
    decoder = FPNDecoder(
        encoder_channels=encoder_channels,
        encoder_depth=encoder_depth,
        pyramid_channels=pyramid_channels,
        segmentation_channels=segmentation_channels,
        dropout=dropout,
        merge_policy=merge_policy
    )
    
    suffix_map = {
        'seg': 'segmentation',
        'det': 'detection',
        'cls': 'classification',
        'reg': 'regression',
    }
    suffix = suffix_map.get(decoder_type, decoder_type)
    print(f"Built FPN decoder for {suffix}")
    
    return decoder


def build_decoders(encoder, config):
    """
    Build all required decoders.
    
    Args:
        encoder: Encoder module
        config: Configuration object
    
    Returns:
        decoders: Dictionary of decoder modules
    """
    decoders = {}
    
    # Segmentation FPN decoder
    decoders['fpn_seg'] = build_fpn_decoder(encoder, config, decoder_type='seg')
    
    # Detection FPN decoder (separate or shared)
    if config.get('model.decoder.separate_detection_fpn', True):
        decoders['fpn_det'] = build_fpn_decoder(encoder, config, decoder_type='det')
        print("Using separate FPN decoder for detection")
    else:
        decoders['fpn_det'] = decoders['fpn_seg']  # Share with segmentation
        print("Sharing FPN decoder between segmentation and detection")

    # Classification FPN decoder (separate or shared)
    if config.get('model.decoder.separate_classification_fpn', True):
        decoders['fpn_cls'] = build_fpn_decoder(encoder, config, decoder_type='cls')
        print("Using separate FPN decoder for classification")
    else:
        decoders['fpn_cls'] = decoders['fpn_seg']  # Share with segmentation
        print("Sharing FPN decoder between segmentation and classification")

    # Regression FPN decoder (separate or shared)
    if config.get('model.decoder.separate_regression_fpn', True):
        decoders['fpn_reg'] = build_fpn_decoder(encoder, config, decoder_type='reg')
        print("Using separate FPN decoder for regression")
    else:
        decoders['fpn_reg'] = decoders['fpn_seg']  # Share with segmentation
        print("Sharing FPN decoder between segmentation and regression")
    
    return decoders
