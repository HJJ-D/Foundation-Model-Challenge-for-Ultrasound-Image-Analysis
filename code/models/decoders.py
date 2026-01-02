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
    pyramid_channels = config.get('model.decoder.pyramid_channels', 256)
    segmentation_channels = config.get('model.decoder.segmentation_channels', 128)
    dropout = config.get('model.decoder.dropout', 0.2)
    merge_policy = config.get('model.decoder.merge_policy', 'cat')
    
    if encoder_name.startswith('swin_'):
        # For Swin encoder, use custom FPN decoder
        encoder_channels = encoder.out_channels
        decoder = FPNDecoder(
            encoder_channels=encoder_channels,
            encoder_depth=5,  # 5 = input + 4 stages
            pyramid_channels=pyramid_channels,
            segmentation_channels=segmentation_channels,
            dropout=dropout,
            merge_policy=merge_policy
        )
    else:
        # For standard encoders, use smp FPN
        encoder_weights = config.get('model.encoder.pretrained')
        temp_model = smp.FPN(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=1
        )
        decoder = temp_model.decoder
    
    suffix = 'detection' if decoder_type == 'det' else 'segmentation'
    print(f"✓ Built FPN decoder for {suffix}")
    
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
        print("✓ Using separate FPN decoder for detection")
    else:
        decoders['fpn_det'] = decoders['fpn_seg']  # Share with segmentation
        print("✓ Sharing FPN decoder between segmentation and detection")
    
    return decoders
