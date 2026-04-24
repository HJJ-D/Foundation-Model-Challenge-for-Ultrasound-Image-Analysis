"""Decoder implementations."""

from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.decoders.fpn.decoder import FPNDecoder


class PromptAttentionBlock(nn.Module):
    """Pre-norm cross-attention + FFN block with gated residual update."""

    def __init__(
        self,
        channels: int,
        condition_dim: int,
        d_model: int,
        num_prompt_tokens: int,
        num_heads: int,
        ffn_ratio: float = 4.0,
        attn_dropout: float = 0.1,
        ffn_dropout: float = 0.1,
        init_scale: float = 0.02,
    ):
        super().__init__()
        self.channels = int(channels)
        self.condition_dim = int(condition_dim)
        self.d_model = int(d_model)
        self.num_prompt_tokens = int(num_prompt_tokens)

        self.feat_in = nn.Conv2d(self.channels, self.d_model, kernel_size=1)
        self.feat_out = nn.Conv2d(self.d_model, self.channels, kernel_size=1)

        self.prompt_tokens = nn.Linear(self.condition_dim, self.num_prompt_tokens * self.d_model)

        self.norm_q = nn.LayerNorm(self.d_model)
        self.norm_ffn = nn.LayerNorm(self.d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=int(num_heads),
            dropout=float(attn_dropout),
            batch_first=True,
        )

        hidden_dim = max(1, int(self.d_model * float(ffn_ratio)))
        self.ffn = nn.Sequential(
            nn.Linear(self.d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(ffn_dropout)),
            nn.Linear(hidden_dim, self.d_model),
        )

        self.gate_proj = nn.Linear(self.condition_dim, 1)
        self.residual_scale = nn.Parameter(torch.tensor(float(init_scale), dtype=torch.float32))
        self._current_epoch = None
        self._last_logged_epoch = None

    def set_current_epoch(self, epoch: Optional[int]) -> None:
        self._current_epoch = None if epoch is None else int(epoch)

    def forward(self, x: torch.Tensor, cond_vec: torch.Tensor) -> torch.Tensor:
        bsz, _, height, width = x.shape

        feat = self.feat_in(x).flatten(2).transpose(1, 2)  # [B, N, d_model]
        prompt = self.prompt_tokens(cond_vec).view(bsz, self.num_prompt_tokens, self.d_model)

        q = self.norm_q(feat)
        attn_out, _ = self.cross_attn(query=q, key=prompt, value=prompt, need_weights=False)
        feat = feat + attn_out

        feat = feat + self.ffn(self.norm_ffn(feat))

        delta = feat.transpose(1, 2).reshape(bsz, self.d_model, height, width)
        delta = self.feat_out(delta)

        gate = torch.sigmoid(self.gate_proj(cond_vec)).view(bsz, 1, 1, 1)
        if (
            self.training
            and self._current_epoch is not None
            and self._last_logged_epoch != self._current_epoch
        ):
            print(
                f"[PromptAttention][epoch {self._current_epoch + 1}] "
                f"residual_scale={self.residual_scale.item():.4f}, "
                f"gate_mean={gate.mean().item():.4f}",
                flush=True,
            )
            self._last_logged_epoch = self._current_epoch
        return x + self.residual_scale * gate * delta


class PromptFPNDecoder(nn.Module):
    """FPN-style decoder with optional prompt-attention at selected pyramid stages."""

    def __init__(
        self,
        encoder_channels: Sequence[int],
        encoder_depth: int,
        pyramid_channels: int = 256,
        segmentation_channels: int = 128,
        dropout: float = 0.2,
        merge_policy: str = "cat",
        condition_dim: int = 128,
        d_model: int = 128,
        num_prompt_tokens: int = 4,
        num_heads: int = 4,
        ffn_ratio: float = 4.0,
        attn_dropout: float = 0.1,
        ffn_dropout: float = 0.1,
        apply_stages: Optional[Sequence[int]] = None,
        init_scale: float = 0.02,
    ):
        super().__init__()
        self.encoder_channels = list(encoder_channels)
        self.encoder_depth = int(encoder_depth)
        self.pyramid_channels = int(pyramid_channels)
        self.out_channels = int(segmentation_channels)
        self.merge_policy = str(merge_policy).lower()
        self.dropout = nn.Dropout2d(float(dropout)) if float(dropout) > 0 else nn.Identity()

        if self.merge_policy not in {"cat", "add"}:
            raise ValueError("PromptFPNDecoder supports merge_policy in {'cat', 'add'}")
        if len(self.encoder_channels) < self.encoder_depth + 1:
            raise ValueError("encoder_channels must include input channel + feature channels")

        self.stage_channels = list(self.encoder_channels[-self.encoder_depth:])

        self.lateral_convs = nn.ModuleList(
            [nn.Conv2d(ch, self.pyramid_channels, kernel_size=1) for ch in self.stage_channels]
        )
        self.seg_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(self.pyramid_channels, self.out_channels, kernel_size=3, padding=1, bias=False),
                    nn.GroupNorm(32, self.out_channels),
                    nn.SiLU(inplace=True),
                )
                for _ in range(self.encoder_depth)
            ]
        )

        merge_channels = self.out_channels * self.encoder_depth if self.merge_policy == "cat" else self.out_channels
        self.merge_conv = nn.Conv2d(merge_channels, self.out_channels, kernel_size=1)

        self.apply_stage_indices = self._normalize_apply_stages(apply_stages)
        self.prompt_blocks = nn.ModuleDict()
        for stage_idx in sorted(self.apply_stage_indices):
            self.prompt_blocks[str(stage_idx)] = PromptAttentionBlock(
                channels=self.pyramid_channels,
                condition_dim=int(condition_dim),
                d_model=int(d_model),
                num_prompt_tokens=int(num_prompt_tokens),
                num_heads=int(num_heads),
                ffn_ratio=float(ffn_ratio),
                attn_dropout=float(attn_dropout),
                ffn_dropout=float(ffn_dropout),
                init_scale=float(init_scale),
            )

    def set_current_epoch(self, epoch: Optional[int]) -> None:
        for block in self.prompt_blocks.values():
            block.set_current_epoch(epoch)

    def _normalize_apply_stages(self, apply_stages: Optional[Sequence[int]]) -> List[int]:
        if apply_stages is None:
            # By default only apply to low-resolution stages to constrain compute and stabilize training.
            return [self.encoder_depth - 1, self.encoder_depth]

        values = [int(v) for v in apply_stages]
        if not values:
            return []

        # Resolve positive indexing scheme at list level to avoid ambiguity.
        # If all positives are in [1, depth], treat as 1-based.
        # Else if all are in [0, depth-1], treat as 0-based and shift by +1.
        positives = [v for v in values if v >= 0]
        use_one_based = bool(positives) and all(1 <= v <= self.encoder_depth for v in positives)
        use_zero_based = bool(positives) and all(0 <= v < self.encoder_depth for v in positives)

        normalized = set()
        for raw_idx in values:
            idx = raw_idx
            if idx < 0:
                # Negative indexing: -1 -> depth, -2 -> depth-1, ...
                idx = self.encoder_depth + idx + 1
            else:
                if use_one_based:
                    idx = idx
                elif use_zero_based:
                    idx = idx + 1
                else:
                    # Fallback for mixed/irregular input: prefer explicit 1-based when valid.
                    if 1 <= idx <= self.encoder_depth:
                        idx = idx
                    elif 0 <= idx < self.encoder_depth:
                        idx = idx + 1

            if 1 <= idx <= self.encoder_depth:
                normalized.add(idx)
        return sorted(normalized)

    def _select_stage_features(self, features: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        feat_list = list(features)
        if len(feat_list) == len(self.encoder_channels):
            feat_list = feat_list[1:]
        if len(feat_list) < self.encoder_depth:
            raise ValueError(
                f"PromptFPNDecoder expected at least {self.encoder_depth} stage features, got {len(feat_list)}"
            )
        return feat_list[-self.encoder_depth:]

    def forward(self, features: Sequence[torch.Tensor], cond_vec: torch.Tensor) -> torch.Tensor:
        if cond_vec is None:
            raise ValueError("PromptFPNDecoder requires cond_vec for prompt-attention conditioning")

        stage_features = self._select_stage_features(features)
        laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, stage_features)]

        pyramid = [None] * self.encoder_depth
        current = laterals[-1]
        stage_idx = self.encoder_depth
        if str(stage_idx) in self.prompt_blocks:
            current = self.prompt_blocks[str(stage_idx)](current, cond_vec)
        pyramid[-1] = current

        for i in range(self.encoder_depth - 2, -1, -1):
            current = F.interpolate(current, size=laterals[i].shape[-2:], mode="nearest") + laterals[i]
            stage_idx = i + 1
            if str(stage_idx) in self.prompt_blocks:
                current = self.prompt_blocks[str(stage_idx)](current, cond_vec)
            pyramid[i] = current

        target_size = pyramid[0].shape[-2:]
        processed = []
        for p_feat, seg_block in zip(pyramid, self.seg_blocks):
            y = seg_block(p_feat)
            if y.shape[-2:] != target_size:
                y = F.interpolate(y, size=target_size, mode="nearest")
            processed.append(y)

        if self.merge_policy == "cat":
            fused = torch.cat(processed, dim=1)
        else:
            fused = torch.stack(processed, dim=0).sum(dim=0)

        fused = self.merge_conv(fused)
        return self.dropout(fused)


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
    # Get FPN configuration
    pyramid_channels = int(config.get('model.decoder.pyramid_channels', 256))
    segmentation_channels = int(config.get('model.decoder.segmentation_channels', 128))
    dropout = float(config.get('model.decoder.dropout', 0.2))
    merge_policy = config.get('model.decoder.merge_policy', 'cat')
    
    # Get encoder output channels in FPN format: [input_channels, stage1, stage2, ...]
    # Custom wrapper encoders (Swin/ViT/DINOv3) already include input channel in out_channels
    # SMP encoders (ResNet/EfficientNet) don't include it, need to prepend input channels.
    encoder_input_channels = int(getattr(encoder, "input_channels", 3))
    if hasattr(encoder, 'is_timm_encoder') and encoder.is_timm_encoder:
        encoder_channels = encoder.out_channels
    else:
        # SMP encoders: [c1, c2, c3, c4, c5] -> [input_channels, c1, c2, c3, c4, c5]
        encoder_channels = [encoder_input_channels] + list(encoder.out_channels)
    
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


def build_prompt_fpn_decoder(encoder, config, decoder_type='seg'):
    """Build prompt-aware FPN decoder from configuration."""
    decoder_cfg = config.get('model.decoder', {}) or {}
    prompt_cfg = config.get('model.prompt_fpn', {}) or {}

    pyramid_channels = int(decoder_cfg.get('pyramid_channels', 256))
    segmentation_channels = int(decoder_cfg.get('segmentation_channels', 128))
    dropout = float(decoder_cfg.get('dropout', 0.2))
    merge_policy = decoder_cfg.get('merge_policy', 'cat')

    encoder_input_channels = int(getattr(encoder, "input_channels", 3))
    if hasattr(encoder, 'is_timm_encoder') and encoder.is_timm_encoder:
        encoder_channels = encoder.out_channels
    else:
        encoder_channels = [encoder_input_channels] + list(encoder.out_channels)
    encoder_depth = len(encoder_channels) - 1

    decoder = PromptFPNDecoder(
        encoder_channels=encoder_channels,
        encoder_depth=encoder_depth,
        pyramid_channels=pyramid_channels,
        segmentation_channels=segmentation_channels,
        dropout=dropout,
        merge_policy=merge_policy,
        condition_dim=int(prompt_cfg.get('condition_dim', 128)),
        d_model=int(prompt_cfg.get('d_model', 128)),
        num_prompt_tokens=int(prompt_cfg.get('num_prompt_tokens', 4)),
        num_heads=int(prompt_cfg.get('num_heads', 4)),
        ffn_ratio=float(prompt_cfg.get('ffn_ratio', 4.0)),
        attn_dropout=float(prompt_cfg.get('attn_dropout', 0.1)),
        ffn_dropout=float(prompt_cfg.get('ffn_dropout', 0.1)),
        apply_stages=prompt_cfg.get('apply_stages', None),
        init_scale=float(prompt_cfg.get('init_scale', 0.02)),
    )

    suffix_map = {
        'seg': 'segmentation',
        'det': 'detection',
        'cls': 'classification',
        'reg': 'regression',
    }
    suffix = suffix_map.get(decoder_type, decoder_type)
    print(f"Built PromptFPN decoder for {suffix}")
    print(f"PromptFPN apply stages: {decoder.apply_stage_indices}")
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
    
    prompt_cfg = config.get('model.prompt_fpn', {}) or {}
    prompt_enabled = bool(prompt_cfg.get('enabled', False))
    decoder_backend = str(prompt_cfg.get('decoder_backend', 'base_fpn')).lower()
    use_prompt_decoder = prompt_enabled and decoder_backend == 'prompt_fpn'

    build_decoder_fn = build_prompt_fpn_decoder if use_prompt_decoder else build_fpn_decoder
    backend_name = 'PromptFPN' if use_prompt_decoder else 'FPN'

    # Segmentation decoder
    decoders['fpn_seg'] = build_decoder_fn(encoder, config, decoder_type='seg')
    
    # Detection FPN decoder (separate or shared)
    if config.get('model.decoder.separate_detection_fpn', True):
        decoders['fpn_det'] = build_decoder_fn(encoder, config, decoder_type='det')
        print(f"Using separate {backend_name} decoder for detection")
    else:
        decoders['fpn_det'] = decoders['fpn_seg']  # Share with segmentation
        print(f"Sharing {backend_name} decoder between segmentation and detection")

    # Classification FPN decoder (separate or shared)
    if config.get('model.decoder.separate_classification_fpn', True):
        decoders['fpn_cls'] = build_decoder_fn(encoder, config, decoder_type='cls')
        print(f"Using separate {backend_name} decoder for classification")
    else:
        decoders['fpn_cls'] = decoders['fpn_seg']  # Share with segmentation
        print(f"Sharing {backend_name} decoder between segmentation and classification")

    # Regression FPN decoder (separate or shared)
    if config.get('model.decoder.separate_regression_fpn', True):
        decoders['fpn_reg'] = build_decoder_fn(encoder, config, decoder_type='reg')
        print(f"Using separate {backend_name} decoder for regression")
    else:
        decoders['fpn_reg'] = decoders['fpn_seg']  # Share with segmentation
        print(f"Sharing {backend_name} decoder between segmentation and regression")
    
    return decoders
