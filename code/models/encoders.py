"""Encoder implementations."""

import math
from typing import Optional, List, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import segmentation_models_pytorch as smp
from .moe import MoEConvBlock


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


def _gn_groups(channels: int) -> int:
    groups = min(32, channels)
    while channels % groups != 0:
        groups -= 1
    return groups


class SwinTransformerEncoder(nn.Module):
    """Swin Transformer encoder wrapper compatible with FPN decoder."""
    is_timm_encoder = True
    
    def __init__(
        self,
        model_name: str = 'swin_b',
        pretrained: bool = True,
        img_size: int = 224,
        moe_config: Optional[dict] = None,
        task_ids: Optional[List[str]] = None,
    ):
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

        # Optional MoE blocks between stages (applied on stage outputs)
        moe_cfg = moe_config or {}
        self.use_moe = moe_cfg.get('enabled', False)
        self.moe_stage_indices = moe_cfg.get('stage_indices', None)
        self.supports_task_id = False
        self.handles_moe = False
        self._moe_warned = False
        if self.use_moe:
            num_experts = moe_cfg.get('num_experts', 4)
            expert_hidden = moe_cfg.get('expert_hidden', None)
            router_hidden = moe_cfg.get('router_hidden', None)
            top_k = moe_cfg.get('top_k', 1)
            use_task_embedding = moe_cfg.get('use_task_embedding', True)
            task_embedding_dim = moe_cfg.get('task_embedding_dim', 32)
            use_residual = moe_cfg.get('use_residual', True)
            dropout = moe_cfg.get('dropout', 0.0)

            if use_task_embedding and not task_ids:
                print("Warning: MoE task embedding enabled but task_ids not provided; disabling task embedding.")
                use_task_embedding = False

            self.moe_blocks = nn.ModuleList([
                MoEConvBlock(
                    in_channels=ch,
                    num_experts=num_experts,
                    expert_hidden=expert_hidden,
                    router_hidden=router_hidden,
                    top_k=top_k,
                    use_task_embedding=use_task_embedding,
                    task_embedding_dim=task_embedding_dim,
                    task_ids=task_ids,
                    use_residual=use_residual,
                    dropout=dropout,
                )
                for ch in self._out_channels
            ])
            self.supports_task_id = True
            self.handles_moe = True
            print(f"✓ Swin MoE enabled on {len(self.moe_blocks)} stages (experts={num_experts}, top_k={top_k})")
        
    def forward(self, x, task_id=None):
        features = self.model(x)
        # Swin outputs features in (B, H, W, C) format, need to convert to (B, C, H, W)
        features = [feat.permute(0, 3, 1, 2).contiguous() for feat in features]
        if self.use_moe:
            features = self._apply_moe(features, task_id)
        return features

    def _apply_moe(self, features, task_id):
        if len(features) != len(self.moe_blocks):
            if not self._moe_warned:
                print("Warning: Swin MoE stage count mismatch, applying to min length.")
                self._moe_warned = True
            max_len = min(len(features), len(self.moe_blocks))
            for idx in range(max_len):
                if self.moe_stage_indices is None or idx in self.moe_stage_indices:
                    features[idx] = self.moe_blocks[idx](features[idx], task_id)
            return features

        out = []
        for idx, (feat, moe) in enumerate(zip(features, self.moe_blocks)):
            if self.moe_stage_indices is None or idx in self.moe_stage_indices:
                out.append(moe(feat, task_id))
            else:
                out.append(feat)
        return out

    def get_moe_aux_loss(self):
        if not self.use_moe:
            device = next(self.parameters()).device
            return torch.tensor(0.0, device=device)
        total = None
        for block in self.moe_blocks:
            aux = getattr(block, "last_aux_loss", None)
            if aux is None:
                continue
            total = aux if total is None else total + aux
        if total is None:
            device = next(self.parameters()).device
            total = torch.tensor(0.0, device=device)
        return total

    def get_moe_stats(self):
        if not self.use_moe:
            return []
        stats = []
        for block in self.moe_blocks:
            block_stats = block.get_stats()
            if block_stats is not None:
                stats.append(block_stats)
        return stats
    
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


class FourScaleAdapter(nn.Module):
    """Adapt arbitrary backbone features to 4 scales: stride 4/8/16/32."""

    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: Optional[int] = None,
        target_strides: Sequence[int] = (4, 8, 16, 32),
    ):
        super().__init__()
        if not in_channels:
            raise ValueError("in_channels cannot be empty")
        self.target_strides = list(target_strides)
        self.in_channels = list(in_channels)
        self.out_channels = [out_channels if out_channels is not None else ch for ch in self.in_channels]
        self.proj = nn.ModuleList([
            nn.Identity() if out_ch == in_ch else nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
            for in_ch, out_ch in zip(self.in_channels, self.out_channels)
        ])

    def forward(self, features: List[torch.Tensor], input_hw):
        if len(features) > len(self.target_strides):
            features = features[:len(self.target_strides)]
        while len(features) < len(self.target_strides):
            features.append(features[-1])

        target_sizes = [
            (max(1, input_hw[0] // stride), max(1, input_hw[1] // stride))
            for stride in self.target_strides
        ]

        out = []
        for feat, proj, target in zip(features, self.proj, target_sizes):
            feat = proj(feat)
            if feat.shape[2] == target[0] and feat.shape[3] == target[1]:
                out.append(feat)
            elif feat.shape[2] >= target[0] and feat.shape[3] >= target[1]:
                out.append(F.adaptive_avg_pool2d(feat, output_size=target))
            else:
                out.append(F.interpolate(feat, size=target, mode="bilinear", align_corners=False))
        return out


class ConvGNAct(nn.Module):
    """Conv + GroupNorm + SiLU block."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.GroupNorm(_gn_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SpatialPyramidModule(nn.Module):
    """
    Build a true CNN pyramid from input image at strides 4/8/16/32.
    """

    def __init__(self, out_channels: Sequence[int], stem_channels: int = 64):
        super().__init__()
        if len(out_channels) != 4:
            raise ValueError("SPM expects 4 output channels for strides 4/8/16/32")
        c2, c3, c4, c5 = out_channels
        self.stem = nn.Sequential(
            ConvGNAct(3, stem_channels, kernel_size=3, stride=2),  # s2
            ConvGNAct(stem_channels, stem_channels, kernel_size=3, stride=1),
        )
        self.stage2 = nn.Sequential(  # s4
            ConvGNAct(stem_channels, c2, kernel_size=3, stride=2),
            ConvGNAct(c2, c2, kernel_size=3, stride=1),
        )
        self.stage3 = nn.Sequential(  # s8
            ConvGNAct(c2, c3, kernel_size=3, stride=2),
            ConvGNAct(c3, c3, kernel_size=3, stride=1),
        )
        self.stage4 = nn.Sequential(  # s16
            ConvGNAct(c3, c4, kernel_size=3, stride=2),
            ConvGNAct(c4, c4, kernel_size=3, stride=1),
        )
        self.stage5 = nn.Sequential(  # s32
            ConvGNAct(c4, c5, kernel_size=3, stride=2),
            ConvGNAct(c5, c5, kernel_size=3, stride=1),
        )

    def forward(self, x):
        x = self.stem(x)
        c2 = self.stage2(x)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)
        c5 = self.stage5(c4)
        return [c2, c3, c4, c5]


class DeformableCrossAttention2D(nn.Module):
    """
    Deformable cross-attention from CNN grid (query) to ViT map (key/value).
    """

    def __init__(self, channels: int, num_heads: int = 8, num_points: int = 4, offset_range: float = 0.25):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels ({channels}) must be divisible by num_heads ({num_heads})")
        self.channels = channels
        self.num_heads = num_heads
        self.num_points = num_points
        self.head_dim = channels // num_heads
        self.offset_range = offset_range

        self.q_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.k_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.v_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.offset_proj = nn.Conv2d(channels, num_heads * num_points * 2, kernel_size=3, padding=1)
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def _base_grid(self, batch, height, width, device, dtype):
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype),
            torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype),
            indexing="ij",
        )
        grid = torch.stack((xx, yy), dim=-1)  # [H, W, 2]
        grid = grid.unsqueeze(0).unsqueeze(1).repeat(batch, self.num_heads, 1, 1, 1)
        return grid

    def forward(self, query_map: torch.Tensor, kv_map: torch.Tensor):
        b, c, h, w = query_map.shape

        q = self.q_proj(query_map).reshape(b, self.num_heads, self.head_dim, h, w)
        k = self.k_proj(kv_map).reshape(b * self.num_heads, self.head_dim, kv_map.shape[2], kv_map.shape[3])
        v = self.v_proj(kv_map).reshape(b * self.num_heads, self.head_dim, kv_map.shape[2], kv_map.shape[3])

        offsets = self.offset_proj(query_map)
        offsets = offsets.view(b, self.num_heads, self.num_points, 2, h, w)
        offsets = torch.tanh(offsets).permute(0, 1, 2, 4, 5, 3).contiguous() * self.offset_range

        base = self._base_grid(b, h, w, query_map.device, query_map.dtype)  # [B, heads, H, W, 2]
        q = q.reshape(b * self.num_heads, self.head_dim, h, w)

        logits = []
        sampled_values = []
        scale = math.sqrt(self.head_dim)
        for p in range(self.num_points):
            grid = base + offsets[:, :, p]  # [B, heads, H, W, 2]
            grid = grid.view(b * self.num_heads, h, w, 2)

            k_sample = F.grid_sample(k, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
            v_sample = F.grid_sample(v, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

            attn_logit = (q * k_sample).sum(dim=1, keepdim=True) / scale
            logits.append(attn_logit)
            sampled_values.append(v_sample)

        attn = torch.softmax(torch.cat(logits, dim=1), dim=1)  # [B*heads, points, H, W]
        out = 0.0
        for p in range(self.num_points):
            out = out + attn[:, p:p + 1] * sampled_values[p]

        out = out.view(b, self.num_heads, self.head_dim, h, w).reshape(b, c, h, w)
        return self.out_proj(out)


class InteractionBlock(nn.Module):
    """Inject ViT semantics into CNN grid with deformable cross-attention."""

    def __init__(self, channels: int, num_heads: int = 8, num_points: int = 4, offset_range: float = 0.25):
        super().__init__()
        self.cross_attn = DeformableCrossAttention2D(
            channels=channels,
            num_heads=num_heads,
            num_points=num_points,
            offset_range=offset_range,
        )
        self.norm1 = nn.GroupNorm(_gn_groups(channels), channels)
        self.norm2 = nn.GroupNorm(_gn_groups(channels), channels)
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, cnn_feat: torch.Tensor, vit_feat: torch.Tensor):
        x = cnn_feat + self.cross_attn(self.norm1(cnn_feat), vit_feat)
        x = x + self.ffn(self.norm2(x))
        return x


class Dinov3Encoder(nn.Module):
    """DINOv3 encoder wrapper with 4-scale adapter (stride 4/8/16/32)."""

    is_timm_encoder = True

    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        img_size: int = 224,
        out_indices: Optional[Sequence[int]] = None,
        adapter_channels: Optional[int] = 256,
        adapter_type: str = "resize",
        spm_stem_channels: int = 64,
        interaction_heads: int = 8,
        interaction_points: int = 4,
        interaction_offset_range: float = 0.25,
        freeze_dino: bool = True,
        vit_layer_mapping: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        self._target_stages = 4
        if out_indices is None:
            out_indices = (2, 5, 8, 11)
        elif not isinstance(out_indices, tuple):
            out_indices = tuple(out_indices)
        self.out_indices = out_indices
        
        # ViT layer mapping for hierarchical interaction
        # Default: [0, 1, 2, 3] means s4->layer[0], s8->layer[1], s16->layer[2], s32->layer[3]
        # Example: [0, 0, 2, 3] means s4/s8 both use layer[0], s16->layer[2], s32->layer[3]
        if vit_layer_mapping is None:
            self.vit_layer_mapping = list(range(self._target_stages))
        else:
            self.vit_layer_mapping = list(vit_layer_mapping)
            if len(self.vit_layer_mapping) != self._target_stages:
                raise ValueError(
                    f"vit_layer_mapping must have {self._target_stages} elements, "
                    f"got {len(self.vit_layer_mapping)}"
                )

        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=self.out_indices,
            img_size=img_size,
        )
        self.freeze_dino = freeze_dino
        if self.freeze_dino:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()

        if not hasattr(self.model, "feature_info"):
            raise ValueError(f"DINOv3 features_only model has no feature_info: {model_name}")
        self._raw_channels = list(self.model.feature_info.channels())
        self._raw_channels = self._normalize_channels(self._raw_channels)

        self.adapter_type = adapter_type
        if self.adapter_type == "resize":
            self.adapter = FourScaleAdapter(
                in_channels=self._raw_channels,
                out_channels=adapter_channels,
                target_strides=(4, 8, 16, 32),
            )
            self._out_channels = list(self.adapter.out_channels)
        elif self.adapter_type == "spm_interaction":
            ch = int(adapter_channels or 256)
            out_channels = [ch, ch, ch, ch]
            self.spm = SpatialPyramidModule(out_channels=out_channels, stem_channels=spm_stem_channels)
            self.vit_proj = nn.ModuleList([
                nn.Conv2d(in_ch, ch, kernel_size=1, bias=False)
                for in_ch in self._raw_channels
            ])
            self.interaction = nn.ModuleList([
                InteractionBlock(
                    channels=ch,
                    num_heads=interaction_heads,
                    num_points=interaction_points,
                    offset_range=interaction_offset_range,
                )
                for _ in range(4)
            ])
            self._out_channels = out_channels
        else:
            raise ValueError(
                f"Unsupported adapter_type: {self.adapter_type}. "
                "Use 'resize' or 'spm_interaction'."
            )
        self.output_stride = 32

    def _normalize_channels(self, channels):
        if not channels:
            raise ValueError("dinov3 features_only returned no feature channels")
        if len(channels) >= self._target_stages:
            return list(channels[:self._target_stages])
        return list(channels) + [channels[-1]] * (self._target_stages - len(channels))

    def _tokens_to_feature_map(self, tokens):
        # tokens: (B, N, C) -> (B, C, H, W)
        grid_size = None
        if hasattr(self.model, "patch_embed") and hasattr(self.model.patch_embed, "grid_size"):
            grid_size = self.model.patch_embed.grid_size

        prefix_tokens = getattr(self.model, "num_prefix_tokens", 0)
        if grid_size is None:
            n_tokens = tokens.shape[1]
            usable = n_tokens - prefix_tokens
            side = int(math.sqrt(max(usable, 1)))
            if side * side != usable:
                side = int(math.sqrt(max(n_tokens, 1)))
                prefix_tokens = max(0, n_tokens - side * side)
            grid_size = (side, side)

        if prefix_tokens > 0:
            tokens = tokens[:, prefix_tokens:, :]
        feat = tokens.reshape(tokens.shape[0], grid_size[0], grid_size[1], tokens.shape[2])
        return feat.permute(0, 3, 1, 2).contiguous()

    def _format_feature(self, feat, expected_channels):
        if feat.ndim == 3:
            feat = self._tokens_to_feature_map(feat)
        if feat.ndim == 4:
            if feat.shape[1] == expected_channels:
                return feat
            if feat.shape[-1] == expected_channels:
                return feat.permute(0, 3, 1, 2).contiguous()
        return feat

    def _extract_features(self, x):
        features = self.model(x)
        if not isinstance(features, (list, tuple)):
            features = [features]
        features = [
            self._format_feature(feat, self._raw_channels[idx] if idx < len(self._raw_channels) else self._raw_channels[-1])
            for idx, feat in enumerate(features)
        ]
        if len(features) > self._target_stages:
            features = features[:self._target_stages]
        while len(features) < self._target_stages:
            features.append(features[-1])
        return features

    def forward(self, x):
        if self.freeze_dino and self.model.training:
            self.model.eval()
        raw_features = self._extract_features(x)
        if self.adapter_type == "resize":
            return self.adapter(raw_features, x.shape[-2:])

        # spm_interaction: true CNN pyramid as query, ViT features as key/value
        cnn_pyramid = self.spm(x)
        if len(raw_features) > 4:
            raw_features = raw_features[:4]
        while len(raw_features) < 4:
            raw_features.append(raw_features[-1])

        # Apply hierarchical ViT layer mapping
        # Map each CNN pyramid scale to its corresponding ViT layer
        vit_maps = []
        for i, cnn_scale_idx in enumerate(range(len(cnn_pyramid))):
            vit_layer_idx = self.vit_layer_mapping[i]
            vit_layer_idx = min(vit_layer_idx, len(raw_features) - 1)  # Clamp to valid range
            vit_feat = raw_features[vit_layer_idx]
            vit_maps.append(self.vit_proj[i](vit_feat))
        
        fused = [
            inter(cnn_feat, vit_feat)
            for cnn_feat, vit_feat, inter in zip(cnn_pyramid, vit_maps, self.interaction)
        ]
        return fused

    @property
    def out_channels(self):
        return [3] + list(self._out_channels)


def build_encoder(config, task_ids=None):
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
            img_size=img_size,
            moe_config=config.get('model.moe', {}),
            task_ids=task_ids,
        )
        print(f"Loaded Swin Transformer: {encoder_name} (img_size={img_size})")
    elif encoder_name.startswith('dinov3') or (encoder_name.startswith('timm:') and 'dinov3' in encoder_name):
        pretrained = (encoder_weights == 'imagenet' or encoder_weights is not None)
        timm_name = config.get('model.encoder.timm_name', encoder_name.replace('timm:', ''))
        adapter_cfg = config.get('model.encoder.adapter', {}) or {}
        adapter_type = adapter_cfg.get('type', 'resize')
        adapter_channels = int(adapter_cfg.get(
            'channels',
            config.get('model.encoder.adapter_channels', 256)
        ))
        spm_stem_channels = int(adapter_cfg.get('spm_stem_channels', 64))
        interaction_heads = int(adapter_cfg.get('interaction_heads', 8))
        interaction_points = int(adapter_cfg.get('interaction_points', 4))
        interaction_offset_range = float(adapter_cfg.get('interaction_offset_range', 0.25))
        vit_layer_mapping = adapter_cfg.get('vit_layer_mapping', None)
        freeze_dino = config.get('model.encoder.freeze_dino', True)
        encoder = Dinov3Encoder(
            model_name=timm_name,
            pretrained=pretrained,
            img_size=img_size,
            out_indices=out_indices,
            adapter_channels=adapter_channels,
            adapter_type=adapter_type,
            spm_stem_channels=spm_stem_channels,
            interaction_heads=interaction_heads,
            interaction_points=interaction_points,
            interaction_offset_range=interaction_offset_range,
            freeze_dino=freeze_dino,
            vit_layer_mapping=vit_layer_mapping,
        )
        mapping_info = f", vit_layer_mapping={vit_layer_mapping}" if vit_layer_mapping else ""
        print(
            f"Loaded DINOv3 encoder: {timm_name} "
            f"(img_size={img_size}, adapter_type={adapter_type}, "
            f"adapter_channels={adapter_channels}, freeze_dino={freeze_dino}{mapping_info})"
        )
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
