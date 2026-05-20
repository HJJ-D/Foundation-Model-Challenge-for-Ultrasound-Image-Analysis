"""Encoder implementations."""

import math
from typing import Optional, List, Sequence, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import segmentation_models_pytorch as smp
from .moe import MoEConvBlock
from .lora import LoraHandle, inject_lora, inject_gated_lora


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
        input_channels: int = 3,
    ):
        super().__init__()
        self.input_channels = int(input_channels)
        # Map simplified name to full timm model name
        full_model_name = SWIN_MODEL_MAPPING.get(model_name, model_name)
        
        self.model = timm.create_model(
            full_model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),  # Swin has 4 stages
            img_size=img_size,  # Specify input image size
            in_chans=self.input_channels,
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
        return [self.input_channels] + list(self._out_channels)


class TimmEncoder(nn.Module):
    """Generic timm encoder wrapper compatible with FPN decoder."""
    is_timm_encoder = True

    def __init__(
        self, 
        model_name: str, 
        pretrained: bool = True, 
        img_size: int = 224, 
        out_indices=None,
        adapter_channels: Optional[int] = None,
        use_adapter: bool = True,
        input_channels: int = 3,
    ):
        super().__init__()
        self.input_channels = int(input_channels)
        full_model_name = VIT_MODEL_MAPPING.get(model_name, model_name)
        if out_indices is not None and not isinstance(out_indices, tuple):
            out_indices = tuple(out_indices)
        self.model = timm.create_model(
            full_model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices,
            img_size=img_size,
            in_chans=self.input_channels,
        )
        self._target_stages = 4
        self._raw_channels = list(self.model.feature_info.channels())
        self._normalized_channels = self._normalize_channels(self._raw_channels)
        self.output_stride = self._infer_output_stride()
        
        # For ViT models, use adapter to create true multi-scale features
        self.use_adapter = use_adapter
        if self.use_adapter and adapter_channels is not None:
            self.adapter = FourScaleAdapter(
                in_channels=self._normalized_channels,
                out_channels=adapter_channels,
                target_strides=(4, 8, 16, 32),
            )
            self._out_channels = list(self.adapter.out_channels)
        else:
            self.adapter = None
            self._out_channels = self._normalized_channels

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
        
        if self.adapter is not None:
            # Use adapter for better multi-scale features (important for ViT)
            return self.adapter(features, x.shape[-2:])
        else:
            # Fallback: simple resize
            features = self._normalize_features(features, x.shape[-2:])
            return features

    @property
    def out_channels(self):
        """Return encoder output channels for FPN decoder."""
        return [self.input_channels] + list(self._out_channels)


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


class InjectorBlock(nn.Module):
    """Inject CNN spatial detail into ViT tokens via deformable cross-attention.

    ViT spatial tokens are reshaped to a 2D grid and used as Q to attend over CNN
    features (KV). The result is projected back to ViT dim and added residually,
    letting ViT attend to fine-grained spatial structure from the CNN pyramid.
    """

    def __init__(
        self,
        vit_dim: int,
        adapter_channels: int,
        num_heads: int = 8,
        num_points: int = 4,
        offset_range: float = 0.25,
    ):
        super().__init__()
        self.vit_dim = vit_dim
        self.adapter_channels = adapter_channels

        self.q_norm = nn.LayerNorm(vit_dim)
        self.q_proj = nn.Linear(vit_dim, adapter_channels, bias=False)
        self.out_proj = nn.Linear(adapter_channels, vit_dim, bias=False)
        nn.init.zeros_(self.out_proj.weight)  # zero-init: identity at step 0

        self.cross_attn = DeformableCrossAttention2D(
            channels=adapter_channels,
            num_heads=num_heads,
            num_points=num_points,
            offset_range=offset_range,
        )
        self.cnn_norm = nn.GroupNorm(_gn_groups(adapter_channels), adapter_channels)

    def forward(
        self,
        vit_tokens: torch.Tensor,  # [B, prefix+N, vit_dim]
        cnn_feat: torch.Tensor,    # [B, adapter_channels, H, W]
        prefix_tokens: int,
    ) -> torch.Tensor:
        B, total_len, _ = vit_tokens.shape
        N = total_len - prefix_tokens
        H_grid = W_grid = int(math.sqrt(N))

        spatial = vit_tokens[:, prefix_tokens:, :]  # [B, N, vit_dim]

        # Project to adapter space and reshape to 2D query map
        q = self.q_proj(self.q_norm(spatial))  # [B, N, adapter_channels]
        q_2d = q.reshape(B, H_grid, W_grid, self.adapter_channels).permute(0, 3, 1, 2)

        # Resize CNN feature to match ViT grid if needed
        cnn_kv = self.cnn_norm(cnn_feat)
        if cnn_kv.shape[2:] != q_2d.shape[2:]:
            cnn_kv = F.interpolate(cnn_kv, size=q_2d.shape[2:], mode='bilinear', align_corners=False)

        # Deformable cross-attention: Q=ViT grid, KV=CNN
        attn_out = self.cross_attn(q_2d, cnn_kv)  # [B, adapter_channels, H_grid, W_grid]

        # Flatten, project back to vit_dim, residual add to spatial tokens
        delta = self.out_proj(
            attn_out.permute(0, 2, 3, 1).reshape(B, N, self.adapter_channels)
        )  # [B, N, vit_dim]

        return torch.cat([vit_tokens[:, :prefix_tokens, :], spatial + delta], dim=1)


class ExtractorBlock(nn.Module):
    """Extract ViT semantics into CNN feature map via deformable cross-attention.

    ViT spatial tokens are reshaped to a 2D map, projected to cnn_channels, then
    used as KV by InteractionBlock (CNN as Q). Wraps InteractionBlock with token
    reshaping and channel projection.
    """

    def __init__(
        self,
        cnn_channels: int,
        vit_dim: int,
        num_heads: int = 8,
        num_points: int = 4,
        offset_range: float = 0.25,
    ):
        super().__init__()
        self.vit_proj = nn.Conv2d(vit_dim, cnn_channels, kernel_size=1, bias=False)
        self.interaction = InteractionBlock(
            channels=cnn_channels,
            num_heads=num_heads,
            num_points=num_points,
            offset_range=offset_range,
        )

    def forward(
        self,
        cnn_feat: torch.Tensor,    # [B, cnn_channels, H, W]
        vit_tokens: torch.Tensor,  # [B, prefix+N, vit_dim]
        prefix_tokens: int,
    ) -> torch.Tensor:
        B, total_len, vit_dim = vit_tokens.shape
        N = total_len - prefix_tokens
        H_vit = W_vit = int(math.sqrt(N))

        spatial = vit_tokens[:, prefix_tokens:, :]  # [B, N, vit_dim]
        vit_2d = spatial.reshape(B, H_vit, W_vit, vit_dim).permute(0, 3, 1, 2)
        vit_kv = self.vit_proj(vit_2d)  # [B, cnn_channels, H_vit, W_vit]

        if vit_kv.shape[2:] != cnn_feat.shape[2:]:
            vit_kv = F.interpolate(vit_kv, size=cnn_feat.shape[2:], mode='bilinear', align_corners=False)

        return self.interaction(cnn_feat, vit_kv)


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
        input_channels: int = 3,
        lora_config: Optional[Dict[str, Any]] = None,
        task_configs: Optional[Sequence[Dict[str, Any]]] = None,
    ):
        super().__init__()
        self.input_channels = int(input_channels)
        # Whether this encoder wants task_id to be passed into forward().
        # Enabled by Gated LoRA below.
        self.supports_task_id = False
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
            in_chans=self.input_channels,
        )
        self.freeze_dino = freeze_dino
        if self.freeze_dino:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()

        # LoRA: inject low-rank trainable branches into frozen DINOv3 layers.
        # Two modes (both implemented in models/lora.py):
        #   - Simple (default): runtime_scale fixed at 1.0 → behaves like
        #     classic fixed-scale LoRA with scaling = alpha / r.
        #   - Gated (lora_config.gated.enabled): per-(layer, group) runtime
        #     scales driven by a prompt controller, written before each forward.
        # Both paths use FusedQKVLoRALinear for qkv (separate Q/K/V B matrices)
        # and LoRALinear for proj/fc1/fc2.
        self.use_lora = False
        self.use_gated_lora = False
        self.lora_handle: Optional[LoraHandle] = None
        self.lora_controller = None
        if lora_config and lora_config.get("enabled", False):
            gated_cfg = lora_config.get("gated", {}) or {}
            if bool(gated_cfg.get("enabled", False)):
                self._inject_gated_lora_branch(lora_config, gated_cfg, task_configs)
            else:
                self._inject_simple_lora_branch(lora_config)

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

    # ------------------------------------------------------------------
    # LoRA injection helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _common_lora_kwargs(lora_config: Dict[str, Any]) -> Dict[str, Any]:
        """Read LoRA hyperparameters shared by simple and gated paths."""
        return dict(
            r=int(lora_config.get("r", 16)),
            alpha=float(lora_config.get("alpha", 32.0)),
            dropout=float(lora_config.get("dropout", 0.0)),
            lora_only=bool(lora_config.get("lora_only", True)),
            target_modules=list(
                lora_config.get("target_modules", ["qkv", "proj", "fc1", "fc2"])
            ),
        )

    def _inject_simple_lora_branch(self, lora_config: Dict[str, Any]) -> None:
        """Simple path: runtime_scale fixed at 1.0, no controller. qkv uses
        FusedQKVLoRALinear so Q/K/V get independent B matrices just like the
        gated path."""
        kw = self._common_lora_kwargs(lora_config)
        handle = inject_lora(vit_model=self.model, **kw)

        self.lora_handle = handle
        self.use_lora = True

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(
            f"✓ LoRA injected into DINOv3: {handle.num_replaced()} layers replaced "
            f"across {len(handle.layer_handles)} blocks "
            f"(targets={kw['target_modules']}, r={kw['r']}, alpha={kw['alpha']}, "
            f"scale={kw['alpha'] / kw['r']:.2f}, lora_only={kw['lora_only']}) | "
            f"trainable params: {trainable:,} / {total:,} "
            f"({100 * trainable / max(total, 1):.2f}%)"
        )

    def _inject_gated_lora_branch(
        self,
        lora_config: Dict[str, Any],
        gated_cfg: Dict[str, Any],
        task_configs: Optional[Sequence[Dict[str, Any]]],
    ) -> None:
        """Gated path: simple injection + a prompt-driven runtime-scale
        controller that is written into every LoRA module before each forward."""
        if task_configs is None:
            raise ValueError(
                "Gated LoRA requires task_configs; pass them through build_encoder."
            )

        kw = self._common_lora_kwargs(lora_config)
        prompt_sources = list(gated_cfg.get("prompt_sources", ["task_type", "task_id"]))
        handle = inject_gated_lora(
            vit_model=self.model,
            task_configs=task_configs,
            **kw,
            prompt_sources=prompt_sources,
            embed_dim=int(gated_cfg.get("embed_dim", 64)),
            hidden_dim=gated_cfg.get("hidden_dim", None),
            activation=str(gated_cfg.get("activation", "sigmoid")),
            max_scale=float(gated_cfg.get("max_scale", 0.5)),
        )
        # Register the controller as a submodule so its params get optimized.
        self.lora_controller = handle.controller
        self.lora_handle = handle
        self.use_lora = True
        self.use_gated_lora = True
        self.supports_task_id = True

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(
            f"✓ Gated LoRA injected into DINOv3: {handle.num_replaced()} layers replaced "
            f"across {len(handle.layer_handles)} blocks "
            f"(targets={kw['target_modules']}, r={kw['r']}, alpha={kw['alpha']}, "
            f"lora_only={kw['lora_only']}, prompts={prompt_sources}, "
            f"activation={gated_cfg.get('activation', 'sigmoid')}, "
            f"max_scale={gated_cfg.get('max_scale', 0.5)}) | "
            f"trainable params: {trainable:,} / {total:,} "
            f"({100 * trainable / max(total, 1):.2f}%)"
        )

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

    def _apply_gated_lora_scales(self, batch_size: int, device: torch.device, task_id: Optional[str]) -> None:
        """Compute per-(layer, group) runtime scales from the controller and
        write them into every LoRA module before the DINO forward runs."""
        if self.lora_controller is None or self.lora_handle is None:
            return
        if task_id is None:
            raise ValueError("Gated LoRA encoder requires a task_id argument in forward().")
        scales = self.lora_controller(
            task_id=task_id, batch_size=batch_size, device=device,
        )
        self.lora_handle.write_scales(scales)
        # Keep a reference to the scales tensor so the autograd graph that feeds
        # the controller is retained through the DINO forward pass.
        self._last_lora_scales = scales

    def forward(self, x, task_id: Optional[str] = None):
        if self.freeze_dino and self.model.training:
            self.model.eval()
            if self.use_lora and self.lora_handle is not None:
                for h in self.lora_handle.layer_handles:
                    for m in (h.qkv, h.proj, h.fc1, h.fc2):
                        if m is not None:
                            m.train()
        if self.use_gated_lora:
            self._apply_gated_lora_scales(batch_size=x.shape[0], device=x.device, task_id=task_id)
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
        return [self.input_channels] + list(self._out_channels)


class ViTAdapterEncoder(nn.Module):
    """ViT-Adapter style encoder: ResNet-50 CNN pyramid + DINOv3 ViT with bidirectional
    interactions between block groups.

    Each block group is followed by:
      - Extractor: CNN ← ViT semantics (deformable cross-attn, CNN as Q, ViT tokens as KV)
      - Injector:  ViT tokens ← CNN spatial detail (deformable cross-attn, ViT as Q, CNN as KV)

    The 4 CNN feature scales (updated by Extractors) are the final encoder output,
    compatible with the existing FPN decoder. LoRA injection into ViT is supported
    via the same interface as Dinov3Encoder.
    """

    is_timm_encoder = True

    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        img_size: int = 224,
        adapter_channels: int = 256,
        interaction_indexes: Optional[List[List[int]]] = None,
        resnet_pretrained: bool = True,
        resnet_freeze_stages: int = 1,
        num_heads: int = 16,
        num_points: int = 4,
        offset_range: float = 0.25,
        freeze_vit: bool = True,
        lora_config: Optional[Dict[str, Any]] = None,
        task_configs: Optional[Sequence[Dict[str, Any]]] = None,
        input_channels: int = 3,
    ):
        super().__init__()
        self.input_channels = int(input_channels)
        self.adapter_channels = adapter_channels
        self.supports_task_id = False

        # Default: ViT-Large (24 blocks) split into 4 groups of 6
        if interaction_indexes is None:
            interaction_indexes = [[0, 5], [6, 11], [12, 17], [18, 23]]
        self.interaction_indexes = interaction_indexes
        num_groups = len(interaction_indexes)

        # ── ViT backbone (loaded without features_only to access individual blocks) ──
        self.vit = timm.create_model(
            model_name,
            pretrained=pretrained,
            img_size=img_size,
            in_chans=self.input_channels,
        )
        self.vit_dim: int = self.vit.embed_dim
        self.num_prefix_tokens: int = getattr(self.vit, 'num_prefix_tokens', 1)

        self.freeze_vit = freeze_vit
        if self.freeze_vit:
            for param in self.vit.parameters():
                param.requires_grad = False
            self.vit.eval()

        # LoRA injection — same interface as Dinov3Encoder
        self.use_lora = False
        self.use_gated_lora = False
        self.lora_handle: Optional[LoraHandle] = None
        self.lora_controller = None
        if lora_config and lora_config.get('enabled', False):
            gated_cfg = lora_config.get('gated', {}) or {}
            if bool(gated_cfg.get('enabled', False)):
                self._inject_gated_lora_branch(lora_config, gated_cfg, task_configs)
            else:
                self._inject_simple_lora_branch(lora_config)

        # ── ResNet-50 CNN pyramid ──
        self.resnet = timm.create_model(
            'resnet50',
            pretrained=resnet_pretrained,
            features_only=True,
            out_indices=(1, 2, 3, 4),  # C2–C5: 256/512/1024/2048 at stride 4/8/16/32
        )
        resnet_channels: List[int] = list(self.resnet.feature_info.channels())
        self._freeze_resnet_stages(resnet_freeze_stages)

        # Project ResNet channels → adapter_channels
        self.cnn_proj = nn.ModuleList([
            nn.Conv2d(ch, adapter_channels, kernel_size=1, bias=False)
            for ch in resnet_channels
        ])

        # ── Bidirectional interaction blocks (one pair per group) ──
        self.injectors = nn.ModuleList([
            InjectorBlock(
                vit_dim=self.vit_dim,
                adapter_channels=adapter_channels,
                num_heads=num_heads,
                num_points=num_points,
                offset_range=offset_range,
            )
            for _ in range(num_groups)
        ])
        self.extractors = nn.ModuleList([
            ExtractorBlock(
                cnn_channels=adapter_channels,
                vit_dim=self.vit_dim,
                num_heads=num_heads,
                num_points=num_points,
                offset_range=offset_range,
            )
            for _ in range(num_groups)
        ])

        self._out_channels = [adapter_channels] * num_groups
        self.output_stride = 32

    # ── Freeze helpers ──

    def _freeze_resnet_stages(self, num_stages: int) -> None:
        """Freeze ResNet-50 stem + first num_stages layers (layer1..layerN)."""
        if num_stages <= 0:
            return
        stem_names = {'conv1', 'bn1', 'act1', 'maxpool', 'stem'}
        layer_names = {f'layer{i}' for i in range(1, num_stages + 1)}
        freeze_names = stem_names | layer_names
        for name, module in self.resnet.named_children():
            if name in freeze_names:
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False

    # ── LoRA helpers (mirrors Dinov3Encoder) ──

    @staticmethod
    def _common_lora_kwargs(lora_config: Dict[str, Any]) -> Dict[str, Any]:
        return dict(
            r=int(lora_config.get('r', 16)),
            alpha=float(lora_config.get('alpha', 32.0)),
            dropout=float(lora_config.get('dropout', 0.0)),
            lora_only=bool(lora_config.get('lora_only', True)),
            target_modules=list(lora_config.get('target_modules', ['qkv', 'proj', 'fc1', 'fc2'])),
        )

    def _inject_simple_lora_branch(self, lora_config: Dict[str, Any]) -> None:
        kw = self._common_lora_kwargs(lora_config)
        handle = inject_lora(vit_model=self.vit, **kw)
        self.lora_handle = handle
        self.use_lora = True
        trainable = sum(p.numel() for p in self.vit.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.vit.parameters())
        print(
            f"✓ LoRA injected into ViT-Adapter: {handle.num_replaced()} layers replaced "
            f"(r={kw['r']}, alpha={kw['alpha']}, lora_only={kw['lora_only']}) | "
            f"trainable: {trainable:,}/{total:,} ({100*trainable/max(total,1):.2f}%)"
        )

    def _inject_gated_lora_branch(
        self,
        lora_config: Dict[str, Any],
        gated_cfg: Dict[str, Any],
        task_configs: Optional[Sequence[Dict[str, Any]]],
    ) -> None:
        if task_configs is None:
            raise ValueError("Gated LoRA requires task_configs.")
        kw = self._common_lora_kwargs(lora_config)
        prompt_sources = list(gated_cfg.get('prompt_sources', ['task_type', 'task_id']))
        handle = inject_gated_lora(
            vit_model=self.vit,
            task_configs=task_configs,
            **kw,
            prompt_sources=prompt_sources,
            embed_dim=int(gated_cfg.get('embed_dim', 64)),
            hidden_dim=gated_cfg.get('hidden_dim', None),
            activation=str(gated_cfg.get('activation', 'sigmoid')),
            max_scale=float(gated_cfg.get('max_scale', 0.5)),
        )
        self.lora_controller = handle.controller
        self.lora_handle = handle
        self.use_lora = True
        self.use_gated_lora = True
        self.supports_task_id = True
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(
            f"✓ Gated LoRA injected into ViT-Adapter: {handle.num_replaced()} layers replaced | "
            f"trainable: {trainable:,}/{total:,} ({100*trainable/max(total,1):.2f}%)"
        )

    def _apply_gated_lora_scales(
        self, batch_size: int, device: torch.device, task_id: Optional[str]
    ) -> None:
        if self.lora_controller is None or self.lora_handle is None:
            return
        if task_id is None:
            raise ValueError("Gated LoRA requires task_id in forward().")
        scales = self.lora_controller(task_id=task_id, batch_size=batch_size, device=device)
        self.lora_handle.write_scales(scales)
        self._last_lora_scales = scales

    # ── ViT manual forward helpers ──

    def _vit_prefix_forward(self, x: torch.Tensor):
        """patch_embed → prepend cls/reg tokens → positional embedding.

        Returns (tokens, rope) where rope may be None for non-EVA ViTs.
        EVA-ViT's _pos_embed returns (x, rope) for rotary position embeddings;
        standard ViT returns just x.
        """
        x = self.vit.patch_embed(x)
        rope = None
        if hasattr(self.vit, '_pos_embed'):
            result = self.vit._pos_embed(x)
            if isinstance(result, tuple):
                x, rope = result
            else:
                x = result
        else:
            # Fallback for older timm builds
            B = x.shape[0]
            prefix = []
            if hasattr(self.vit, 'cls_token') and self.vit.cls_token is not None:
                prefix.append(self.vit.cls_token.expand(B, -1, -1))
            if hasattr(self.vit, 'reg_token') and self.vit.reg_token is not None:
                prefix.append(self.vit.reg_token.expand(B, -1, -1))
            if prefix:
                x = torch.cat(prefix + [x], dim=1)
            if hasattr(self.vit, 'pos_embed') and self.vit.pos_embed is not None:
                x = x + self.vit.pos_embed
            if hasattr(self.vit, 'pos_drop'):
                x = self.vit.pos_drop(x)
        if hasattr(self.vit, 'patch_drop') and self.vit.patch_drop is not None:
            x = self.vit.patch_drop(x)
        if hasattr(self.vit, 'norm_pre') and self.vit.norm_pre is not None:
            x = self.vit.norm_pre(x)
        return x, rope

    # ── Main forward ──

    def forward(self, x: torch.Tensor, task_id: Optional[str] = None) -> List[torch.Tensor]:
        if self.freeze_vit and self.vit.training:
            self.vit.eval()
            if self.use_lora and self.lora_handle is not None:
                for h in self.lora_handle.layer_handles:
                    for m in (h.qkv, h.proj, h.fc1, h.fc2):
                        if m is not None:
                            m.train()
        if self.use_gated_lora:
            self._apply_gated_lora_scales(x.shape[0], x.device, task_id)

        # CNN pyramid: 4 scales at stride 4/8/16/32, projected to adapter_channels
        cnn_scales = [proj(feat) for proj, feat in zip(self.cnn_proj, self.resnet(x))]

        # ViT: patch embed + prefix tokens + positional encoding
        # rope is None for standard ViT; EVA-ViT returns rotary position embeddings
        vit_tokens, rope = self._vit_prefix_forward(x)

        # Run block groups with bidirectional Extractor→Injector between each group
        for group_idx, (start, end) in enumerate(self.interaction_indexes):
            for block_idx in range(start, end + 1):
                out = self.vit.blocks[block_idx](vit_tokens, rope=rope) if rope is not None else self.vit.blocks[block_idx](vit_tokens)
                vit_tokens = out[0] if isinstance(out, tuple) else out

            # Extractor: CNN ← ViT semantics (updates cnn_scales in-place for this group)
            cnn_scales[group_idx] = self.extractors[group_idx](
                cnn_scales[group_idx], vit_tokens, self.num_prefix_tokens
            )
            # Injector: ViT tokens ← CNN spatial detail (CNN now enriched by Extractor)
            vit_tokens = self.injectors[group_idx](
                vit_tokens, cnn_scales[group_idx], self.num_prefix_tokens
            )

        return cnn_scales

    def get_moe_aux_loss(self) -> torch.Tensor:
        device = next(self.parameters()).device
        return torch.tensor(0.0, device=device)

    def get_moe_stats(self) -> List:
        return []

    @property
    def out_channels(self) -> List[int]:
        return [self.input_channels] + list(self._out_channels)


def build_encoder(config, task_ids=None, input_channels: int = 3, task_configs=None):
    """
    Build encoder from configuration.

    Args:
        config: Configuration object
        task_ids: Optional list of task IDs (used by MoE task embedding).
        input_channels: Number of input channels for encoder stem.
        task_configs: Optional list of full task config dicts (task_id, task_name,
            num_classes, …). Required for Gated LoRA prompt vocabularies.

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
            input_channels=input_channels,
        )
        print(f"Loaded Swin Transformer: {encoder_name} (img_size={img_size})")
    
    elif encoder_name.startswith('vit_'):
        pretrained = (encoder_weights == 'imagenet' or encoder_weights is not None)
        timm_name = config.get('model.encoder.timm_name', None) or VIT_MODEL_MAPPING.get(encoder_name, encoder_name)
        adapter_cfg = config.get('model.encoder.adapter', {}) or {}
        adapter_type = adapter_cfg.get('type', None)
        freeze_encoder = config.get('model.encoder.freeze_encoder', False)
        lora_config = config.get('model.encoder.lora', None)

        # Advanced path: freeze/LoRA/ViT-Adapter — reuse the same classes as DINOv3
        if adapter_type == 'vit_adapter':
            adapter_channels = int(adapter_cfg.get('channels', 256))
            interaction_indexes = adapter_cfg.get(
                'interaction_indexes', [[0, 2], [3, 5], [6, 8], [9, 11]]
            )
            encoder = ViTAdapterEncoder(
                model_name=timm_name,
                pretrained=pretrained,
                img_size=img_size,
                adapter_channels=adapter_channels,
                interaction_indexes=interaction_indexes,
                resnet_pretrained=bool(adapter_cfg.get('pyramid_pretrained', True)),
                resnet_freeze_stages=int(adapter_cfg.get('pyramid_freeze_stages', 1)),
                num_heads=int(adapter_cfg.get('interaction_heads', 16)),
                num_points=int(adapter_cfg.get('interaction_points', 4)),
                offset_range=float(adapter_cfg.get('interaction_offset_range', 0.25)),
                freeze_vit=freeze_encoder,
                lora_config=lora_config,
                task_configs=task_configs,
                input_channels=input_channels,
            )
            print(
                f"Loaded ViT + ViT-Adapter encoder: {timm_name} "
                f"(img_size={img_size}, adapter_channels={adapter_channels}, "
                f"groups={interaction_indexes}, freeze={freeze_encoder})"
            )
        elif freeze_encoder or (lora_config and lora_config.get('enabled', False)):
            # Freeze + LoRA with resize/spm adapter (same as Dinov3Encoder but
            # backed by an ImageNet-supervised ViT instead of DINOv3)
            adapter_channels = int(adapter_cfg.get(
                'channels', config.get('model.encoder.adapter_channels', 256)
            ))
            spm_stem_channels = int(adapter_cfg.get('spm_stem_channels', 64))
            vit_layer_mapping = adapter_cfg.get('vit_layer_mapping', None)
            a_type = adapter_cfg.get('type', 'resize') or 'resize'
            encoder = Dinov3Encoder(
                model_name=timm_name,
                pretrained=pretrained,
                img_size=img_size,
                out_indices=out_indices,
                adapter_channels=adapter_channels,
                adapter_type=a_type,
                spm_stem_channels=spm_stem_channels,
                interaction_heads=int(adapter_cfg.get('interaction_heads', 8)),
                interaction_points=int(adapter_cfg.get('interaction_points', 4)),
                interaction_offset_range=float(adapter_cfg.get('interaction_offset_range', 0.25)),
                freeze_dino=freeze_encoder,
                vit_layer_mapping=vit_layer_mapping,
                input_channels=input_channels,
                lora_config=lora_config,
                task_configs=task_configs,
            )
            print(
                f"Loaded ViT encoder (freeze+adapt): {timm_name} "
                f"(img_size={img_size}, adapter={a_type}, channels={adapter_channels}, "
                f"freeze={freeze_encoder})"
            )
        else:
            # Simple path: full finetune with FourScaleAdapter (original behavior)
            adapter_channels = config.get('model.encoder.adapter_channels', None)
            if adapter_channels is not None:
                adapter_channels = int(adapter_channels)

            encoder = TimmEncoder(
                model_name=encoder_name,
                pretrained=pretrained,
                img_size=img_size,
                out_indices=out_indices,
                adapter_channels=adapter_channels,
                use_adapter=True,
                input_channels=input_channels,
            )
            full_name = VIT_MODEL_MAPPING.get(encoder_name, encoder_name)
            adapter_info = f" with adapter (channels={adapter_channels})" if adapter_channels else ""
            print(f"Loaded ViT encoder: {encoder_name} -> {full_name} (img_size={img_size}){adapter_info}")
    
    elif encoder_name.startswith('dinov3') or (encoder_name.startswith('timm:') and 'dinov3' in encoder_name):
        pretrained = (encoder_weights == 'imagenet' or encoder_weights is not None)
        timm_name = config.get('model.encoder.timm_name', encoder_name.replace('timm:', ''))
        adapter_cfg = config.get('model.encoder.adapter', {}) or {}
        adapter_type = adapter_cfg.get('type', 'resize')
        adapter_channels = int(adapter_cfg.get(
            'channels',
            config.get('model.encoder.adapter_channels', 256)
        ))
        freeze_dino = config.get('model.encoder.freeze_dino', True)
        lora_config = config.get('model.encoder.lora', None)

        if adapter_type == 'vit_adapter':
            interaction_indexes = adapter_cfg.get(
                'interaction_indexes', [[0, 5], [6, 11], [12, 17], [18, 23]]
            )
            encoder = ViTAdapterEncoder(
                model_name=timm_name,
                pretrained=pretrained,
                img_size=img_size,
                adapter_channels=adapter_channels,
                interaction_indexes=interaction_indexes,
                resnet_pretrained=bool(adapter_cfg.get('pyramid_pretrained', True)),
                resnet_freeze_stages=int(adapter_cfg.get('pyramid_freeze_stages', 1)),
                num_heads=int(adapter_cfg.get('interaction_heads', 16)),
                num_points=int(adapter_cfg.get('interaction_points', 4)),
                offset_range=float(adapter_cfg.get('interaction_offset_range', 0.25)),
                freeze_vit=freeze_dino,
                lora_config=lora_config,
                task_configs=task_configs,
                input_channels=input_channels,
            )
            print(
                f"Loaded ViT-Adapter encoder: {timm_name} "
                f"(img_size={img_size}, adapter_channels={adapter_channels}, "
                f"groups={interaction_indexes}, freeze_vit={freeze_dino})"
            )
        else:
            spm_stem_channels = int(adapter_cfg.get('spm_stem_channels', 64))
            interaction_heads = int(adapter_cfg.get('interaction_heads', 8))
            interaction_points = int(adapter_cfg.get('interaction_points', 4))
            interaction_offset_range = float(adapter_cfg.get('interaction_offset_range', 0.25))
            vit_layer_mapping = adapter_cfg.get('vit_layer_mapping', None)
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
                input_channels=input_channels,
                lora_config=lora_config,
                task_configs=task_configs,
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
            adapter_channels = config.get('model.encoder.adapter_channels', None)
            if adapter_channels is not None:
                adapter_channels = int(adapter_channels)
            
            encoder = TimmEncoder(
                model_name=timm_name,
                pretrained=pretrained,
                img_size=img_size,
                out_indices=out_indices,
                adapter_channels=adapter_channels,
                use_adapter=(adapter_channels is not None),
                input_channels=input_channels,
            )
            adapter_info = f" with adapter (channels={adapter_channels})" if adapter_channels else ""
            print(f"Loaded timm encoder: {timm_name} (img_size={img_size}){adapter_info}")
        else:
            # Use standard segmentation_models_pytorch encoder, fallback to timm if not found
            try:
                encoder = smp.encoders.get_encoder(
                    name=encoder_name,
                    in_channels=input_channels,
                    depth=5,
                    weights=encoder_weights
                )
                print(f"Loaded encoder: {encoder_name}")
            except Exception:
                pretrained = (encoder_weights == 'imagenet' or encoder_weights is not None)
                adapter_channels = config.get('model.encoder.adapter_channels', None)
                if adapter_channels is not None:
                    adapter_channels = int(adapter_channels)
                
                encoder = TimmEncoder(
                    model_name=timm_name,
                    pretrained=pretrained,
                    img_size=img_size,
                    out_indices=out_indices,
                    adapter_channels=adapter_channels,
                    use_adapter=(adapter_channels is not None),
                    input_channels=input_channels,
                )
                adapter_info = f" with adapter (channels={adapter_channels})" if adapter_channels else ""
                print(f"Loaded timm encoder: {timm_name} (img_size={img_size}){adapter_info}")
    
    # Keep a unified attribute for downstream modules (e.g., decoder channel metadata).
    encoder.input_channels = int(input_channels)
    return encoder
