"""Multi-task model architecture."""

import torch
import torch.nn as nn
from .encoders import build_encoder
from .decoders import build_decoders
from .heads import build_all_heads
from .moe import MoEConvBlock
from .film_layer import TaskFiLMGenerator, TaskEmbeddingFiLMGenerator, FiLMLayer


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
        task_ids = [cfg['task_id'] for cfg in self.task_configs]
        self.encoder = build_encoder(config, task_ids=task_ids)
        
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
        self.fpn_decoder_cls = decoders['fpn_cls']
        self.fpn_decoder_reg = decoders['fpn_reg']
        self.use_fpn_for_cls = config.get('model.decoder.use_fpn_for_classification', True)
        self.use_fpn_for_reg = config.get('model.decoder.use_fpn_for_regression', True)
        
        # Get FPN output channels
        self.fpn_out_channels = self.fpn_decoder_seg.out_channels
        
        # Build FiLM modulation (optional)
        self.use_film = config.get('model.use_film', False)
        if self.use_film:
            task_ids = [cfg['task_id'] for cfg in self.task_configs]
            film_config = config.get('model.film', {})
            use_embedding = film_config.get('use_task_embedding', False)
            embedding_dim = int(film_config.get('embedding_dim', 64))
            use_affine = film_config.get('use_affine', True)
            
            if use_embedding:
                self.film_generator = TaskEmbeddingFiLMGenerator(
                    task_ids=task_ids,
                    num_features=self.fpn_out_channels,
                    embedding_dim=embedding_dim,
                    use_affine=use_affine
                )
                print(f"✓ Using FiLM with task embeddings (dim={embedding_dim})")
            else:
                self.film_generator = TaskFiLMGenerator(
                    task_ids=task_ids,
                    num_features=self.fpn_out_channels,
                    use_affine=use_affine
                )
                print(f"✓ Using FiLM with per-task parameters")
            
            self.film_layer = FiLMLayer(
                num_features=self.fpn_out_channels,
                use_affine=use_affine
            )

        # Build MoE blocks (optional)
        moe_cfg = config.get('model.moe', {}) or {}
        self.use_moe = moe_cfg.get('enabled', False) and not getattr(self.encoder, "handles_moe", False)
        self.moe_stage_indices = moe_cfg.get('stage_indices', None)
        self._moe_warned = False
        if self.use_moe:
            task_ids = [cfg['task_id'] for cfg in self.task_configs]
            num_experts = int(moe_cfg.get('num_experts', 4))
            expert_hidden = int(moe_cfg.get('expert_hidden')) if moe_cfg.get('expert_hidden') is not None else None
            router_hidden = int(moe_cfg.get('router_hidden')) if moe_cfg.get('router_hidden') is not None else None
            top_k = int(moe_cfg.get('top_k', 1))
            use_task_embedding = moe_cfg.get('use_task_embedding', True)
            task_embedding_dim = int(moe_cfg.get('task_embedding_dim', 32))
            use_residual = moe_cfg.get('use_residual', True)
            dropout = float(moe_cfg.get('dropout', 0.0))

            moe_channels = list(encoder_channels)
            if getattr(self.encoder, "is_timm_encoder", False) and len(moe_channels) > 1:
                moe_channels = moe_channels[1:]

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
                for ch in moe_channels
            ])
            print(f"✓ Using MoE on {len(self.moe_blocks)} encoder stages (experts={num_experts}, top_k={top_k})")
        
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
        if getattr(self.encoder, "supports_task_id", False):
            features = self.encoder(x, task_id)
        else:
            features = self.encoder(x)
        if self.use_moe:
            features = self._apply_moe(features, task_id)
        
        # Route through appropriate decoder and head
        if task_name == 'segmentation':
            fpn_features = self.fpn_decoder_seg(features)
            
            # Apply FiLM modulation if enabled
            if self.use_film:
                gamma, beta = self.film_generator(task_id)
                fpn_features = self.film_layer(fpn_features, condition=(gamma, beta))
            
            output = self.heads[task_id](fpn_features)
        
        elif task_name == 'detection':
            fpn_features = self.fpn_decoder_det(features)
            
            # Apply FiLM modulation if enabled
            if self.use_film:
                gamma, beta = self.film_generator(task_id)
                fpn_features = self.film_layer(fpn_features, condition=(gamma, beta))
            
            output = self.heads[task_id](fpn_features)
        
        elif task_name == 'classification':
            if self.use_fpn_for_cls:
                fpn_features = self.fpn_decoder_cls(features)
                if self.use_film:
                    gamma, beta = self.film_generator(task_id)
                    fpn_features = self.film_layer(fpn_features, condition=(gamma, beta))
                output = self.heads[task_id](fpn_features)
            else:
                output = self.heads[task_id](features)

        else:  # regression
            if self.use_fpn_for_reg:
                fpn_features = self.fpn_decoder_reg(features)
                if self.use_film:
                    gamma, beta = self.film_generator(task_id)
                    fpn_features = self.film_layer(fpn_features, condition=(gamma, beta))
                output = self.heads[task_id](fpn_features)
            else:
                output = self.heads[task_id](features)
        
        return output

    def _apply_moe(self, features, task_id):
        if len(features) == len(self.moe_blocks):
            out = []
            for idx, (feat, moe) in enumerate(zip(features, self.moe_blocks)):
                if self.moe_stage_indices is None or idx in self.moe_stage_indices:
                    out.append(moe(feat, task_id))
                else:
                    out.append(feat)
            return out

        if len(features) == len(self.moe_blocks) + 1:
            out = [features[0]]
            for local_idx, (feat, moe) in enumerate(zip(features[1:], self.moe_blocks)):
                idx = local_idx + 1
                if self.moe_stage_indices is None or idx in self.moe_stage_indices:
                    out.append(moe(feat, task_id))
                else:
                    out.append(feat)
            return out

        # Fallback: apply MoE to as many stages as possible.
        if not self._moe_warned:
            print("Warning: MoE stage count does not match encoder features, applying to min length.")
            self._moe_warned = True
        out = list(features)
        for idx in range(min(len(features), len(self.moe_blocks))):
            if self.moe_stage_indices is None or idx in self.moe_stage_indices:
                out[idx] = self.moe_blocks[idx](features[idx], task_id)
        return out
    
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
        if self.use_fpn_for_cls and self.fpn_decoder_cls not in (self.fpn_decoder_seg, self.fpn_decoder_det):
            head_params += list(self.fpn_decoder_cls.parameters())
        if self.use_fpn_for_reg and self.fpn_decoder_reg not in (
            self.fpn_decoder_seg,
            self.fpn_decoder_det,
            self.fpn_decoder_cls,
        ):
            head_params += list(self.fpn_decoder_reg.parameters())
        head_params += list(self.heads.parameters())
        
        return encoder_params, head_params

    def get_moe_aux_loss(self):
        device = next(self.parameters()).device
        total = torch.tensor(0.0, device=device)
        if hasattr(self.encoder, "get_moe_aux_loss"):
            total = total + self.encoder.get_moe_aux_loss()
        if self.use_moe:
            for block in self.moe_blocks:
                aux = getattr(block, "last_aux_loss", None)
                if aux is not None:
                    total = total + aux
        return total

    def get_moe_stats(self):
        stats = []
        if hasattr(self.encoder, "get_moe_stats"):
            stats.extend(self.encoder.get_moe_stats())
        if self.use_moe:
            for block in self.moe_blocks:
                block_stats = block.get_stats()
                if block_stats is not None:
                    stats.append(block_stats)
        return stats
    
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
