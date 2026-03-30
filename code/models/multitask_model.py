"""Multi-task model architecture."""

import torch
import torch.nn as nn
from .encoders import build_encoder
from .decoders import build_decoders
from .heads import build_all_heads
from .moe import MoEConvBlock
from .film_layer import TaskFiLMGenerator, TaskEmbeddingFiLMGenerator, FiLMLayer
from .task_prompt import TaskPrompt2D
from .task_condition import TaskConditionEncoder
from .prompt_modulators import EncoderFeaturePromptModulator


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

        # Parse task prompt config early because concat mode changes encoder input channels.
        task_prompt_cfg = config.get('model.task_prompt', {}) or {}
        self.use_task_prompt = bool(task_prompt_cfg.get('enabled', False))
        self.task_prompt_inject_mode = str(task_prompt_cfg.get('inject_mode', 'add')).lower()
        self.task_prompt_channels = int(task_prompt_cfg.get('channels', 1))
        self.task_prompt_use_task_name = bool(task_prompt_cfg.get('use_task_name', True))
        self.task_prompt_use_organ = bool(task_prompt_cfg.get('use_organ', True))
        self.task_prompt_use_num_classes = bool(task_prompt_cfg.get('use_num_classes', False))
        apply_task_names = task_prompt_cfg.get('apply_to_task_names', None)
        if apply_task_names is None:
            self.task_prompt_apply_task_names = None
        else:
            self.task_prompt_apply_task_names = {str(name).lower() for name in apply_task_names}
        self.encoder_input_channels = 3
        if self.use_task_prompt and self.task_prompt_inject_mode == "concat":
            self.encoder_input_channels = 3 + self.task_prompt_channels

        # Parse encoder-feature prompt config (task-conditioned modulation after encoder output).
        feature_prompt_cfg = config.get('model.feature_prompt', {}) or {}
        self.use_feature_prompt = bool(feature_prompt_cfg.get('enabled', False))
        self.feature_prompt_condition_source = str(feature_prompt_cfg.get('condition_source', 'task_type_only'))
        self.feature_prompt_condition_dim = int(feature_prompt_cfg.get('condition_dim', 128))
        feature_prompt_hidden_dim = feature_prompt_cfg.get('hidden_dim', None)
        self.feature_prompt_hidden_dim = (
            int(feature_prompt_hidden_dim) if feature_prompt_hidden_dim is not None else None
        )
        self.feature_prompt_init_scale = float(feature_prompt_cfg.get('init_scale', 0.1))
        self.feature_prompt_apply_stages = feature_prompt_cfg.get('apply_stages', None)

        # Parse prompt-aware decoder config (decoder-internal cross-attention)
        prompt_fpn_cfg = config.get('model.prompt_fpn', {}) or {}
        prompt_decoder_backend = str(prompt_fpn_cfg.get('decoder_backend', 'base_fpn')).lower()
        self.use_prompt_fpn = bool(prompt_fpn_cfg.get('enabled', False)) and prompt_decoder_backend == 'prompt_fpn'
        self.prompt_fpn_condition_source = str(prompt_fpn_cfg.get('condition_source', 'task_type_only'))
        self.prompt_fpn_condition_dim = int(prompt_fpn_cfg.get('condition_dim', 128))
        prompt_fpn_hidden_dim = prompt_fpn_cfg.get('hidden_dim', None)
        self.prompt_fpn_hidden_dim = int(prompt_fpn_hidden_dim) if prompt_fpn_hidden_dim is not None else None
        
        # Build encoder
        task_ids = [cfg['task_id'] for cfg in self.task_configs]
        self.encoder = build_encoder(config, task_ids=task_ids, input_channels=self.encoder_input_channels)
        
        # Normalize encoder channel metadata to [in, c1, c2, ...]
        raw_channels = list(self.encoder.out_channels)
        encoder_channels = (
            raw_channels
            if (raw_channels and raw_channels[0] == self.encoder_input_channels)
            else [self.encoder_input_channels] + raw_channels
        )
        
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

        # Build TaskPrompt modulation (optional, input-level)
        if self.use_task_prompt:
            if hasattr(config, 'tasks_from_dataset') and not config.tasks_from_dataset():
                raise ValueError(
                    "TaskPrompt2D requires dataset-derived task configs. "
                    "Load dataset metadata and override config tasks before building the model."
                )
            self.task_prompt = TaskPrompt2D(
                task_configs=self.task_configs,
                out_channels=self.task_prompt_channels,
                prompt_size=int(task_prompt_cfg.get('prompt_size', 32)),
                inject_mode=self.task_prompt_inject_mode,
                init_scale=float(task_prompt_cfg.get('init_scale', 0.1)),
                use_tanh=bool(task_prompt_cfg.get('use_tanh', True)),
                use_task_name=self.task_prompt_use_task_name,
                use_organ=self.task_prompt_use_organ,
                use_num_classes=self.task_prompt_use_num_classes,
            )
            print(
                "Using TaskPrompt2D "
                f"(dim={self.task_prompt.prompt_dim}, prompt_size={self.task_prompt.prompt_size}, "
                f"mode={self.task_prompt.inject_mode})"
            )
            print(f"TaskPrompt2D metadata components: {self.task_prompt.vocab_info.get('enabled_components', [])}")
            if self.task_prompt.inject_mode == "concat":
                print(f"TaskPrompt2D concat mode enabled: encoder input channels = {self.encoder_input_channels}")
            if self.task_prompt_apply_task_names is None:
                print("TaskPrompt2D apply scope: all task types")
            else:
                print(f"TaskPrompt2D apply scope: {sorted(self.task_prompt_apply_task_names)}")

        # Build encoder-feature prompt modulation (optional, encoder output-level)
        self.task_condition_encoder = None
        self.prompt_fpn_condition_encoder = None
        if self.use_feature_prompt:
            encoder_feature_channels = list(encoder_channels[1:])
            self.task_condition_encoder = TaskConditionEncoder(
                task_configs=self.task_configs,
                condition_dim=self.feature_prompt_condition_dim,
                condition_source=self.feature_prompt_condition_source,
                hidden_dim=self.feature_prompt_hidden_dim,
            )
            self.encoder_feature_prompt = EncoderFeaturePromptModulator(
                feature_channels=encoder_feature_channels,
                condition_dim=self.feature_prompt_condition_dim,
                init_scale=self.feature_prompt_init_scale,
                apply_stages=self.feature_prompt_apply_stages,
            )
            print(
                "Using encoder-feature prompt modulation "
                f"(source={self.task_condition_encoder.condition_source}, "
                f"cond_dim={self.feature_prompt_condition_dim}, "
                f"stages={sorted(self.encoder_feature_prompt.apply_stage_indices)})"
            )

        # Build condition encoder for prompt-aware decoder (optional).
        if self.use_prompt_fpn:
            can_share_condition_encoder = (
                self.use_feature_prompt
                and self.feature_prompt_condition_source == self.prompt_fpn_condition_source
                and self.feature_prompt_condition_dim == self.prompt_fpn_condition_dim
                and self.feature_prompt_hidden_dim == self.prompt_fpn_hidden_dim
            )
            if can_share_condition_encoder:
                self.prompt_fpn_condition_encoder = self.task_condition_encoder
            else:
                self.prompt_fpn_condition_encoder = TaskConditionEncoder(
                    task_configs=self.task_configs,
                    condition_dim=self.prompt_fpn_condition_dim,
                    condition_source=self.prompt_fpn_condition_source,
                    hidden_dim=self.prompt_fpn_hidden_dim,
                )
            print(
                "Using PromptFPN decoder conditioning "
                f"(source={self.prompt_fpn_condition_encoder.condition_source}, "
                f"cond_dim={self.prompt_fpn_condition_dim})"
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

        # Optional task-conditioned input prompt (MTUS-Net-inspired)
        use_task_prompt_for_current = self.use_task_prompt
        if use_task_prompt_for_current and self.task_prompt_apply_task_names is not None:
            use_task_prompt_for_current = task_name.lower() in self.task_prompt_apply_task_names

        if use_task_prompt_for_current:
            x = self.task_prompt.apply(x, task_id)
        elif self.use_task_prompt and self.task_prompt.inject_mode == "concat":
            x = self.task_prompt.concat_zeros(x)
        
        # Encode features
        if getattr(self.encoder, "supports_task_id", False):
            features = self.encoder(x, task_id)
        else:
            features = self.encoder(x)
        if self.use_moe:
            features = self._apply_moe(features, task_id)
        if self.use_feature_prompt:
            feature_dtype = x.dtype
            if isinstance(features, (list, tuple)) and len(features) > 0:
                feature_dtype = features[0].dtype
            cond_vec = self.task_condition_encoder(
                task_id=task_id,
                batch_size=x.shape[0],
                device=x.device,
                dtype=feature_dtype,
            )
            features = self.encoder_feature_prompt(features, cond_vec)

        prompt_cond_vec = None
        if self.use_prompt_fpn:
            feature_dtype = x.dtype
            if isinstance(features, (list, tuple)) and len(features) > 0:
                feature_dtype = features[0].dtype
            if self.task_condition_encoder is not None and self.prompt_fpn_condition_encoder is self.task_condition_encoder:
                prompt_cond_vec = cond_vec
            else:
                prompt_cond_vec = self.prompt_fpn_condition_encoder(
                    task_id=task_id,
                    batch_size=x.shape[0],
                    device=x.device,
                    dtype=feature_dtype,
                )
        
        # Route through appropriate decoder and head
        if task_name == 'segmentation':
            if self.use_prompt_fpn:
                fpn_features = self.fpn_decoder_seg(features, prompt_cond_vec)
            else:
                fpn_features = self.fpn_decoder_seg(features)
            
            # Apply FiLM modulation if enabled
            if self.use_film:
                gamma, beta = self.film_generator(task_id)
                fpn_features = self.film_layer(fpn_features, condition=(gamma, beta))
            
            output = self.heads[task_id](fpn_features)
        
        elif task_name == 'detection':
            if self.use_prompt_fpn:
                fpn_features = self.fpn_decoder_det(features, prompt_cond_vec)
            else:
                fpn_features = self.fpn_decoder_det(features)
            
            # Apply FiLM modulation if enabled
            if self.use_film:
                gamma, beta = self.film_generator(task_id)
                fpn_features = self.film_layer(fpn_features, condition=(gamma, beta))
            
            output = self.heads[task_id](fpn_features)
        
        elif task_name == 'classification':
            if self.use_fpn_for_cls:
                if self.use_prompt_fpn:
                    fpn_features = self.fpn_decoder_cls(features, prompt_cond_vec)
                else:
                    fpn_features = self.fpn_decoder_cls(features)
                if self.use_film:
                    gamma, beta = self.film_generator(task_id)
                    fpn_features = self.film_layer(fpn_features, condition=(gamma, beta))
                output = self.heads[task_id](fpn_features)
            else:
                output = self.heads[task_id](features)

        else:  # regression
            if self.use_fpn_for_reg:
                if self.use_prompt_fpn:
                    fpn_features = self.fpn_decoder_reg(features, prompt_cond_vec)
                else:
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
        if self.use_task_prompt:
            head_params += list(self.task_prompt.parameters())
        if self.use_feature_prompt:
            head_params += list(self.task_condition_encoder.parameters())
            head_params += list(self.encoder_feature_prompt.parameters())
        if self.use_prompt_fpn and self.prompt_fpn_condition_encoder is not None:
            if self.prompt_fpn_condition_encoder is not self.task_condition_encoder:
                head_params += list(self.prompt_fpn_condition_encoder.parameters())
        
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

    def set_current_epoch(self, epoch: int) -> None:
        """Propagate current epoch to modules that need epoch-scoped logging."""
        seen = set()
        for decoder in [self.fpn_decoder_seg, self.fpn_decoder_det, self.fpn_decoder_cls, self.fpn_decoder_reg]:
            if id(decoder) in seen:
                continue
            seen.add(id(decoder))
            if hasattr(decoder, "set_current_epoch"):
                decoder.set_current_epoch(epoch)

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
