"""Prompt modulation modules for encoder features."""

from typing import Iterable, Sequence

import torch
import torch.nn as nn


def _normalize_apply_stages(apply_stages: Iterable[int], num_stages: int):
    if apply_stages is None:
        return set(range(num_stages))

    values = [int(v) for v in apply_stages]
    if not values:
        return set()

    if all(1 <= v <= num_stages for v in values):
        values = [v - 1 for v in values]
    elif not all(0 <= v < num_stages for v in values):
        raise ValueError(
            f"Invalid apply_stages={values} for num_stages={num_stages}. "
            "Use either 0-based [0..N-1] or 1-based [1..N]."
        )

    return set(sorted(set(values)))


class EncoderFeaturePromptModulator(nn.Module):
    """
    Stage-wise FiLM-style modulation over encoder feature pyramid.
    """

    def __init__(
        self,
        feature_channels: Sequence[int],
        condition_dim: int = 128,
        init_scale: float = 0.1,
        apply_stages: Iterable[int] = None,
    ):
        super().__init__()
        if not feature_channels:
            raise ValueError("feature_channels cannot be empty.")

        self.feature_channels = [int(ch) for ch in feature_channels]
        self.condition_dim = int(condition_dim)
        self.num_stages = len(self.feature_channels)
        self.apply_stage_indices = _normalize_apply_stages(apply_stages, self.num_stages)
        self._stage_mismatch_warned = False

        self.film_generators = nn.ModuleList(
            [
                nn.Linear(self.condition_dim, 2 * channels)
                for channels in self.feature_channels
            ]
        )
        self.stage_scales = nn.Parameter(
            torch.full((self.num_stages,), float(init_scale), dtype=torch.float32)
        )

    def forward(self, features_list, cond_vec: torch.Tensor):
        if not isinstance(features_list, (list, tuple)):
            raise TypeError(f"features_list must be list/tuple, got {type(features_list)}")
        if cond_vec.dim() != 2:
            raise ValueError(f"cond_vec must be [B, D], got shape={tuple(cond_vec.shape)}")
        if cond_vec.shape[1] != self.condition_dim:
            raise ValueError(
                f"cond_vec dim mismatch: expected {self.condition_dim}, got {cond_vec.shape[1]}"
            )
        if len(features_list) == 0:
            return features_list

        max_modulated = min(len(features_list), self.num_stages)
        if len(features_list) != self.num_stages and not self._stage_mismatch_warned:
            print(
                "Warning: EncoderFeaturePromptModulator feature-stage mismatch "
                f"(features={len(features_list)}, modulators={self.num_stages}); "
                "applying to minimum shared stages."
            )
            self._stage_mismatch_warned = True

        outputs = []
        for stage_idx, feat in enumerate(features_list):
            if stage_idx < max_modulated and stage_idx in self.apply_stage_indices:
                gamma_beta = self.film_generators[stage_idx](cond_vec)
                gamma, beta = gamma_beta.chunk(2, dim=1)
                gamma = gamma.to(dtype=feat.dtype).unsqueeze(-1).unsqueeze(-1)
                beta = beta.to(dtype=feat.dtype).unsqueeze(-1).unsqueeze(-1)
                alpha = self.stage_scales[stage_idx].to(dtype=feat.dtype)
                feat = (1.0 + alpha * gamma) * feat + alpha * beta
            outputs.append(feat)

        if isinstance(features_list, tuple):
            return tuple(outputs)
        return outputs
