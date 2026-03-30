"""Reusable task-condition encoder for prompt-based modulation."""

from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn

from .task_prompt import build_task_prompt_metadata


_CONDITION_SOURCE_TO_FLAGS = {
    # task_type one-hot only (4 distinct vectors) — baseline, minimal signal
    "task_type_only": dict(
        use_task_name=True, use_organ=False, use_num_classes=False,
        use_is_pathology=False, use_is_fetal=False,
    ),
    # adds organ — known to cause detection negative transfer when organ overlaps task types
    "task_type_plus_organ": dict(
        use_task_name=True, use_organ=True, use_num_classes=False,
        use_is_pathology=False, use_is_fetal=False,
    ),
    # recommended: task_type + semantic flags + num_classes; no organ (avoids negative transfer)
    "task_type_plus_semantic": dict(
        use_task_name=True, use_organ=False, use_num_classes=True,
        use_is_pathology=True, use_is_fetal=True,
    ),
    # full: task_type + organ + all semantic flags (use with caution — organ may still cause issues)
    "full_semantic": dict(
        use_task_name=True, use_organ=True, use_num_classes=True,
        use_is_pathology=True, use_is_fetal=True,
    ),
}


def _normalize_condition_source(condition_source: str) -> str:
    source = str(condition_source).strip().lower()
    if source not in _CONDITION_SOURCE_TO_FLAGS:
        valid = ", ".join(sorted(_CONDITION_SOURCE_TO_FLAGS.keys()))
        raise ValueError(f"Unsupported condition_source: {condition_source}. Valid options: {valid}")
    return source


def build_condition_metadata(
    task_configs: Sequence[Dict],
    condition_source: str = "task_type_only",
) -> Tuple[torch.Tensor, Dict[str, int], Dict[str, List[str]]]:
    """
    Build per-task metadata vectors for condition encoding.

    Args:
        task_configs: Task configurations
        condition_source: Condition recipe name

    Returns:
        metadata_table: [num_tasks, metadata_dim]
        task_id_to_idx: task_id -> row index
        vocab_info: Debug vocab metadata
    """
    source = _normalize_condition_source(condition_source)
    metadata, task_id_to_idx, vocab_info = build_task_prompt_metadata(
        task_configs=task_configs,
        **_CONDITION_SOURCE_TO_FLAGS[source],
    )
    vocab_info = dict(vocab_info)
    vocab_info["condition_source"] = source
    return metadata, task_id_to_idx, vocab_info


class TaskConditionEncoder(nn.Module):
    """
    Convert task_id to a learned condition vector.

    The metadata source is configurable (task_type_only / task_type_plus_organ).
    """

    def __init__(
        self,
        task_configs: Sequence[Dict],
        condition_dim: int = 128,
        condition_source: str = "task_type_only",
        hidden_dim: int = None,
    ):
        super().__init__()
        metadata, task_id_to_idx, vocab_info = build_condition_metadata(
            task_configs=task_configs,
            condition_source=condition_source,
        )
        if metadata.numel() == 0:
            raise ValueError("TaskConditionEncoder received empty metadata.")

        self.task_id_to_idx = task_id_to_idx
        self.vocab_info = vocab_info
        self.condition_source = _normalize_condition_source(condition_source)
        self.condition_dim = int(condition_dim)
        self.metadata_dim = int(metadata.shape[1])
        self.hidden_dim = int(hidden_dim) if hidden_dim is not None else max(self.condition_dim, self.metadata_dim)

        self.register_buffer("task_metadata", metadata, persistent=True)
        self.condition_proj = nn.Sequential(
            nn.Linear(self.metadata_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.condition_dim),
        )

    def _lookup_metadata(self, task_id: str, batch_size: int, device: torch.device) -> torch.Tensor:
        if task_id not in self.task_id_to_idx:
            raise ValueError(f"Unknown task_id for TaskConditionEncoder: {task_id}")
        idx = self.task_id_to_idx[task_id]
        metadata = self.task_metadata[idx].to(device=device)
        return metadata.unsqueeze(0).expand(batch_size, -1)

    def forward(
        self,
        task_id: str,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = None,
    ) -> torch.Tensor:
        metadata = self._lookup_metadata(task_id=task_id, batch_size=batch_size, device=device)
        cond_vec = self.condition_proj(metadata)
        if dtype is not None:
            cond_vec = cond_vec.to(dtype=dtype)
        return cond_vec
