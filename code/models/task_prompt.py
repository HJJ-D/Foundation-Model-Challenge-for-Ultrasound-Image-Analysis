"""Task prompt modules (reference implementation inspired by MTUS-Net style prompts)."""

import re
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


_TASK_PREFIX_RE = re.compile(r"^t\d+[a-z]?$", re.IGNORECASE)
# _LOW_INFO_TOKEN_RE = re.compile(r"^\d+cls$", re.IGNORECASE)
#_LOW_INFO_TOKENS = {"det", "multi", "cls"}


def _tokenize_task_id(task_id: str) -> List[str]:
    parts = [p.strip().lower() for p in str(task_id).split("_") if p.strip()]
    return [
        p
        for p in parts
        if not _TASK_PREFIX_RE.match(p)
        # and not _LOW_INFO_TOKEN_RE.match(p)
        # and p not in _LOW_INFO_TOKENS
    ]


def build_task_prompt_metadata(task_configs: Sequence[Dict]) -> Tuple[torch.Tensor, Dict[str, int], Dict[str, List[str]]]:
    """
    Build per-task metadata vectors from task type + task_id tokens.

    Returns:
        metadata_table: [num_tasks, prompt_dim] float tensor
        task_id_to_idx: task_id -> row index
        vocab_info: debug info for task types/tokens used
    """
    task_ids = [str(cfg["task_id"]) for cfg in task_configs]
    task_names = [str(cfg.get("task_name", "unknown")).lower() for cfg in task_configs]
    num_classes_tags = [f"num_classes_{int(cfg.get('num_classes', -1))}" for cfg in task_configs]

    type_vocab = sorted(set(task_names))
    class_vocab = sorted(set(num_classes_tags))

    token_sets = []
    token_vocab = set()
    for task_id in task_ids:
        tokens = _tokenize_task_id(task_id)
        token_sets.append(tokens)
        token_vocab.update(tokens)
    token_vocab = sorted(token_vocab)

    type_to_idx = {name: i for i, name in enumerate(type_vocab)}
    token_to_idx = {tok: i for i, tok in enumerate(token_vocab)}
    task_id_to_idx = {task_id: i for i, task_id in enumerate(task_ids)}

    prompt_dim = len(type_vocab) + len(class_vocab) + len(token_vocab)
    metadata = torch.zeros(len(task_ids), prompt_dim, dtype=torch.float32)

    class_to_idx = {name: i for i, name in enumerate(class_vocab)}

    for row, (task_name, class_tag, tokens) in enumerate(zip(task_names, num_classes_tags, token_sets)):
        metadata[row, type_to_idx[task_name]] = 1.0
        metadata[row, len(type_vocab) + class_to_idx[class_tag]] = 1.0
        for tok in tokens:
            metadata[row, len(type_vocab) + len(class_vocab) + token_to_idx[tok]] = 1.0

    vocab_info = {
        "task_types": type_vocab,
        "num_classes_tags": class_vocab,
        "task_tokens": token_vocab,
    }
    return metadata, task_id_to_idx, vocab_info


class TaskPrompt2D(nn.Module):
    """
    Generate a spatial prompt map from task metadata and inject it into the input image.

    This mirrors the idea in MTUS-Net (task/organ vectors -> linear projection -> 2D prompt),
    but derives metadata automatically from this project's task configuration.
    """

    def __init__(
        self,
        task_configs: Sequence[Dict],
        out_channels: int = 1,
        prompt_size: int = 32,
        inject_mode: str = "add",
        init_scale: float = 0.1,
        use_tanh: bool = True,
    ):
        super().__init__()
        if inject_mode not in {"add", "mul"}:
            raise ValueError(f"Unsupported inject_mode: {inject_mode}")

        metadata, task_id_to_idx, vocab_info = build_task_prompt_metadata(task_configs)
        if metadata.numel() == 0:
            raise ValueError("TaskPrompt2D received empty task metadata.")

        self.task_id_to_idx = task_id_to_idx
        self.vocab_info = vocab_info
        self.out_channels = int(out_channels)
        self.prompt_size = int(prompt_size)
        self.inject_mode = inject_mode
        self.use_tanh = bool(use_tanh)

        self.register_buffer("task_metadata", metadata, persistent=True)
        self.prompt_proj = nn.Linear(metadata.shape[1], self.out_channels * self.prompt_size * self.prompt_size)
        self.prompt_scale = nn.Parameter(torch.tensor(float(init_scale), dtype=torch.float32))

    @property
    def prompt_dim(self) -> int:
        return int(self.task_metadata.shape[1])

    def _get_prompt_vec(self, task_id: str, device: torch.device, batch_size: int) -> torch.Tensor:
        if task_id not in self.task_id_to_idx:
            raise ValueError(f"Unknown task_id for TaskPrompt2D: {task_id}")
        idx = self.task_id_to_idx[task_id]
        vec = self.task_metadata[idx].to(device=device)
        return vec.unsqueeze(0).expand(batch_size, -1)

    def forward(self, task_id: str, batch_size: int, spatial_size: Tuple[int, int], device: torch.device, return_vec: bool = False):
        prompt_vec = self._get_prompt_vec(task_id, device=device, batch_size=batch_size)
        prompt = self.prompt_proj(prompt_vec).view(batch_size, self.out_channels, self.prompt_size, self.prompt_size)
        if self.use_tanh:
            prompt = torch.tanh(prompt)
        if prompt.shape[-2:] != tuple(spatial_size):
            prompt = F.interpolate(prompt, size=spatial_size, mode="bilinear", align_corners=False)
        if return_vec:
            return prompt, prompt_vec
        return prompt

    def apply(self, x: torch.Tensor, task_id: str) -> torch.Tensor:
        prompt = self.forward(
            task_id=task_id,
            batch_size=x.shape[0],
            spatial_size=(x.shape[-2], x.shape[-1]),
            device=x.device,
        )
        scale = self.prompt_scale.to(dtype=x.dtype)
        prompt = prompt.to(dtype=x.dtype)
        if self.inject_mode == "add":
            return x + scale * prompt
        return x * (1.0 + scale * prompt)
