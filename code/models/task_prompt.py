"""Task prompt modules (reference implementation inspired by MTUS-Net style prompts)."""

import re
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


_TASK_PREFIX_RE = re.compile(r"^t\d+[a-z]?$", re.IGNORECASE)
_TASK_NAME_TO_GROUP = {
    "detection": "detection",
    "segmentation": "segmentation",
    "classification": "classification",
    "regression": "regression",
}
_TASK_GROUP_VOCAB = ["detection", "segmentation", "classification", "regression"]

# Pathology tasks: involve detecting/classifying/segmenting abnormalities
_PATHOLOGY_TASK_IDS = {
    "breast_2cls", "breast_3cls", "breast_lesion",
    "liver_lesion_2cls", "lung_2cls", "lung_disease_3cls",
    "ovary_tumor",
    "spinal_cord_injury_loc",
    "thyroid_nodule", "thyroid_nodule_det",
    "uterine_fibroid_det",
}

# Fetal tasks: involve fetal imaging (including obstetric measurements)
_FETAL_TASK_IDS = {
    "FUGC", "IUGC", "fetal_femur",
    "fetal_head_pos_cls", "fetal_plane_cls", "fetal_sacral_pos_cls",
    "fetal_abdomen_multi", "fetal_head", "fetal_heart",
    "head_symphysis_multi",
}


def _is_pathology_task(task_id: str) -> bool:
    return str(task_id) in _PATHOLOGY_TASK_IDS


def _is_fetal_task(task_id: str) -> bool:
    tid = str(task_id)
    return tid in _FETAL_TASK_IDS or tid.lower().startswith("fetal")


def _tokenize_task_id(task_id: str) -> List[str]:
    parts = [p.strip().lower() for p in str(task_id).split("_") if p.strip()]
    return [p for p in parts if not _TASK_PREFIX_RE.match(p)]


def _normalize_task_name(task_name: str) -> str:
    normalized = str(task_name).strip().lower()
    if normalized not in _TASK_NAME_TO_GROUP:
        raise ValueError(f"Unsupported task_name for TaskPrompt2D: {task_name}")
    return _TASK_NAME_TO_GROUP[normalized]


def _extract_organ_label(task_id: str) -> str:
    tokens = _tokenize_task_id(task_id)
    if not tokens:
        raise ValueError(f"Cannot extract organ label from task_id: {task_id}")

    first_token = tokens[0]
    if first_token == "fetal":
        return "fetal" if len(tokens) == 1 else f"fetal {tokens[1]}"
    return first_token


def build_task_prompt_metadata(
    task_configs: Sequence[Dict],
    use_task_name: bool = True,
    use_organ: bool = True,
    use_num_classes: bool = False,
    use_is_pathology: bool = False,
    use_is_fetal: bool = False,
) -> Tuple[torch.Tensor, Dict[str, int], Dict[str, List[str]]]:
    """
    Build per-task metadata vectors from selected metadata components.

    Components:
        use_task_name:    one-hot over 4 task types (seg/cls/det/reg)       → dim 4
        use_organ:        one-hot over organ labels extracted from task_id   → dim N
        use_num_classes:  single normalized float (num_classes/max)          → dim 1
        use_is_pathology: binary flag — 1 if task involves pathology          → dim 1
        use_is_fetal:     binary flag — 1 if task involves fetal imaging      → dim 1

    Returns:
        metadata_table: [num_tasks, metadata_dim] float tensor
        task_id_to_idx: task_id -> row index
        vocab_info: debug info for selected metadata vocabularies
    """
    if not (use_task_name or use_organ or use_num_classes or use_is_pathology or use_is_fetal):
        raise ValueError("At least one metadata component must be enabled.")

    task_ids = [str(cfg["task_id"]) for cfg in task_configs]
    task_groups = [_normalize_task_name(cfg.get("task_name", "unknown")) for cfg in task_configs] if use_task_name else []
    organ_labels = [_extract_organ_label(task_id) for task_id in task_ids] if use_organ else []

    type_vocab = list(_TASK_GROUP_VOCAB) if use_task_name else []
    organ_vocab = sorted(set(organ_labels)) if use_organ else []

    type_to_idx = {name: i for i, name in enumerate(type_vocab)}
    organ_to_idx = {organ: i for i, organ in enumerate(organ_vocab)}
    task_id_to_idx = {task_id: i for i, task_id in enumerate(task_ids)}

    # num_classes: single normalized scalar, divide by max across all tasks
    num_classes_values = [int(cfg.get("num_classes", 1)) for cfg in task_configs]
    max_num_classes = max(num_classes_values) if num_classes_values else 1

    metadata_dim = (
        len(type_vocab)
        + len(organ_vocab)
        + (1 if use_num_classes else 0)
        + (1 if use_is_pathology else 0)
        + (1 if use_is_fetal else 0)
    )
    metadata = torch.zeros(len(task_ids), metadata_dim, dtype=torch.float32)

    for row, task_id in enumerate(task_ids):
        col = 0
        if use_task_name:
            metadata[row, col + type_to_idx[task_groups[row]]] = 1.0
        col += len(type_vocab)

        if use_organ:
            metadata[row, col + organ_to_idx[organ_labels[row]]] = 1.0
        col += len(organ_vocab)

        if use_num_classes:
            metadata[row, col] = num_classes_values[row] / max_num_classes
            col += 1

        if use_is_pathology:
            metadata[row, col] = 1.0 if _is_pathology_task(task_id) else 0.0
            col += 1

        if use_is_fetal:
            metadata[row, col] = 1.0 if _is_fetal_task(task_id) else 0.0
            col += 1

    vocab_info = {
        "task_types": type_vocab,
        "organ_labels": organ_vocab,
        "max_num_classes": max_num_classes,
        "enabled_components": [
            name
            for name, flag in (
                ("task_name", use_task_name),
                ("organ", use_organ),
                ("num_classes_normalized", use_num_classes),
                ("is_pathology", use_is_pathology),
                ("is_fetal", use_is_fetal),
            )
            if flag
        ],
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
        use_task_name: bool = True,
        use_organ: bool = True,
        use_num_classes: bool = False,
        use_is_pathology: bool = False,
        use_is_fetal: bool = False,
    ):
        super().__init__()
        if inject_mode not in {"add", "mul", "concat"}:
            raise ValueError(f"Unsupported inject_mode: {inject_mode}")

        metadata, task_id_to_idx, vocab_info = build_task_prompt_metadata(
            task_configs,
            use_task_name=use_task_name,
            use_organ=use_organ,
            use_num_classes=use_num_classes,
            use_is_pathology=use_is_pathology,
            use_is_fetal=use_is_fetal,
        )
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
        if self.inject_mode == "concat":
            return torch.cat([x, scale * prompt], dim=1)
        return x * (1.0 + scale * prompt)

    def concat_zeros(self, x: torch.Tensor) -> torch.Tensor:
        """
        Keep input channels consistent when concat mode is enabled but prompt is skipped.
        """
        if self.inject_mode != "concat":
            return x
        zeros = torch.zeros(
            x.shape[0],
            self.out_channels,
            x.shape[-2],
            x.shape[-1],
            device=x.device,
            dtype=x.dtype,
        )
        return torch.cat([x, zeros], dim=1)
