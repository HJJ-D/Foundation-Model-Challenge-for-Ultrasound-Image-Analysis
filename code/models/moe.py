"""MoE (Mixture of Experts) blocks for CNN feature maps."""

from __future__ import annotations

from typing import Optional, List

import torch
import torch.nn as nn


class ConvExpert(nn.Module):
    """Lightweight convolutional expert that preserves input channels."""

    def __init__(self, in_channels: int, hidden_channels: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=1, bias=False),
        ]
        if dropout > 0:
            layers.insert(4, nn.Dropout2d(p=dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MoEConvBlock(nn.Module):
    """
    Mixture-of-Experts block for CNN features.
    Routing is per-sample (global pooled), optionally conditioned on task embedding.
    """

    def __init__(
        self,
        in_channels: int,
        num_experts: int = 4,
        expert_hidden: Optional[int] = None,
        router_hidden: Optional[int] = None,
        top_k: int = 1,
        use_task_embedding: bool = False,
        task_embedding_dim: int = 32,
        task_ids: Optional[List[str]] = None,
        use_residual: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        if num_experts < 1:
            raise ValueError("num_experts must be >= 1")
        if top_k < 1 or top_k > num_experts:
            raise ValueError("top_k must be in [1, num_experts]")

        self.in_channels = in_channels
        self.num_experts = num_experts
        self.top_k = top_k
        self.use_task_embedding = use_task_embedding
        self.use_residual = use_residual
        self.last_aux_loss = None
        self.last_importance = None
        self.last_load = None

        hidden = expert_hidden or max(8, in_channels // 2)
        self.experts = nn.ModuleList(
            [ConvExpert(in_channels, hidden, dropout=dropout) for _ in range(num_experts)]
        )

        self.task_id_to_idx = None
        if use_task_embedding:
            if not task_ids:
                raise ValueError("task_ids must be provided when use_task_embedding=True")
            self.task_id_to_idx = {task_id: idx for idx, task_id in enumerate(task_ids)}
            self.task_embed = nn.Embedding(len(task_ids), task_embedding_dim)

        router_in_dim = in_channels + (task_embedding_dim if use_task_embedding else 0)
        router_hidden_dim = router_hidden or max(16, router_in_dim // 2)
        self.router = nn.Sequential(
            nn.Linear(router_in_dim, router_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(router_hidden_dim, num_experts),
        )

    def _get_task_embedding(self, task_id: str, batch_size: int, device: torch.device) -> torch.Tensor:
        if self.task_id_to_idx is None:
            raise ValueError("task embedding is disabled")
        if task_id not in self.task_id_to_idx:
            raise ValueError(f"Unknown task_id: {task_id}")
        idx = torch.tensor(self.task_id_to_idx[task_id], device=device, dtype=torch.long)
        emb = self.task_embed(idx)  # [D]
        return emb.unsqueeze(0).expand(batch_size, -1)

    def forward(self, x: torch.Tensor, task_id: Optional[str] = None) -> torch.Tensor:
        """
        Args:
            x: Feature map [B, C, H, W]
            task_id: Task identifier string (required if use_task_embedding=True)
        """
        batch_size = x.size(0)
        pooled = x.mean(dim=(2, 3))  # [B, C]

        if self.use_task_embedding:
            if task_id is None:
                raise ValueError("task_id must be provided when use_task_embedding=True")
            task_emb = self._get_task_embedding(task_id, batch_size, x.device)
            router_in = torch.cat([pooled, task_emb], dim=1)
        else:
            router_in = pooled

        logits = self.router(router_in)
        probs = torch.softmax(logits, dim=1)

        if self.top_k < self.num_experts:
            topk_vals, topk_idx = torch.topk(probs, self.top_k, dim=1)
            dispatch_mask = torch.zeros_like(probs).scatter(1, topk_idx, 1.0)
            mask = torch.zeros_like(probs).scatter(1, topk_idx, topk_vals)
            probs = mask / (mask.sum(dim=1, keepdim=True) + 1e-9)
        else:
            dispatch_mask = torch.ones_like(probs)

        # Switch-style load balancing loss: E * sum(importance * load)
        importance = probs.mean(dim=0)
        load = dispatch_mask.mean(dim=0)
        self.last_aux_loss = self.num_experts * torch.sum(importance * load)
        self.last_importance = importance.detach()
        self.last_load = load.detach()

        expert_outs = [expert(x) for expert in self.experts]
        stacked = torch.stack(expert_outs, dim=1)  # [B, E, C, H, W]
        weights = probs.view(batch_size, self.num_experts, 1, 1, 1)
        out = (stacked * weights).sum(dim=1)

        if self.use_residual:
            out = out + x

        return out

    def get_stats(self):
        if self.last_importance is None or self.last_load is None:
            return None
        return {
            "importance": self.last_importance,
            "load": self.last_load,
        }
