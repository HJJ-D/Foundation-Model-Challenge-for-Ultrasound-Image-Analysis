"""Runtime-Scaled LoRA (simple and prompt-gated variants).

Overview
--------
Standard LoRA injects ``y = W x + (alpha/r) * B(A(x))`` into frozen Linear layers.
This module adds a *runtime* scale to every LoRA branch, so the effective
contribution can be dynamically gated per task on a per-(layer, group) basis:

    y = W x + (alpha/r) * runtime_scale * B(A(x))

In the **simple** variant the runtime scale is fixed at 1.0, giving the same
behavior as classic fixed-scale LoRA. In the **gated** variant a lightweight
controller produces per-(layer, group) runtime scales from one or more task
"prompts" (task_type, task_id, organ, ...) and writes them into every LoRA
module before each forward; the controller is trained end-to-end.

Layout
------
* ``LoRALinear``             — single-group runtime-scaled LoRA.
* ``FusedQKVLoRALinear``     — shared A + 3 B's for timm's fused qkv Linear,
                               with 3 independent runtime scales (q/k/v).
* ``GatedLoraController``    — embeddings + ctrl heads producing (L, G) scales.
* ``inject_lora``            — top-level entry: replace Linear layers in every
                               transformer block; returns a :class:`LoraHandle`.
* ``inject_gated_lora``      — same as above, but also builds the controller
                               and attaches it to the handle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .task_prompt import (
    _TASK_GROUP_VOCAB,
    _extract_organ_label,
    _is_fetal_task,
    _is_pathology_task,
    _normalize_task_name,
)

# ---------------------------------------------------------------------------
# Scale groups
# ---------------------------------------------------------------------------

SCALE_GROUPS: Tuple[str, ...] = ("q", "k", "v", "proj", "fc1", "fc2")
NUM_SCALE_GROUPS: int = len(SCALE_GROUPS)
SCALE_GROUP_TO_IDX: Dict[str, int] = {g: i for i, g in enumerate(SCALE_GROUPS)}


# ---------------------------------------------------------------------------
# Section 1: Runtime-scaled LoRA layers
# ---------------------------------------------------------------------------

class _LoRABase(nn.Module):
    """Shared book-keeping for runtime-scaled LoRA layers."""

    def __init__(self, linear: nn.Linear, r: int, alpha: float, lora_only: bool):
        super().__init__()
        if r <= 0:
            raise ValueError(f"LoRA rank r must be > 0, got {r}")
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.r = int(r)
        self.base_scale = float(alpha) / float(r)

        # Re-use the original frozen weights / bias
        self.weight = linear.weight
        self.bias = linear.bias
        if lora_only:
            self.weight.requires_grad = False
            if self.bias is not None:
                self.bias.requires_grad = False

    @staticmethod
    def _init_A(layer: nn.Linear) -> None:
        nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))

    @staticmethod
    def _init_B(layer: nn.Linear) -> None:
        # Zero-init B so that delta = 0 at step 0 → model behaves identically.
        nn.init.zeros_(layer.weight)

    @staticmethod
    def _to_scalar_tensor(scale: Union[float, torch.Tensor]) -> torch.Tensor:
        if isinstance(scale, torch.Tensor):
            return scale
        return torch.tensor(float(scale))


class LoRALinear(_LoRABase):
    """Runtime-scaled LoRA wrapper around a single ``nn.Linear``.

    ``set_runtime_scale`` can accept either a Python float or a 0-d Tensor.
    If a Tensor is given, its gradient flows back to whatever produced it
    (e.g. the ``GatedLoraController``).
    """

    def __init__(
        self,
        linear: nn.Linear,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        lora_only: bool = True,
    ):
        super().__init__(linear, r=r, alpha=alpha, lora_only=lora_only)
        self.lora_A = nn.Linear(self.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, self.out_features, bias=False)
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self._init_A(self.lora_A)
        self._init_B(self.lora_B)

        # Live runtime scale. Stored as a plain attribute (not buffer/parameter)
        # so gradient flows from loss back through it to the controller.
        # The controller overwrites this every forward before the encoder runs.
        self.runtime_scale: torch.Tensor = torch.tensor(1.0)

    def set_runtime_scale(self, scale: Union[float, torch.Tensor]) -> None:
        self.runtime_scale = self._to_scalar_tensor(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        delta = self.lora_B(self.lora_A(self.lora_dropout(x)))
        scale = self.runtime_scale.to(dtype=delta.dtype, device=delta.device)
        return base + delta * (self.base_scale * scale)

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"r={self.r}, base_scale={self.base_scale:.3f}"
        )


class FusedQKVLoRALinear(_LoRABase):
    """Runtime-scaled LoRA for timm's fused ``qkv`` Linear (``out = 3*in``).

    A shared ``lora_A`` projects input to r dims; three independent B's map
    back to the q/k/v output segments, each with its own runtime scale.
    """

    def __init__(
        self,
        linear: nn.Linear,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        lora_only: bool = True,
    ):
        super().__init__(linear, r=r, alpha=alpha, lora_only=lora_only)
        D = self.in_features
        if self.out_features != 3 * D:
            raise ValueError(
                f"FusedQKVLoRALinear expects out_features == 3 * in_features, "
                f"got in={D}, out={self.out_features}"
            )
        self.head_dim = D

        self.lora_A = nn.Linear(D, r, bias=False)
        self.lora_B_q = nn.Linear(r, D, bias=False)
        self.lora_B_k = nn.Linear(r, D, bias=False)
        self.lora_B_v = nn.Linear(r, D, bias=False)
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self._init_A(self.lora_A)
        self._init_B(self.lora_B_q)
        self._init_B(self.lora_B_k)
        self._init_B(self.lora_B_v)

        self.runtime_scale_q: torch.Tensor = torch.tensor(1.0)
        self.runtime_scale_k: torch.Tensor = torch.tensor(1.0)
        self.runtime_scale_v: torch.Tensor = torch.tensor(1.0)

    def set_runtime_scale_qkv(
        self,
        s_q: Union[float, torch.Tensor],
        s_k: Union[float, torch.Tensor],
        s_v: Union[float, torch.Tensor],
    ) -> None:
        self.runtime_scale_q = self._to_scalar_tensor(s_q)
        self.runtime_scale_k = self._to_scalar_tensor(s_k)
        self.runtime_scale_v = self._to_scalar_tensor(s_v)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        a = self.lora_A(self.lora_dropout(x))

        def _scaled(b_layer: nn.Linear, scale: torch.Tensor) -> torch.Tensor:
            delta = b_layer(a)
            s = scale.to(dtype=delta.dtype, device=delta.device)
            return delta * (self.base_scale * s)

        dq = _scaled(self.lora_B_q, self.runtime_scale_q)
        dk = _scaled(self.lora_B_k, self.runtime_scale_k)
        dv = _scaled(self.lora_B_v, self.runtime_scale_v)
        return base + torch.cat([dq, dk, dv], dim=-1)

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"r={self.r}, base_scale={self.base_scale:.3f}"
        )


# ---------------------------------------------------------------------------
# Section 2: Prompt sources
# ---------------------------------------------------------------------------

# All supported categorical prompt sources. Each maps a task_id to a single
# category index. Continuous metadata (num_classes) is bucketized by unique
# observed values so it can share the nn.Embedding interface.
PROMPT_SOURCES: Tuple[str, ...] = (
    "task_type",
    "task_id",
    "organ",
    "is_pathology",
    "is_fetal",
    "num_classes",
)


def _vocab_task_type(task_configs: Sequence[Dict]) -> Tuple[Dict[str, int], List[str]]:
    vocab = list(_TASK_GROUP_VOCAB)
    name_to_idx = {name: i for i, name in enumerate(vocab)}
    mapping = {
        str(c["task_id"]): name_to_idx[_normalize_task_name(c.get("task_name", "unknown"))]
        for c in task_configs
    }
    return mapping, vocab


def _vocab_task_id(task_configs: Sequence[Dict]) -> Tuple[Dict[str, int], List[str]]:
    task_ids = [str(c["task_id"]) for c in task_configs]
    return {tid: i for i, tid in enumerate(task_ids)}, list(task_ids)


def _vocab_organ(task_configs: Sequence[Dict]) -> Tuple[Dict[str, int], List[str]]:
    task_ids = [str(c["task_id"]) for c in task_configs]
    organs = [_extract_organ_label(tid) for tid in task_ids]
    vocab = sorted(set(organs))
    o_to_idx = {o: i for i, o in enumerate(vocab)}
    return {tid: o_to_idx[org] for tid, org in zip(task_ids, organs)}, vocab


def _vocab_is_pathology(task_configs: Sequence[Dict]) -> Tuple[Dict[str, int], List[str]]:
    vocab = ["non_pathology", "pathology"]
    mapping = {
        str(c["task_id"]): int(_is_pathology_task(str(c["task_id"])))
        for c in task_configs
    }
    return mapping, vocab


def _vocab_is_fetal(task_configs: Sequence[Dict]) -> Tuple[Dict[str, int], List[str]]:
    vocab = ["non_fetal", "fetal"]
    mapping = {
        str(c["task_id"]): int(_is_fetal_task(str(c["task_id"])))
        for c in task_configs
    }
    return mapping, vocab


def _vocab_num_classes(task_configs: Sequence[Dict]) -> Tuple[Dict[str, int], List[str]]:
    # Bucketize by unique num_classes values observed across all tasks.
    per_task: List[Tuple[str, int]] = [
        (str(c["task_id"]), int(c.get("num_classes", 1))) for c in task_configs
    ]
    unique = sorted({v for _, v in per_task})
    v_to_idx = {v: i for i, v in enumerate(unique)}
    mapping = {tid: v_to_idx[v] for tid, v in per_task}
    vocab = [f"nc_{v}" for v in unique]
    return mapping, vocab


_VOCAB_BUILDERS: Dict[str, Callable[[Sequence[Dict]], Tuple[Dict[str, int], List[str]]]] = {
    "task_type": _vocab_task_type,
    "task_id": _vocab_task_id,
    "organ": _vocab_organ,
    "is_pathology": _vocab_is_pathology,
    "is_fetal": _vocab_is_fetal,
    "num_classes": _vocab_num_classes,
}


@dataclass
class PromptSourceSpec:
    name: str
    task_id_to_idx: Dict[str, int]
    vocab: List[str]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)


def build_prompt_source_specs(
    prompt_sources: Sequence[str],
    task_configs: Sequence[Dict],
) -> List[PromptSourceSpec]:
    specs: List[PromptSourceSpec] = []
    for name in prompt_sources:
        if name not in _VOCAB_BUILDERS:
            raise ValueError(
                f"Unknown prompt source: '{name}'. "
                f"Supported: {list(_VOCAB_BUILDERS.keys())}"
            )
        mapping, vocab = _VOCAB_BUILDERS[name](task_configs)
        specs.append(PromptSourceSpec(name=name, task_id_to_idx=mapping, vocab=vocab))
    return specs


# ---------------------------------------------------------------------------
# Section 3: Gated LoRA controller
# ---------------------------------------------------------------------------

def _sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)


def _softplus(x: torch.Tensor) -> torch.Tensor:
    return F.softplus(x)


def _tanh01(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (torch.tanh(x) + 1.0)


_ACTIVATIONS: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "sigmoid": _sigmoid,
    "softplus": _softplus,
    "tanh01": _tanh01,
}


class GatedLoraController(nn.Module):
    """Produce per-(layer, group) runtime scales from task prompts.

    Forward procedure:
      1. For each enabled prompt source, look up per-sample category index.
      2. Embed → ctrl head → logits of shape (B, L*G).
      3. Sum source logits element-wise (late fusion of prompts).
      4. Average across batch → (L*G).
      5. Apply activation (sigmoid / softplus / tanh01), clamp to max_scale.
      6. Reshape to (L, G) and return.

    Parameter naming uses the ``lora_`` prefix so the outer training loop can
    select them into a dedicated optimizer group.
    """

    def __init__(
        self,
        task_configs: Sequence[Dict],
        num_layers: int,
        prompt_sources: Sequence[str] = ("task_type", "task_id"),
        embed_dim: int = 64,
        hidden_dim: Optional[int] = None,
        activation: str = "sigmoid",
        max_scale: float = 0.5,
        num_groups: int = NUM_SCALE_GROUPS,
    ):
        super().__init__()
        if not prompt_sources:
            raise ValueError("prompt_sources must not be empty")
        if activation not in _ACTIVATIONS:
            raise ValueError(
                f"Unsupported activation '{activation}'. "
                f"Choose from {list(_ACTIVATIONS.keys())}"
            )
        self.num_layers = int(num_layers)
        self.num_groups = int(num_groups)
        self.embed_dim = int(embed_dim)
        self.hidden_dim = int(hidden_dim) if hidden_dim is not None else self.embed_dim
        self.activation_name = activation
        self._activation_fn = _ACTIVATIONS[activation]
        self.max_scale = float(max_scale)

        self._specs: List[PromptSourceSpec] = build_prompt_source_specs(
            prompt_sources=prompt_sources, task_configs=task_configs,
        )

        self.lora_prompt_embeddings = nn.ModuleDict()
        self.lora_ctrl_heads = nn.ModuleDict()
        out_dim = self.num_layers * self.num_groups
        for spec in self._specs:
            self.lora_prompt_embeddings[spec.name] = nn.Embedding(spec.vocab_size, self.embed_dim)
            self.lora_ctrl_heads[spec.name] = self._make_ctrl_head(self.embed_dim, self.hidden_dim, out_dim)

    # ------- helpers ---------------------------------------------------------

    @staticmethod
    def _make_ctrl_head(embed_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
        """Two-layer MLP. Final layer zero-init → initial scales concentrate near
        activation(0): sigmoid→0.5, softplus→0.69, tanh01→0.5. That keeps the
        first-step behaviour close to unit gating regardless of chosen activation."""
        head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )
        nn.init.zeros_(head[-1].weight)
        nn.init.zeros_(head[-1].bias)
        return head

    @property
    def prompt_source_names(self) -> List[str]:
        return [s.name for s in self._specs]

    def vocab_info(self) -> Dict[str, List[str]]:
        return {s.name: list(s.vocab) for s in self._specs}

    def _lookup_indices(
        self, task_id: str, batch_size: int, device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for spec in self._specs:
            if task_id not in spec.task_id_to_idx:
                raise ValueError(
                    f"Prompt source '{spec.name}' has no entry for task_id='{task_id}'"
                )
            idx = spec.task_id_to_idx[task_id]
            out[spec.name] = torch.full(
                (batch_size,), idx, dtype=torch.long, device=device,
            )
        return out

    # ------- forward ---------------------------------------------------------

    def forward(
        self, task_id: str, batch_size: int, device: torch.device,
    ) -> torch.Tensor:
        """Return runtime scales of shape ``(num_layers, num_groups)``."""
        indices = self._lookup_indices(task_id, batch_size=batch_size, device=device)

        total_logits: Optional[torch.Tensor] = None
        for spec in self._specs:
            emb = self.lora_prompt_embeddings[spec.name](indices[spec.name])  # [B, E]
            logits = self.lora_ctrl_heads[spec.name](emb)                     # [B, L*G]
            total_logits = logits if total_logits is None else total_logits + logits

        assert total_logits is not None
        # Batch-mean → shared scales for all samples in the batch (which by
        # construction come from a single task).
        mean_logits = total_logits.mean(dim=0)                                # [L*G]
        # scales = self._activation_fn(mean_logits).clamp(max=self.max_scale)
        scales = 1.0 + self.max_scale * (2.0 * torch.sigmoid(mean_logits) - 1.0)
        # 范围 [1 - max_scale, 1 + max_scale]，零点就是 1.0（= 原非 gated 行为）

        return scales.view(self.num_layers, self.num_groups)

    def scale_l1(self, scales: torch.Tensor) -> torch.Tensor:
        """For optional sparsity regularisation on the current scales tensor."""
        return scales.abs().sum()


# ---------------------------------------------------------------------------
# Section 4: Injection into timm ViT blocks
# ---------------------------------------------------------------------------

@dataclass
class LoraLayerHandle:
    """Per-block registry of LoRA modules. The controller writes scales here."""
    qkv: Optional[FusedQKVLoRALinear] = None
    proj: Optional[LoRALinear] = None
    fc1: Optional[LoRALinear] = None
    fc2: Optional[LoRALinear] = None


@dataclass
class LoraHandle:
    """Bundle returned by :func:`inject_lora` / :func:`inject_gated_lora`.

    ``controller`` is populated only for the gated variant; in the simple
    variant it stays ``None`` and :meth:`write_scales` is a no-op because each
    wrapped LoRA module keeps its default ``runtime_scale = 1.0``.
    """
    controller: Optional[GatedLoraController] = None
    layer_handles: List[LoraLayerHandle] = field(default_factory=list)

    @property
    def is_gated(self) -> bool:
        return self.controller is not None

    # ---- Scale write-back --------------------------------------------------

    def write_scales(self, scales: torch.Tensor) -> None:
        """Write ``scales`` of shape ``(num_layers, num_groups)`` into every
        LoRA module. Gradient paths through ``scales`` are preserved."""
        if scales.ndim != 2:
            raise ValueError(f"scales must be 2-D (L, G), got shape {tuple(scales.shape)}")
        L, G = scales.shape
        if L != len(self.layer_handles):
            raise ValueError(
                f"scales has {L} layers but controller is bound to "
                f"{len(self.layer_handles)} blocks"
            )
        if G != NUM_SCALE_GROUPS:
            raise ValueError(
                f"scales has {G} groups, expected {NUM_SCALE_GROUPS} ({SCALE_GROUPS})"
            )
        for li, handle in enumerate(self.layer_handles):
            self._write_row(handle, scales[li])

    @staticmethod
    def _write_row(handle: LoraLayerHandle, row: torch.Tensor) -> None:
        if handle.qkv is not None:
            handle.qkv.set_runtime_scale_qkv(
                row[SCALE_GROUP_TO_IDX["q"]],
                row[SCALE_GROUP_TO_IDX["k"]],
                row[SCALE_GROUP_TO_IDX["v"]],
            )
        if handle.proj is not None:
            handle.proj.set_runtime_scale(row[SCALE_GROUP_TO_IDX["proj"]])
        if handle.fc1 is not None:
            handle.fc1.set_runtime_scale(row[SCALE_GROUP_TO_IDX["fc1"]])
        if handle.fc2 is not None:
            handle.fc2.set_runtime_scale(row[SCALE_GROUP_TO_IDX["fc2"]])

    # ---- Summary -----------------------------------------------------------

    def num_replaced(self) -> int:
        total = 0
        for h in self.layer_handles:
            total += sum(m is not None for m in (h.qkv, h.proj, h.fc1, h.fc2))
        return total


def _wrap_linear_in_block(
    parent: nn.Module,
    attr: str,
    *,
    wrapper_cls,
    r: int,
    alpha: float,
    dropout: float,
    lora_only: bool,
) -> Optional[nn.Module]:
    """Replace ``parent.attr`` (nn.Linear) with ``wrapper_cls`` in place."""
    layer = getattr(parent, attr, None)
    if not isinstance(layer, nn.Linear):
        return None
    wrapped = wrapper_cls(layer, r=r, alpha=alpha, dropout=dropout, lora_only=lora_only)
    setattr(parent, attr, wrapped)
    return wrapped


_DEFAULT_TARGET_MODULES: Tuple[str, ...] = ("qkv", "proj", "fc1", "fc2")

# Attribute names timm uses to store the inner backbone inside a features-only
# wrapper (FeatureListNet / FeatureHookNet).
_TIMM_INNER_ATTRS: Tuple[str, ...] = ("model", "backbone", "encoder", "net")


def _unwrap_vit(model: nn.Module) -> nn.Module:
    """Return the first nn.Module in the ownership chain that has `.blocks`.

    timm's ``features_only=True`` wraps the real ViT in a ``FeatureListNet``
    (or similar) that hides the ``blocks`` attribute.  We try a fixed set of
    common inner-model attribute names before giving up.
    """
    if hasattr(model, "blocks"):
        return model
    for attr in _TIMM_INNER_ATTRS:
        inner = getattr(model, attr, None)
        if inner is not None and hasattr(inner, "blocks"):
            return inner
    raise ValueError(
        "Could not find `.blocks` in the supplied model or any of its "
        f"sub-attributes {_TIMM_INNER_ATTRS}. "
        "Pass a timm VisionTransformer (or its features-only wrapper)."
    )


def inject_lora(
    vit_model: nn.Module,
    *,
    target_modules: Sequence[str] = _DEFAULT_TARGET_MODULES,
    r: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    lora_only: bool = True,
) -> LoraHandle:
    """Inject runtime-scaled LoRA into every block of a timm ViT.

    The model is expected to expose ``.blocks[i].attn.{qkv,proj}`` and
    ``.blocks[i].mlp.{fc1,fc2}``.  If ``vit_model`` is a timm
    ``features_only=True`` wrapper, the inner ViT is located automatically.
    Leaf names listed in ``target_modules`` get replaced: ``qkv`` uses
    :class:`FusedQKVLoRALinear` (3 independent runtime scales for q/k/v via
    one shared A and three B's); ``proj``/``fc1``/``fc2`` use
    :class:`LoRALinear` (one scale each). Non-matching or missing leaves are
    silently skipped.

    Every wrapper starts with ``runtime_scale = 1.0``, so without a controller
    the forward is equivalent to classic fixed-scale LoRA with ``alpha/r``.

    Returns a :class:`LoraHandle` with ``controller=None``.
    """
    vit_model = _unwrap_vit(vit_model)

    targets = set(target_modules)
    kwargs = dict(r=r, alpha=alpha, dropout=dropout, lora_only=lora_only)
    layer_handles: List[LoraLayerHandle] = []
    for block in vit_model.blocks:
        handle = LoraLayerHandle()

        attn = getattr(block, "attn", None)
        if attn is not None:
            if "qkv" in targets:
                handle.qkv = _wrap_linear_in_block(
                    attn, "qkv", wrapper_cls=FusedQKVLoRALinear, **kwargs,
                )
            if "proj" in targets:
                handle.proj = _wrap_linear_in_block(
                    attn, "proj", wrapper_cls=LoRALinear, **kwargs,
                )

        mlp = getattr(block, "mlp", None)
        if mlp is not None:
            if "fc1" in targets:
                handle.fc1 = _wrap_linear_in_block(
                    mlp, "fc1", wrapper_cls=LoRALinear, **kwargs,
                )
            if "fc2" in targets:
                handle.fc2 = _wrap_linear_in_block(
                    mlp, "fc2", wrapper_cls=LoRALinear, **kwargs,
                )

        layer_handles.append(handle)

    return LoraHandle(controller=None, layer_handles=layer_handles)


def inject_gated_lora(
    vit_model: nn.Module,
    task_configs: Sequence[Dict],
    *,
    target_modules: Sequence[str] = _DEFAULT_TARGET_MODULES,
    r: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    lora_only: bool = True,
    prompt_sources: Sequence[str] = ("task_type", "task_id"),
    embed_dim: int = 64,
    hidden_dim: Optional[int] = None,
    activation: str = "sigmoid",
    max_scale: float = 0.5,
) -> LoraHandle:
    """Same as :func:`inject_lora`, but also builds a :class:`GatedLoraController`
    and attaches it to the returned handle.

    The caller is responsible for registering the controller as a submodule of
    the enclosing encoder and invoking
    ``handle.write_scales(controller(task_id, ...))`` before each forward.
    """
    handle = inject_lora(
        vit_model,
        target_modules=target_modules,
        r=r,
        alpha=alpha,
        dropout=dropout,
        lora_only=lora_only,
    )
    handle.controller = GatedLoraController(
        task_configs=task_configs,
        num_layers=len(handle.layer_handles),
        prompt_sources=prompt_sources,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        activation=activation,
        max_scale=max_scale,
    )
    return handle
