"""LoRA target mapping for DeepSeek-V4 attention projections and shared experts.

Adapters may use checkpoint names (``layers.N.attn.wq_a``,
``layers.N.ffn.shared_experts.w1``) or SGLang module names. Normalized target
names are parent-qualified because the bare leaf names also exist under the
indexer (``wq_b``), the compressor and Engram (``wkv``) and the routed experts
(``w1``/``w2``/``w3``).
"""

from typing import Dict, Iterable, Optional, Tuple

import torch

WQKV_A = "self_attn.wqkv_a"
WQ_B = "self_attn.wq_b"
WO_B = "self_attn.wo_b"
SHARED_GATE_UP = "shared_experts.gate_up_proj"
SHARED_DOWN = "shared_experts.down_proj"

SUPPORTED_TARGET_MODULES = frozenset({WQKV_A, WQ_B, WO_B, SHARED_GATE_UP, SHARED_DOWN})

TARGET_MODULE_ALIASES = {
    "wq_a": WQKV_A,
    "wkv": WQKV_A,
    "wq_b": WQ_B,
    "wo_b": WO_B,
    "w1": SHARED_GATE_UP,
    "w3": SHARED_GATE_UP,
    "gate_proj": SHARED_GATE_UP,
    "up_proj": SHARED_GATE_UP,
    "gate_up_proj": SHARED_GATE_UP,
    "w2": SHARED_DOWN,
    "down_proj": SHARED_DOWN,
}

STACKED_MULTIPLY = {WQKV_A: 2, SHARED_GATE_UP: 2}


def remap_weight_name(name: str) -> str:
    name = name.replace(".attn.", ".self_attn.").replace(".ffn.", ".mlp.")
    for src, dst in (
        (".w1.", ".gate_proj."),
        (".w3.", ".up_proj."),
        (".w2.", ".down_proj."),
    ):
        name = name.replace(src, dst)
    return name


def fuse_wqa_wkv_weights(
    weights: Dict[str, torch.Tensor], q_dim: int, kv_dim: int
) -> None:
    """Stack wq_a and wkv LoRA factors into the fused wqkv_a projection.

    A becomes [A_q; A_kv] (2r x hidden) and B becomes [B_q; B_kv]
    ((q_dim + kv_dim) x r); a missing side is zero-filled.
    """
    pairs: Dict[str, Dict[str, str]] = {}
    for name in weights:
        for part, module in (("q", ".self_attn.wq_a."), ("kv", ".self_attn.wkv.")):
            if module in name:
                fused = name.replace(module, f".{WQKV_A}.")
                pairs.setdefault(fused, {})[part] = name

    for fused, parts in pairs.items():
        q = weights.pop(parts["q"]) if "q" in parts else None
        kv = weights.pop(parts["kv"]) if "kv" in parts else None
        present = q if q is not None else kv
        if "lora_A" in fused:
            q = torch.zeros_like(present) if q is None else q
            kv = torch.zeros_like(present) if kv is None else kv
        else:
            rank = present.shape[-1]
            if q is None:
                q = present.new_zeros(q_dim, rank)
            if kv is None:
                kv = present.new_zeros(kv_dim, rank)
        weights[fused] = torch.cat((q, kv), dim=0)


def normalize_layer_weights(
    weights: Dict[str, torch.Tensor], q_dim: int, kv_dim: int
) -> None:
    for name in list(weights):
        weights[remap_weight_name(name)] = weights.pop(name)
    fuse_wqa_wkv_weights(weights, q_dim, kv_dim)


def hidden_dims(module_name: str, config) -> Optional[Tuple[int, int]]:
    hidden = config.hidden_size
    q_rank = config.q_lora_rank
    head_dim = config.head_dim
    shared_inter = config.moe_intermediate_size * config.n_shared_experts
    return {
        WQKV_A: (hidden, q_rank + head_dim),
        WQ_B: (q_rank, config.num_attention_heads * head_dim),
        WO_B: (config.o_groups * config.o_lora_rank, hidden),
        SHARED_GATE_UP: (hidden, 2 * shared_inter),
        SHARED_DOWN: (shared_inter, hidden),
    }.get(module_name)


def validate_target_modules(
    target_modules: Iterable[str],
    *,
    quantized: bool,
    attn_cp_size: int,
    fuse_wqa_wkv: bool,
    fused_shared_experts: bool,
) -> None:
    target_modules = set(target_modules)
    unsupported = target_modules - SUPPORTED_TARGET_MODULES
    if unsupported:
        raise ValueError(
            f"DeepSeek-V4 LoRA does not support target modules {sorted(unsupported)}; "
            "supported targets are wq_a, wkv, wq_b, wo_b and the shared-expert "
            "w1, w2, w3."
        )
    if quantized:
        # Quantized linears can hand FP8/MXFP8 activations to the wrapped
        # projections, which the LoRA kernels cannot consume.
        raise ValueError("DeepSeek-V4 LoRA requires BF16 (unquantized) weights.")
    if attn_cp_size > 1:
        raise ValueError(
            "DeepSeek-V4 LoRA does not support attention context parallelism."
        )
    if not fuse_wqa_wkv:
        raise ValueError("DeepSeek-V4 LoRA requires SGLANG_OPT_FUSE_WQA_WKV=1.")
    if fused_shared_experts and target_modules & {SHARED_GATE_UP, SHARED_DOWN}:
        raise ValueError("Shared-expert LoRA requires unfused shared experts.")
