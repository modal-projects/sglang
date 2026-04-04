"""Shared Qwen3.5 checkpoint/load-time name mapping helpers."""

from __future__ import annotations

from typing import Final

QWEN3_5_STACKED_PARAMS_MAPPING: Final = (
    ("qkv_proj", "q_proj", "q"),
    ("qkv_proj", "k_proj", "k"),
    ("qkv_proj", "v_proj", "v"),
    ("gate_up_proj", "gate_proj", 0),
    ("gate_up_proj", "up_proj", 1),
    ("in_proj_qkvz.", "in_proj_qkv.", (0, 1, 2)),
    ("in_proj_qkvz.", "in_proj_z.", 3),
    ("in_proj_ba.", "in_proj_b.", 0),
    ("in_proj_ba.", "in_proj_a.", 1),
)

QWEN3_5_IGNORE_SUFFIXES: Final = (
    ".bias",
    "_bias",
    ".k_scale",
    "_k_scale",
    ".v_scale",
    "_v_scale",
    ".weight_scale",
    "_weight_scale",
    ".input_scale",
    "_input_scale",
)

QWEN3_5_FUSED_EXPERT_PARAMS_MAPPING: Final = (
    ("experts.w13_weight", "experts.gate_up_proj", 0, "w1"),
    ("experts.w2_weight", "experts.down_proj", 0, "w2"),
)


def normalize_qwen3_5_checkpoint_name(name: str) -> str:
    if "language_model" in name:
        name = name.replace(r"model.language_model.", r"model.")
    if ".self_attn." in name:
        name = name.replace(".self_attn", "")
    return name
