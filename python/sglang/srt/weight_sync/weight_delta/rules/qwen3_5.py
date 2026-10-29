"""Qwen3.5 family (Qwen3.5/3.6/3.8, dense and MoE): Gated DeltaNet and gated attention layers,
the vision tower and the MTP draft."""

from __future__ import annotations

import re

from sglang.srt.weight_sync.weight_delta.map import Copy, Derived, Destination, Ignored

# (shards in checkpoint order, fused parameter), as the loader stacks them along dim 0.
FUSED = (
    (("q_proj", "k_proj", "v_proj"), "qkv_proj"),
    (("gate_proj", "up_proj"), "gate_up_proj"),
    (("in_proj_qkv", "in_proj_z"), "in_proj_qkvz"),
    (("in_proj_b", "in_proj_a"), "in_proj_ba"),
)
# Each fused parameter's scales (block FP8) are fused the same way as its weights.
FUSED_KINDS = ("weight", "weight_scale_inv")
EXPERT = re.compile(
    r"(?P<module>.+\.mlp\.experts)\.(?P<expert>\d+)\.(?P<projection>gate|up|down)_proj\."
    r"(?P<kind>weight|weight_scale|weight_scale_2|weight_scale_inv|input_scale)$"
)
# MTP tensors the draft keeps at its top level instead of under model.
MTP_TOP = ("fc.", "pre_fc_norm_embedding.", "pre_fc_norm_hidden.")


class Qwen3_5:
    architectures = {
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
    }

    def __init__(self, models, inventory):
        self.models = models

    def destination(self, name: str) -> Destination | Ignored:
        if name.endswith("rotary_emb.inv_freq"):
            return Ignored("the loader skips it")
        if name.startswith("model.visual."):
            path = name.replace("model.visual.", "visual.", 1)
            return Destination(
                "target", param=path.replace(".attn.qkv.", ".attn.qkv_proj.")
            )
        role, path = "target", name.replace("model.language_model.", "model.", 1)
        if name.startswith("mtp."):
            if "draft" not in self.models:
                return Ignored(
                    "the engine runs no draft model (speculative decoding is off)"
                )
            role, rest = "draft", name.removeprefix("mtp.")
            path = rest if rest.startswith(MTP_TOP) else f"model.{rest}"
        path = path.replace(".self_attn.", ".")
        expert = EXPERT.match(path)
        if expert:
            if expert["kind"] == "input_scale":
                return Ignored("activation calibration, never trained")
            return Destination(
                role,
                module=expert["module"],
                expert=int(expert["expert"]),
                projection=expert["projection"],
                kind=expert["kind"],
            )
        for shards, fused in FUSED:
            for shard, leaf in enumerate(shards):
                for kind in FUSED_KINDS:
                    suffix = f".{leaf}.{kind}"
                    if path.endswith(suffix):
                        prefix = name[: -len(suffix)]
                        return Destination(
                            role,
                            param=f"{path[: -len(suffix)]}.{fused}.{kind}",
                            shards=tuple(f"{prefix}.{each}.{kind}" for each in shards),
                            shard=shard,
                        )
        return Destination(role, param=path)

    def copies(self, module) -> list[Copy]:
        return []

    def derived(self, role, path, module, shadows) -> list[Derived]:
        return []
