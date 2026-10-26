"""DeepSeek MLA family (DeepSeek V3/V3.2, GLM-5.x, Kimi K2 LM) and its MTP draft."""

from __future__ import annotations

import re
from dataclasses import dataclass

from sglang.srt.weight_sync.stitch_delta.map import Copy, Ignored

# (shard 0, shard 1, fused parameter), as the loader stacks them along dim 0.
FUSED = (
    (
        "self_attn.q_a_proj",
        "self_attn.kv_a_proj_with_mqa",
        "self_attn.fused_qkv_a_proj_with_mqa",
    ),
    (
        "self_attn.indexer.wk",
        "self_attn.indexer.weights_proj",
        "self_attn.indexer.wk_weights_proj",
    ),
    (
        "mlp.shared_experts.gate_proj",
        "mlp.shared_experts.up_proj",
        "mlp.shared_experts.gate_up_proj",
    ),
    ("mlp.gate_proj", "mlp.up_proj", "mlp.gate_up_proj"),
)
EXPERT = re.compile(
    r"(?P<module>.+\.mlp\.experts)\.(?P<expert>\d+)\."
    r"(?P<projection>gate|up|down)_proj\.(?P<kind>weight|weight_scale|weight_scale_2|input_scale)$"
)
LAYER = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
# MTP tensors the draft keeps at its top level instead of under model.decoder.
MTP_TOP = ("eh_proj.", "enorm.", "hnorm.", "shared_head.norm.")


@dataclass(frozen=True)
class Target:
    runner: str
    param: str = ""
    shards: tuple[str, ...] = ()
    shard: int = 0
    module: str = ""
    expert: int | None = None
    projection: str = ""
    kind: str = ""


class DeepSeekMla:
    architectures = {
        "DeepseekV3ForCausalLM",
        "DeepseekV32ForCausalLM",
        "GlmMoeDsaForCausalLM",
    }

    def __init__(self, config):
        self.num_layers = config.num_hidden_layers

    def target(self, name: str) -> Target | Ignored:
        runner, path = "target", name
        match = LAYER.match(name)
        if match and int(match[1]) >= self.num_layers:
            if int(match[1]) > self.num_layers:
                raise ValueError(f"{name}: only one MTP layer is supported")
            rest = match[2]
            if rest.startswith(("embed_tokens.", "shared_head.head.")):
                return Ignored("the draft uses the target's embedding and lm_head")
            runner = "draft"
            path = (
                f"model.{rest}" if rest.startswith(MTP_TOP) else f"model.decoder.{rest}"
            )
        if path.endswith("rotary_emb.inv_freq"):
            return Ignored("the loader skips it")
        expert = EXPERT.match(path)
        if expert:
            if expert["kind"] == "input_scale":
                return Ignored("activation calibration, never trained")
            return Target(
                runner,
                module=expert["module"],
                expert=int(expert["expert"]),
                projection=expert["projection"],
                kind=expert["kind"],
            )
        for shard0, shard1, fused in FUSED:
            for shard, leaf in enumerate((shard0, shard1)):
                suffix = f".{leaf}.weight"
                if path.endswith(suffix):
                    prefix = name[: -len(suffix)]
                    return Target(
                        runner,
                        param=f"{path[: -len(suffix)]}.{fused}.weight",
                        shards=(
                            f"{prefix}.{shard0}.weight",
                            f"{prefix}.{shard1}.weight",
                        ),
                        shard=shard,
                    )
        return Target(runner, param=path)

    def copies(self, module) -> list[Copy]:
        """w_kc and w_vc are kv_b_proj's key and value parts, stored separately."""
        if getattr(module, "w_kc", None) is None or not hasattr(module, "kv_b_proj"):
            return []
        weight = module.kv_b_proj.weight
        if weight.dtype != module.w_kc.dtype:
            raise ValueError("w_kc/w_vc are computed from kv_b_proj, not copied")
        nope, v = module.qk_nope_head_dim, module.v_head_dim
        heads = weight.unflatten(0, (-1, nope + v))
        return [
            Copy(heads[:, :nope], module.w_kc),
            Copy(heads[:, nope:].transpose(1, 2), module.w_vc),
        ]
