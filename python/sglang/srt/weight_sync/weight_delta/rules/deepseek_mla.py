"""DeepSeek MLA family (DeepSeek V3/V3.2, GLM-5.x) and its MTP draft."""

from __future__ import annotations

import re

import torch

from sglang.srt.weight_sync.weight_delta.map import Copy, Derived, Destination, Ignored

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
# Each fused parameter's scales (block FP8) are fused the same way as its weights.
FUSED_KINDS = ("weight", "weight_scale_inv")
EXPERT = re.compile(
    r"(?P<module>.+\.mlp\.experts)\.(?P<expert>\d+)\.(?P<projection>gate|up|down)_proj\."
    r"(?P<kind>weight|weight_scale|weight_scale_2|weight_scale_inv|input_scale)$"
)
# The DSA indexer's wk (top rows) and weights_proj (bottom rows) share one BF16 parameter;
# the loader dequantizes a block-FP8 checkpoint's shard into it.
INDEXER = re.compile(
    r"^(?P<module>.+\.indexer)\.(?P<shard>wk|weights_proj)\.(?P<kind>weight|weight_scale_inv)$"
)
LAYER = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
# MTP tensors the draft keeps at its top level instead of under model.decoder.
MTP_TOP = ("eh_proj.", "enorm.", "hnorm.", "shared_head.norm.")


class DeepSeekMla:
    architectures = {
        "DeepseekV3ForCausalLM",
        "DeepseekV32ForCausalLM",
        "GlmMoeDsaForCausalLM",
    }

    def __init__(self, models, inventory):
        config = models["target"].config
        self.num_layers = config.num_hidden_layers
        self.num_routed_experts = config.n_routed_experts
        self.inventory = inventory
        self.models = models
        self.params = {
            role: dict(model.named_parameters(remove_duplicate=False))
            for role, model in models.items()
        }
        # Per role (the draft decides on its own): shared experts loaded as expert
        # n_routed_experts of the MoE layer instead of as their own MLP.
        self.fused_shared = {
            role: any(
                getattr(module, "num_fused_shared_experts", 0) > 0
                for module in model.modules()
            )
            for role, model in models.items()
        }

    def destination(self, name: str) -> Destination | Ignored:
        role, path = "target", name
        match = LAYER.match(name)
        if match and int(match[1]) >= self.num_layers:
            if int(match[1]) > self.num_layers:
                raise ValueError(f"{name}: only one MTP layer is supported")
            rest = match[2]
            if rest.startswith(("embed_tokens.", "shared_head.head.")):
                return Ignored("the draft uses the target's embedding and lm_head")
            if "draft" not in self.models:
                return Ignored(
                    "the engine runs no draft model (speculative decoding is off)"
                )
            role = "draft"
            path = (
                f"model.{rest}" if rest.startswith(MTP_TOP) else f"model.decoder.{rest}"
            )
        if path.endswith("rotary_emb.inv_freq"):
            return Ignored("the loader skips it")
        if self.fused_shared[role] and ".mlp.shared_experts." in path:
            path = path.replace(
                ".mlp.shared_experts.", f".mlp.experts.{self.num_routed_experts}."
            )
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
        indexer = INDEXER.match(path)
        if indexer and self._dequantized(role, name, indexer):
            return Destination(role, param=path, shadow=True)
        for shard0, shard1, fused in FUSED:
            for shard, leaf in enumerate((shard0, shard1)):
                for kind in FUSED_KINDS:
                    suffix = f".{leaf}.{kind}"
                    if path.endswith(suffix):
                        prefix = name[: -len(suffix)]
                        return Destination(
                            role,
                            param=f"{path[: -len(suffix)]}.{fused}.{kind}",
                            shards=(
                                f"{prefix}.{shard0}.{kind}",
                                f"{prefix}.{shard1}.{kind}",
                            ),
                            shard=shard,
                        )
        return Destination(role, param=path)

    def _dequantized(self, role, name, match) -> bool:
        """Whether the loader dequantizes this indexer shard into the fused BF16 parameter."""
        fused = self.params[role].get(f"{match['module']}.wk_weights_proj.weight")
        weight = self.inventory[name.removesuffix(match["kind"]) + "weight"]
        return (
            fused is not None
            and fused.dtype == torch.bfloat16
            and weight.dtype != "BF16"
        )

    def copies(self, module) -> list[Copy]:
        """w_kc and w_vc are kv_b_proj's key and value parts, stored separately."""
        if getattr(module, "w_kc", None) is None or not hasattr(module, "kv_b_proj"):
            return []
        weight = module.kv_b_proj.weight
        if weight.dtype != module.w_kc.dtype:
            return []  # computed: see derived
        nope, v = module.qk_nope_head_dim, module.v_head_dim
        heads = weight.unflatten(0, (-1, nope + v))
        return [
            Copy(heads[:, :nope], module.w_kc),
            Copy(heads[:, nope:].transpose(1, 2), module.w_vc),
        ]

    def derived(self, role, path, module, shadows) -> list[Derived]:
        """What the loader computes from block-FP8 tensors: w_kc/w_vc from kv_b_proj
        (post_load_weights) and the indexer's fused BF16 wk_weights_proj."""
        from sglang.srt.layers.quantization.fp8_utils import block_quant_dequant
        from sglang.srt.models.deepseek_common.deepseek_weight_loader import (
            _dequant_fused_indexer_shard,
        )

        found = []
        kv_b = getattr(module, "kv_b_proj", None)
        if getattr(module, "w_kc", None) is not None and kv_b is not None:
            if kv_b.weight.dtype != module.w_kc.dtype:
                if module.w_kc.dtype != torch.bfloat16 or not hasattr(
                    kv_b, "weight_scale_inv"
                ):
                    raise ValueError(
                        f"{path}: w_kc is {module.w_kc.dtype} from a "
                        f"{kv_b.weight.dtype} kv_b_proj; only block-FP8 -> BF16 is mapped"
                    )
                block = kv_b.quant_method.weight_block_size
                nope, v = module.qk_nope_head_dim, module.v_head_dim

                def attention_values(module=module, kv_b=kv_b, block=block):
                    weight = block_quant_dequant(
                        kv_b.weight, kv_b.weight_scale_inv, block, torch.bfloat16
                    )
                    heads = weight.unflatten(0, (-1, nope + v))
                    return [
                        (heads[:, :nope], module.w_kc),
                        (heads[:, nope:].transpose(1, 2), module.w_vc),
                    ]

                found.append(Derived(attention_values))
        fused = getattr(getattr(module, "wk_weights_proj", None), "weight", None)
        if fused is not None:
            quant_config = getattr(self.models[role], "quant_config", None)
            for shard in ("wk", "weights_proj"):
                weight = shadows.get((role, f"{path}.{shard}.weight"))
                if weight is None:
                    continue
                scale = shadows.get((role, f"{path}.{shard}.weight_scale_inv"))
                if scale is None:
                    raise ValueError(
                        f"{path}.{shard}: weight shadowed without its scale"
                    )

                def indexer_values(weight=weight, scale=scale, shard=shard):
                    value = _dequant_fused_indexer_shard(weight, scale, quant_config)
                    rows = value.shape[0]
                    return [(value, fused[:rows] if shard == "wk" else fused[-rows:])]

                found.append(Derived(indexer_values))
        return found
