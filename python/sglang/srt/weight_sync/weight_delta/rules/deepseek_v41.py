"""DeepSeek-V4.1 (models/deepseek_v4.py) and its vision tower."""

from __future__ import annotations

import re

from sglang.srt.weight_sync.weight_delta.map import (
    TORCH_DTYPES,
    Derived,
    Destination,
    Ignored,
)

EXPERT = re.compile(
    r"(?P<module>.+\.mlp\.experts)\.(?P<expert>\d+)\.(?P<projection>gate|up|down)_proj\."
    r"(?P<kind>weight|weight_scale|weight_scale_2|input_scale)$"
)
# (checkpoint shards in the loader's order, fused parameter), stacked along dim 0 wherever
# the model built the fused parameter: wq_a/wkv under SGLANG_OPT_FUSE_WQA_WKV, a ratio-2
# compressor's wkv/wgate, the shared expert's w1/w3.
FUSED = (
    (("wq_a", "wkv"), "wqkv_a"),
    (("wkv", "wgate"), "wkv_gate"),
    (("w1", "w3"), "gate_up_proj"),
)


class DeepSeekV41:
    # V4 and V4.1 share DeepseekV4ForCausalLM; the model type tells them apart.
    model_types = {"deepseek_v41"}

    def __init__(self, models, inventory):
        from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM

        self.remap = DeepseekV4ForCausalLM.remap_weight_name_to_dpsk_hf_format
        self.models = models
        self.params = dict(models["target"].named_parameters(remove_duplicate=False))
        self.inventory = inventory

    def destination(self, name: str) -> Destination | Ignored:
        if ".engram.embed." in name:
            return Ignored("the engram table is frozen: the trainer never exports it")
        if name.startswith("mtp.") and "draft" not in self.models:
            return Ignored(
                "the engine runs no draft model (speculative decoding is off)"
            )
        path = self.remap(name)
        expert = EXPERT.match(path)
        if expert:
            if expert["kind"] == "input_scale":
                return Ignored("activation calibration, never trained")
            return Destination(
                "target",
                module=expert["module"],
                expert=int(expert["expert"]),
                projection=expert["projection"],
                kind=expert["kind"],
            )
        for shards, fused in FUSED:
            for shard, leaf in enumerate(shards):
                suffix = f".{leaf}.weight"
                param = f"{path.rsplit('.', 2)[0]}.{fused}.weight"
                if name.endswith(suffix) and param in self.params:
                    prefix = name[: -len(suffix)]
                    return Destination(
                        "target",
                        param=param,
                        shards=tuple(f"{prefix}.{each}.weight" for each in shards),
                        shard=shard,
                    )
        served, tensor = self.params.get(path), self.inventory[name]
        if (
            served is not None
            and served.dtype != TORCH_DTYPES[tensor.dtype]
            and len(tensor.shape) > 1
        ):
            # Served in another dtype (Engram's FP32 q/k weights as BF16): XORed into a copy.
            return Destination("target", param=path, shadow=True)
        return Destination("target", param=path)

    def copies(self, module) -> list:
        return []

    def derived(self, role, path, module, shadows) -> list[Derived]:
        """Parameters served from a copy, cast as the loader casts; under FlashMLA's mega
        attention, the FP8 wo_a setup_mega_attention makes."""
        found = [
            Derived(lambda param=param, shadow=shadow: [(shadow, param)])
            for leaf, param in module.named_parameters(recurse=False)
            if (shadow := shadows.get((role, f"{path}.{leaf}"))) is not None
        ]
        if not getattr(module, "use_mega_attention", False):
            return found
        from sglang.srt.layers.attention.deepseek_v4_mega_backend import (
            interleave_output_columns,
            quantize_wo_a,
        )

        def wo_a_values(module=module):
            wo_a = module.wo_a.weight.view(
                module.n_local_groups, module.o_lora_rank, -1
            )
            served = quantize_wo_a(interleave_output_columns(wo_a))
            return list(zip(served, module.mega_wo_a, strict=True))

        return [*found, Derived(wo_a_values)]
