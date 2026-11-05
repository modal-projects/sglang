"""GLM-5.3-Flash (glm5_next): KDA and DSA layers, its vision tower and MTP draft."""

from __future__ import annotations

import itertools
import re

from sglang.srt.weight_sync.weight_delta.map import (
    TORCH_DTYPES,
    Copy,
    Derived,
    Destination,
    Ignored,
)
from sglang.srt.weight_sync.weight_delta.rules.deepseek_mla import kv_b_copies

# (shards in checkpoint order, fused parameter) in the loader's order; the first fused parameter
# the model built takes the shard, stacked along dim 0.
FUSED = (
    (("gate_proj", "up_proj"), "gate_up_proj"),
    (
        ("q_proj", "k_proj", "v_proj", "b_proj", "f_a_proj", "g_a_proj"),
        "fused_qkvbfg_a_proj",
    ),
    (("f_b_proj", "g_b_proj"), "fused_fg_b_proj"),
    (("b_proj", "f_a_proj", "g_a_proj"), "fused_bfg_a_proj"),
    (("q_proj", "k_proj", "v_proj"), "qkv_proj"),
    (("q_conv1d", "k_conv1d", "v_conv1d"), "qkv_conv1d"),
    (("q_a_proj", "kv_a_proj_with_mqa"), "fused_qkv_a_proj_with_mqa"),
)
FUSED_KINDS = ("weight", "bias")
EXPERT = re.compile(
    r"(?P<module>.+\.mlp\.experts)\.(?P<expert>\d+)\.(?P<projection>gate|up|down)_proj\."
    r"(?P<kind>weight|weight_scale|weight_scale_2|input_scale)$"
)
LAYER = re.compile(r"^model\.language_model\.layers\.(\d+)\.(.+)$")
# MTP tensors the draft keeps at its top level instead of under model.decoder.
MTP_TOP = ("eh_proj.", "enorm.", "hnorm.", "shared_head.norm.")


class Glm5Next:
    architectures = {"Glm5NextForConditionalGeneration"}

    def __init__(self, models, inventory):
        self.num_layers = models["target"].config.num_hidden_layers
        self.models = models
        self.inventory = inventory
        self.params = {
            role: dict(model.named_parameters(remove_duplicate=False))
            for role, model in models.items()
        }

    def destination(self, name: str) -> Destination | Ignored:
        role = "target"
        if name.startswith("model.visual."):
            path = name.replace("model.visual.", "visual.", 1)
            path = path.replace(".attn.qkv.", ".attn.qkv_proj.")
        else:
            path = name.replace("model.language_model.", "model.", 1)
            match = LAYER.match(name)
            if match and int(match[1]) >= self.num_layers:
                if "draft" not in self.models:
                    return Ignored(
                        "the engine runs no draft model (speculative decoding is off)"
                    )
                role, rest = "draft", match[2]
                path = (
                    f"model.{rest}"
                    if rest.startswith(MTP_TOP)
                    else f"model.decoder.{rest}"
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
        param, shards, shard = self._fused(role, name, path)
        served, tensor = self.params[role].get(param), self.inventory[name]
        if (
            served is not None
            and served.dtype != TORCH_DTYPES[tensor.dtype]
            and len(tensor.shape) > 1
        ):
            # The loader upcasts this matrix: it is XORed into a copy that derived recasts.
            return Destination(role, param=path, shadow=True)
        return Destination(role, param=param, shards=shards, shard=shard)

    def _fused(self, role, name, path):
        """The parameter holding the tensor, with its fused shards and its index among them."""
        for shards, fused in FUSED:
            for shard, leaf in enumerate(shards):
                for kind in FUSED_KINDS:
                    suffix = f".{leaf}.{kind}"
                    param = f"{path[: -len(suffix)]}.{fused}.{kind}"
                    if path.endswith(suffix) and param in self.params[role]:
                        prefix = name[: -len(suffix)]
                        return (
                            param,
                            tuple(f"{prefix}.{each}.{kind}" for each in shards),
                            shard,
                        )
        return path, (), 0

    def copies(self, module) -> list[Copy]:
        return kv_b_copies(module)

    def derived(self, role, path, module, shadows) -> list[Derived]:
        """The upcast matrices (KDA conv1d, the indexer's weights_proj), recast from their copies."""
        parent, _, leaf = path.rpartition(".")
        shards = next((each for each, fused in FUSED if fused == leaf), (leaf,))
        found = []
        for kind, param in module.named_parameters(recurse=False):
            copies = [shadows.get((role, f"{parent}.{each}.{kind}")) for each in shards]
            if any(copy is None for copy in copies):
                continue

            def values(param=param, copies=copies):
                starts = itertools.accumulate(
                    (copy.shape[0] for copy in copies), initial=0
                )
                return [
                    (copy.to(param.dtype), param.narrow(0, start, copy.shape[0]))
                    for copy, start in zip(copies, starts)
                ]

            found.append(Derived(values))
        return found
