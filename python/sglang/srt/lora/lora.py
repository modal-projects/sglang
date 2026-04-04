# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Integrates "S-LoRA: Serving Thousands of Concurrent LoRA Adapters"
# and "Punica: Multi-Tenant LoRA Serving"

# LoRA layers class inheritance adapted from:
# https://github.com/vllm-project/vllm/blob/4abf6336ec65c270343eb895e7b18786e9274176/vllm/lora/layers.py

import logging
from typing import Dict, List

import torch
from torch import nn

from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.layers.utils import get_layer_id
from sglang.srt.lora.backend.base_backend import BaseLoRABackend
from sglang.srt.lora.backend.lora_registry import LORA_SUPPORTED_BACKENDS
from sglang.srt.lora.lora_config import LoRAConfig
from sglang.srt.lora.utils import (
    normalize_lora_gate_up_weights,
    normalize_lora_qkv_weights,
    rename_lora_expert_w_to_proj_name,
    rewrite_lora_embedding_aliases_in_weight_name,
)
from sglang.srt.model_loader.loader import DefaultModelLoader
from sglang.srt.utils.hf_transformers_utils import AutoConfig

logger = logging.getLogger(__name__)


class LoRALayer(nn.Module):
    def __init__(self, config: LoRAConfig, base_hf_config: AutoConfig):
        super().__init__()
        self.config: LoRAConfig = config
        self.base_hf_config: AutoConfig = base_hf_config

        # lora weights in cpu. The weights are loaded from checkpoint.
        self.weights: Dict[str, torch.Tensor] = {}


class LoRAAdapter(nn.Module):
    def __init__(
        self,
        uid: str,
        config: LoRAConfig,
        base_hf_config: AutoConfig,
        load_config: LoadConfig,
        lora_backend: BaseLoRABackend,
    ):
        super().__init__()
        self.uid: str = uid
        self.config: LoRAConfig = config
        assert self.config.hf_config["peft_type"].lower() == "lora"
        self.base_hf_config: AutoConfig = base_hf_config
        self.load_config: LoadConfig = load_config
        self.lora_backend: BaseLoRABackend = lora_backend
        self.scaling: float = self.config.lora_alpha / self.config.r

        self.layers: List[LoRALayer] = nn.ModuleList(
            [
                LoRALayer(config, base_hf_config)
                for _ in range(base_hf_config.num_hidden_layers)
            ]
        )

        self.embedding_layers: Dict[str, torch.Tensor] = {}
        self.added_tokens_embeddings: Dict[str, torch.Tensor] = {}

    def initialize_weights(self):
        model_path = self.config.path
        loader = DefaultModelLoader(self.load_config)
        revision = getattr(self.config.hf_config, "revision", None)

        # Get normalized target modules for filtering
        for name, loaded_weight in loader._get_weights_iterator(
            DefaultModelLoader.Source(
                model_path, revision=revision, fall_back_to_pt=True
            )
        ):
            self._process_weight(name, loaded_weight)

        self._normalize_weights()

    def initialize_weights_from_tensors(self, tensors: Dict[str, torch.Tensor]):
        for name, tensor in tensors.items():
            self._process_weight(name, tensor)

        self._normalize_weights()

    def _process_weight(self, name: str, loaded_weight: torch.Tensor):
        from sglang.srt.lora.utils import get_normalized_target_modules

        normalized_target_modules = get_normalized_target_modules(
            self.config.target_modules
        )

        # Remap PEFT aliases so the weight is recognized and loaded into the
        # correct buffer or later normalization pass.
        name = rewrite_lora_embedding_aliases_in_weight_name(name)

        layer_id = get_layer_id(name)
        if layer_id is not None:
            self.layers[layer_id].weights[name] = loaded_weight.cpu()
        elif "embed_tokens" in name or "lm_head" in name:
            # Check if this module is declared in target_modules before loading.
            # When normalized_target_modules is {"all"} (e.g. target_modules was
            # "all-linear"), we allow loading since the server-level
            # --lora-target-modules will govern which modules are active.
            module_name = "embed_tokens" if "embed_tokens" in name else "lm_head"
            if (
                "all" in normalized_target_modules
                or module_name in normalized_target_modules
            ):
                self.embedding_layers[name] = loaded_weight.cpu()
            else:
                logger.debug(
                    f"Skipping {name} as '{module_name}' is not in adapter's target_modules: {self.config.target_modules}"
                )
        elif "input_embeddings" in name or "output_embeddings" in name:
            # added/extra token emb
            self.added_tokens_embeddings[name] = loaded_weight.cpu()
            assert loaded_weight.shape[0] == self.config.lora_added_tokens_size, (
                f"LoRA adapter {self.uid} has lora_added_tokens_size {self.config.lora_added_tokens_size} specified in the config, "
                f"but the loaded weight '{name}' has shape {loaded_weight.shape[0]} in first dimension"
            )

    def _normalize_weights(self):
        # normalize kv_proj and gate_up_proj
        for layer in self.layers:
            self.normalize_qkv_proj(list(layer.weights.keys()), layer.weights)
            self._rename_expert_w_to_proj(layer.weights)
            self.normalize_gate_up_proj(list(layer.weights.keys()), layer.weights)

    def normalize_qkv_proj(
        self, weight_names: List[str], weights: Dict[str, torch.Tensor]
    ):
        del weight_names
        normalize_lora_qkv_weights(weights)

    def _rename_expert_w_to_proj(self, weights: Dict[str, torch.Tensor]):
        """Rename w1 -> gate_proj, w3 -> up_proj, w2 -> down_proj so that
        normalize_gate_up_proj can stack them into gate_up_proj."""
        renames = {}
        for name in list(weights.keys()):
            new_name = rename_lora_expert_w_to_proj_name(name)
            if new_name != name:
                renames[name] = new_name
        for old_name, new_name in renames.items():
            weights[new_name] = weights.pop(old_name)

    def normalize_gate_up_proj(
        self, weight_names: List[str], weights: Dict[str, torch.Tensor]
    ):
        del weight_names
        normalize_lora_gate_up_weights(
            weights,
            backend_name=self.lora_backend.name,
            supported_backend_names=set(LORA_SUPPORTED_BACKENDS),
        )

    def pin_weights_in_cpu(self):
        for layer in self.layers:
            for name, weight in layer.weights.items():
                layer.weights[name] = weight.pin_memory()

        for name, weight in self.embedding_layers.items():
            self.embedding_layers[name] = weight.pin_memory()

        for name, weight in self.added_tokens_embeddings.items():
            self.added_tokens_embeddings[name] = weight.pin_memory()
