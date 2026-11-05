import unittest
from types import SimpleNamespace

from sglang.srt.layers.linear import LinearBase
from sglang.srt.layers.quantization.modelopt_quant import (
    ModelOptFp4Config,
    ModelOptFp4LinearMethod,
)
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# A routed-expert NVFP4 checkpoint's ignore list, in the checkpoint's module names.
EXCLUDE_MODULES = [
    "head",
    "layers.0.attn.wq_a",
    "layers.0.attn.wkv",
    "layers.0.attn.wq_b",
    "layers.0.attn.wo_b",
    "layers.0.attn.indexer.wq_b",
    "layers.0.ffn.shared_experts.w1",
    "layers.0.ffn.shared_experts.w2",
    "layers.0.ffn.shared_experts.w3",
    "layers.1.engram.wkv",
]


class TestDeepseekV4ModelOptNvFp4(CustomTestCase):
    def _config(self, quant_method: str):
        config = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
            exclude_modules=list(EXCLUDE_MODULES),
            packed_modules_mapping={},
        )
        hf_config = SimpleNamespace(
            quantization_config={"quant_method": quant_method, "quant_algo": "NVFP4"}
        )
        config.apply_weight_name_mapper(
            DeepseekV4ForCausalLM.get_hf_to_sglang_mapper(hf_config)
        )
        return config

    def test_checkpoint_exclusions_keep_sglang_linears_unquantized(self):
        linear = LinearBase.__new__(LinearBase)
        head = ParallelLMHead.__new__(ParallelLMHead)
        # The checkpoint's quant_method, and the one ModelConfig resolves before the load.
        for quant_method in ("modelopt", "modelopt_fp4"):
            config = self._config(quant_method)
            for layer, prefix in (
                *(
                    (linear, f"model.layers.0.self_attn.{name}")
                    for name in ("wq_a", "wkv", "wq_b", "wo_b", "indexer.wq_b")
                ),
                (linear, "model.layers.0.mlp.shared_experts.gate_up_proj"),
                (linear, "model.layers.0.mlp.shared_experts.down_proj"),
                (linear, "model.layers.1.engram.wkv"),
                (head, "lm_head"),
            ):
                with self.subTest(quant_method=quant_method, prefix=prefix):
                    self.assertIsInstance(
                        config.get_quant_method(layer, prefix), UnquantizedLinearMethod
                    )

    def test_unlisted_modules_stay_nvfp4(self):
        config = self._config("modelopt_fp4")
        self.assertFalse(config.is_layer_excluded("model.layers.0.mlp.experts"))
        linear = LinearBase.__new__(LinearBase)
        self.assertIsInstance(
            config.get_quant_method(linear, "model.layers.2.self_attn.wq_a"),
            ModelOptFp4LinearMethod,
        )


if __name__ == "__main__":
    unittest.main()
