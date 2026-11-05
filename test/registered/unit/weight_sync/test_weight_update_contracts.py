"""Run the adjacent weight-update and sampling contract tests in CPU CI."""

from pathlib import Path

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=120, suite="base-a-test-cpu")

_SRT_ROOT = Path(__file__).resolve().parents[4] / "python" / "sglang" / "srt"
_TEST_MODULES = (
    "arg_groups/validation_hook_test.py",
    "layers/moe/fused_moe_triton/layer_test.py",
    "layers/quantization/fp8_test.py",
    "layers/quantization/fp8_utils_test.py",
    "layers/quantization/kv_cache_test.py",
    "layers/quantization/modelopt_quant_test.py",
    "layers/quantization/mxfp4_test.py",
    "layers/utils/common_test.py",
    "managers/scheduler_components/weight_updater_test.py",
    "model_executor/model_runner_components/weight_updater_test.py",
    "models/deepseek_common/deepseek_weight_loader_test.py",
    "models/deepseek_v4_test.py",
    "models/glm4_moe_test.py",
    "models/glm4_moe_lite_test.py",
    "models/kimi_k3_test.py",
    "models/qwen3_5_test.py",
    "models/registry_test.py",
    "speculative/spec_sampling_mask_test.py",
    "weight_sync/canonical_checkpoint_test.py",
    "weight_sync/canonical_delta_test.py",
    "weight_sync/disk_checkpoint_test.py",
    "weight_sync/host_local_buffer_test.py",
    "weight_sync/rank_weight_compiler_test.py",
    "weight_sync/rank_weight_image_test.py",
    "weight_sync/rank_weight_stager_test.py",
    "weight_sync/safetensors_buffer_test.py",
    "weight_sync/weight_load_isolation_test.py",
)


if __name__ == "__main__":
    raise SystemExit(
        pytest.main(
            [
                *(str(_SRT_ROOT / module) for module in _TEST_MODULES),
                "--import-mode=importlib",
                "-v",
            ]
        )
    )
