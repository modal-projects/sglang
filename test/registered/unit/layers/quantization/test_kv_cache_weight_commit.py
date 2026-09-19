from types import SimpleNamespace

import torch

from sglang.srt.layers.quantization.kv_cache import BaseKVCacheMethod
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_weight_commit_refreshes_runtime_kv_scales():
    layer = SimpleNamespace(
        k_scale=torch.tensor(2.0),
        v_scale=torch.tensor(3.0),
        k_scale_float=1.0,
        v_scale_float=1.0,
    )

    BaseKVCacheMethod(None).process_weights_after_weight_commit(layer)

    assert layer.k_scale_float == 2.0
    assert layer.v_scale_float == 3.0
