from types import SimpleNamespace

import torch

from sglang.srt.layers.quantization.kv_cache import BaseKVCacheMethod


def test_weight_commit_refreshes_runtime_kv_scales():
    layer = SimpleNamespace(
        k_scale=torch.tensor(2.0),
        v_scale=torch.tensor(3.0),
        k_scale_float=1.0,
        v_scale_float=1.0,
    )

    method = BaseKVCacheMethod(None)
    method.process_weights_after_weight_commit(layer)

    assert layer.k_scale_float == 2.0
    assert layer.v_scale_float == 3.0
    assert method.weight_staging_postprocess_device(layer) == "cpu"
