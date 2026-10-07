from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.quantization import fp8


@pytest.mark.parametrize("mxfp8_layout, expected", [(False, "cpu"), (True, "cuda")])
def test_block_fp8_staging_uses_layout_device(monkeypatch, mxfp8_layout, expected):
    monkeypatch.setattr(fp8, "_is_fp8_fnuz", False)
    monkeypatch.setattr(fp8, "_use_aiter_bpreshuffle_gfx95", False)
    method = object.__new__(fp8.Fp8LinearMethod)
    method.block_quant = True
    method.is_checkpoint_fp8_serialized = True
    method.use_mxfp8 = False
    method.convert_mxfp8_to_block = False
    method.block_fp8_as_mxfp8 = mxfp8_layout
    method.w8a8_block_fp8_linear = object()
    layer = SimpleNamespace(weight=torch.empty((128, 128)))

    assert method.weight_staging_postprocess_device(layer) == expected
