from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.layers.quantization import unquant
from sglang.srt.model_loader.weight_utils import default_weight_loader


@pytest.mark.parametrize(
    "backend",
    [MoeRunnerBackend.FLASHINFER_TRTLLM, MoeRunnerBackend.FLASHINFER_TRTLLM_ROUTED],
)
@pytest.mark.parametrize(
    "name,shape", [("w13_weight", (2, 256, 256)), ("w2_weight", (2, 256, 128))]
)
def test_trtllm_bf16_reload_accepts_checkpoint_layout(
    monkeypatch, backend, name, shape
):
    monkeypatch.setattr(unquant, "get_moe_runner_backend", lambda: backend)
    method = object.__new__(unquant.UnquantizedFusedMoEMethod)
    layer = SimpleNamespace(
        num_local_experts=2,
        hidden_size=256,
        intermediate_size_per_partition=128,
        moe_runner_config=SimpleNamespace(is_gated=True),
    )
    parameter = torch.nn.Parameter(
        torch.zeros(*shape[:-1], shape[-1] // 32, 32, dtype=torch.bfloat16),
        requires_grad=False,
    )
    checkpoint = (
        torch.arange(parameter.numel(), dtype=torch.float32)
        .reshape(shape)
        .to(torch.bfloat16)
    )
    pointer = parameter.data_ptr()
    method.maybe_restore_flashinfer_trtllm_bf16_weight_shape_for_load(
        layer, parameter, "model.layers.0.mlp.experts." + name
    )
    default_weight_loader(parameter, checkpoint)
    assert parameter.data_ptr() == pointer
    assert torch.equal(parameter, checkpoint)
