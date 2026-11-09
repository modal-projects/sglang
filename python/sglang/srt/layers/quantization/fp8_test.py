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


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="MXFP8 online quantization requires Blackwell",
)
@pytest.mark.parametrize("kind", ["linear", "moe"])
def test_online_mxfp8_reload_matches_fresh_quantization(monkeypatch, kind):
    from sglang.srt.layers.moe.utils import MoeRunnerBackend
    from sglang.srt.layers.quantization import fp8_utils
    from sglang.srt.model_loader.weight_utils import default_weight_loader
    from sglang.srt.runtime_context import get_flags, get_parallel

    monkeypatch.setattr(
        fp8_utils,
        "FP8_GEMM_RUNNER_BACKEND",
        fp8_utils.Fp8GemmRunnerBackend.FLASHINFER_CUTLASS,
    )
    monkeypatch.setattr(get_flags().moe, "runner_backend", MoeRunnerBackend.TRITON)
    config = fp8.Fp8Config(use_mxfp8=True)

    def make(weights):
        layer = torch.nn.Module()
        if kind == "linear":
            method = fp8.Fp8LinearMethod(config)
            with torch.device("cuda"), get_parallel().override(tp_size=1):
                method.create_weights(
                    layer,
                    256,
                    [128],
                    256,
                    128,
                    torch.bfloat16,
                    weight_loader=default_weight_loader,
                )
        else:
            method = fp8.Fp8MoEMethod(config)
            layer.moe_runner_config = SimpleNamespace(is_gated=True)
            layer.num_local_experts = 2
            with torch.device("cuda"), get_parallel().override(tp_size=1):
                method.create_weights(layer, 2, 256, 128, torch.bfloat16)
        for name, weight in weights.items():
            default_weight_loader(getattr(layer, name), weight)
        method.process_weights_after_loading(layer)
        return method, layer

    shapes = (
        {"weight": (128, 256)}
        if kind == "linear"
        else {
            "w13_weight": (2, 256, 256),
            "w2_weight": (2, 256, 128),
        }
    )
    torch.manual_seed(1234)
    weights = {
        name: torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.02
        for name, shape in shapes.items()
    }
    method, reloaded = make(weights)
    parameter_ids = {name: id(getattr(reloaded, name)) for name in weights}
    for _ in range(2):
        weights = {
            name: value + torch.randn_like(value) * 0.001
            for name, value in weights.items()
        }
        _, fresh = make(weights)
        method.restore_weights_before_loading(reloaded)
        for name, weight in weights.items():
            assert id(getattr(reloaded, name)) == parameter_ids[name]
            assert getattr(reloaded, name).dtype == torch.bfloat16
            default_weight_loader(getattr(reloaded, name), weight)
        method.process_weights_after_loading(reloaded)
        actual = dict(reloaded.named_parameters())
        expected = dict(fresh.named_parameters())
        assert actual.keys() == expected.keys()
        for name in expected:
            assert torch.equal(actual[name], expected[name]), name
