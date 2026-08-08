from types import SimpleNamespace

import torch

import sglang.srt.layers.quantization.mxfp4 as mxfp4
from sglang.srt.layers.quantization.mxfp4 import (
    Mxfp4MoEMethod,
    _compose_trtllm_gate_up_permutation,
    _parallel_cpu_zero_,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _make_method() -> Mxfp4MoEMethod:
    method = object.__new__(Mxfp4MoEMethod)
    method.use_marlin = False
    method.use_deep_gemm = False
    method.use_flashinfer = True
    method._fi_kernel = "trtllm_sm100"
    method.num_experts = 2
    method.intermediate_size_per_partition = 128
    method.hidden_size = 128
    return method


def _make_layer(seed: int) -> torch.nn.Module:
    generator = torch.Generator().manual_seed(seed)
    layer = torch.nn.Module()
    layer.num_local_experts = 2
    layer.intermediate_size_per_partition = 128
    layer.hidden_size = 128
    layer.moe_runner_config = SimpleNamespace(
        gate_up_interleaved=True,
        gemm1_alpha=None,
        gemm1_clamp_limit=None,
    )

    def parameter(shape, dtype):
        if dtype == torch.bfloat16:
            value = torch.randn(shape, generator=generator, dtype=dtype)
        else:
            value = torch.randint(0, 256, shape, generator=generator, dtype=dtype)
        return torch.nn.Parameter(value, requires_grad=False)

    layer.w13_weight = parameter((2, 256, 64), torch.uint8)
    layer.w13_weight_scale = parameter((2, 256, 4), torch.uint8)
    layer.w13_weight_bias = parameter((2, 256), torch.bfloat16)
    layer.w2_weight = parameter((2, 128, 64), torch.uint8)
    layer.w2_weight_scale = parameter((2, 128, 4), torch.uint8)
    layer.w2_weight_bias = parameter((2, 128), torch.bfloat16)
    return layer


def test_cpu_staging_rebuilds_runtime_layout_in_place(monkeypatch):
    monkeypatch.setattr(
        mxfp4,
        "_get_flashinfer_mxfp4_device_permute_indices",
        lambda tensor, *_args, **_kwargs: torch.arange(tensor.shape[0]),
    )
    method = _make_method()
    layer = _make_layer(seed=1)
    target = _make_layer(seed=2)
    reference = _make_layer(seed=2)
    method.process_weights_after_loading(layer)
    runtime_pointers = {
        name: parameter.data_ptr() for name, parameter in layer.named_parameters()
    }

    method.restore_weights_before_loading(layer)
    assert layer.w13_weight_scale.dtype == torch.uint8
    assert layer.w2_weight_scale.dtype == torch.uint8
    for name in (
        "w13_weight",
        "w13_weight_scale",
        "w13_weight_bias",
        "w2_weight",
        "w2_weight_scale",
        "w2_weight_bias",
    ):
        assert torch.count_nonzero(getattr(layer, name)) == 0
    for name, parameter in target.named_parameters():
        getattr(layer, name).data.copy_(parameter)
    method.process_weights_after_loading(layer)
    method.process_weights_after_loading(reference)

    for name, expected in reference.named_parameters():
        actual = getattr(layer, name)
        if actual.dtype == torch.float8_e4m3fn:
            torch.testing.assert_close(
                actual.view(torch.uint8),
                expected.view(torch.uint8),
            )
        else:
            torch.testing.assert_close(actual, expected)
        assert actual.data_ptr() == runtime_pointers[name]


def test_cpu_staging_keeps_runtime_weight_buffers(monkeypatch):
    monkeypatch.setattr(
        mxfp4,
        "_get_flashinfer_mxfp4_device_permute_indices",
        lambda tensor, *_args, **_kwargs: torch.arange(tensor.shape[0]),
    )
    method = _make_method()
    layer = _make_layer(seed=1)
    target = _make_layer(seed=2)
    reference = _make_layer(seed=2)
    method.process_weights_after_loading(layer)
    runtime_pointers = {
        name: getattr(layer, name).data_ptr() for name in ("w13_weight", "w2_weight")
    }

    method.restore_weights_before_cpu_staging(layer)
    for name, pointer in runtime_pointers.items():
        parameter = getattr(layer, name)
        assert parameter.data_ptr() != pointer
        assert parameter._weight_staging_runtime_buffer.data_ptr() == pointer
    for name, parameter in target.named_parameters():
        getattr(layer, name).data.copy_(parameter)
    method.process_weights_after_loading(layer)
    method.process_weights_after_loading(reference)

    for name, pointer in runtime_pointers.items():
        actual = getattr(layer, name)
        torch.testing.assert_close(actual, getattr(reference, name))
        assert actual.data_ptr() == pointer
        assert not hasattr(actual, "_weight_staging_runtime_buffer")


def test_cpu_staging_zeroes_padded_checkpoint_buffers(monkeypatch):
    monkeypatch.setattr(mxfp4, "_cpu_weight_transform_workers", lambda: 2)
    method = _make_method()
    layer = _make_layer(seed=1)
    layer.intermediate_size_per_partition = 96

    method.restore_weights_before_cpu_staging(layer)

    for name, parameter in layer.named_parameters():
        assert torch.count_nonzero(parameter) == 0, name


def test_parallel_cpu_zero_preserves_tensor_boundaries(monkeypatch):
    monkeypatch.setattr(mxfp4, "_cpu_weight_transform_workers", lambda: 2)
    storage = torch.full((32,), 7, dtype=torch.uint8)
    first = storage[4:12]
    second = storage[16:28].view(torch.bfloat16)

    _parallel_cpu_zero_((first, first, second))

    torch.testing.assert_close(storage[:4], torch.full((4,), 7, dtype=torch.uint8))
    torch.testing.assert_close(storage[4:12], torch.zeros(8, dtype=torch.uint8))
    torch.testing.assert_close(storage[12:16], torch.full((4,), 7, dtype=torch.uint8))
    torch.testing.assert_close(storage[16:28], torch.zeros(12, dtype=torch.uint8))
    torch.testing.assert_close(storage[28:], torch.full((4,), 7, dtype=torch.uint8))


def test_cpu_staging_capability_is_backend_specific():
    method = _make_method()
    assert method.weight_staging_postprocess_device(torch.nn.Module()) == "cpu"

    method._fi_kernel = "cutlass_sm90"
    assert method.weight_staging_postprocess_device(torch.nn.Module()) is None


def test_gate_up_reordering_is_composed_with_runtime_permutation():
    indices = torch.tensor([3, 0, 2, 1])
    rows = torch.arange(4)

    interleaved = _compose_trtllm_gate_up_permutation(
        indices,
        gate_up_interleaved=True,
    )
    torch.testing.assert_close(
        rows.index_select(0, interleaved),
        rows.view(2, 2).flip(1).reshape(-1).index_select(0, indices),
    )

    blocked_gate_up = torch.tensor([2, 0, 3, 1])
    blocked = _compose_trtllm_gate_up_permutation(
        indices,
        gate_up_interleaved=False,
    )
    torch.testing.assert_close(
        rows.index_select(0, blocked),
        rows.index_select(0, blocked_gate_up).index_select(0, indices),
    )
