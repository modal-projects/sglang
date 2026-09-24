"""Tests for the CuTe DSL TGV BF16 GEMM kernel."""

import sys

import pytest
import torch

from sglang.kernels.jit.utils import (
    get_ci_test_range,
    get_jit_cuda_arch,
    is_hip_runtime,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from sglang.kernels.ops.gemm.cutedsl_bf16_gemm import (  # noqa: E402
    _K3_TGV_WIN_SHAPES,
    _run_tgv,
    cutedsl_bf16_gemm,
    use_cutedsl_bf16_gemm,
)

SHAPES = [(n, k) for n in [1024, 2624, 6144] for k in [2048, 6144]] + [(2048, 4096)]
NUM_TOKENS = get_ci_test_range(list(range(1, 33)), [1, 15, 16, 32])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("n,k", SHAPES)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
def test_cutedsl_bf16_gemm(num_tokens, k, n, has_bias):
    if is_hip_runtime() or get_jit_cuda_arch().major != 10:
        pytest.skip("SM10x required")

    torch.manual_seed(num_tokens)
    x = torch.randn(num_tokens, k, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
    bias = torch.randn(n, dtype=torch.bfloat16, device="cuda") if has_bias else None
    if bias is not None:
        bias.requires_grad_(True)

    with torch.no_grad():
        out = cutedsl_bf16_gemm(x, weight, bias)
    assert out.shape == (num_tokens, n)
    assert out.dtype == torch.bfloat16

    ref = x.float() @ weight.float().T
    if bias is not None:
        ref = ref + bias.detach().float()
    torch.testing.assert_close(out, ref.bfloat16(), rtol=2e-2, atol=2.5)


@pytest.mark.parametrize("n, k", sorted(_K3_TGV_WIN_SHAPES) + [(1024, 2048)])
def test_empty_batch_not_tgv_eligible(n, k):
    """DP-attention idle groups run a 0-token dummy forward to keep the
    mlp-sync lockstep; every m == 0 shape must route to cuBLAS."""
    assert not use_cutedsl_bf16_gemm(0, n, k)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("has_bias", [False, True])
def test_cutedsl_bf16_gemm_empty_batch(has_bias):
    """Empty input must yield the empty [0, N] output, mirroring F.linear —
    launching TGV with a 0-CTA grid fails with CUDA_ERROR_INVALID_VALUE."""
    if is_hip_runtime() or get_jit_cuda_arch().major != 10:
        pytest.skip("SM10x required")

    n, k = 6144, 2048
    x = torch.empty(0, k, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
    bias = torch.randn(n, dtype=torch.bfloat16, device="cuda") if has_bias else None

    out = cutedsl_bf16_gemm(x, weight, bias)
    torch.cuda.synchronize()
    assert out.shape == (0, n)
    assert out.dtype == torch.bfloat16


@pytest.mark.parametrize("tactic", [1, 18, 27])
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("pdl", [False, True])
def test_tgv_tactics_graph_replay(tactic, out_dtype, has_bias, pdl):
    """Graph replay must refresh every output and preserve FP32 accumulator precision."""
    if is_hip_runtime() or get_jit_cuda_arch().major != 10:
        pytest.skip("SM10x required")

    torch.manual_seed(17)
    x = torch.randn(17, 2048, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(1024, 2048, dtype=torch.bfloat16, device="cuda")
    bias = torch.randn(1024, dtype=torch.bfloat16, device="cuda") if has_bias else None
    out = torch.empty(17, 1024, dtype=out_dtype, device="cuda")

    def run():
        return _run_tgv(x, weight.t(), bias, out, pdl=pdl, tactic=tactic)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        run()

    rtol, atol = (5e-5, 2e-3) if out_dtype == torch.float32 else (2e-2, 2e-2)
    for _ in range(3):
        x.normal_()
        out.fill_(float("nan"))
        graph.replay()
        ref = x.double() @ weight.double().T
        if bias is not None:
            ref = ref + bias.double()
        torch.testing.assert_close(out, ref.to(out_dtype), rtol=rtol, atol=atol)
        first_output = out.clone()
        out.fill_(float("nan"))
        graph.replay()
        torch.testing.assert_close(out, first_output, rtol=0, atol=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
