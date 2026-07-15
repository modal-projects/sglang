import pytest
import torch

from sglang.jit_kernel.inkling_attn_prologue import inkling_attn_prologue_verify
from sglang.srt.kernels.mxfp8_quant import to_mxfp8
from sglang.srt.models.inkling_common.attn import compute_log_scaling_tau
from sglang.test.ci.ci_register import register_cuda_ci


register_cuda_ci(est_time=60, suite="nightly-4-gpu-b200", nightly=True)


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] < 10,
    reason="Inkling MXFP8 fused prologue requires an SM100 GPU",
)
def test_mxfp8_fused_prologue_scales_q_before_quantization():
    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_tokens = 3
    head_dim = 128
    conv_width = 4
    page_size = 128

    # Constant Q makes the RMSNorm reference deterministic. A 1.5 gamma puts
    # the final row close enough to an E8M0 boundary that the 1M-context tau
    # must change the selected MXFP8 exponent.
    q = torch.ones((num_tokens, head_dim), dtype=dtype, device=device)
    k = torch.zeros_like(q)
    v = torch.zeros_like(q)
    qkvr = torch.cat((q, k, v), dim=-1)
    q_gamma = torch.full((head_dim,), 1.5, dtype=dtype, device=device)
    k_gamma = torch.ones((head_dim,), dtype=dtype, device=device)

    cache_shape = (num_tokens, conv_width - 1, head_dim)
    k_cache = torch.zeros(cache_shape, dtype=dtype, device=device)
    v_cache = torch.zeros_like(k_cache)
    cache_indices = torch.arange(num_tokens, dtype=torch.int32, device=device)
    cache_mask = torch.zeros(num_tokens, dtype=torch.bool, device=device)
    weight = torch.zeros((head_dim, conv_width), dtype=dtype, device=device)
    inter_shape = (num_tokens, 1, conv_width - 1, head_dim)
    k_inter = torch.zeros(inter_shape, dtype=dtype, device=device)
    v_inter = torch.zeros_like(k_inter)

    loc = torch.arange(num_tokens, dtype=torch.int64, device=device)
    cache = torch.empty(
        (page_size, 1, head_dim), dtype=torch.float8_e4m3fn, device=device
    )
    scale_shape = (1, 1, 32, page_size // 32, head_dim // 32)
    sfk = torch.empty(scale_shape, dtype=torch.float8_e8m0fnu, device=device)
    sfv = torch.empty_like(sfk)

    positions = torch.tensor(
        [127_999, 255_999, 1_048_575], dtype=torch.int64, device=device
    )
    tau = compute_log_scaling_tau(positions, n_floor=128_000, alpha=0.1)

    q_fused, _, _, sfq_fused = inkling_attn_prologue_verify(
        qkvr,
        k_cache,
        v_cache,
        cache_indices,
        cache_mask,
        weight,
        weight,
        k_inter,
        v_inter,
        q_gamma,
        k_gamma,
        1e-6,
        loc,
        cache,
        cache.clone(),
        0,
        head_dim,
        2 * head_dim,
        head_dim,
        head_dim,
        1,
        use_residual=False,
        do_store=True,
        mxfp8_quant=True,
        sfk=sfk,
        sfv=sfv,
        page_size=page_size,
        q_log_scaling_tau=tau,
    )

    q_norm = (
        q.float()
        * torch.rsqrt(q.float().square().mean(dim=-1, keepdim=True) + 1e-6)
        * q_gamma.float()
    ).to(dtype)
    q_scaled = (q_norm.float() * tau[:, None]).to(dtype)
    reference = to_mxfp8(q_scaled)

    assert sfq_fused is not None
    assert torch.equal(q_fused.view(torch.uint8), reference.data.view(torch.uint8))
    assert torch.equal(
        sfq_fused.view(torch.uint8),
        reference.scale.view(torch.uint8).view_as(sfq_fused.view(torch.uint8)),
    )
    assert not torch.equal(
        sfq_fused[0].view(torch.uint8), sfq_fused[-1].view(torch.uint8)
    )
