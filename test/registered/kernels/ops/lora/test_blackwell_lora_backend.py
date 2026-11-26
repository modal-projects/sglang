"""Blackwell fused multi-LoRA backend (``--lora-backend blackwell``) against a torch reference.

Runs ``BlackwellLoRABackend.forward_with_base`` on decode-shaped batches for the projection
shapes the backend specializes: a plain projection, a gate/up pair, a three-slice QKV with
256-wide heads, and an MXFP8 base. Pool slots carry garbage past each adapter's rank (as the
memory pool leaves them) and one slot is rank 0 (base only).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=600, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="the blackwell LoRA backend needs an SM100/SM103 GPU",
)

DEVICE = torch.device("cuda")
SLOTS = 8
RANK = 32
RANKS = [32, 16, 0, 8, 32, 32, 24, 32]  # slot 2 carries no adapter
SCALINGS = [1.5, 0.5, 1.0, 2.0, 1.0, 0.75, 1.25, 1.0]


def _backend(T, slots):
    from sglang.srt.lora.backend.blackwell_backend import BlackwellLoRABackend

    be = BlackwellLoRABackend.__new__(BlackwellLoRABackend)  # no server context needed
    be._kernels, be._mirrored, be._weights_version, be._rejected = {}, [], None, set()
    be.batch_info = SimpleNamespace(
        bs=T, is_prefill=False, req_weight_indices=slots,
        lora_ranks=torch.tensor(RANKS, device=DEVICE, dtype=torch.int32),
        scalings=torch.tensor(SCALINGS, device=DEVICE, dtype=torch.float32))
    return be


def _pool(slices, K, N, seed):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    a = torch.randn(SLOTS, slices * RANK, K, generator=g, device=DEVICE) * 100
    b = torch.randn(SLOTS, N, RANK, generator=g, device=DEVICE) * 100
    for s, rk in enumerate(RANKS):
        for i in range(slices):
            a[s, i * RANK: i * RANK + rk] = torch.randn(rk, K, generator=g, device=DEVICE) * K ** -0.5
        b[s, :, :rk] = torch.randn(N, rk, generator=g, device=DEVICE) * 0.05
    return a.bfloat16().contiguous(), b.bfloat16().contiguous()


def _reference(x, w_f32, a, b, slots, offsets):
    y = x.float() @ w_f32.T
    starts, ends = offsets[:-1], offsets[1:]
    for t in range(x.shape[0]):
        s = int(slots[t])
        rk = RANKS[s]
        if rk == 0:
            continue
        for i, (lo, hi) in enumerate(zip(starts, ends)):
            z = a[s, i * RANK: i * RANK + rk].float() @ x[t].float()
            y[t, lo:hi] += SCALINGS[s] * (b[s, lo:hi, :rk].float() @ z)
    return y


def _run(N, K, offsets, T, mx=False, seed=0):
    from sglang.srt.lora.backend import blackwell_backend as bb
    from sglang.srt.lora.dense.plan import DenseLoraKind

    g = torch.Generator(device=DEVICE).manual_seed(seed)
    w = (torch.randn(N, K, generator=g, device=DEVICE) * K ** -0.5).bfloat16()
    x = torch.randn(T, K, generator=g, device=DEVICE).bfloat16()
    a, b = _pool(len(offsets) - 1, K, N, seed + 1)
    slots = torch.randint(0, SLOTS, (T,), generator=g, device=DEVICE, dtype=torch.int32)
    base = SimpleNamespace(weight=w, bias=None)
    w_ref = w.float()
    if mx:
        import flashinfer

        wq, wsf = flashinfer.mxfp8_quantize(w, True)
        _, lin = flashinfer.mxfp8_quantize(w, False)
        lin = lin.view(torch.uint8).reshape(-1)[: N * (K // 32)].reshape(N, K // 32)
        w_ref = wq.float() * torch.pow(2.0, lin.float() - 127).repeat_interleave(32, dim=1)
        base = SimpleNamespace(weight=wq, bias=None, weight_scale_inv_swizzled=wsf)
        bb._mxfp8_base = lambda layer: True
        # the activation is quantized too: compare against the dequantized activation
        xq, _ = flashinfer.mxfp8_quantize(x, True)
        _, xl = flashinfer.mxfp8_quantize(x, False)
        xl = xl.view(torch.uint8).reshape(-1)[: T * (K // 32)].reshape(T, K // 32)
        x_base = xq.float() * torch.pow(2.0, xl.float() - 127).repeat_interleave(32, dim=1)
    layer = SimpleNamespace(base_layer=base, lora_a=a, lora_b=b)
    be = _backend(T, slots)

    def base_fn():
        raise AssertionError("the fused path must not call the base GEMM")

    y = be.forward_with_base(layer, x, base_fn, a, b, None, tuple(offsets),
                             kind=DenseLoraKind.LINEAR)
    ref = _reference(x, w_ref, a, b, slots, offsets)
    if mx:  # base from the quantized activation, LoRA from the BF16 one
        ref = ref - x.float() @ w_ref.T + x_base @ w_ref.T
    torch.cuda.synchronize()
    assert torch.isfinite(y.float()).all()
    rel = float((y.float() - ref).norm() / ref.norm())
    again = be.forward_with_base(layer, x, base_fn, a, b, None, tuple(offsets),
                                 kind=DenseLoraKind.LINEAR)
    assert torch.equal(y, again), "fused projection is not bitwise repeatable"
    return rel


@pytest.mark.parametrize("T", [1, 8, 128, 256])
def test_projection(T):
    assert _run(2560, 4096, (0, 2560), T) < 5e-3


@pytest.mark.parametrize("T", [8, 128, 256])
def test_gate_up_pair(T):
    assert _run(2 * 4864, 2560, (0, 4864, 2 * 4864), T) < 5e-3


@pytest.mark.parametrize("T", [8, 256])
def test_qkv_256_heads(T):
    assert _run(10240, 4096, (0, 8192, 9216, 10240), T) < 5e-3


@pytest.mark.parametrize("T", [8, 128])
def test_mxfp8_base(T):
    assert _run(4096, 1280, (0, 4096), T, mx=True) < 5e-3


def test_prefill_uses_csgmv():
    from sglang.srt.lora.dense.plan import DenseLoraKind

    be = _backend(4, torch.zeros(4, device=DEVICE, dtype=torch.int32))
    be.batch_info.is_prefill = True
    layer = SimpleNamespace(base_layer=SimpleNamespace(weight=torch.zeros(256, 256, device=DEVICE,
                                                                         dtype=torch.bfloat16),
                                                       bias=None))
    x = torch.zeros(4, 256, device=DEVICE, dtype=torch.bfloat16)
    assert be._fused(layer, x, None, None, (0, 256)) is None
