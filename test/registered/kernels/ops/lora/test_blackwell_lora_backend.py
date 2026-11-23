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
    be._split_layers = []
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


@pytest.mark.parametrize("T", [1, 8, 64, 128])
def test_mxfp8_split_gate_up(T):
    # DeepSeek-V4.1 shared expert at TP8: gate/up slices of 288 rows (not 128-aligned)
    assert _run(576, 5120, (0, 288, 576), T, mx=True) < 5e-3


@pytest.mark.parametrize("T", [1, 8, 64, 128])
def test_mxfp8_padded_k(T):
    # ... and its down projection, K = 288
    assert _run(5120, 288, (0, 5120), T, mx=True) < 5e-3


def _mx_layer(N, K, slices, seed, **base_kw):
    import flashinfer

    g = torch.Generator(device=DEVICE).manual_seed(seed)
    w = (torch.randn(N, K, generator=g, device=DEVICE) * K ** -0.5).bfloat16()
    wq, wsf = flashinfer.mxfp8_quantize(w, True)
    a, b = _pool(slices, K, N, seed + 1)
    offsets = tuple(N // slices * i for i in range(slices + 1))
    base = SimpleNamespace(weight=wq, bias=None, weight_scale_inv_swizzled=wsf, **base_kw)
    return SimpleNamespace(base_layer=base, lora_a=a, lora_b=b, lora_offsets=offsets)


@pytest.mark.parametrize("T", [1, 8, 100, 128])
def test_mxfp8_quant_hint(T):
    """DeepSeek-V4.1's fused q_norm hands wq_b its MXFP8 quant (``layer.mxfp8_hint``): the
    result equals quantizing the BF16 row in the backend, and the hint is consumed."""
    from sglang.kernels.ops.layernorm.rmsnorm_mxfp8 import rmsnorm_mxfp8
    from sglang.srt.lora.backend import blackwell_backend as bb

    bb._mxfp8_base = lambda layer: True
    layer = _mx_layer(4096, 1280, 1, 21)
    g = torch.Generator(device=DEVICE).manual_seed(22)
    q = torch.randn(T, 1280, generator=g, device=DEVICE).bfloat16() * 3
    gamma = (1 + 0.1 * torch.randn(1280, generator=g, device=DEVICE)).bfloat16()
    y, yq, ys = rmsnorm_mxfp8(q, gamma, 1e-6)
    slots = torch.randint(0, SLOTS, (T,), generator=g, device=DEVICE, dtype=torch.int32)
    be = _backend(T, slots)
    plain = be._fused(layer, y, layer.lora_a, layer.lora_b, layer.lora_offsets)
    layer.mxfp8_hint = (y, yq, ys)
    hinted = be._fused(layer, y, layer.lora_a, layer.lora_b, layer.lora_offsets)
    assert layer.mxfp8_hint is None
    assert torch.equal(plain, hinted)
    layer.mxfp8_hint = (y.clone(), yq, ys)  # another tensor: ignored
    assert torch.equal(plain, be._fused(layer, y, layer.lora_a, layer.lora_b,
                                        layer.lora_offsets))


@pytest.mark.parametrize("T", [1, 8, 100, 128])
def test_mlp_mx_shared_expert(T):
    """DeepSeek-V4.1's TP8 shared expert (gate/up slices of 288, down K 288, swiglu limit 10)
    through forward_mlp_mx (gate/up launch, fused activation + K pad + MXFP8, down launch)
    equals the unfused LoRA path bit for bit."""
    from sglang.kernels.ops.activation.silu_and_mul_clamp import silu_and_mul_clamp
    from sglang.srt.lora.backend import blackwell_backend as bb

    bb._mxfp8_base = lambda layer: True
    H, Is, L = 5120, 288, 10.0
    gu = _mx_layer(2 * Is, H, 2, 31)
    dn = _mx_layer(H, Is, 1, 33, reduce_results=False, input_is_parallel=True)
    g = torch.Generator(device=DEVICE).manual_seed(35)
    x = torch.randn(T, H, generator=g, device=DEVICE).bfloat16() * 4
    slots = torch.randint(0, SLOTS, (T,), generator=g, device=DEVICE, dtype=torch.int32)
    be = _backend(T, slots)
    fused = be.forward_mlp_mx(gu, dn, x, L)
    act = torch.empty(T, Is, device=DEVICE, dtype=torch.bfloat16)
    silu_and_mul_clamp(be._fused(gu, x, gu.lora_a, gu.lora_b, gu.lora_offsets), act, L)
    unfused = be._fused(dn, act, dn.lora_a, dn.lora_b, dn.lora_offsets)
    torch.cuda.synchronize()
    assert fused is not None and fused.shape == (T, H)
    assert torch.isfinite(fused.float()).all()
    assert torch.equal(fused, unfused)


def _run_block_fp8(N, K, offsets, T, packed, seed=0):
    """128x128 block-FP8 base with UE8M0 scales (GLM-5.3 Flash after SGLang's DeepGEMM requant
    on Blackwell): FP32 power-of-two scales or DeepGEMM's packed int32 layout."""
    from sglang.srt.lora.backend import blackwell_backend as bb
    from sglang.srt.lora.dense.plan import DenseLoraKind
    from sglang.srt.layers.quantization.fp8_utils import (
        _get_mn_major_tma_aligned_packed_ue8m0_tensor_torch_impl as pack)
    import flashinfer

    bb._mxfp8_base = lambda layer: False
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    wf = torch.randn(N, K, generator=g, device=DEVICE) * K ** -0.5
    blk = wf.view(N // 128, 128, K // 128, 128)
    sc = torch.exp2(torch.ceil(torch.log2(blk.abs().amax(dim=(1, 3)) / 448.0)))
    wq = (blk / sc[:, None, :, None]).to(torch.float8_e4m3fn).view(N, K)
    w_ref = (wq.float().view(N // 128, 128, K // 128, 128) * sc[:, None, :, None]).view(N, K)
    x = torch.randn(T, K, generator=g, device=DEVICE).bfloat16()
    a, b = _pool(len(offsets) - 1, K, N, seed + 1)
    slots = torch.randint(0, SLOTS, (T,), generator=g, device=DEVICE, dtype=torch.int32)
    if packed:
        s_in = pack(sc.repeat_interleave(128, dim=0)[:N].contiguous())
        s_in.format_ue8m0 = True
    else:
        s_in = sc
    layer = SimpleNamespace(base_layer=SimpleNamespace(weight=wq, bias=None,
                                                       weight_scale_inv=s_in),
                            lora_a=a, lora_b=b)
    xq, _ = flashinfer.mxfp8_quantize(x, True)
    _, xl = flashinfer.mxfp8_quantize(x, False)
    xl = xl.view(torch.uint8).reshape(-1)[: T * (K // 32)].reshape(T, K // 32)
    x_base = xq.float() * torch.pow(2.0, xl.float() - 127).repeat_interleave(32, dim=1)
    be = _backend(T, slots)

    def base_fn():
        raise AssertionError("the fused path must not call the base GEMM")

    y = be.forward_with_base(layer, x, base_fn, a, b, None, tuple(offsets),
                             kind=DenseLoraKind.LINEAR)
    assert y is not None
    ref = _reference(x, w_ref, a, b, slots, offsets) - x.float() @ w_ref.T + x_base @ w_ref.T
    torch.cuda.synchronize()
    return float((y.float() - ref).norm() / ref.norm())


@pytest.mark.parametrize("T", [1, 8, 64])
@pytest.mark.parametrize("packed", [False, True])
def test_block_fp8_ue8m0(T, packed):
    # GLM-5.3 Flash TP8 shapes: o_proj (N 4096, K 2048), shared expert gate/up (2 x 256, K 4096)
    assert _run_block_fp8(4096, 2048, (0, 4096), T, packed) < 5e-3
    assert _run_block_fp8(512, 4096, (0, 256, 512), T, packed, seed=3) < 5e-3
    # shared expert down: K = 256, one packed scale word per row (size-1 dim, aligned stride)
    assert _run_block_fp8(4096, 256, (0, 4096), T, packed, seed=4) < 5e-3


def test_prefill_uses_csgmv():
    from sglang.srt.lora.dense.plan import DenseLoraKind

    be = _backend(4, torch.zeros(4, device=DEVICE, dtype=torch.int32))
    be.batch_info.is_prefill = True
    layer = SimpleNamespace(base_layer=SimpleNamespace(weight=torch.zeros(256, 256, device=DEVICE,
                                                                         dtype=torch.bfloat16),
                                                       bias=None))
    x = torch.zeros(4, 256, device=DEVICE, dtype=torch.bfloat16)
    assert be._fused(layer, x, None, None, (0, 256)) is None


# ---------------------------------------------------------------------- fused layer kinds
def _merged_layer(N, K, offsets, seed):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    w = (torch.randn(N, K, generator=g, device=DEVICE) * K ** -0.5).bfloat16()
    a, b = _pool(len(offsets) - 1, K, N, seed + 1)
    layer = SimpleNamespace(base_layer=SimpleNamespace(weight=w, bias=None), lora_a=a,
                            lora_b=b, lora_offsets=tuple(offsets))
    return layer, w, a, b


@pytest.mark.parametrize("T", [1, 8, 128, 256])
def test_gate_up_swiglu(T):
    I, K = 4864, 2560
    layer, w, a, b = _merged_layer(2 * I, K, (0, I, 2 * I), 5)
    g = torch.Generator(device=DEVICE).manual_seed(6)
    x = torch.randn(T, K, generator=g, device=DEVICE).bfloat16()
    slots = torch.randint(0, SLOTS, (T,), generator=g, device=DEVICE, dtype=torch.int32)
    act = _backend(T, slots).forward_gate_up_swiglu(layer, x)
    gu = _reference(x, w.float(), a, b, slots, (0, I, 2 * I))
    ref = torch.nn.functional.silu(gu[:, :I]) * gu[:, I:]
    rel = float((act.float() - ref).norm() / ref.norm())
    assert act.shape == (T, I) and rel < 5e-3, rel


class RMSNorm(SimpleNamespace):
    pass


class GemmaRMSNorm(SimpleNamespace):
    pass


def _rope_cache(rot, positions=4096):
    inv = 1.0 / (1e6 ** (torch.arange(0, rot, 2, device=DEVICE).float() / rot))
    ang = torch.arange(positions, device=DEVICE).float()[:, None] * inv[None]
    return torch.cat([ang.cos(), ang.sin()], -1).contiguous()


def _interleaved_axis_map(rot, section=(11, 11, 10)):
    pairs = rot // 2
    axes = [0] * pairs
    for axis in (1, 2):
        for lane in range(axis, min(3 * section[axis], pairs), 3):
            axes[lane] = axis
    return torch.tensor(axes, device=DEVICE)


@pytest.mark.parametrize(
    "T,qh,kvh,hd,rot,gated,mrope,K",
    [(8, 32, 8, 128, 128, False, False, 2560), (128, 32, 8, 128, 128, False, False, 2560),
     (1, 16, 4, 256, 64, True, False, 4096), (64, 16, 4, 256, 64, True, False, 4096),
     (256, 16, 4, 256, 64, True, False, 4096), (32, 16, 4, 256, 64, True, True, 4096),
     # Qwen3.5-9B / Qwen3.8-27B attention layers as served (multi-axis RoPE)
     (64, 16, 4, 256, 64, True, True, 4096), (8, 24, 4, 256, 64, True, False, 5120),
     (8, 24, 4, 256, 64, True, True, 5120), (1, 16, 4, 256, 64, True, True, 4096),
     (40, 24, 4, 256, 64, True, True, 5120),
     (128, 24, 4, 256, 64, True, True, 5120)])
def test_attention_inputs(T, qh, kvh, hd, rot, gated, mrope, K):
    from sglang.srt.lora.utils import AttentionInputSpec

    qw = qh * hd * (2 if gated else 1)
    N = qw + 2 * kvh * hd
    offsets = (0, qw, qw + kvh * hd, N)
    layer, w, a, b = _merged_layer(N, K, offsets, 9)
    g = torch.Generator(device=DEVICE).manual_seed(10)
    x = torch.randn(T, K, generator=g, device=DEVICE).bfloat16()
    slots = torch.randint(0, SLOTS, (T,), generator=g, device=DEVICE, dtype=torch.int32)
    pos1 = torch.randint(0, 4000, (T,), generator=g, device=DEVICE)
    positions = torch.stack([pos1, (pos1 + 7) % 4000, (pos1 * 3) % 4000]) if mrope else pos1
    cache = 4 * T + 64
    kc = torch.zeros(cache, kvh, hd, device=DEVICE, dtype=torch.bfloat16)
    vc = torch.zeros_like(kc)
    loc = torch.randperm(cache, device=DEVICE)[:T]
    norm_cls = GemmaRMSNorm if gated else RMSNorm
    qn = norm_cls(weight=(0.1 * torch.randn(hd, generator=g, device=DEVICE)).bfloat16(),
                  variance_epsilon=1e-6)
    kn = norm_cls(weight=(0.1 * torch.randn(hd, generator=g, device=DEVICE)).bfloat16(),
                  variance_epsilon=1e-6)
    cos_sin = _rope_cache(rot)
    axis_map = _interleaved_axis_map(rot) if mrope else None
    rope = SimpleNamespace(rotary_dim=rot, is_neox_style=True, cos_sin_cache=cos_sin,
                           axis_map=axis_map)
    fb = SimpleNamespace(token_to_kv_pool=SimpleNamespace(get_kv_buffer=lambda lid: (kc, vc)),
                         out_cache_loc=loc)
    spec = AttentionInputSpec(positions=positions, forward_batch=fb, layer_id=0, num_heads=qh,
                              num_kv_heads=kvh, head_dim=hd, q_norm=qn, k_norm=kn,
                              rotary_emb=rope, gated=gated)
    q, gate = _backend(T, slots).forward_attention_inputs(layer, x, spec)

    proj = _reference(x, w.float(), a, b, slots, offsets).bfloat16().float()
    if gated:
        qg = proj[:, :qw].view(T, qh, 2, hd)
        q_r, g_r = qg[:, :, 0], qg[:, :, 1]
    else:
        q_r, g_r = proj[:, :qw].view(T, qh, hd), None
    k_r = proj[:, qw: qw + kvh * hd].view(T, kvh, hd)
    v_r = proj[:, qw + kvh * hd:].view(T, kvh, hd)

    def gamma(n):
        return n.weight.float() + (1.0 if isinstance(n, GemmaRMSNorm) else 0.0)

    def norm_rope(t, n):
        t = (t * torch.rsqrt((t * t).mean(-1, keepdim=True) + 1e-6) * gamma(n)).bfloat16().float()
        h = rot // 2
        if mrope:
            p = positions[axis_map, :].T  # [T, rot/2]: each pair's own axis
        else:
            p = positions[:, None].expand(T, h)
        idx = torch.arange(h, device=DEVICE)[None].expand(T, h)
        co = cos_sin[p, idx][:, None]
        si = cos_sin[p, h + idx][:, None]
        x1, x2 = t[..., :h].clone(), t[..., h:rot].clone()
        t[..., :h] = x1 * co - x2 * si
        t[..., h:rot] = x2 * co + x1 * si
        return t

    def rel(u, v):
        return float((u.float() - v).norm() / v.norm())

    assert rel(q, norm_rope(q_r, qn).reshape(T, -1)) < 6e-3
    if gated:
        assert rel(gate, g_r.reshape(T, -1)) < 6e-3
    assert rel(kc[loc], norm_rope(k_r, kn)) < 6e-3
    assert rel(vc[loc], v_r) < 6e-3


def test_attention_inputs_ignore_padded_rows():
    """Rows past the real batch (graph warm-up, padding) may carry any position or cache
    slot: positions are clamped into the cos/sin table, out-of-range slots are not written."""
    from sglang.srt.lora.utils import AttentionInputSpec

    T, qh, kvh, hd, rot, K = 16, 24, 4, 256, 64, 5120
    qw = qh * hd * 2
    N = qw + 2 * kvh * hd
    layer, w, a, b = _merged_layer(N, K, (0, qw, qw + kvh * hd, N), 11)
    x = torch.randn(T, K, device=DEVICE).bfloat16()
    slots = torch.zeros(T, device=DEVICE, dtype=torch.int32)
    positions = torch.full((3, T), 1 << 40, device=DEVICE, dtype=torch.int64)
    positions[:, ::2] = -(1 << 40)
    kc = torch.zeros(64, kvh, hd, device=DEVICE, dtype=torch.bfloat16)
    vc = torch.zeros_like(kc)
    loc = torch.full((T,), 1 << 33, device=DEVICE, dtype=torch.int64)
    loc[0] = 3
    norm = GemmaRMSNorm(weight=torch.zeros(hd, device=DEVICE, dtype=torch.bfloat16),
                        variance_epsilon=1e-6)
    rope = SimpleNamespace(rotary_dim=rot, is_neox_style=True, cos_sin_cache=_rope_cache(rot),
                           axis_map=_interleaved_axis_map(rot))
    fb = SimpleNamespace(token_to_kv_pool=SimpleNamespace(get_kv_buffer=lambda lid: (kc, vc)),
                         out_cache_loc=loc)
    spec = AttentionInputSpec(positions=positions, forward_batch=fb, layer_id=0, num_heads=qh,
                              num_kv_heads=kvh, head_dim=hd, q_norm=norm, k_norm=norm,
                              rotary_emb=rope, gated=True)
    q, _ = _backend(T, slots).forward_attention_inputs(layer, x, spec)
    torch.cuda.synchronize()
    assert torch.isfinite(q.float()).all()
    written = (kc.float().abs().sum((1, 2)) > 0).nonzero().flatten().tolist()
    assert written == [3]
