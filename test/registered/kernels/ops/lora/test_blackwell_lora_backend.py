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
    "T,qh,kvh,hd,rot,gated,mrope",
    [(8, 32, 8, 128, 128, False, False), (128, 32, 8, 128, 128, False, False),
     (1, 16, 4, 256, 64, True, False), (64, 16, 4, 256, 64, True, False),
     (256, 16, 4, 256, 64, True, False), (32, 16, 4, 256, 64, True, True)])
def test_attention_inputs(T, qh, kvh, hd, rot, gated, mrope):
    from sglang.srt.lora.utils import AttentionInputSpec

    K = 4096 if hd == 256 else 2560
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
