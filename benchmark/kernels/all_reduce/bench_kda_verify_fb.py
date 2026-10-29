"""Fused KDA chain-verify with in-kernel f_b (HAS_FB) vs tiny_k_gemm + verify.
K3 TP8 shape: H=HV=12, K=V=128, f_a dim 128 (slice of the [.., 144] bfa output)."""

import torch

from sglang.kernels.ops.attention.linear.fused_kda_conv_recurrent_verify import (
    fused_kda_conv_gating_verify,
)
from sglang.kernels.ops.gemm import kimi_k3_tiny_gemm

H = HV = 12
K = V = 128
DIM = 2 * H * K + HV * V
dev = "cuda"


def make(B, T, lb):
    torch.manual_seed(B * 100 + T)
    S = B * T
    d = dict(
        mixed_qkv=torch.randn(S, DIM, device=dev, dtype=torch.bfloat16),
        conv_weight=torch.randn(DIM, 4, device=dev) * 0.3,
        conv_bias=torch.randn(DIM, device=dev) * 0.1,
        conv_state=torch.randn(
            B + 2, 3, DIM, device=dev, dtype=torch.bfloat16
        ).transpose(1, 2),
        conv_state_indices=torch.arange(B, device=dev, dtype=torch.int32),
        intermediate_conv_window=None,
        intermediate_state_indices=torch.arange(B, device=dev, dtype=torch.int32),
        b=torch.randn(S, HV, device=dev, dtype=torch.bfloat16),
        A_log=torch.randn(HV, device=dev) * 0.5,
        dt_bias=torch.randn(HV * K, device=dev) * 0.5,
        ssm_states=torch.randn(B + 2, HV, V, K, device=dev) * 0.1,
        cache_indices=torch.arange(B, device=dev, dtype=torch.int32),
        intermediate_states_buffer=torch.zeros(B, T, HV, V, K, device=dev),
        scale=K**-0.5,
        T=T,
        num_q_heads=H,
        num_v_heads=HV,
        head_k_dim=K,
        head_v_dim=V,
        lower_bound=lb,
    )
    bfa = torch.randn(S, 144, device=dev, dtype=torch.bfloat16)
    fbw = (torch.randn(HV * K, 128, device=dev) * 0.08).to(torch.bfloat16)
    return d, bfa[:, :128], fbw


def ref(d, f_a, fbw):
    gate = kimi_k3_tiny_gemm(f_a, fbw)
    return fused_kda_conv_gating_verify(a=gate, **d)


def fused(d, f_a, fbw):
    return fused_kda_conv_gating_verify(a=f_a, f_b_weight=fbw, **d)


def clone(d):
    return {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in d.items()}


def graph_time(fn, iters=50, reps=5):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(iters):
            fn()
    best = 1e9
    for _ in range(reps):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        g.replay()
        b.record()
        torch.cuda.synchronize()
        best = min(best, a.elapsed_time(b) * 1e3 / iters)
    return best


for lb in (None, -5.0):
    for B in (1, 2, 4, 8, 16):
        T = 8
        d, f_a, fbw = make(B, T, lb)
        dr, df = clone(d), clone(d)
        o_r = ref(dr, f_a, fbw).float()
        o_f = fused(df, f_a, fbw).float()
        diff = (o_r - o_f).abs()
        rel = (diff / o_r.abs().clamp_min(1e-3)).max().item()
        st = (
            (dr["intermediate_states_buffer"] - df["intermediate_states_buffer"])
            .abs()
            .max()
            .item()
        )
        gate_r = kimi_k3_tiny_gemm(f_a, fbw).float()
        gate_t = (f_a.float() @ fbw.float().t()).to(torch.bfloat16).float()
        gmis = (gate_r != gate_t).float().mean().item()
        tr = graph_time(lambda: ref(d, f_a, fbw))
        tf = graph_time(lambda: fused(d, f_a, fbw))
        tg = graph_time(lambda: kimi_k3_tiny_gemm(f_a, fbw))
        print(
            f"lb={lb} B={B} T={T} ref_us={tr:.2f} fused_us={tf:.2f} save_us={tr - tf:.2f} "
            f"tinygemm_us={tg:.2f} o_maxabs={diff.max().item():.3e} o_maxrel={rel:.3e} "
            f"o_mismatch={(diff > 0).float().mean().item():.2e} states_maxabs={st:.3e} "
            f"gate_ref_vs_fp32dot_mismatch={gmis:.2e}",
            flush=True,
        )
