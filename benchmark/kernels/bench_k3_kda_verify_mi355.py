"""Kimi-K3 TP8 KDA chain-verify kernels on MI355X: timing and parity.

Shapes: 12 heads of 128, conv width 4, T=8 draft tokens, bf16 SSM, ReplaySSM ring.
Compares the shipped paths (fused conv+recurrence for bs<=16, causal_conv1d_update
+ recurrent kernel above) against candidates. Each call reads a distinct state set
so the timing is not served from L2.
"""

import argparse
import json

import torch

import sglang.kernels.ops.attention.linear.fused_kda_conv_recurrent_verify as fmod
import sglang.kernels.ops.attention.linear.fused_sigmoid_gating_recurrent as rmod
from sglang.kernels.ops.attention.fla.kda_verify_closed_form import (
    kda_verify_closed_form,
)
from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_update

H = HV = 12
K = V = 128
W = 4
DIM = 3 * H * K
LB = -5.0
SCALE = K**-0.5
NSETS = 8
_SHIPPED_SELECT = None


def make_inputs(B, T, L, seed=0, ssm_dtype=torch.bfloat16, conv_dtype=torch.bfloat16):
    g = torch.Generator(device="cuda").manual_seed(seed)
    dev = "cuda"
    S = NSETS * B + 3

    def rn(*shape, dtype=torch.bfloat16, s=1.0):
        return (torch.randn(*shape, generator=g, device=dev) * s).to(dtype)

    x = dict(
        mixed_qkv=rn(NSETS, B * T, DIM),
        a=rn(NSETS, B * T, HV * K),
        b=rn(NSETS, B * T, HV),
        conv_w=rn(DIM, W, dtype=conv_dtype, s=0.5),
        A_log=rn(HV, dtype=torch.float32, s=0.5),
        dt_bias=rn(HV * K, dtype=torch.float32, s=0.5),
        conv_states=rn(S, W - 1, DIM),
        ssm=rn(S, HV, V, K, dtype=ssm_dtype, s=0.1),
        icw=torch.zeros(S, T, W - 1, DIM, dtype=torch.bfloat16, device=dev),
        rawv=torch.zeros(S, HV, L, V, dtype=torch.bfloat16, device=dev),
        rawk=torch.zeros(S, H, L, K, dtype=torch.bfloat16, device=dev),
        rg=torch.zeros(S, HV, L, K, dtype=torch.float32, device=dev),
        rbeta=torch.zeros(S, HV, L, dtype=torch.float32, device=dev),
        qsl=torch.arange(0, B * T + 1, T, dtype=torch.int32, device=dev),
    )
    perm = torch.randperm(NSETS * B, generator=g, device=dev).to(torch.int32)
    x["cache_idx"] = [perm[i * B : (i + 1) * B].contiguous() for i in range(NSETS)]
    x["icw_idx"] = x["cache_idx"]
    return x


HF_SNAP = (
    "/hf/hub/models--moonshotai--Kimi-K3-MTP-Release/snapshots/"
    "8455440e191f81ff4c6338208a2012701356e46b"
)


def load_real_layer(x, layer, tp_rank=0):
    """TP-rank shard of a checkpoint KDA layer; evolve the states over 512 tokens."""
    import json

    from safetensors import safe_open

    pre = f"language_model.model.layers.{layer}.self_attn."
    wm = json.load(open(f"{HF_SNAP}/model.safetensors.index.json"))["weight_map"]

    def get(name):
        with safe_open(f"{HF_SNAP}/{wm[pre + name]}", "pt", device="cpu") as f:
            return f.get_tensor(pre + name)

    lo, hi = tp_rank * H * K, (tp_rank + 1) * H * K
    conv = [get(f"{c}_conv1d.weight").reshape(-1, W)[lo:hi] for c in "qkv"]
    x["conv_w"] = torch.cat(conv).to("cuda", torch.bfloat16).contiguous()
    a_log = get("A_log").reshape(-1)[tp_rank * HV : (tp_rank + 1) * HV]
    x["A_log"] = a_log.to("cuda", torch.float32).contiguous()
    x["dt_bias"] = get("dt_bias").reshape(-1)[lo:hi].to("cuda", torch.float32)
    S = x["ssm"].shape[0]
    x["ssm"].zero_()
    T = 8
    idx = torch.arange(S, dtype=torch.int32, device="cuda")
    qsl = torch.arange(0, S * T + 1, T, dtype=torch.int32, device="cuda")
    for _ in range(64):
        q, k, v = (
            torch.nn.functional.silu(torch.randn(1, S * T, H, K, device="cuda")).to(
                torch.bfloat16
            )
            for _ in range(3)
        )
        rmod.fused_sigmoid_gating_delta_rule_update(
            A_log=x["A_log"],
            dt_bias=x["dt_bias"],
            q=q,
            k=k,
            v=v,
            a=torch.randn(S * T, HV * K, device="cuda").to(torch.bfloat16),
            b=torch.randn(S * T, HV, device="cuda").to(torch.bfloat16),
            initial_state_source=x["ssm"],
            initial_state_indices=idx,
            cu_seqlens=qsl,
            use_qk_l2norm_in_kernel=True,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            is_kda=True,
            lower_bound=LB,
        )
    torch.cuda.synchronize()
    x["ssm_absmax"] = x["ssm"].abs().max().item()


def ring_kw(x):
    return dict(
        cache_ring=True,
        replayssm_rawv=x["rawv"],
        replayssm_rawk=x["rawk"],
        replayssm_g=x["rg"],
        replayssm_beta=x["rbeta"],
    )


def run_fused(x, i, B, T):
    return fmod.fused_kda_conv_gating_verify(
        mixed_qkv=x["mixed_qkv"][i],
        conv_weight=x["conv_w"],
        conv_bias=None,
        conv_state=x["conv_states"].transpose(-1, -2),
        conv_state_indices=x["cache_idx"][i],
        intermediate_conv_window=x["icw"].transpose(-1, -2),
        intermediate_state_indices=x["icw_idx"][i],
        a=x["a"][i],
        b=x["b"][i],
        A_log=x["A_log"],
        dt_bias=x["dt_bias"],
        ssm_states=x["ssm"],
        cache_indices=x["cache_idx"][i],
        intermediate_states_buffer=None,
        scale=SCALE,
        T=T,
        num_q_heads=H,
        num_v_heads=HV,
        head_k_dim=K,
        head_v_dim=V,
        lower_bound=LB,
        **ring_kw(x),
    ).view(B * T, HV, V)


def run_conv(x, i, B, T):
    xin = x["mixed_qkv"][i].view(B, T, DIM).transpose(1, 2)
    out = causal_conv1d_update(
        xin,
        x["conv_states"].transpose(-1, -2),
        x["conv_w"],
        None,
        activation="silu",
        conv_state_indices=x["cache_idx"][i],
        intermediate_conv_window=x["icw"].transpose(-1, -2),
        intermediate_state_indices=x["icw_idx"][i],
    )
    flat = out.transpose(1, 2).reshape(B * T, DIM)
    q, k, v = flat.split([H * K, H * K, HV * V], dim=-1)
    return (
        q.unflatten(-1, (H, K)).unsqueeze(0),
        k.unflatten(-1, (H, K)).unsqueeze(0),
        v.unflatten(-1, (HV, V)).unsqueeze(0),
    )


def run_unfused(x, i, B, T):
    q, k, v = run_conv(x, i, B, T)
    o = rmod.fused_sigmoid_gating_delta_rule_update(
        A_log=x["A_log"],
        dt_bias=x["dt_bias"],
        q=q,
        k=k,
        v=v,
        a=x["a"][i],
        b=x["b"][i],
        initial_state_source=x["ssm"],
        initial_state_indices=x["cache_idx"][i],
        cu_seqlens=x["qsl"],
        use_qk_l2norm_in_kernel=True,
        softplus_beta=1.0,
        softplus_threshold=20.0,
        is_kda=True,
        lower_bound=LB,
        disable_state_update=True,
        intermediate_states_buffer=None,
        intermediate_state_indices=x["icw_idx"][i],
        cache_steps=T,
        retrieve_parent_token=None,
        **ring_kw(x),
    )
    return o.view(B * T, HV, V)


def run_closed(x, i, B, T, **kw):
    q, k, v = run_conv(x, i, B, T)
    o = kda_verify_closed_form(
        x["A_log"],
        x["a"][i],
        x["dt_bias"],
        q,
        k,
        v,
        x["b"][i],
        x["ssm"],
        x["cache_idx"][i],
        x["qsl"],
        LB,
        x["rawv"],
        x["rawk"],
        x["rg"],
        x["rbeta"],
        x["cache_idx"][i],
        scale=SCALE,
        **kw,
    )
    return o.view(B * T, HV, V)


def _frozen_unfused(sel):
    def fn(*a):
        cur = rmod._select_recurrent_launch_config
        rmod._select_recurrent_launch_config = sel
        try:
            return run_unfused(*a)
        finally:
            rmod._select_recurrent_launch_config = cur

    return fn


def ref_of(B):
    return run_fused if B <= 16 else _frozen_unfused(_SHIPPED_SELECT)


def time_us(fn, reps=20, iters=20):
    for i in range(NSETS):
        fn(i)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for r in range(reps):
            fn(r % NSETS)
    graph.replay()
    torch.cuda.synchronize()
    st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(iters):
        graph.replay()
    en.record()
    torch.cuda.synchronize()
    return st.elapsed_time(en) * 1000 / (reps * iters)


def snapshot(x):
    return {k: x[k].clone() for k in ("rawv", "rawk", "rg", "rbeta", "icw", "ssm")}


def reset(x):
    for k in ("rawv", "rawk", "rg", "rbeta", "icw"):
        x[k].zero_()


def parity(x, B, T, ref_fn, fn):
    reset(x)
    o_ref = ref_fn(x, 0, B, T).float().clone()
    s_ref = snapshot(x)
    reset(x)
    o = fn(x, 0, B, T).float().clone()
    s = snapshot(x)
    d = (o - o_ref).abs()
    res = {
        "o_max_abs": d.max().item(),
        "o_max_rel": (d / o_ref.abs().clamp_min(1e-3)).max().item(),
        "o_bitexact": bool(torch.equal(o, o_ref)),
        "o_ref_absmax": o_ref.abs().max().item(),
    }
    for k in s:
        res[f"{k}_eq"] = bool(torch.equal(s[k], s_ref[k]))
    return res


def _shipped_select(n, h, hv, k, v, is_kda, target_verify=False, *, cache_steps=0):
    return min(v, 32), 1


def main():
    global _SHIPPED_SELECT
    _SHIPPED_SELECT = _shipped_select
    ap = argparse.ArgumentParser()
    ap.add_argument("--bs", default="1,2,4,8,16,32")
    ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--L", type=int, default=32)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--ssm", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--real-layers", default="", help="checkpoint KDA layer ids")
    args = ap.parse_args()
    T = args.T
    out = []
    for B in [int(b) for b in args.bs.split(",")]:
        x = make_inputs(
            B,
            T,
            args.L,
            ssm_dtype=torch.float32 if args.ssm == "fp32" else torch.bfloat16,
        )
        row = {"B": B, "ssm": args.ssm}
        if args.real_layers:
            for layer in [int(v) for v in args.real_layers.split(",")]:
                load_real_layer(x, layer)
                row[f"L{layer}"] = {
                    "ssm_absmax": x["ssm_absmax"],
                    "closed": parity(x, B, T, ref_of(B), run_closed),
                    "unfused_tuned": parity(x, B, T, ref_of(B), run_unfused),
                }
            print(json.dumps(row), flush=True)
            continue
        arms = {"unfused": run_unfused, "closed": run_closed, "conv_only": run_conv}
        if B <= 16:
            arms["fused"] = run_fused
        for name, fn in arms.items():
            row[f"{name}_us"] = round(time_us(lambda i, x=x, fn=fn: fn(x, i, B, T)), 2)
        row["closed_vs_ref"] = parity(x, B, T, ref_of(B), run_closed)
        row["unfused_tuned_vs_ref"] = parity(x, B, T, ref_of(B), run_unfused)
        if B <= 16:
            row["unfused_vs_fused"] = parity(x, B, T, run_fused, run_unfused)
        if args.sweep:
            sw = {}
            base_bv, base_hip = fmod.KDA_VERIFY_BLOCK_V, fmod.KDA_VERIFY_BLOCK_V_HIP
            if B <= 16:
                for bv in (4, 8, 16, 32):
                    fmod.KDA_VERIFY_BLOCK_V = fmod.KDA_VERIFY_BLOCK_V_HIP = bv
                    sw[f"fused_bv{bv}"] = round(
                        time_us(lambda i, x=x: run_fused(x, i, B, T)), 2
                    )
                fmod.KDA_VERIFY_BLOCK_V, fmod.KDA_VERIFY_BLOCK_V_HIP = (
                    base_bv,
                    base_hip,
                )
            sel = rmod._select_recurrent_launch_config
            ref_bitexact = ref_of(B)
            for bv, nw in ((8, 1), (16, 1), (32, 1), (32, 2), (64, 2), (16, 2)):
                rmod._select_recurrent_launch_config = lambda *a, _r=(bv, nw), **k: _r
                try:
                    sw[f"unfused_bv{bv}_w{nw}"] = round(
                        time_us(lambda i, x=x: run_unfused(x, i, B, T)), 2
                    )
                    sw[f"unfused_bv{bv}_w{nw}_eq"] = parity(
                        x, B, T, ref_bitexact, run_unfused
                    )["o_bitexact"]
                except Exception as e:  # noqa: BLE001
                    sw[f"unfused_bv{bv}_w{nw}"] = repr(e)[:200]
            rmod._select_recurrent_launch_config = sel
            for bv in (32, 64, 128):
                for nw in (2, 4, 8):
                    for kc in (32, 64):
                        kw = dict(block_v=bv, num_warps=nw, block_k=kc, num_stages=1)
                        try:
                            sw[f"closed_bv{bv}_w{nw}_kc{kc}"] = round(
                                time_us(
                                    lambda i, x=x, kw=kw: run_closed(x, i, B, T, **kw)
                                ),
                                2,
                            )
                        except Exception as e:  # noqa: BLE001
                            sw[f"closed_bv{bv}_w{nw}_kc{kc}"] = repr(e)[:120]
            row["sweep"] = sw
        print(json.dumps(row), flush=True)
        out.append(row)
        del x
        torch.cuda.empty_cache()
    return out


if __name__ == "__main__":
    main()
