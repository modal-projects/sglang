"""K3 MI355 microbench: k3_ar_agg (AR / AG + attn-res aggregation, now up to
128 tokens) vs the unfused path (AITER custom all-reduce or one-barrier
all-gather + attn_res_smallm) under HIP graphs, with exactness checks.

  torchrun --nproc_per_node=8 bench_k3_ar_agg_t128.py --tokens 8,32,64,128
"""

import argparse
import json
import os

import torch
import torch.distributed as dist

from sglang.srt.distributed.parallel_state import (
    get_tp_group,
    graph_capture,
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.test.test_utils import publish_build_topology

H = 7168


def unfused_ar(ca, tp, partial, prefix, bank, cw, ow, nvb, fp8):
    from sglang.kernels.ops.attention.attn_res_hip import attn_res_hip
    from sglang.kernels.ops.attention.attn_res_smallm_hip import (
        attn_res_smallm_hip,
    )
    from sglang.srt.layers.attn_residual import use_hip_smallm

    ar = tp.all_reduce(partial)
    out = torch.empty_like(prefix)
    pout = torch.empty_like(prefix)
    out8 = (
        torch.empty(out.shape, dtype=torch.float8_e4m3fn, device=out.device)
        if fp8
        else None
    )
    k = attn_res_smallm_hip if use_hip_smallm(prefix.shape[0], H, nvb) else attn_res_hip
    k(
        prefix,
        bank,
        cw,
        ow,
        out,
        nvb,
        1e-6,
        1e-6,
        addend=ar,
        prefix_out=pout,
        write_prefix=False,
        out_fp8=out8,
    )
    return out, pout, out8


def fused_ar(mod, ca, partial, prefix, bank, cw, ow, nvb, fp8):
    out = torch.empty_like(prefix)
    pout = torch.empty_like(prefix)
    out8 = (
        torch.empty(out.shape, dtype=torch.float8_e4m3fn, device=out.device)
        if fp8
        else None
    )
    mod.ar_agg(
        ca,
        partial,
        prefix,
        bank,
        cw,
        ow,
        nvb,
        1e-6,
        1e-6,
        out=out,
        prefix_out=pout,
        out_fp8=out8,
    )
    return out, pout, out8


def unfused_ag(ca, y, b, c, bank, cw, ow, nvb, fp8):
    from sglang.kernels.ops.attention.attn_res_hip import attn_res_hip
    from sglang.kernels.ops.attention.attn_res_smallm_hip import (
        attn_res_smallm_hip,
    )
    from sglang.srt.layers.attn_residual import use_hip_smallm

    if torch.cuda.is_current_stream_capturing():
        # the serving path at T > 64 (ag_add_one_barrier); same rounding as AITER's
        from sglang.kernels.ops.communication.k3_ag_one_barrier_hip import (
            all_gather_lastdim_add_1sync,
        )

        row = all_gather_lastdim_add_1sync(ca, y, b, c)
    else:
        row = ca.all_gather_lastdim_add(y, b, c)
    out = torch.empty_like(b)
    out8 = (
        torch.empty(out.shape, dtype=torch.float8_e4m3fn, device=out.device)
        if fp8
        else None
    )
    k = attn_res_smallm_hip if use_hip_smallm(b.shape[0], H, nvb) else attn_res_hip
    k(
        row,
        bank,
        cw,
        ow,
        out,
        nvb,
        1e-6,
        1e-6,
        prefix_out=row,
        write_prefix=True,
        out_fp8=out8,
    )
    return out, row, out8


def fused_ag(mod, ca, y, b, c, bank, cw, ow, nvb, fp8):
    out = torch.empty_like(b)
    pout = torch.empty_like(b)
    out8 = (
        torch.empty(out.shape, dtype=torch.float8_e4m3fn, device=out.device)
        if fp8
        else None
    )
    mod.ag_agg(
        ca,
        y,
        b,
        c,
        bank,
        cw,
        ow,
        nvb,
        1e-6,
        1e-6,
        out=out,
        prefix_out=pout,
        out_fp8=out8,
        write_bank=True,
    )
    return out, pout, out8


def time_graph(fn, iters, reps, inner=10):
    with graph_capture() as gc:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=gc.stream):
            for _ in range(inner):
                fn()
    torch.cuda.synchronize()
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    res = []
    for _ in range(reps):
        s, e = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        dist.barrier()
        torch.cuda.synchronize()
        s.record()
        for _ in range(iters):
            g.replay()
        e.record()
        e.synchronize()
        res.append(s.elapsed_time(e) * 1e3 / iters / inner)
    t = torch.tensor([sorted(res)[len(res) // 2]], device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item()


def err(a, b):
    a, b = a.float(), b.float()
    d = (a - b).abs()
    return dict(
        max_abs=d.max().item(),
        max_rel=(d / b.abs().clamp_min(1e-2)).max().item(),
        mismatch=(d > 0).float().mean().item(),
        exact=bool(torch.equal(a, b)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default="8,32,64,72,96,128")
    ap.add_argument("--nvb", default="1,4,8")
    ap.add_argument("--modes", default="ar,ag")
    ap.add_argument("--fp8", default="0")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    init_distributed_environment(
        world_size=8,
        rank=rank,
        local_rank=int(os.environ["LOCAL_RANK"]),
        backend="nccl",
    )
    publish_build_topology(world_rank=rank, tp_size=8)
    initialize_model_parallel()
    tp = get_tp_group()
    ca = tp.ca_comm
    from sglang.kernels.ops.communication import k3_ar_agg_hip

    lt = k3_ar_agg_hip

    if rank == 0:
        print(
            "ca",
            type(ca).__name__,
            "max_size",
            getattr(ca, "max_size", None),
            flush=True,
        )
    rows = []
    for mode in args.modes.split(","):
        for nvb in map(int, args.nvb.split(",")):
            for fp8 in map(int, args.fp8.split(",")):
                for T in map(int, args.tokens.split(",")):
                    g = torch.Generator(device="cuda").manual_seed(1234 + rank)
                    gs = torch.Generator(device="cuda").manual_seed(
                        99
                    )  # shared across ranks
                    bank = torch.randn(
                        T, nvb + 1, H, device="cuda", dtype=torch.bfloat16, generator=gs
                    )
                    cw = (
                        torch.randn(H, device="cuda", dtype=torch.float32, generator=gs)
                        * 0.05
                    )
                    ow = (
                        1 + 0.1 * torch.randn(H, device="cuda", generator=gs)
                    ).bfloat16()
                    if mode == "ar":
                        partial = torch.randn(
                            T, H, device="cuda", dtype=torch.bfloat16, generator=g
                        )
                        prefix = torch.randn(
                            T, H, device="cuda", dtype=torch.bfloat16, generator=gs
                        )
                        uf = lambda: unfused_ar(
                            ca, tp, partial, prefix, bank, cw, ow, nvb, fp8
                        )
                        fu = lambda: fused_ar(
                            lt, ca, partial, prefix, bank, cw, ow, nvb, fp8
                        )
                    else:
                        y = torch.randn(
                            T, H // 8, device="cuda", dtype=torch.bfloat16, generator=g
                        )
                        b = torch.randn(
                            T, H, device="cuda", dtype=torch.bfloat16, generator=gs
                        )
                        c = torch.randn(
                            T, H, device="cuda", dtype=torch.bfloat16, generator=gs
                        )
                        bank_u = bank.clone()
                        uf = lambda: unfused_ag(ca, y, b, c, bank_u, cw, ow, nvb, fp8)
                        fu = lambda: fused_ag(lt, ca, y, b, c, bank, cw, ow, nvb, fp8)
                    ro, rp, r8 = uf()
                    fo, fp, f8 = fu()
                    torch.cuda.synchronize()
                    e_out, e_pre = err(fo, ro), err(fp, rp)
                    e8 = err(f8.float(), r8.float()) if fp8 else None
                    if mode == "ag":
                        e_bank = err(bank[:, nvb], bank_u[:, nvb])
                    t_u = time_graph(uf, args.iters, args.reps)
                    t_f = time_graph(fu, args.iters, args.reps)
                    # graph-mode outputs equal too (the captured kernels re-ran)
                    row = dict(
                        mode=mode,
                        nvb=nvb,
                        fp8=fp8,
                        T=T,
                        unfused_us=round(t_u, 2),
                        fused_us=round(t_f, 2),
                        speedup=round(t_u / t_f, 3),
                        out=e_out,
                        prefix=e_pre,
                    )
                    if e8:
                        row["out8"] = e8
                    if mode == "ag":
                        row["bank_row"] = e_bank
                    rows.append(row)
                    if rank == 0:
                        print(json.dumps(row), flush=True)
    if rank == 0 and args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
