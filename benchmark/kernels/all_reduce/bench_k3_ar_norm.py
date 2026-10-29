"""K3 MI355 microbench: k3_ar_norm (MoE [latent | shared] AR + latent RMSNorm
in one kernel) vs AITER custom AR + the standalone RMSNorm, under HIP graphs.

  torchrun --nproc_per_node=8 bench_k3_ar_norm.py --tokens 8,32,64,128
"""

import argparse
import json
import os

import torch
import torch.distributed as dist
from bench_k3_ar_agg_t128 import err, time_graph

from sglang.srt.distributed.parallel_state import (
    get_tp_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.test.test_utils import publish_build_topology

L, HS = 3584, 7168


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default="8,32,64,128")
    ap.add_argument("--tpb", default="0")
    ap.add_argument("--unit-weight", default="1,0")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--reps", type=int, default=7)
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
    from sglang.kernels.ops.communication import k3_ar_norm_hip as m
    from sglang.srt.layers.layernorm import RMSNorm

    for unit in map(int, args.unit_weight.split(",")):
        norm = RMSNorm(L, eps=1e-6).cuda().to(torch.bfloat16)
        if not unit:
            gs = torch.Generator(device="cuda").manual_seed(7)
            norm.weight.data.copy_(
                1 + 0.1 * torch.randn(L, device="cuda", generator=gs)
            )
        for T in map(int, args.tokens.split(",")):
            g = torch.Generator(device="cuda").manual_seed(1234 + rank)
            buf = torch.randn(
                T * (L + HS), device="cuda", dtype=torch.bfloat16, generator=g
            )

            def uf():
                o = tp.all_reduce(buf)
                return o, norm(o[: T * L].view(T, L))

            for tpb in map(int, args.tpb.split(",")):
                if tpb and (
                    -(-T // tpb) > m.MAX_BLOCKS or tpb * (L + HS) // 64 > m.THREADS
                ):
                    continue

                def fu():
                    return m.ar_norm(ca, buf, T, L, norm.weight, 1e-6, tpb=tpb)

                ro, rn = uf()
                fo, fn = fu()
                torch.cuda.synchronize()
                row = dict(
                    T=T,
                    bs=T // 8,
                    tpb=tpb or m.tokens_per_block(T, L + HS),
                    unit_weight=unit,
                    out=err(fo, ro),
                    norm=err(fn, rn),
                )
                row["unfused_us"] = round(time_graph(uf, args.iters, args.reps), 2)
                row["fused_us"] = round(time_graph(fu, args.iters, args.reps), 2)
                row["saved_ms_per_step_92"] = round(
                    (row["unfused_us"] - row["fused_us"]) * 92 / 1e3, 3
                )
                if rank == 0:
                    print(json.dumps(row), flush=True)
    dist.barrier()


if __name__ == "__main__":
    main()
