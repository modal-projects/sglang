"""K3 MLA verify (HK regime) breakdown vs HBM roofline on MI355X:1, HIP graphs.
Full k3_mla_verify_auto (default / cta_fit) plus hk_stage1 / hk_reduce alone over an nsplit sweep."""

import sys

import torch

from sglang.kernels.ops.attention.k3_mla_verify_auto import k3_mla_verify_auto
from sglang.kernels.ops.attention.k3_mla_verify_common import (
    cta_fit_num_splits,
    default_num_splits,
)
from sglang.kernels.ops.attention.k3_mla_verify_hk import hk_reduce, hk_stage1

H, QLEN, D, DV = 12, 8, 576, 512
SCALE = D**-0.5
KVS = 0.5
dev = "cuda"


def gtime(fn, iters=20, reps=5):
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


def hbm_bw():
    x = torch.empty(1 << 29, dtype=torch.uint8, device=dev)
    y = torch.empty_like(x)
    t = gtime(lambda: y.copy_(x), iters=10)
    return 2 * x.numel() / (t * 1e-6)


bw = hbm_bw()
print(f"copy BW {bw / 1e12:.2f} TB/s", flush=True)
pool = 90000 * 16 + 4096
kv = (torch.randn(pool, 1, D, device=dev) * 0.5).to(torch.float8_e4m3fn)
ks = torch.tensor([KVS], device=dev)
perm = torch.randperm(pool, device=dev).to(torch.int32)
contig = torch.arange(pool, device=dev, dtype=torch.int32)
bss = (
    [int(x) for x in sys.argv[1].split(",")]
    if len(sys.argv) > 1
    else [1, 2, 3, 4, 8, 16]
)
for layout, idx in (("perm", perm), ("contig", contig)):
    for ctx in (64000, 90000):
        for bs in bss:
            lens = [ctx - 97 * b for b in range(bs)]
            indptr = torch.tensor([0] + lens, device=dev).cumsum(0).to(torch.int32)
            slots = idx[: sum(lens)].contiguous()
            q = torch.randn(bs * QLEN, H, D, device=dev, dtype=torch.bfloat16)
            roof = sum(lens) * D / bw * 1e6
            o_def = k3_mla_verify_auto(q, kv, indptr, slots, SCALE, ks, KVS, thresh=0)
            t_def = gtime(
                lambda: k3_mla_verify_auto(
                    q, kv, indptr, slots, SCALE, ks, KVS, thresh=0
                )
            )
            t_fit = gtime(
                lambda: k3_mla_verify_auto(
                    q, kv, indptr, slots, SCALE, ks, KVS, thresh=0, cta_fit=True
                )
            )
            print(
                f"{layout} L={ctx} bs={bs} roof_us={roof:.2f} auto_def={t_def:.2f} (ns {max(2, default_num_splits(bs))}) "
                f"auto_fit={t_fit:.2f} (ns {cta_fit_num_splits(bs)})",
                flush=True,
            )
            if bs > 4 or layout == "contig" and ctx == 64000:
                continue
            for ns in sorted(
                {
                    32,
                    48,
                    64,
                    96,
                    128,
                    160,
                    192,
                    256,
                    cta_fit_num_splits(bs),
                    max(2, default_num_splits(bs)),
                }
            ):
                if bs * ns > 1024:
                    continue
                out = torch.empty(bs * QLEN, H, DV, device=dev, dtype=torch.bfloat16)
                op = torch.empty(bs, ns, H * QLEN, DV, device=dev, dtype=torch.bfloat16)
                lp = torch.empty(bs, ns, H * QLEN, device=dev)
                p8 = 96
                st = lambda: hk_stage1(
                    q, kv, indptr, slots, ks, op, lp, out, ns, 256, SCALE, p8
                )
                rd = lambda: hk_reduce(op, lp, indptr, out, ns, 256, ks, p8)
                st()
                rd()
                torch.cuda.synchronize()
                d = (out.float() - o_def.float()).abs().max().item()
                t1, t2 = gtime(st), gtime(rd)
                t12 = gtime(lambda: (st(), rd()))
                print(
                    f"   ns={ns:3d} stage1={t1:.2f} reduce={t2:.2f} both={t12:.2f} vs_def_maxabs={d:.2e}",
                    flush=True,
                )
