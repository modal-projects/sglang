"""K3 MLA verify: HK stage1/reduce L-sweep (fixed overhead vs streaming) and HK-only vs auto at short L."""

import torch

from sglang.kernels.ops.attention.k3_mla_verify_auto import k3_mla_verify_auto
from sglang.kernels.ops.attention.k3_mla_verify_common import cta_fit_num_splits
from sglang.kernels.ops.attention.k3_mla_verify_hk import hk_reduce, hk_stage1

H, QLEN, D, DV, SCALE, KVS, dev = 12, 8, 576, 512, 576**-0.5, 0.5, "cuda"


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


pool = 130000 * 4 + 4096
kv = (torch.randn(pool, 1, D, device=dev) * 0.5).to(torch.float8_e4m3fn)
ks = torch.tensor([KVS], device=dev)
perm = torch.randperm(pool, device=dev).to(torch.int32)
empty = torch.empty(1, device=dev)
print("empty-kernel graph us", gtime(lambda: empty.add_(1.0)), flush=True)
for bs in (1, 2, 4):
    ns = cta_fit_num_splits(bs) if bs != 2 else 128
    for L in (2000, 8000, 16000, 27000, 32000, 48000, 64000, 90000, 128000):
        lens = [L - 97 * b for b in range(bs)]
        indptr = torch.tensor([0] + lens, device=dev).cumsum(0).to(torch.int32)
        slots = perm[: sum(lens)].contiguous()
        q = torch.randn(bs * QLEN, H, D, device=dev, dtype=torch.bfloat16)
        out = torch.empty(bs * QLEN, H, DV, device=dev, dtype=torch.bfloat16)
        op = torch.empty(bs, ns, H * QLEN, DV, device=dev, dtype=torch.bfloat16)
        lp = torch.empty(bs, ns, H * QLEN, device=dev)
        st = lambda: hk_stage1(
            q, kv, indptr, slots, ks, op, lp, out, ns, 256, SCALE, 96
        )
        rd = lambda: hk_reduce(op, lp, indptr, out, ns, 256, ks, 96)
        t1, t2 = gtime(st), gtime(rd)
        t12 = gtime(lambda: (st(), rd()))
        ta = gtime(
            lambda: k3_mla_verify_auto(
                q, kv, indptr, slots, SCALE, ks, KVS, cta_fit=True
            )
        )
        st()
        rd()
        oa = k3_mla_verify_auto(q, kv, indptr, slots, SCALE, ks, KVS, cta_fit=True)
        d = (out.float() - oa.float()).abs().max().item()
        mb = sum(lens) * D / 1e6
        print(
            f"bs={bs} ns={ns} L={L:6d} kvMB={mb:7.1f} stage1={t1:6.2f} reduce={t2:5.2f} hk_only={t12:6.2f} auto={ta:6.2f} hk-auto_maxabs={d:.1e}",
            flush=True,
        )
