"""Benchmark + correctness for the bf16 routed-expert grouped GEMM.

Production shapes (Inkling, TP4): E=256 routed experts, top-6;
GEMM1 [M,6144] x [E,1536,6144] -> [M,1536]; GEMM2 [M,768] x [E,6144,768].
Decode M = 6*T (<=384, <=6 rows per active expert); prefill M up to ~100K.

Impls:
  current        grouped_gemm_triton with its shipped fixed config
  cfg:BM-BN-BK-W-S  same kernel, explicit config (tuning sweep)
  torch_grouped  torch._grouped_mm (CUTLASS), if available on this build
  ref            per-expert torch.mm loop (correctness reference; also timed)

Run:
  CUDA_VISIBLE_DEVICES=4 python -m sglang.tml.kernels.benchmark.bench_grouped_gemm \
      --tokens 1,4,16,64 --gemm 1
"""

from __future__ import annotations

import argparse

import torch
import triton
import triton.testing

from sglang.srt.kernels.inkling_moe import (
    BLOCK_SIZE_M,
    _grouped_gemm_kernel,
    compute_grouped_gemm_metadata,
    grouped_gemm_triton,
)

E = 256
TOPK = 6
HIDDEN = 6144
F_TP = 768  # 3072 / TP4


def make_case(tokens: int, gemm: int, seed: int = 0):
    torch.manual_seed(seed)
    dev = torch.device("cuda")
    ids = torch.stack(
        [torch.randperm(E, device=dev)[:TOPK] for _ in range(tokens)]
    ).to(torch.int32)  # [T, 6], distinct experts per token like real topk
    sorted_ids, reorder = torch.sort(ids.view(-1).to(torch.int16), stable=True)
    meta = compute_grouped_gemm_metadata(sorted_ids, E)
    m = tokens * TOPK
    if gemm == 1:
        k, n = HIDDEN, 2 * F_TP
    else:
        k, n = F_TP, HIDDEN
    a = (torch.randn(m, k, device=dev) * 0.05).to(torch.bfloat16)
    b = (torch.randn(E, n, k, device=dev) * 0.02).to(torch.bfloat16)
    return a, b, sorted_ids, meta


def ref_out(a, b, sorted_ids):
    out = torch.empty(a.shape[0], b.shape[1], dtype=a.dtype, device=a.device)
    ids64 = sorted_ids.long()
    for e in ids64.unique():
        rows = (ids64 == e).nonzero(as_tuple=True)[0]
        out[rows] = a[rows] @ b[e].T
    return out


def run_current(a, b, meta):
    return grouped_gemm_triton(a, b, E, *meta)


def run_cfg(a, b, meta, bm, bn, bk, warps, stages):
    m, k = a.shape
    _, n, _ = b.shape
    c = torch.empty((m, n), device=a.device, dtype=a.dtype)
    num_tokens_per_expert, expert_token_offs, expert_block_offs, expert_block_schedule = meta
    if bm != BLOCK_SIZE_M:
        # block schedule is BLOCK_SIZE_M-dependent; rebuild for this bm
        from sglang.srt.kernels.inkling_moe import compute_expert_block_metadata

        expert_block_offs, expert_block_schedule = compute_expert_block_metadata(
            num_tokens_per_expert, m, block_size_m=bm
        )
    grid_m = expert_block_schedule.numel()
    _grouped_gemm_kernel[(grid_m * triton.cdiv(n, bn),)](
        A=a, B=b, C=c,
        NumTokensPerExpert=num_tokens_per_expert,
        ExpertTokenOffs=expert_token_offs,
        ExpertBlockOffs=expert_block_offs,
        ExpertBlockSchedule=expert_block_schedule,
        a_stride_0=a.stride(0), b_stride_0=b.stride(0), b_stride_1=b.stride(1),
        c_stride_0=c.stride(0),
        E=E, N=n, K=k, grid_m=grid_m,
        BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn, BLOCK_SIZE_K=bk, GROUP_SIZE_M=8,
        KN_MASK=k % bk != 0 or n % bn != 0,
        INT64_INDEX=a.nbytes >= 2**31 or b.nbytes >= 2**31 or c.nbytes >= 2**31,
        num_warps=warps, num_stages=stages,
    )
    return c


def has_torch_grouped():
    return hasattr(torch, "_grouped_mm")


def run_torch_grouped(a, b, meta):
    # offs = expert_token_offs[1:]; b needs [E, K, N]
    offs = meta[1][1:].to(torch.int32)
    return torch._grouped_mm(a, b.transpose(1, 2), offs=offs)


def gtime(fn, iters=30):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(iters):
            fn()
    g.replay()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(5):
        g.replay()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1e3 / (5 * iters)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=str, default="1,4,16,64,512,4096")
    ap.add_argument("--gemm", type=int, default=1, choices=[1, 2])
    ap.add_argument("--sweep", action="store_true", help="config sweep")
    args = ap.parse_args()

    print(f"gemm{args.gemm}  torch._grouped_mm available: {has_torch_grouped()}")
    cfgs = [(128, 256, 64, 8, 3)]  # current
    if args.sweep:
        cfgs += [
            (16, 64, 128, 4, 4), (16, 64, 256, 4, 4), (16, 64, 128, 4, 6),
            (16, 128, 128, 4, 4), (16, 128, 128, 4, 6), (16, 128, 256, 4, 3),
            (16, 256, 128, 8, 4), (32, 128, 128, 4, 4), (32, 64, 128, 4, 4),
            (64, 128, 128, 4, 4), (128, 128, 64, 4, 3), (128, 256, 128, 8, 3),
        ]
    for tokens in [int(t) for t in args.tokens.split(",")]:
        a, b, sorted_ids, meta = make_case(tokens, args.gemm)
        ref = ref_out(a, b, sorted_ids)

        def check(c):
            return (c - ref).abs().max().item()

        cur = run_current(a, b, meta)
        line = [f"T={tokens:5} M={a.shape[0]:6}"]
        line.append(f"current {gtime(lambda: run_current(a, b, meta)):8.2f}us(err {check(cur):.1e})")
        if has_torch_grouped():
            try:
                tg = run_torch_grouped(a, b, meta)
                line.append(
                    f"torch_grouped {gtime(lambda: run_torch_grouped(a, b, meta)):8.2f}us(err {check(tg):.1e})"
                )
            except Exception as ex:
                line.append(f"torch_grouped FAIL({type(ex).__name__})")
        print(" | ".join(line))
        if args.sweep:
            results = []
            for bm, bn, bk, w, s in cfgs[1:]:
                try:
                    c = run_cfg(a, b, meta, bm, bn, bk, w, s)
                    err = check(c)
                    us = gtime(lambda: run_cfg(a, b, meta, bm, bn, bk, w, s))
                    results.append((us, f"{bm}-{bn}-{bk}-w{w}-s{s}", err))
                except Exception as ex:
                    results.append((float("inf"), f"{bm}-{bn}-{bk}-w{w}-s{s} {type(ex).__name__}", -1))
            for us, tag, err in sorted(results)[:5]:
                print(f"      cfg {tag:>22}: {us:8.2f} us (err {err:.1e})")


if __name__ == "__main__":
    main()
