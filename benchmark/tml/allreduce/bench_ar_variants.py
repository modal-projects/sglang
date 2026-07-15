"""Complete kernel-variant comparison for the Inkling custom all-reduce family.

Benches every variant at every shape with a per-variant config sweep and
reports the full matrix: torch multimem ("mm"), v3 (two-shot multimem,
single-leader barriers), v3b (v3 + per-block barriers), v4 (full one-shot
ld_reduce), v5 (push one-shot, leader + per-block flavors). v4/v5 are
region-bound to <= V5_MAX_TOKENS rows; larger shapes bench mm/v3/v3b only.

Run (TP4):
  CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc-per-node 4 \
      benchmark/tml/allreduce/bench_ar_variants.py
"""

import torch
import torch.distributed as dist

import validate_inkling_all_reduce as V
from sglang.jit_kernel.inkling_all_reduce import (
    compile_inkling_all_reduce,
    inkling_multimem_full_oneshot,
)

SIZES_FULL = [1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024]
SIZES_LARGE = [1536, 2048, 3072, 4096, 8192, 16384]  # mm / v3 / v3b only

V3_CFGS = [(16, 1024), (24, 1024), (32, 512), (32, 1024), (64, 512),
           (96, 512), (96, 256)]
V3B_CFGS = V.V3PB_CONFIGS
V4_CFGS = [(1, 1024), (2, 1024), (4, 1024), (8, 1024), (16, 1024), (0, 1024)]
V5_CFGS = V.V5_CONFIGS  # (nb, bs, per_block) -- both flavors


def best_of(dev, make_fn, cfgs):
    t_best, c_best = float("inf"), None
    for cfg in cfgs:
        t = V.bench_us(dev, make_fn(cfg))
        if t < t_best:
            t_best, c_best = t, cfg
    return t_best, c_best


def fmt(t, cfg=None):
    if t == float("inf"):
        return f"{'-':>7} {'':>10}"
    c = "" if cfg is None else "/".join(
        str(x) if not isinstance(x, bool) else ("pb" if x else "ld") for x in cfg
    )
    return f"{t:>7.1f} {c:>10}"


def main():
    import os

    rank = int(os.environ["LOCAL_RANK"])
    dev = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(dev)
    dist.init_process_group("nccl")
    if rank == 0:
        compile_inkling_all_reduce(torch.bfloat16, dist.get_world_size())
    dist.barrier()
    compile_inkling_all_reduce(torch.bfloat16, dist.get_world_size())
    h = V.Harness(dev)
    V.quiesce(dev)

    V.log(f"AR variant matrix (us, min-of-5x20, world={h.world}, "
          f"hidden={V.HIDDEN}, best config per variant):")
    V.log(f"{'tokens':>7} | {'mm':>7} | {'v3':>18} | {'v3b':>18} | "
          f"{'v4':>18} | {'v5':>18} | {'best':>5} {'vs mm':>6}")
    es = h.elem_size
    for tk in SIZES_FULL + SIZES_LARGE:
        n = tk * V.HIDDEN
        buf = h.buffer[:n]
        buf.copy_(h.pattern(n, 7))
        t_mm = V.bench_us(dev, lambda: torch.ops.symm_mem.multimem_all_reduce_(
            buf, "sum", h.group_name))
        t_v3, c_v3 = best_of(dev, lambda c: lambda: h.run_v3(
            tk, 7, nb=c[0], bs=c[1], pb=False, check=False, fill=False), V3_CFGS)
        t_v3b, c_v3b = best_of(dev, lambda c: lambda: h.run_v3(
            tk, 7, nb=c[0], bs=c[1], pb=True, check=False, fill=False), V3B_CFGS)
        t_v4 = t_v5 = float("inf")
        c_v4 = c_v5 = None
        if tk <= V.V5_MAX_TOKENS:
            # v4 borrows the (amply sized) v5 staging/out regions for in/out.
            in_off = h.v5_stage[0]
            in_view = h.buffer[in_off : in_off + n]
            out_view = h.buffer[h.v5_out : h.v5_out + n]
            in_view.copy_(h.pattern(n, 7))
            mc = h.hdl.multicast_ptr + in_off * es
            t_v4, c_v4 = best_of(dev, lambda c: lambda: inkling_multimem_full_oneshot(
                in_view, out_view, mc, h.hflags.buffer_ptrs_dev,
                h.state.data_ptr(), h.rank, h.world, n, c[0], c[1]), V4_CFGS)
            in_t = h.pattern(n, 7)
            t_v5, c_v5 = best_of(dev, lambda c: lambda: h.run_v5(
                tk, 7, nb=c[0], bs=c[1], pb=c[2], check=False, in_t=in_t), V5_CFGS)
        times = {"mm": t_mm, "v3": t_v3, "v3b": t_v3b, "v4": t_v4, "v5": t_v5}
        best = min(times, key=times.get)
        V.log(f"{tk:>7} | {t_mm:>7.1f} | {fmt(t_v3, c_v3)} | {fmt(t_v3b, c_v3b)} | "
              f"{fmt(t_v4, c_v4)} | {fmt(t_v5, c_v5)} | {best:>5} "
              f"{t_mm / times[best]:>5.2f}x")
        V.quiesce(dev)
    V.log("ALL_OK")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
