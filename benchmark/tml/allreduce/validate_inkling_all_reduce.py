"""Correctness + stress validation for the Inkling custom JIT all-reduce kernels.

Validates the kernel family in ``python/sglang/jit_kernel/inkling_all_reduce.py``
(v2 two-shot explicit, v3 two-shot multimem, v4 full one-shot) plus the
auto-select dispatch, against exact integer reference sums:

  1. correctness across the tuned token grid (auto-dispatch, all kernels)
  2. uint32 epoch wraparound (preloads xepoch/flags/releases near 2**32)
  3. v4 A/B rotation stress (double-buffer reuse-distance safety)
  4. CUDA-graph capture + replay (v3 in-place and v4 out-of-place)
  5. quick bench: tuned v3 / v4 / v5 vs torch multimem_all_reduce_
  6. v5 (push one-shot) band correctness + staging rotation stress
  7. v5 CUDA-graph capture + replay

Run (TP4):
  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc-per-node 4 \
      benchmark/tml/allreduce/validate_inkling_all_reduce.py
TP8: CUDA_VISIBLE_DEVICES=0,...,7 torchrun --nproc-per-node 8 <same>
"""

import os

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as torch_symm_mem

from sglang.jit_kernel.inkling_all_reduce import (
    STATE_SIZE,
    compile_inkling_all_reduce,
    flags_numel,
    select_ar_config,
    inkling_multimem_full_oneshot,
    inkling_multimem_one_shot_fused,
    inkling_multimem_push_oneshot,
    inkling_two_shot_all_reduce_fused,
)

HIDDEN = 6144
V4_REGION = 16 * HIDDEN  # mirrors _INKLING_AR_V4_REGION in tml/kernels/comm.py
V5_MAX_TOKENS = 1024  # staging headroom for the push one-shot (v5) phases
MAX_TOKENS = 16384
TOKEN_GRID = [1, 2, 3, 8, 24, 64, 96, 128, 192, 256, 512, 1024, 1536, 2048,
              3072, 4096, 8192, 16384]
# v5 bench config sweep (num_blocks, block_size, per_block_barrier); nb=0 =
# auto (full-range grid). Small grids (1-4 blocks) matter for the leader
# barrier (solo path skips the grid funnel); the per-block barrier ("pb") has
# no funnel at any block count.
V5_CONFIGS = [
    (0, 1024, False), (1, 1024, False), (4, 1024, False), (8, 512, False),
    (16, 1024, False), (32, 1024, False), (48, 1024, False),
    (1, 1024, True), (2, 512, True), (2, 1024, True), (4, 512, True),
    (4, 1024, True), (8, 512, True), (8, 1024, True), (16, 512, True),
    (16, 1024, True), (32, 512, True), (32, 1024, True), (48, 1024, True),
    (64, 1024, True), (0, 1024, True),
]
# v3 with per-block barriers: sweep for the mid band where the grid funnel is
# the two-shot's handicap.
V3PB_CONFIGS = [(8, 512), (16, 512), (16, 1024), (32, 512), (32, 1024),
                (48, 512), (64, 512), (64, 1024)]


def log(*a):
    if dist.get_rank() == 0:
        print(*a, flush=True)


class Harness:
    """Symm buffer + barrier resources, mirroring tml/kernels/comm.py setup."""

    def __init__(self, device):
        self.rank = dist.get_rank()
        self.world = dist.get_world_size()
        self.group_name = dist.group.WORLD.group_name
        v5_stage = self.world * V5_MAX_TOKENS * HIDDEN  # one rotation buffer
        v5_out = V5_MAX_TOKENS * HIDDEN
        total = MAX_TOKENS * HIDDEN + 3 * V4_REGION + 2 * v5_stage + v5_out
        self.buffer = torch_symm_mem.empty(
            total, device=device, dtype=torch.bfloat16
        )
        self.hdl = torch_symm_mem.rendezvous(self.buffer, self.group_name)
        self.flags = torch_symm_mem.empty(
            flags_numel(self.world), device=device, dtype=torch.uint32
        )
        self.flags.zero_()
        self.hflags = torch_symm_mem.rendezvous(self.flags, self.group_name)
        self.hflags.barrier()
        self.state = torch.zeros(STATE_SIZE, device=device, dtype=torch.uint32)
        base = MAX_TOKENS * HIDDEN
        self.v4_in = (base, base + V4_REGION)
        self.v4_out = base + 2 * V4_REGION
        self.v4_cur = 0
        self.v5_stage = (base + 3 * V4_REGION, base + 3 * V4_REGION + v5_stage)
        self.v5_out = base + 3 * V4_REGION + 2 * v5_stage
        self.v5_cur = 0
        self.elem_size = self.buffer.element_size()

    def pattern(self, n, salt, rank=None):
        """Integer pattern exact in bf16 whose cross-rank sum is exact too."""
        rank = self.rank if rank is None else rank
        i = torch.arange(n, device=self.buffer.device, dtype=torch.float32)
        return (((i + salt) % 9 - 4) * (rank + 1)).to(torch.bfloat16)

    def expected(self, n, salt):
        scale = self.world * (self.world + 1) // 2
        i = torch.arange(n, device=self.buffer.device, dtype=torch.float32)
        return (((i + salt) % 9 - 4) * scale).to(torch.bfloat16)

    def run_auto(self, num_tokens, salt, check=True):
        """Fill producer data, dispatch like symm_mem_all_reduce, verify."""
        n = num_tokens * HIDDEN
        kernel, nb, bs = select_ar_config(num_tokens, self.world)
        if kernel == "v4" and n <= V4_REGION:
            in_off = self.v4_in[self.v4_cur]
            in_view = self.buffer[in_off : in_off + n]
            out_view = self.buffer[self.v4_out : self.v4_out + n]
            in_view.copy_(self.pattern(n, salt))
            mc = self.hdl.multicast_ptr + in_off * self.elem_size
            inkling_multimem_full_oneshot(
                in_view, out_view, mc, self.hflags.buffer_ptrs_dev,
                self.state.data_ptr(), self.rank, self.world, n, nb, bs,
            )
            self.v4_cur = 1 - self.v4_cur
            result = out_view
        else:
            buf = self.buffer[:n]
            buf.copy_(self.pattern(n, salt))
            if kernel == "v3":
                inkling_multimem_one_shot_fused(
                    buf, self.hdl.multicast_ptr, self.hflags.buffer_ptrs_dev,
                    self.state.data_ptr(), self.rank, self.world, n, nb, bs,
                )
            elif kernel == "v2":
                inkling_two_shot_all_reduce_fused(
                    buf, self.hdl.buffer_ptrs_dev, self.hflags.buffer_ptrs_dev,
                    self.state.data_ptr(), self.rank, self.world, n, nb, bs,
                )
            else:  # "mm"
                torch.ops.symm_mem.multimem_all_reduce_(
                    buf, "sum", self.group_name
                )
            result = buf
        if check:
            bad = (result != self.expected(n, salt)).sum().item()
            assert bad == 0, (
                f"rank{self.rank} {kernel} tokens={num_tokens} salt={salt}: "
                f"{bad}/{n} mismatches"
            )
        return kernel

    def run_v5(self, num_tokens, salt, nb=0, bs=0, pb=False, check=True, in_t=None):
        """Push one-shot (v5) with A/B staging rotation; input is a plain LOCAL
        tensor (exercises the no-symm-input path)."""
        n = num_tokens * HIDDEN
        stage_off = self.v5_stage[self.v5_cur]
        out_view = self.buffer[self.v5_out : self.v5_out + n]
        if in_t is None:
            in_t = self.pattern(n, salt)
        mc = self.hdl.multicast_ptr + stage_off * self.elem_size
        local = self.buffer.data_ptr() + stage_off * self.elem_size
        inkling_multimem_push_oneshot(
            in_t, out_view, mc, local, self.hflags.buffer_ptrs_dev,
            self.state.data_ptr(), self.rank, self.world, n, nb, bs,
            per_block_barrier=pb,
        )
        self.v5_cur = 1 - self.v5_cur
        if check:
            bad = (out_view != self.expected(n, salt)).sum().item()
            assert bad == 0, (
                f"rank{self.rank} v5 tokens={num_tokens} salt={salt}: "
                f"{bad}/{n} mismatches"
            )
        return out_view

    def run_v3(self, num_tokens, salt, nb=0, bs=0, pb=False, check=True, fill=True):
        """Two-shot multimem (v3) in place on buf[:n], either barrier flavor."""
        n = num_tokens * HIDDEN
        buf = self.buffer[:n]
        if fill:
            buf.copy_(self.pattern(n, salt))
        inkling_multimem_one_shot_fused(
            buf, self.hdl.multicast_ptr, self.hflags.buffer_ptrs_dev,
            self.state.data_ptr(), self.rank, self.world, n, nb, bs,
            per_block_barrier=pb,
        )
        if check:
            bad = (buf != self.expected(n, salt)).sum().item()
            assert bad == 0, (
                f"rank{self.rank} v3(pb={pb}) tokens={num_tokens} salt={salt}: "
                f"{bad}/{n} mismatches"
            )
        return buf


def quiesce(dev):
    torch.cuda.synchronize(dev)
    dist.barrier()


def phase_grid(h, dev):
    used = {}
    for salt, tk in enumerate(TOKEN_GRID):
        kernel = h.run_auto(tk, salt)
        used.setdefault(kernel, []).append(tk)
    quiesce(dev)
    log(f"[1] token-grid correctness OK; dispatch: {used}")


def phase_wrap(h, dev):
    # Preload the monotonic counters just below 2**32 so the test crosses the
    # wrap; every rank must preload identically while no kernel is in flight.
    quiesce(dev)
    x = (1 << 32) - 8
    h.state[2] = x  # release0
    h.state[3] = x  # release1
    h.state[4] = x  # xepoch
    h.state[8:] = x  # per-block epochs (must stay consistent with the flags)
    h.flags.fill_(x)
    quiesce(dev)
    # Mixed 1- and 2-barrier kernels (incl. v5 pushes), crossing the wrap early.
    for salt, tk in enumerate([1, 2, 256, 1024, 3072, 1, 64, 192, 512, 2048] * 3):
        h.run_auto(tk, salt=100 + salt)
        if salt % 2 == 0:
            h.run_v5(16, salt=150 + salt, pb=bool(salt % 4))
    quiesce(dev)
    assert int(h.state[4].item()) < x, "xepoch did not wrap"
    log("[2] uint32 epoch wraparound OK (correct across the wrap, no hang)")


def phase_rotation(h, dev):
    for it in range(400):
        h.run_auto(1 + (it % 2), salt=200 + it)
    quiesce(dev)
    log("[3] v4 A/B rotation stress (400 iters) OK")


def phase_graph(h, dev):
    n = HIDDEN  # 1 token
    # --- v4 graph: two ARs (A then B), even count, distinct outputs.
    in_a = h.buffer[h.v4_in[0] : h.v4_in[0] + n]
    in_b = h.buffer[h.v4_in[1] : h.v4_in[1] + n]
    out_a = h.buffer[h.v4_out : h.v4_out + n]
    out_b = h.buffer[h.v4_out + n : h.v4_out + 2 * n]
    mc_a = h.hdl.multicast_ptr + h.v4_in[0] * h.elem_size
    mc_b = h.hdl.multicast_ptr + h.v4_in[1] * h.elem_size
    args = (h.hflags.buffer_ptrs_dev, h.state.data_ptr(), h.rank, h.world, n, 1, 1024)
    in_a.copy_(h.pattern(n, 0))
    in_b.copy_(h.pattern(n, 1))
    quiesce(dev)
    g4 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g4):
        inkling_multimem_full_oneshot(in_a, out_a, mc_a, *args)
        inkling_multimem_full_oneshot(in_b, out_b, mc_b, *args)
    quiesce(dev)
    for rep in range(25):
        in_a.copy_(h.pattern(n, 300 + rep))
        in_b.copy_(h.pattern(n, 400 + rep))
        g4.replay()
        torch.cuda.synchronize(dev)
        assert torch.equal(out_a, h.expected(n, 300 + rep)), f"v4 graph A rep{rep}"
        assert torch.equal(out_b, h.expected(n, 400 + rep)), f"v4 graph B rep{rep}"
    quiesce(dev)
    # --- v3 graph: in-place on buf[:n3].
    tk = 256
    n3 = tk * HIDDEN
    _, nb, bs = select_ar_config(tk, h.world)
    if select_ar_config(tk, h.world)[0] != "v3":
        nb, bs = 24, 1024
    buf = h.buffer[:n3]
    buf.copy_(h.pattern(n3, 0))
    quiesce(dev)
    g3 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g3):
        inkling_multimem_one_shot_fused(
            buf, h.hdl.multicast_ptr, h.hflags.buffer_ptrs_dev,
            h.state.data_ptr(), h.rank, h.world, n3, nb, bs,
        )
    quiesce(dev)
    for rep in range(25):
        buf.copy_(h.pattern(n3, 500 + rep))
        g3.replay()
        torch.cuda.synchronize(dev)
        assert torch.equal(buf, h.expected(n3, 500 + rep)), f"v3 graph rep{rep}"
    quiesce(dev)
    log("[4] CUDA-graph capture + 25 replays (v4 x2, v3) OK")


def phase_v5(h, dev):
    # Correctness across the target band incl. untuned odd shapes, both
    # barrier flavors (leader + per-block).
    for pb in (False, True):
        for salt, tk in enumerate([3, 4, 8, 16, 64, 100, 128, 192, 256, 300, 512, 1000, 1024]):
            h.run_v5(tk, 600 + salt, pb=pb)
    # Staging A/B rotation stress at mixed sizes, interleaving barrier flavors.
    for it in range(200):
        h.run_v5(1 + (it % 3) * 31, 700 + it, pb=bool(it % 2))
    # v3 with per-block barriers, incl. sizes above the pb block cap.
    for salt, tk in enumerate([3, 64, 192, 256, 1024, 3000, 4096]):
        h.run_v3(tk, 640 + salt, nb=32, bs=512, pb=True)
        h.run_v3(tk, 660 + salt, pb=True)  # auto nb (capped at 64)
    quiesce(dev)
    log("[6] v5 push one-shot + v3-pb: band correctness + 200-iter rotation OK")


def phase_v5_graph(h, dev):
    n = 64 * HIDDEN
    in_a = h.pattern(n, 0)
    in_b = h.pattern(n, 1)
    out_a = h.buffer[h.v5_out : h.v5_out + n]
    out_b = h.buffer[h.v5_out + n : h.v5_out + 2 * n]
    es = h.elem_size
    args = (h.hflags.buffer_ptrs_dev, h.state.data_ptr(), h.rank, h.world, n, 16, 1024)
    quiesce(dev)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        inkling_multimem_push_oneshot(
            in_a, out_a, h.hdl.multicast_ptr + h.v5_stage[0] * es,
            h.buffer.data_ptr() + h.v5_stage[0] * es, *args)
        inkling_multimem_push_oneshot(
            in_b, out_b, h.hdl.multicast_ptr + h.v5_stage[1] * es,
            h.buffer.data_ptr() + h.v5_stage[1] * es, *args,
            per_block_barrier=True)
    quiesce(dev)
    for rep in range(25):
        in_a.copy_(h.pattern(n, 800 + rep))
        in_b.copy_(h.pattern(n, 900 + rep))
        g.replay()
        torch.cuda.synchronize(dev)
        assert torch.equal(out_a, h.expected(n, 800 + rep)), f"v5 graph A rep{rep}"
        assert torch.equal(out_b, h.expected(n, 900 + rep)), f"v5 graph B rep{rep}"
    quiesce(dev)
    log("[7] v5 CUDA-graph capture + 25 replays OK")


def bench_us(dev, fn, iters=20, reps=5):
    fn()  # warm
    quiesce(dev)
    best = float("inf")
    for _ in range(reps):
        dist.barrier()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize(dev)
        best = min(best, start.elapsed_time(end) / iters * 1000)
    return best


def phase_bench(h, dev):
    log(f"[5] bench (us, min-of-5x20, world={h.world}, hidden={HIDDEN}):")
    log(f"{'tokens':>8} {'torch-mm':>10} {'custom':>10} {'kernel':>7} "
        f"{'v5-best':>10} {'v5-cfg':>12} {'v5-vs-mm':>9} "
        f"{'v3pb-best':>10} {'v3pb-cfg':>10}")
    for tk in [1, 4, 8, 16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024,
               4096, 16384]:
        n = tk * HIDDEN
        buf = h.buffer[:n]
        buf.copy_(h.pattern(n, 7))
        t_mm = bench_us(dev, lambda: torch.ops.symm_mem.multimem_all_reduce_(
            buf, "sum", h.group_name))
        kernel, nb, bs = select_ar_config(tk, h.world)
        if kernel == "v4":
            in_off = h.v4_in[0]
            in_view = h.buffer[in_off : in_off + n]
            out_view = h.buffer[h.v4_out : h.v4_out + n]
            in_view.copy_(h.pattern(n, 7))
            mc = h.hdl.multicast_ptr + in_off * h.elem_size
            fn = lambda: inkling_multimem_full_oneshot(
                in_view, out_view, mc, h.hflags.buffer_ptrs_dev,
                h.state.data_ptr(), h.rank, h.world, n, nb, bs)
        else:
            if kernel == "mm":  # bench v3 anyway to probe the mm band
                kernel, nb, bs = "v3", 24, 1024
            if kernel == "v3":
                fn = lambda: inkling_multimem_one_shot_fused(
                    buf, h.hdl.multicast_ptr, h.hflags.buffer_ptrs_dev,
                    h.state.data_ptr(), h.rank, h.world, n, nb, bs)
            else:
                fn = lambda: inkling_two_shot_all_reduce_fused(
                    buf, h.hdl.buffer_ptrs_dev, h.hflags.buffer_ptrs_dev,
                    h.state.data_ptr(), h.rank, h.world, n, nb, bs)
        t_c = bench_us(dev, fn)
        t_v5, cfg_v5 = float("inf"), None
        if tk <= V5_MAX_TOKENS:
            in_t = h.pattern(n, 7)
            for v5_nb, v5_bs, v5_pb in V5_CONFIGS:
                t = bench_us(dev, lambda: h.run_v5(
                    tk, 7, nb=v5_nb, bs=v5_bs, pb=v5_pb, check=False, in_t=in_t))
                if t < t_v5:
                    t_v5, cfg_v5 = t, (v5_nb, v5_bs, "pb" if v5_pb else "ld")
        v5_col = (
            f"{t_v5:>10.1f} {'/'.join(map(str, cfg_v5)):>12} {t_mm / t_v5:>8.2f}x"
            if cfg_v5 else f"{'-':>10} {'-':>12} {'-':>9}"
        )
        t_v3pb, cfg_v3pb = float("inf"), None
        buf.copy_(h.pattern(n, 7))
        for pb_nb, pb_bs in V3PB_CONFIGS:
            t = bench_us(dev, lambda: h.run_v3(
                tk, 7, nb=pb_nb, bs=pb_bs, pb=True, check=False, fill=False))
            if t < t_v3pb:
                t_v3pb, cfg_v3pb = t, (pb_nb, pb_bs)
        log(f"{tk:>8} {t_mm:>10.1f} {t_c:>10.1f} {kernel:>7} {v5_col} "
            f"{t_v3pb:>10.1f} {'/'.join(map(str, cfg_v3pb)):>10}")
        quiesce(dev)


def main():
    rank = int(os.environ["LOCAL_RANK"])
    dev = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(dev)
    dist.init_process_group("nccl")
    if rank == 0:  # compile once, peers hit the JIT cache
        compile_inkling_all_reduce(torch.bfloat16, dist.get_world_size())
    dist.barrier()
    compile_inkling_all_reduce(torch.bfloat16, dist.get_world_size())
    h = Harness(dev)
    quiesce(dev)
    phase_grid(h, dev)
    phase_wrap(h, dev)
    phase_rotation(h, dev)
    phase_graph(h, dev)
    phase_v5(h, dev)
    phase_v5_graph(h, dev)
    phase_bench(h, dev)
    quiesce(dev)
    log("ALL_OK")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
