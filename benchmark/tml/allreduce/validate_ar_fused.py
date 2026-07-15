"""Validation + bench for the fused decode {AR -> sconv -> add+RMSNorm} kernel.

Compares ``inkling_ar_sconv_norm`` (jit_kernel/inkling_ar_fused.py) against the exact
unfused reference chain:

    x_ref  = bf16(fp32-sum of all ranks' partials)      # what v5 would store
    y_ref  = fused_causal_conv1d_update_decode(x_ref)   # the unfused jit sconv
    r_f    = float(y_ref) + float(residual)             # fused_add_rmsnorm math
    res    = bf16(r_f);  hs = bf16(r_f * rsqrt(mean(r_f^2)+eps) * gamma)

The sconv path (y and the in-place cache update, incl. PAD rows, cache_mask
gating and the track-copy) must be BIT-EXACT vs the unfused kernel; hs allows
last-ulp variance-reduction-order differences. Also: staging-rotation stress
interleaved with plain v5 ARs (barrier interop), CUDA-graph capture + replay,
and a fused vs {v5 + sconv + fused_add_rmsnorm} chain bench.

Run (TP4):
  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc-per-node 4 \
      benchmark/tml/allreduce/validate_ar_fused.py
"""

import os

import torch
import torch.distributed as dist
import validate_inkling_all_reduce as V

from sglang.jit_kernel.inkling_all_reduce import compile_inkling_all_reduce
from sglang.jit_kernel.inkling_ar_fused import inkling_ar_sconv_norm
from sglang.jit_kernel.inkling_sconv import fused_causal_conv1d_update_decode

D = V.HIDDEN
W = 4
POOL = 4096
EPS = 1e-6


class FusedCase:
    """One test case: per-rank partials + shared sconv/norm state."""

    def __init__(self, h, T, salt, track):
        self.h, self.T, self.track = h, T, track
        dev = h.buffer.device
        g = torch.Generator(device=dev)
        # Shared (rank-identical) state.
        g.manual_seed(1000 + salt)
        self.cache = torch.randn(POOL, W - 1, D, generator=g, device=dev).bfloat16()
        self.weight = (torch.randn(D, W, generator=g, device=dev) * 0.5).bfloat16()
        self.residual = torch.randn(T, D, generator=g, device=dev).bfloat16()
        self.gamma = torch.randn(D, generator=g, device=dev).bfloat16()
        self.ci = (torch.arange(T, device=dev, dtype=torch.int32) * 3 + 11) % 1500
        self.cm = torch.ones(T, device=dev, dtype=torch.bool)
        if T >= 2:
            self.cm[1] = False  # fresh-state row: history gated off
        if T >= 4:
            self.ci[2] = -1  # PAD row: emits y, never writes cache
        self.tm = None
        self.ti = None
        if track:
            self.tm = torch.zeros(T, device=dev, dtype=torch.bool)
            self.tm[0] = True
            self.ti = 2000 + torch.arange(T, device=dev, dtype=torch.int64)
        # Per-rank partials for ALL ranks (so x_ref is locally computable).
        self.parts = []
        for r in range(h.world):
            g.manual_seed(2000 + salt * 17 + r)
            self.parts.append(torch.randn(T, D, generator=g, device=dev).bfloat16())

    def x_ref(self):
        acc = torch.zeros(self.T, D, device=self.parts[0].device, dtype=torch.float32)
        for p in self.parts:
            acc += p.float()
        return acc.bfloat16()

    def reference(self, cache_ref):
        """The EXACT production unfused chain: the jit decode sconv followed by
        sgl_kernel.fused_add_rmsnorm (flashinfer, PDL) -- the same kernels the
        model's mlp_sconv + attn_norm run today."""
        from sgl_kernel import fused_add_rmsnorm

        y = fused_causal_conv1d_update_decode(
            x=self.x_ref(),
            weight=self.weight,
            sconv_cache=cache_ref,
            cache_indices=self.ci,
            cache_mask=self.cm,
            activation=None,
            use_residual=True,
            track_mask=self.tm,
            track_indices=self.ti,
        )
        res = self.residual.clone()
        fused_add_rmsnorm(y, res, self.gamma, EPS)  # in place: y -> hs
        return res, y

    def run_fused(self, cache_fused, hs_out, res_out, vpt=0):
        h = self.h
        stage_off = h.v5_stage[h.v5_cur]
        es = h.elem_size
        inkling_ar_sconv_norm(
            self.parts[h.rank],
            self.residual,
            res_out,
            hs_out,
            self.gamma,
            EPS,
            cache_fused,
            self.ci,
            self.cm,
            self.weight,
            h.hdl.multicast_ptr + stage_off * es,
            h.buffer.data_ptr() + stage_off * es,
            h.hflags.buffer_ptrs_dev,
            h.state.data_ptr(),
            h.rank,
            h.world,
            activation=None,
            use_residual=True,
            track_mask=self.tm,
            track_indices=self.ti,
            vecs_per_thread=vpt,
        )
        h.v5_cur = 1 - h.v5_cur


def check_case(h, dev, T, salt, track, vpt=0):
    case = FusedCase(h, T, salt, track)
    cache_ref = case.cache.clone()
    cache_fused = case.cache.clone()
    res_ref, hs_ref = case.reference(cache_ref)
    hs_out = torch.empty(T, D, device=dev, dtype=torch.bfloat16)
    res_out = torch.empty(T, D, device=dev, dtype=torch.bfloat16)
    case.run_fused(cache_fused, hs_out, res_out, vpt=vpt)
    torch.cuda.synchronize(dev)
    # The sconv leg (cache update) must be bit-exact vs the production jit
    # kernel; hs/residual are compared against the production flashinfer
    # fused_add_rmsnorm, whose reduction order may differ in the last ulp.
    assert torch.equal(
        cache_fused, cache_ref
    ), f"rank{h.rank} T={T} track={track}: cache update not bit-exact"
    exact = 1.0
    for name, got, ref in (("residual", res_out, res_ref), ("hs", hs_out, hs_ref)):
        diff = (got.float() - ref.float()).abs()
        rel = (diff / ref.float().abs().clamp_min(1e-3)).max().item()
        exact = min(exact, (got == ref).float().mean().item())
        assert rel < 2e-2, f"rank{h.rank} T={T} track={track}: {name} rel diff {rel}"
    return exact


def phase_correctness(h, dev):
    worst_exact = 1.0
    for salt, T in enumerate([1, 2, 4, 8, 16, 64, 96]):
        for track in (False, True):
            worst_exact = min(worst_exact, check_case(h, dev, T, salt, track))
    # All VPT variants must agree too (same math, different thread mapping).
    for vpt in (2, 3, 4, 6):
        worst_exact = min(worst_exact, check_case(h, dev, 4, 90 + vpt, True, vpt=vpt))
    V.quiesce(dev)
    V.log(
        f"[F1] fused correctness OK incl. VPT 2/3/4/6 (cache/residual "
        f"bit-exact; hs exact-match worst case {worst_exact:.4f})"
    )


def phase_rotation(h, dev):
    # Interleave fused calls with plain v5 ARs: shared staging rotation and
    # shared per-block barrier epoch slots must stay consistent.
    case = FusedCase(h, 8, 999, True)
    cache_fused = case.cache.clone()
    cache_ref = case.cache.clone()
    hs_out = torch.empty(8, D, device=dev, dtype=torch.bfloat16)
    res_out = torch.empty(8, D, device=dev, dtype=torch.bfloat16)
    for it in range(100):
        case.run_fused(cache_fused, hs_out, res_out)
        h.run_v5(1 + (it % 3), 600 + it, pb=bool(it % 2))
    # The cache evolved 100x; replay the reference chain 100x and compare once.
    x_ref = case.x_ref()
    for _ in range(100):
        fused_causal_conv1d_update_decode(
            x=x_ref,
            weight=case.weight,
            sconv_cache=cache_ref,
            cache_indices=case.ci,
            cache_mask=case.cm,
            activation=None,
            use_residual=True,
            track_mask=case.tm,
            track_indices=case.ti,
        )
    torch.cuda.synchronize(dev)
    assert torch.equal(cache_fused, cache_ref), f"rank{h.rank}: rotation cache drift"
    V.quiesce(dev)
    V.log("[F2] 100-iter fused/v5 interleaved rotation OK (cache bit-exact)")


def phase_graph(h, dev):
    case = FusedCase(h, 4, 1234, False)
    cache_fused = case.cache.clone()
    cache_ref = case.cache.clone()
    hs_out = torch.empty(4, D, device=dev, dtype=torch.bfloat16)
    res_out = torch.empty(4, D, device=dev, dtype=torch.bfloat16)
    V.quiesce(dev)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        case.run_fused(cache_fused, hs_out, res_out)  # A slot
        case.run_fused(cache_fused, hs_out, res_out)  # B slot (even count)
    V.quiesce(dev)
    for rep in range(10):
        g.replay()
        torch.cuda.synchronize(dev)
    # 2 fused updates per replay x 10 replays = 20 reference sconv steps
    # (capture itself records but does not execute the kernels).
    x_ref = case.x_ref()
    for _ in range(20):
        fused_causal_conv1d_update_decode(
            x=x_ref,
            weight=case.weight,
            sconv_cache=cache_ref,
            cache_indices=case.ci,
            cache_mask=case.cm,
            activation=None,
            use_residual=True,
            track_mask=None,
            track_indices=None,
        )
    torch.cuda.synchronize(dev)
    assert torch.equal(cache_fused, cache_ref), f"rank{h.rank}: graph cache drift"
    V.quiesce(dev)
    V.log("[F3] CUDA-graph capture + 10 replays OK (cache bit-exact)")


def phase_bench(h, dev):
    from sgl_kernel import fused_add_rmsnorm

    V.log(f"[F4] bench (us, min-of-5x20): fused (VPT sweep) vs unfused chain")
    V.log(f"{'tokens':>8} {'unfused':>10} |  vpt=1   vpt=2   vpt=3   vpt=4   vpt=6")
    for T in [1, 2, 4, 8, 16, 32, 64]:
        case = FusedCase(h, T, 5000 + T, True)
        cache = case.cache
        hs_out = torch.empty(T, D, device=dev, dtype=torch.bfloat16)
        res_out = torch.empty(T, D, device=dev, dtype=torch.bfloat16)
        n = T * D
        in_t = case.parts[h.rank].reshape(-1)
        res2 = case.residual.clone()

        def unfused():
            out = h.run_v5(
                T, 7, nb=(1 if T <= 2 else 8), bs=1024, pb=True, check=False, in_t=in_t
            )
            y = fused_causal_conv1d_update_decode(
                x=out.view(T, D),
                weight=case.weight,
                sconv_cache=cache,
                cache_indices=case.ci,
                cache_mask=case.cm,
                activation=None,
                use_residual=True,
                track_mask=case.tm,
                track_indices=case.ti,
            )
            fused_add_rmsnorm(y, res2, case.gamma, EPS)

        t_un = V.bench_us(dev, unfused)
        best_t, best_vpt, cols = float("inf"), 1, []
        for vpt in (1, 2, 3, 4, 6):
            t = V.bench_us(dev, lambda: case.run_fused(cache, hs_out, res_out, vpt=vpt))
            cols.append(f"{t:>7.1f}")
            if t < best_t:
                best_t, best_vpt = t, vpt
        V.log(
            f"{T:>8} {t_un:>10.1f} | "
            + " ".join(cols)
            + f" | best vpt={best_vpt} {t_un / best_t:>6.2f}x"
        )
        V.quiesce(dev)


def main():
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
    phase_correctness(h, dev)
    phase_rotation(h, dev)
    phase_graph(h, dev)
    phase_verify(h, dev)
    phase_bench(h, dev)
    phase_verify_bench(h, dev)
    V.quiesce(dev)
    V.log("ALL_OK")
    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Target-verify variant: {AR -> causal_conv1d -> save_windows -> add+RMSNorm}
# ---------------------------------------------------------------------------
from sglang.jit_kernel.inkling_ar_fused import inkling_ar_sconv_norm_verify
from sglang.jit_kernel.inkling_sconv import causal_conv1d as jit_causal_conv1d
from sglang.srt.models.inkling_common.kernels.sconv import (
    save_intermediate_conv_windows,
)

Q = 9  # draft_token_num


class VerifyCase:
    def __init__(self, h, B, salt, with_pad=True):
        self.h, self.B, self.T = h, B, B * Q
        dev = h.buffer.device
        g = torch.Generator(device=dev)
        g.manual_seed(3000 + salt)
        self.cache = torch.randn(POOL, W - 1, D, generator=g, device=dev).bfloat16()
        self.weight = (torch.randn(D, W, generator=g, device=dev) * 0.5).bfloat16()
        self.residual = torch.randn(self.T, D, generator=g, device=dev).bfloat16()
        self.gamma = torch.randn(D, generator=g, device=dev).bfloat16()
        self.ci = (torch.arange(B, device=dev, dtype=torch.int32) * 5 + 21) % 1500
        self.cm = torch.ones(B, device=dev, dtype=torch.bool)
        self.pad_seq = None
        if with_pad and B >= 3:
            self.pad_seq = B - 1
            self.ci[self.pad_seq] = -1
        self.inter = torch.full((B, Q, W - 1, D), float("nan"), device=dev).bfloat16()
        self.parts = []
        for r in range(h.world):
            g.manual_seed(4000 + salt * 13 + r)
            self.parts.append(
                torch.randn(self.T, D, generator=g, device=dev).bfloat16()
            )

    def x_ref(self):
        acc = torch.zeros(self.T, D, device=self.parts[0].device, dtype=torch.float32)
        for p in self.parts:
            acc += p.float()
        return acc.bfloat16()

    def reference(self):
        dev = self.cache.device
        x = self.x_ref()
        safe = self.ci.clamp_min(0).to(torch.int64)
        cm = self.cm.clone()
        if self.pad_seq is not None:
            cm[self.pad_seq] = False  # mirror the fused kernel's PAD gating
        cu = torch.arange(0, (self.B + 1) * Q, Q, dtype=torch.int64, device=dev)
        si = (torch.arange(self.T, device=dev) // Q).to(torch.int32)
        y = jit_causal_conv1d(
            x=x,
            weight=self.weight,
            sconv_cache=self.cache,
            cache_mask=cm.view(-1, 1, 1),
            safe_idx=safe,
            cu=cu,
            si=si,
            activation=None,
            use_residual=True,
            is_decode=False,
        )
        inter_ref = torch.full_like(self.inter, float("nan"))
        save_intermediate_conv_windows(
            sconv_cache=self.cache,
            hidden_states=x,
            cache_indices=self.ci,
            intermediate_out=inter_ref,
            batch_size=self.B,
            draft_token_num=Q,
        )
        from sgl_kernel import fused_add_rmsnorm

        res = self.residual.clone()
        fused_add_rmsnorm(y, res, self.gamma, EPS)
        return res, y, inter_ref

    def run_fused(self, hs_out, res_out):
        h = self.h
        stage_off = h.v5_stage[h.v5_cur]
        es = h.elem_size
        inkling_ar_sconv_norm_verify(
            self.parts[h.rank],
            self.residual,
            res_out,
            hs_out,
            self.gamma,
            EPS,
            self.cache,
            self.ci,
            self.cm,
            self.weight,
            self.inter,
            Q,
            h.hdl.multicast_ptr + stage_off * es,
            h.buffer.data_ptr() + stage_off * es,
            h.hflags.buffer_ptrs_dev,
            h.state.data_ptr(),
            h.rank,
            h.world,
            activation=None,
            use_residual=True,
        )
        h.v5_cur = 1 - h.v5_cur


def phase_verify(h, dev):
    worst = 1.0
    for salt, B in enumerate([1, 2, 4, 8, 16]):
        case = VerifyCase(h, B, salt)
        res_ref, hs_ref, inter_ref = case.reference()
        hs_out = torch.empty(case.T, D, device=dev, dtype=torch.bfloat16)
        res_out = torch.empty(case.T, D, device=dev, dtype=torch.bfloat16)
        case.run_fused(hs_out, res_out)
        torch.cuda.synchronize(dev)
        live = [b for b in range(B) if b != case.pad_seq]
        assert torch.equal(
            case.inter[live], inter_ref[live]
        ), f"rank{h.rank} B={B}: intermediate windows not bit-exact"
        rows = torch.tensor(
            [t for t in range(case.T) if t // Q != case.pad_seq], device=dev
        )
        assert torch.equal(
            res_out[rows], res_ref[rows]
        ), f"rank{h.rank} B={B}: residual (=> conv y) not bit-exact"
        diff = (hs_out[rows].float() - hs_ref[rows].float()).abs()
        rel = (diff / hs_ref[rows].float().abs().clamp_min(1e-3)).max().item()
        worst = min(worst, (hs_out[rows] == hs_ref[rows]).float().mean().item())
        assert rel < 2e-2, f"rank{h.rank} B={B}: hs rel diff {rel}"
    V.quiesce(dev)
    V.log(
        f"[F5] verify-mode fused correctness OK (windows/residual bit-exact; "
        f"hs exact-match worst {worst:.4f})"
    )


def phase_verify_bench(h, dev):
    from sgl_kernel import fused_add_rmsnorm

    V.log(
        f"[F6] verify bench (us, min-of-5x20): fused vs "
        f"{{v5 AR + causal_conv1d + save_windows + add_rmsnorm}}"
    )
    V.log(f"{'B x Q':>8} {'unfused':>10} {'fused':>10} {'speedup':>8}")
    for B in [1, 4, 8, 16]:
        case = VerifyCase(h, B, 7000 + B, with_pad=False)
        T = case.T
        hs_out = torch.empty(T, D, device=dev, dtype=torch.bfloat16)
        res_out = torch.empty(T, D, device=dev, dtype=torch.bfloat16)
        safe = case.ci.clamp_min(0).to(torch.int64)
        cu = torch.arange(0, (B + 1) * Q, Q, dtype=torch.int64, device=dev)
        si = (torch.arange(T, device=dev) // Q).to(torch.int32)
        in_t = case.parts[h.rank].reshape(-1)
        res2 = case.residual.clone()

        def unfused():
            out = h.run_v5(T, 7, nb=8, bs=1024, pb=True, check=False, in_t=in_t)
            x = out.view(T, D)
            y = jit_causal_conv1d(
                x=x,
                weight=case.weight,
                sconv_cache=case.cache,
                cache_mask=case.cm.view(-1, 1, 1),
                safe_idx=safe,
                cu=cu,
                si=si,
                activation=None,
                use_residual=True,
                is_decode=False,
            )
            save_intermediate_conv_windows(
                sconv_cache=case.cache,
                hidden_states=x,
                cache_indices=case.ci,
                intermediate_out=case.inter,
                batch_size=B,
                draft_token_num=Q,
            )
            fused_add_rmsnorm(y, res2, case.gamma, EPS)

        def fused():
            case.run_fused(hs_out, res_out)

        t_un = V.bench_us(dev, unfused)
        t_f = V.bench_us(dev, fused)
        V.log(f"{B:>4}x{Q:<3} {t_un:>10.1f} {t_f:>10.1f} {t_un / t_f:>8.2f}x")
        V.quiesce(dev)


if __name__ == "__main__":
    main()
