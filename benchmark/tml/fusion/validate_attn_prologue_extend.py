"""Correctness + graph-replay bench for the EXTEND fused attention prologue:
{k/v varlen sconv + per-head q/k RMSNorm + KV store} in the main kernel plus
the tiny trailing {k/v conv-cache update (+ prefix-cache track)} kernel
(``inkling_attn_prologue_extend``). Single GPU, no communication.

Reference = the unfused production chain: 2x ``causal_conv1d`` +
``fused_inplace_qknorm`` + 2x ``update_sconv_cache`` + the indexed KV store.
Integer-valued inputs make the conv exact in both pipelines, so conv outputs,
cache updates, stores and track copies are compared bit-exact; the q/k norms
are compared with a tight allclose (both sides reduce in fp32, but reduction
shapes differ).

Run:
    python benchmark/tml/fusion/validate_attn_prologue_extend.py           # correctness
    python benchmark/tml/fusion/validate_attn_prologue_extend.py --bench   # + graph-replay bench
"""

import sys

import torch

from sglang.jit_kernel.inkling_attn_prologue import inkling_attn_prologue_extend
from sglang.jit_kernel.norm import fused_inplace_qknorm
from sglang.srt.models.inkling_common.kernels.sconv import (
    _seq_idx_from_cu_seqlens,
    causal_conv1d,
    update_sconv_cache,
)

HEAD = 128
W = 4
POOL = 512  # conv-cache slots
KV_POOL = 20000  # KV-cache slots


class Case:
    def __init__(self, T, qsl, has_init, cache_idx, hq, hkv, salt, track=None):
        dev = torch.device("cuda")
        self.T, self.hq, self.hkv = T, hq, hkv
        self.dq, self.dkv = hq * HEAD, hkv * HEAD
        g = torch.Generator(device=dev)
        g.manual_seed(4200 + salt)
        # q | k | v | tail(r stand-in); integer-ish values, exact in bf16.
        row = self.dq + 2 * self.dkv + 2 * HEAD
        self.q_off, self.k_off, self.v_off = 0, self.dq, self.dq + self.dkv
        self.qkvr = (
            torch.randint(-8, 9, (T, row), generator=g, device=dev) * 0.5
        ).bfloat16()
        self.kw = (
            torch.randint(-8, 9, (self.dkv, W), generator=g, device=dev) * 0.25
        ).bfloat16()
        self.vw = (
            torch.randint(-8, 9, (self.dkv, W), generator=g, device=dev) * 0.25
        ).bfloat16()
        self.k_cache = (
            torch.randint(-16, 17, (POOL, W - 1, self.dkv), generator=g, device=dev)
            * 0.5
        ).bfloat16()
        self.v_cache = (
            torch.randint(-16, 17, (POOL, W - 1, self.dkv), generator=g, device=dev)
            * 0.5
        ).bfloat16()
        self.qg = (
            torch.randint(-4, 5, (HEAD,), generator=g, device=dev) * 0.25 + 1.0
        ).bfloat16()
        self.kg = (
            torch.randint(-4, 5, (HEAD,), generator=g, device=dev) * 0.25 + 1.0
        ).bfloat16()
        self.eps = 1e-6
        self.qsl = torch.tensor(qsl, device=dev, dtype=torch.int32)
        b = len(qsl) - 1
        self.has_init = torch.tensor(has_init, device=dev, dtype=torch.bool)
        self.ci = torch.tensor(cache_idx, device=dev, dtype=torch.int32)
        assert self.has_init.numel() == b and self.ci.numel() == b
        self.cache_mask = self.has_init & (self.ci != -1)
        self.cu = self.qsl.to(torch.int64)
        self.si = _seq_idx_from_cu_seqlens(self.cu, T)
        # KV store locations: unique slots, with a few -1 sentinels (the SWA
        # full->swa translation emits those; the store must skip them).
        loc = torch.randperm(KV_POOL, generator=g, device=dev)[:T].to(torch.int64)
        loc[:: max(1, T // 7)] = -1
        self.loc = loc
        self.k_buf = torch.zeros(KV_POOL, hkv, HEAD, dtype=torch.bfloat16, device=dev)
        self.v_buf = torch.zeros(KV_POOL, hkv, HEAD, dtype=torch.bfloat16, device=dev)
        if track is None:
            self.trows = torch.empty((0, W - 1), dtype=torch.int64, device=dev)
            self.tmask = torch.empty((0,), dtype=torch.bool, device=dev)
            self.tdst = torch.empty((0,), dtype=torch.int64, device=dev)
        else:
            self.trows, self.tmask, self.tdst = track

    def slices(self):
        q = self.qkvr[:, self.q_off : self.q_off + self.dq]
        k = self.qkvr[:, self.k_off : self.k_off + self.dkv]
        v = self.qkvr[:, self.v_off : self.v_off + self.dkv]
        return q, k, v

    def reference(self):
        """Unfused production chain on fresh cache/buffer clones."""
        q, k, v = self.slices()
        kc, vc = self.k_cache.clone(), self.v_cache.clone()
        kx, vx = k.contiguous(), v.contiguous()
        args = dict(
            cache_mask=self.cache_mask[:, None, None],
            safe_idx=self.ci.clamp(min=0).long(),
            cu=self.cu,
            si=self.si,
            activation="silu",
            use_residual=True,
        )
        k_conv = causal_conv1d(x=kx, weight=self.kw, sconv_cache=kc, **args)
        v_conv = causal_conv1d(x=vx, weight=self.vw, sconv_cache=vc, **args)
        update_sconv_cache(
            x=kx,
            sconv_cache=kc,
            cache_indices=self.ci,
            has_initial_state=self.has_init,
            query_start_loc=self.qsl,
        )
        update_sconv_cache(
            x=vx,
            sconv_cache=vc,
            cache_indices=self.ci,
            has_initial_state=self.has_init,
            query_start_loc=self.qsl,
        )
        if self.tmask.numel():
            for b in range(self.tmask.numel()):
                if self.tmask[b]:
                    for w in range(W - 1):
                        kc[self.tdst[b], w] = kx[self.trows[b, w]]
                        vc[self.tdst[b], w] = vx[self.trows[b, w]]
        q_ref, k_ref = q.contiguous(), k_conv.clone()
        fused_inplace_qknorm(
            q_ref.view(self.T, -1, HEAD),
            k_ref.view(self.T, -1, HEAD),
            self.qg,
            self.kg,
            self.eps,
        )
        kb_ref, vb_ref = torch.zeros_like(self.k_buf), torch.zeros_like(self.v_buf)
        valid = self.loc >= 0
        kb_ref[self.loc[valid]] = k_ref[valid].view(-1, self.hkv, HEAD)
        vb_ref[self.loc[valid]] = v_conv[valid].view(-1, self.hkv, HEAD)
        return q_ref, k_ref, v_conv, kc, vc, kb_ref, vb_ref

    def run_fused(self, do_store=True):
        kc, vc = self.k_cache.clone(), self.v_cache.clone()
        self.k_buf.zero_()
        self.v_buf.zero_()
        q_out, k_out, v_out, _ = inkling_attn_prologue_extend(
            self.qkvr,
            kc,
            vc,
            self.ci,
            self.cache_mask,
            self.has_init,
            self.cu,
            self.si,
            self.kw,
            self.vw,
            self.trows,
            self.tmask,
            self.tdst,
            self.qg,
            self.kg,
            self.eps,
            self.loc,
            self.k_buf,
            self.v_buf,
            self.q_off,
            self.k_off,
            self.v_off,
            self.dq,
            self.dkv,
            activation="silu",
            use_residual=True,
            do_store=do_store,
        )
        return q_out, k_out, v_out, kc, vc


def check(tag, case):
    q_ref, k_ref, v_ref, kc_ref, vc_ref, kb_ref, vb_ref = case.reference()
    q_out, k_out, v_out, kc, vc = case.run_fused()
    # Store check = internal consistency (stored rows == emitted rows); the
    # norm ACCURACY is covered by the q/k allclose below (the fused norm and
    # the jit qknorm kernel can differ by 1 bf16 ulp on rounding boundaries,
    # which torch.equal against the reference store would spuriously trip).
    kb_int = torch.zeros_like(case.k_buf)
    vb_int = torch.zeros_like(case.v_buf)
    valid = case.loc >= 0
    kb_int[case.loc[valid]] = k_out[valid].view(-1, case.hkv, HEAD)
    vb_int[case.loc[valid]] = v_out[valid].view(-1, case.hkv, HEAD)
    del kb_ref, vb_ref
    checks = {
        "v": torch.equal(v_out, v_ref),  # pure conv: exact with integer data
        "kcache": torch.equal(kc, kc_ref),
        "vcache": torch.equal(vc, vc_ref),
        "kbuf": torch.equal(case.k_buf, kb_int),
        "vbuf": torch.equal(case.v_buf, vb_int),
        # fp32 reductions with different shapes -> tight allclose.
        "q": torch.allclose(q_out.float(), q_ref.float(), rtol=2e-2, atol=2e-2),
        "k": torch.allclose(k_out.float(), k_ref.float(), rtol=2e-2, atol=2e-2),
    }
    if all(checks.values()):
        qmax = (q_out.float() - q_ref.float()).abs().max().item()
        kmax = (k_out.float() - k_ref.float()).abs().max().item()
        print(f"  ok {tag} (norm max diff q={qmax:.2e} k={kmax:.2e})")
        return True
    bad = [name for name, ok in checks.items() if not ok]
    print(f"  FAIL {tag}: {bad}")
    return False


def bench_graph(fn, iters=200):
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    for _ in range(10):
        g.replay()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        g.replay()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000.0 / iters  # us


def main():
    torch.cuda.set_device(0)
    fails = 0

    print("attn prologue EXTEND correctness (hq=16 hkv=2, TP4 shapes):")
    cases = [
        ("ext1 T=17", 17, [0, 17], [True], [7]),
        ("ext1 T=250", 250, [0, 250], [True], [7]),
        ("fresh T=40", 40, [0, 40], [False], [3]),
        ("pad T=40", 40, [0, 40], [True], [-1]),
        ("short T=2", 2, [0, 2], [True], [5]),  # qlen < W-1 shift path
        (
            "multi4 T=301",
            301,
            [0, 5, 21, 150, 301],
            [True, False, True, True],
            [1, 2, -1, 9],
        ),
        ("multi2 T=4096", 4096, [0, 1500, 4096], [True, True], [11, 12]),
        (
            "decode-shaped T=64",
            64,
            list(range(65)),
            [True] * 64,
            [(i * 3 + 1) % POOL for i in range(64)],
        ),
    ]
    for tag, T, qsl, hi, ci in cases:
        fails += 0 if check(tag, Case(T, qsl, hi, ci, 16, 2, salt=T)) else 1
    # TP8 shapes (hq=8, hkv=1: the k/v roles split a warp).
    print("attn prologue EXTEND correctness (hq=8 hkv=1, TP8 shapes):")
    for tag, T, qsl, hi, ci in cases[:6]:
        fails += 0 if check(tag, Case(T, qsl, hi, ci, 8, 1, salt=900 + T)) else 1
    # prefix-cache track
    dev = torch.device("cuda")
    trows = torch.tensor(
        [[0, 1, 2], [64, 65, 66], [200, 201, 202]], dtype=torch.int64, device=dev
    )
    tmask = torch.tensor([True, False, True], dtype=torch.bool, device=dev)
    tdst = torch.tensor([100, 101, 102], dtype=torch.int64, device=dev)
    c = Case(
        301,
        [0, 5, 150, 301],
        [True, True, True],
        [1, 2, 9],
        16,
        2,
        salt=77,
        track=(trows, tmask, tdst),
    )
    fails += 0 if check("track T=301", c) else 1

    print(f"RESULT: {'PASS' if fails == 0 else f'FAIL ({fails})'}")
    if fails or "--bench" not in sys.argv:
        return 1 if fails else 0

    print("graph-replay bench, fused vs unfused chain (us):")
    print("T | hq/hkv | unfused | fused | delta")
    for hq, hkv in ((16, 2), (8, 1)):
        for T in (512, 1024, 2048, 4096, 8192, 16384):
            c = Case(T, [0, T], [True], [7], hq, hkv, salt=5000 + T)
            q, k, v = c.slices()
            kc, vc = c.k_cache.clone(), c.v_cache.clone()
            loc = c.loc.clamp(min=0)  # bench the store on all-valid locs
            args = dict(
                cache_mask=c.cache_mask[:, None, None],
                safe_idx=c.ci.clamp(min=0).long(),
                cu=c.cu,
                si=c.si,
                activation="silu",
                use_residual=True,
            )

            def unfused():
                kx, vx = k.contiguous(), v.contiguous()
                k_conv = causal_conv1d(x=kx, weight=c.kw, sconv_cache=kc, **args)
                v_conv = causal_conv1d(x=vx, weight=c.vw, sconv_cache=vc, **args)
                update_sconv_cache(
                    x=kx,
                    sconv_cache=kc,
                    cache_indices=c.ci,
                    has_initial_state=c.has_init,
                    query_start_loc=c.qsl,
                )
                update_sconv_cache(
                    x=vx,
                    sconv_cache=vc,
                    cache_indices=c.ci,
                    has_initial_state=c.has_init,
                    query_start_loc=c.qsl,
                )
                qn = q.contiguous()
                fused_inplace_qknorm(
                    qn.view(T, -1, HEAD), k_conv.view(T, -1, HEAD), c.qg, c.kg, c.eps
                )
                c.k_buf[loc] = k_conv.view(-1, hkv, HEAD)
                c.v_buf[loc] = v_conv.view(-1, hkv, HEAD)

            def fused():
                inkling_attn_prologue_extend(
                    c.qkvr,
                    kc,
                    vc,
                    c.ci,
                    c.cache_mask,
                    c.has_init,
                    c.cu,
                    c.si,
                    c.kw,
                    c.vw,
                    c.trows,
                    c.tmask,
                    c.tdst,
                    c.qg,
                    c.kg,
                    c.eps,
                    loc,
                    c.k_buf,
                    c.v_buf,
                    c.q_off,
                    c.k_off,
                    c.v_off,
                    c.dq,
                    c.dkv,
                    activation="silu",
                    use_residual=True,
                    do_store=True,
                )

            tu = bench_graph(unfused)
            tf = bench_graph(fused)
            print(
                f"BENCH {T} | {hq}/{hkv} | {tu:.1f} | {tf:.1f} | "
                f"{(tf - tu) / tu * 100:+.1f}%"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
