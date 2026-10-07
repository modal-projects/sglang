import sys, time, torch
from sglang.kernels.ops.attention.extend_attention import extend_attention_fwd
from sglang.kernels.ops.attention.k3_draft_swa_verify import draft_swa_verify_fwd, can_handle

torch.manual_seed(0)
dev = "cuda"
HQ, HKV, D, L, W = 4, 1, 128, 8, 4096
SLOTS = 200000
kbuf = torch.randn(SLOTS, HKV, D, device=dev, dtype=torch.bfloat16)
vbuf = torch.randn(SLOTS, HKV, D, device=dev, dtype=torch.bfloat16)


def make(bs, seqs):
    T = bs * L
    qkv = torch.randn(T, (HQ + 2 * HKV) * D, device=dev, dtype=torch.bfloat16)
    q, k, v = qkv.split([HQ * D, HKV * D, HKV * D], dim=-1)
    q = q.view(T, HQ, D); k = k.view(T, HKV, D); v = v.view(T, HKV, D)
    P = [min(s, W) for s in seqs]
    kv_indptr = torch.zeros(bs + 1, dtype=torch.int32, device=dev)
    kv_indptr[1:] = torch.cumsum(torch.tensor(P, device=dev), 0)
    kv_indices = torch.randperm(SLOTS, device=dev)[: int(sum(P))].to(torch.int64)
    qo_indptr = torch.arange(0, (bs + 1) * L, L, dtype=torch.int32, device=dev)
    return q, k, v, qo_indptr, kv_indptr, kv_indices


def ref(q, k, v, qo, kvp, kvi, sw):
    o = torch.empty(q.shape[0], HQ, D, device=dev, dtype=q.dtype)
    extend_attention_fwd(q, k.contiguous(), v.contiguous(), o, kbuf, vbuf, qo, kvp, kvi, None, True, None, L, 1.0, 1.0, D ** -0.5, sliding_window_size=sw, page_size=64)
    return o


def new(q, k, v, qo, kvp, kvi, sw, ns=None):
    o = torch.empty(q.shape[0], HQ, D, device=dev, dtype=q.dtype)
    draft_swa_verify_fwd(q, k, v, o, kbuf, vbuf, qo, kvp, kvi, L, D ** -0.5, sw, 64, n_splits=ns)
    return o


def bench(fn, iters=50):
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn(); fn()
    torch.cuda.current_stream().wait_stream(s)
    with torch.cuda.graph(g):
        for _ in range(10):
            fn()
    g.replay(); torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        g.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters / 10 * 1e6


for bs, seqs in [(1, [4096]), (1, [90000]), (1, [100]), (1, [4100]), (3, [5000, 64, 4096]), (4, [9000] * 4), (8, [3000, 9000, 4096, 4095, 1, 70000, 4097, 20000]), (16, [6000] * 16)]:
    q, k, v, qo, kvp, kvi = make(bs, seqs)
    assert can_handle(q, k, v, kbuf, vbuf, qo, L, causal=True, sinks=None, logit_cap=0.0, xai_temperature_len=-1, score_mod=None)
    for sw in (W,):
        a = ref(q, k, v, qo, kvp, kvi, sw).float()
        b = new(q, k, v, qo, kvp, kvi, sw).float()
        err = (a - b).abs().max().item()
        rel = ((a - b).norm() / a.norm()).item()
        ta = bench(lambda: ref(q, k, v, qo, kvp, kvi, sw))
        tb = bench(lambda: new(q, k, v, qo, kvp, kvi, sw))
        print(f"bs={bs} seqs={seqs[:4]} sw={sw} maxerr={err:.2e} rel={rel:.2e} ref_us={ta:.1f} (incl 2 copies) new_us={tb:.1f}", flush=True)
        assert err < 2e-2, err
# split sweep at bs1/4/8 window
for bs in (1, 4, 8):
    q, k, v, qo, kvp, kvi = make(bs, [9000] * bs)
    for ns in (8, 16, 32, 64):
        print(f"  bs={bs} n_splits={ns} us={bench(lambda: new(q, k, v, qo, kvp, kvi, W, ns)):.1f}")
print("OK")
