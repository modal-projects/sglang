import torch
from sglang.srt.layers import layernorm as _ln
from sglang.srt.layers.rotary_embedding.factory import get_rope
from sglang.srt.mem_cache.memory_pool import _set_kv_buffer_prefix_valid_impl
from sglang.kernels.ops.speculative.k3_dflash_kv_store import rope_store_prefix_valid
from sglang.kernels.ops.speculative.k3_dflash_post import mamba_track_steps, mamba_track_steps_reference, verify_argmax

torch.manual_seed(0)
dev = "cuda"
HD, L = 128, 8
rope = get_rope(HD, HD, 262144, 2000000, True, dtype=torch.bfloat16).to(dev)
print("rope", type(rope).__name__, "fallback", getattr(rope, "use_fallback_kernel", None))
norm = _ln.RMSNorm(HD, eps=1e-5).to(dev).to(torch.bfloat16)
norm.weight.data.copy_(torch.rand(HD, device=dev) * 2)
SLOTS = 50000
tot = 0
for bs in (1, 3, 8, 16):
    T = bs * L
    qkv = (torch.randn(T, 768, device=dev) * 3).to(torch.bfloat16)
    k = qkv[:, 512:640]
    v = qkv[:, 640:768]
    pos = torch.randint(0, 200000, (bs,), device=dev)[:, None] + torch.arange(L, device=dev)[None]
    pos = pos.reshape(-1).to(torch.int64)
    loc = torch.randperm(SLOTS, device=dev)[:T].view(bs, L).to(torch.int64)
    commit = torch.randint(1, L + 1, (bs,), device=dev, dtype=torch.int32)
    kb0 = torch.zeros(SLOTS, 1, HD, device=dev, dtype=torch.bfloat16); vb0 = kb0.clone()
    kb1 = kb0.clone(); vb1 = kb0.clone()
    # baseline (as DFlashAttention.apply_k_norm / apply_k_rope / set_kv_buffer_prefix_valid)
    kn = norm(k.reshape(-1, HD)).view_as(k)
    kn_saved = kn.clone()
    dq = kn.new_empty(kn.shape)
    _, kr = rope(pos, dq, kn)
    _set_kv_buffer_prefix_valid_impl(kr.view(-1, 1, HD).contiguous(), v.view(-1, 1, HD).contiguous(), kb0, vb0, loc, commit, row_dim=HD, store_dtype=torch.bfloat16)
    # fused
    kn2 = _ln.rms_norm(k, norm.weight.data, norm.variance_epsilon)
    if not torch.equal(kn2.view(torch.int16), kn_saved.reshape(-1, HD).view(torch.int16)):
        kn3 = _ln.rms_norm(k.contiguous(), norm.weight.data, norm.variance_epsilon)
        print("DIFF", norm._forward_method, (kn2.float()-kn.reshape(-1,HD).float()).abs().max().item(), torch.equal(kn3, kn.reshape(-1,HD)), torch.equal(kn3,kn2))
        raise SystemExit(1)
    cs = rope._hip_cos_sin_cache_as(torch.bfloat16)
    rope_store_prefix_valid(kn2.view(-1, 1, HD), v.view(-1, 1, HD), pos, cs, loc, commit, kb1, vb1)
    eqk = torch.equal(kb0.view(torch.int16), kb1.view(torch.int16))
    eqv = torch.equal(vb0.view(torch.int16), vb1.view(torch.int16))
    nd = (kb0.view(torch.int16) != kb1.view(torch.int16)).sum().item()
    print(f"bs={bs} K bitexact={eqk} (ndiff={nd}) V bitexact={eqv}")
    assert eqk and eqv
print("KV fuse bit-exact")

# mamba track
for trial in range(50):
    n = [1, 4, 8, 16, 33][trial % 5]
    commit = torch.randint(1, 9, (n,), device=dev, dtype=torch.int32)
    pre = torch.randint(0, 100000, (n,), device=dev, dtype=torch.int64)
    if trial % 4 == 0:
        pre[0] = -5
    post = pre + commit.to(torch.int64)
    for interval in (256, 64, 1):
        for has in (True, False):
            a = mamba_track_steps(commit, pre, post, interval, has)
            b = mamba_track_steps_reference(commit, pre, post, interval, has)
            assert torch.equal(a[0], b[0])
            if has:
                assert torch.equal(a[1], b[1]), (trial, interval, a[1], b[1], pre, post)
print("mamba track fused == reference")

# verify argmax
import time
for rows in (8, 32, 64, 128):
    for trial in range(10):
        x = torch.randn(rows, 163840, device=dev)
        if trial % 3 == 1:
            x = x.round()
        if trial % 3 == 2:
            x[0, 1000] = float("nan"); x[min(1, rows - 1), 5] = float("nan"); x[min(1, rows - 1), 3] = float("nan")
        assert torch.equal(verify_argmax(x), torch.argmax(x, dim=-1)), (rows, trial)
    def bench(fn, it=200):
        for _ in range(5): fn()
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(it): fn()
        torch.cuda.synchronize(); return (time.perf_counter() - t) / it * 1e6
    print(f"argmax rows={rows}: torch {bench(lambda: torch.argmax(x, dim=-1)):.1f}us fused {bench(lambda: verify_argmax(x)):.1f}us")
print("verify argmax == torch.argmax")

# compact lens
from sglang.kernels.ops.speculative.k3_dflash_post import compact_draft_lens
def ref_compact(seq, W, page):
    vis = torch.clamp(seq.to(torch.int32), max=W)
    if page <= 1:
        d = vis
    else:
        s64 = seq.to(torch.int64); vs = s64 - vis.to(torch.int64)
        al = vs - torch.remainder(vs, page)
        d = (s64 - al).to(torch.int32)
    return d, seq.to(torch.int64) - d.to(torch.int64)
for trial in range(100):
    n = [1, 3, 8, 16, 64][trial % 5]
    seq = torch.randint(0, 300000, (n,), device=dev, dtype=torch.int64)
    seq[0] = [0, 1, 4095, 4096, 4097, 4159, 4160, 4161, 63, 64][trial % 10]
    for W, page in ((4096, 64), (4096, 1), (100, 16)):
        a = compact_draft_lens(seq, W, page); b = ref_compact(seq, W, page)
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]), (seq, W, page)
print("compact lens == reference")

# fused verify metadata
from sglang.kernels.ops.attention.k3_draft_verify_metadata import fill_draft_verify_metadata
R2T = torch.randint(0, 10**6, (70, 9000), device=dev, dtype=torch.int32)
for bs in (1, 2, 5, 8, 16, 64):
    for W in (4096, 0):
        seq = torch.randint(1, 8900, (bs,), device=dev, dtype=torch.int32)
        seq[0] = 4160 if bs > 1 else 4100
        rpi = torch.randperm(70, device=dev)[:bs].to(torch.int64)
        L = 8
        qo = torch.full((bs + 2,), -7, dtype=torch.int64, device=dev); kvp = torch.full((bs + 2,), -7, dtype=torch.int32, device=dev)
        kvi = torch.full((bs * 9000,), -7, dtype=torch.int64, device=dev); mip = torch.full((bs + 2,), -7, dtype=torch.int64, device=dev)
        wp = torch.full((bs + 2,), -7, dtype=torch.int32, device=dev); wi = torch.full((bs * 9000,), -7, dtype=torch.int64, device=dev); wo = torch.full((bs + 1,), -7, dtype=torch.int32, device=dev)
        fill_draft_verify_metadata(bs=bs, seq_lens=seq, req_pool_indices=rpi, req_to_token=R2T, num_tokens_per_req=L, qo_indptr=qo, kv_indptr=kvp, kv_indices=kvi, mask_indptr=mip,
                                   window_size=W or None, window_kv_indptr=wp, window_kv_indices=wi, window_kv_offsets=wo)
        s64 = seq.to(torch.int64)
        assert torch.equal(qo[: bs + 1], torch.arange(0, (bs + 1) * L, L, device=dev))
        ref_kvp = torch.zeros(bs + 1, dtype=torch.int64, device=dev); ref_kvp[1:] = torch.cumsum(s64, 0)
        assert torch.equal(kvp[: bs + 1].to(torch.int64), ref_kvp)
        ref_kvi = torch.cat([R2T[rpi[b], : seq[b]].to(torch.int64) for b in range(bs)])
        assert torch.equal(kvi[: ref_kvi.numel()], ref_kvi) and int(kvi[ref_kvi.numel()]) == -7
        ref_mip = torch.zeros(bs + 1, dtype=torch.int64, device=dev); ref_mip[1:] = torch.cumsum(L * (s64 + L), 0)
        assert torch.equal(mip[: bs + 1], ref_mip)
        if W:
            wl = torch.clamp(s64, max=W); off = s64 - wl
            ref_wp = torch.zeros(bs + 1, dtype=torch.int64, device=dev); ref_wp[1:] = torch.cumsum(wl, 0)
            assert torch.equal(wp[: bs + 1].to(torch.int64), ref_wp)
            assert torch.equal(wo[:bs].to(torch.int64), off)
            ref_wi = torch.cat([R2T[rpi[b], off[b]: off[b] + wl[b]].to(torch.int64) for b in range(bs)])
            assert torch.equal(wi[: ref_wi.numel()], ref_wi)
print("fused verify metadata == reference")
