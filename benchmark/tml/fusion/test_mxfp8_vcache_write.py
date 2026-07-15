"""Correctness testbed for the legacy incremental token-blocked V-scale writer
(`update_mxfp8_v_cache_seqblocked`).

Current paged FA4 v_dequant uses BlockScaledBasicChunk SFV laid out like SFK:
per token, per head-dim block. This helper exercises the older seq-blocked
V-scale layout where sfv is shaped (num_pages, h_kv, head_dim, page_size // 32).

Property under test: incrementally appending tokens (decode 1-at-a-time,
extend chunks, whole blocks at once) reproduces offline whole-block
quantization — scale bytes exactly (the ceil-log2 scale rule is monotone, so
the running max equals the offline max), payload up to one subnormal
double-rounding (rescale-on-bump is an exact power-of-two shift for normal
values).

Also covered: stale-scale reset on page reuse (a block's first token has
slot % 32 == 0 and resets the ratchet), negative-loc padding entries, and an
FA4 QV integration A/B (attention over the incremental cache vs the offline
cache).
"""

import os

import torch

from sglang.srt.kernels.mxfp8_quant import (
    from_mxfp8,
    to_mxfp8,
    update_mxfp8_v_cache_seqblocked,
    MXFP8Tensor,
)

H_KV, HEAD_DIM, PAGE = 1, 128, 128
BLK = 32
DEV = "cuda"


def offline_quant(v_seq, num_pages):
    """v_seq (num_pages*PAGE, h, d) bf16 -> (cache fp8, sfv uint8) page-blocked."""
    v_pages = v_seq.view(num_pages, PAGE, H_KV, HEAD_DIM)
    q = to_mxfp8(v_pages.permute(0, 2, 3, 1).contiguous())  # quant along tokens
    cache = q.data.permute(0, 3, 1, 2).contiguous()
    sfv = q.scale.contiguous()  # (num_pages, h, d, PAGE//32) uint8
    return cache, sfv


def check_against_offline(name, cache, sfv, v_seq, written_slots, num_pages):
    """Scales bitwise; payload within one subnormal double-rounding."""
    cache_ref, sfv_ref = offline_quant(v_seq, num_pages)
    wmask = torch.zeros(num_pages * PAGE, dtype=torch.bool, device=DEV)
    wmask[written_slots] = True
    # only fully/partially written blocks have defined scales; untouched
    # blocks were zeroed at init on both sides via v_seq zeros
    bmask = wmask.view(num_pages, PAGE // BLK, BLK).any(-1)  # (pages, blocks)
    sfv_ok = (
        sfv.permute(0, 3, 1, 2)[bmask] == sfv_ref.permute(0, 3, 1, 2)[bmask]
    ).all()
    assert sfv_ok, f"{name}: scale bytes differ from offline quantization"

    got = from_mxfp8(
        MXFP8Tensor(
            data=cache.permute(0, 2, 3, 1).contiguous(),
            scale=sfv,
        ),
        out_dtype=torch.float32,
    ).permute(0, 3, 1, 2).reshape(num_pages * PAGE, H_KV, HEAD_DIM)
    want = from_mxfp8(
        MXFP8Tensor(
            data=cache_ref.permute(0, 2, 3, 1).contiguous(),
            scale=sfv_ref,
        ),
        out_dtype=torch.float32,
    ).permute(0, 3, 1, 2).reshape(num_pages * PAGE, H_KV, HEAD_DIM)
    # bound: one lsb of the e4m3 subnormal grid at the block scale
    scale_per_slot = (
        torch.exp2(sfv.float() - 127.0 - 8.0)
        .permute(0, 3, 1, 2)  # (pages, blocks, h, d)
        .repeat_interleave(BLK, dim=1)
        .reshape(num_pages * PAGE, H_KV, HEAD_DIM)
    )
    diff = (got - want).abs()[wmask]
    bound = scale_per_slot[wmask]
    nbad = int((diff > bound).sum())
    print(
        f"  {name}: scales bitwise OK, payload max_diff={diff.max().item():.3e} "
        f"(subnormal-bound violations: {nbad})"
    )
    assert nbad == 0, f"{name}: payload exceeds subnormal double-rounding bound"


def fill_garbage(cache, sfv, g):
    cache.view(torch.uint8).random_(0, 256, generator=g)
    sfv.random_(0, 256, generator=g)


def run_pattern(name, chunks, num_pages, garbage=False, pad_every=0):
    """chunks: list of (start_slot, length) writes, in order."""
    g = torch.Generator(device=DEV).manual_seed(sum(map(ord, name)))
    total = num_pages * PAGE
    v_seq = torch.zeros(total, H_KV, HEAD_DIM, device=DEV, dtype=torch.bfloat16)
    cache = torch.zeros(num_pages, PAGE, H_KV, HEAD_DIM, device=DEV, dtype=torch.float8_e4m3fn)
    sfv = torch.zeros(num_pages, H_KV, HEAD_DIM, PAGE // BLK, device=DEV, dtype=torch.uint8)
    if garbage:
        fill_garbage(cache, sfv, g)

    written = []
    for start, length in chunks:
        v = torch.randn(length, H_KV, HEAD_DIM, device=DEV, dtype=torch.bfloat16, generator=g)
        # per-token magnitude spread so scale bumps and subnormals both occur
        v *= torch.exp2(
            torch.randint(-6, 7, (length, 1, 1), device=DEV, generator=g).float()
        )
        loc = torch.arange(start, start + length, device=DEV, dtype=torch.int64)
        v_seq[start : start + length] = v
        written.extend(range(start, start + length))
        if pad_every:
            # interleave padding entries (negative loc) like a padded batch
            v_pad = torch.zeros(length + pad_every, H_KV, HEAD_DIM, device=DEV, dtype=torch.bfloat16)
            loc_pad = torch.full((length + pad_every,), -1, device=DEV, dtype=torch.int64)
            v_pad[:length] = v
            loc_pad[:length] = loc
            v, loc = v_pad, loc_pad
        update_mxfp8_v_cache_seqblocked(v, loc, cache, sfv)

    check_against_offline(name, cache, sfv, v_seq, torch.tensor(written, device=DEV), num_pages)
    return cache, sfv, v_seq


def multi_seq_decode(name, num_seqs, steps):
    """num_seqs sequences appending 1 token/step, each on its own pages."""
    g = torch.Generator(device=DEV).manual_seed(99)
    pages_per_seq = (steps + PAGE - 1) // PAGE
    num_pages = num_seqs * pages_per_seq
    total = num_pages * PAGE
    v_seq = torch.zeros(total, H_KV, HEAD_DIM, device=DEV, dtype=torch.bfloat16)
    cache = torch.zeros(num_pages, PAGE, H_KV, HEAD_DIM, device=DEV, dtype=torch.float8_e4m3fn)
    sfv = torch.zeros(num_pages, H_KV, HEAD_DIM, PAGE // BLK, device=DEV, dtype=torch.uint8)

    written = []
    for t in range(steps):
        v = torch.randn(num_seqs, H_KV, HEAD_DIM, device=DEV, dtype=torch.bfloat16, generator=g)
        v *= torch.exp2(
            torch.randint(-6, 7, (num_seqs, 1, 1), device=DEV, generator=g).float()
        )
        loc = torch.tensor(
            [s * pages_per_seq * PAGE + t for s in range(num_seqs)],
            device=DEV,
            dtype=torch.int64,
        )
        v_seq[loc] = v
        written.extend(loc.tolist())
        update_mxfp8_v_cache_seqblocked(v, loc, cache, sfv)

    check_against_offline(name, cache, sfv, v_seq, torch.tensor(written, device=DEV), num_pages)


def integration_fa4(cache, sfv, v_seq, num_pages, kv_len):
    """FA4 QV attention over the incremental cache vs the offline cache."""
    from sglang.jit_kernel.flash_attention_v4 import flash_attn_varlen_func

    os.environ["FA4_DEBUG_QSTAGE1"] = "1"
    g = torch.Generator(device=DEV).manual_seed(5)
    q = torch.randn(1, 8, HEAD_DIM, device=DEV, dtype=torch.bfloat16, generator=g)
    k = torch.randn(kv_len, H_KV, HEAD_DIM, device=DEV, dtype=torch.bfloat16, generator=g)
    kq = to_mxfp8(k)
    qq = to_mxfp8(q)
    k_pad = torch.zeros(num_pages * PAGE, H_KV, HEAD_DIM, device=DEV, dtype=torch.bfloat16)
    k_pad[:kv_len] = k
    kq_pad = to_mxfp8(k_pad)
    k_p = kq_pad.data.view(num_pages, PAGE, H_KV, HEAD_DIM)
    sfk = kq_pad.scale.view(torch.float8_e8m0fnu).view(-1, H_KV, HEAD_DIM // 32)
    page_table = torch.arange(num_pages, device=DEV, dtype=torch.int32).view(1, -1)
    common = dict(
        cu_seqlens_q=torch.tensor([0, 1], device=DEV, dtype=torch.int32),
        max_seqlen_q=1,
        seqused_k=torch.tensor([kv_len], device=DEV, dtype=torch.int32),
        page_table=page_table,
        softmax_scale=1.0 / HEAD_DIM,
        causal=True,
        num_splits=1,
        sfk=sfk,
        sfq=qq.scale.view(torch.float8_e8m0fnu),
    )
    out_inc = flash_attn_varlen_func(
        qq.data, k_p, cache, sfv=sfv.view(torch.float8_e8m0fnu), **common
    )
    cache_ref, sfv_ref = offline_quant(v_seq, num_pages)
    out_off = flash_attn_varlen_func(
        qq.data, k_p, cache_ref, sfv=sfv_ref.view(torch.float8_e8m0fnu), **common
    )
    diff = (out_inc.float() - out_off.float()).abs().max().item()
    print(f"  fa4 integration: incremental-vs-offline cache max_abs={diff:.3e}")
    assert diff <= 1e-2, "FA4 output over incremental cache diverges from offline cache"


def main():
    torch.manual_seed(0)
    print("decode 1-token appends (multi-seq, 3 pages/seq):")
    multi_seq_decode("decode x8 seqs x 300 steps", num_seqs=8, steps=300)

    print("extend chunks (contiguous, misaligned sizes):")
    chunks, pos = [], 0
    for length in [1, 31, 32, 33, 97, 5, 128, 60, 13]:
        chunks.append((pos, length))
        pos += length
    run_pattern("extend mixed chunks", chunks, num_pages=4)

    print("whole-block collisions in one call + padding entries:")
    run_pattern("block-at-once + padding", [(0, 256), (256, 64)], num_pages=3, pad_every=7)

    print("garbage-initialized cache/scales (page reuse):")
    cache, sfv, v_seq = run_pattern(
        "garbage reuse", [(0, 128), (128, 200)], num_pages=3, garbage=True
    )

    print("fa4 QV integration (incremental cache feeds the kernel):")
    cache, sfv, v_seq = run_pattern("fa4 feed", [(0, 300)], num_pages=3)
    integration_fa4(cache, sfv, v_seq, num_pages=3, kv_len=300)

    print("ALL OK")


if __name__ == "__main__":
    main()
