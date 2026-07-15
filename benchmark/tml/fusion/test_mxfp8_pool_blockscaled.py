"""Pool-level test for the MHATokenToKVPoolMXFP8 dequant-mode KV cache.

The pool stores K and V pre-quantized (fp8 e4m3) with per-token UE8M0 scales
(quantized in the model layer, not the pool). For page_size==128 the scales
are written interleaved into the FA4 BlockScaledBasicChunk atom layout.

Covers set_kv_buffer (extend chunks + decode 1-token appends across two
sequences, two layers), the interleaved scale roundtrip via
_read_sf_interleaved, move_kv_cache relocation (data + scales travel together),
and the guard that set_kv_buffer requires both K and V scale tensors.
"""

import torch

from sglang.srt.mem_cache.memory_pool import MHATokenToKVPoolMXFP8
from sglang.srt.kernels.mxfp8_quant import to_mxfp8

H_KV, HEAD_DIM, PAGE, LAYERS = 1, 128, 128, 2
SF_DIM = HEAD_DIM // 32
DEV = "cuda"


def make_pool(size, layer_num):
    return MHATokenToKVPoolMXFP8(
        size=size,
        page_size=PAGE,
        dtype=torch.float8_e4m3fn,
        head_num=H_KV,
        head_dim=HEAD_DIM,
        layer_num=layer_num,
        device=DEV,
        enable_memory_saver=False,
        enable_alt_stream=False,
    )


def u8(x):
    return x.view(torch.uint8)


def check_layer(name, pool, idx, k_data_ref, k_sf_ref, v_data_ref, v_sf_ref, written):
    """Every written slot holds exactly the fp8 bytes + UE8M0 scale we wrote."""
    loc = torch.tensor(written, device=DEV, dtype=torch.int64)
    kb, vb = pool._get_key_buffer(idx), pool._get_value_buffer(idx)
    assert (u8(kb[loc]) == u8(k_data_ref[loc])).all(), f"{name}: K data differs"
    assert (u8(vb[loc]) == u8(v_data_ref[loc])).all(), f"{name}: V data differs"

    k_scale_buf, v_scale_buf = pool.get_kv_scale_buffer(idx + pool.start_layer)
    k_sf = pool._read_sf_interleaved(k_scale_buf, loc)
    v_sf = pool._read_sf_interleaved(v_scale_buf, loc)
    assert (u8(k_sf) == u8(k_sf_ref[loc])).all(), f"{name}: K scales differ"
    assert (u8(v_sf) == u8(v_sf_ref[loc])).all(), f"{name}: V scales differ"
    print(f"  {name}: data + interleaved scales roundtrip OK ({len(written)} slots)")


def main():
    torch.manual_seed(0)
    g = torch.Generator(device=DEV).manual_seed(1)
    pool = make_pool(PAGE * 15, LAYERS)
    num_pages = (pool.size + PAGE) // PAGE
    total = num_pages * PAGE

    # Per-slot reference of the exact quantized bytes/scales written per layer.
    k_data = [torch.zeros(total, H_KV, HEAD_DIM, device=DEV, dtype=torch.float8_e4m3fn) for _ in range(LAYERS)]
    v_data = [torch.zeros_like(k_data[0]) for _ in range(LAYERS)]
    k_sf = [torch.zeros(total, H_KV, SF_DIM, device=DEV, dtype=torch.float8_e8m0fnu) for _ in range(LAYERS)]
    v_sf = [torch.zeros_like(k_sf[0]) for _ in range(LAYERS)]

    def rand(n):
        x = torch.randn(n, H_KV, HEAD_DIM, device=DEV, dtype=torch.bfloat16, generator=g)
        return x * torch.exp2(torch.randint(-6, 7, (n, 1, 1), device=DEV, generator=g).float())

    def write(loc):
        n = loc.shape[0]
        for lid in range(LAYERS):
            kq, vq = to_mxfp8(rand(n)), to_mxfp8(rand(n))
            ks = kq.scale.view(torch.float8_e8m0fnu)
            vs = vq.scale.view(torch.float8_e8m0fnu)
            pool.set_kv_buffer(
                None, loc, kq.data, vq.data,
                k_scale=ks, v_scale=vs, layer_id_override=lid,
            )
            k_data[lid][loc], v_data[lid][loc] = kq.data, vq.data
            k_sf[lid][loc], v_sf[lid][loc] = ks, vs

    written = []
    seq_base = [PAGE, 4 * PAGE]  # slot 0..PAGE-1 is the pool's guard region
    seq_len = [0, 0]
    for s, chunk in [(0, 200), (1, 130)]:  # prefill / extend chunks
        loc = torch.arange(seq_base[s], seq_base[s] + chunk, device=DEV, dtype=torch.int64)
        write(loc)
        written.extend(loc.tolist())
        seq_len[s] = chunk
    for _ in range(70):  # decode: both sequences append one token per step
        loc = torch.tensor([seq_base[s] + seq_len[s] for s in range(2)], device=DEV, dtype=torch.int64)
        write(loc)
        written.extend(loc.tolist())
        seq_len = [l + 1 for l in seq_len]

    for lid in range(LAYERS):
        check_layer(f"layer {lid} extend+decode", pool, lid, k_data[lid], k_sf[lid], v_data[lid], v_sf[lid], written)

    # move_kv_cache: relocate seq 0's rows to a fresh page-aligned region;
    # data and scales must travel together (extra_buffer relocation).
    n_move = seq_len[0]
    src = torch.arange(seq_base[0], seq_base[0] + n_move, device=DEV, dtype=torch.int64)
    tgt = torch.arange(12 * PAGE, 12 * PAGE + n_move, device=DEV, dtype=torch.int64)
    pool.move_kv_cache(tgt, src)
    for lid in range(LAYERS):
        k_data[lid][tgt], v_data[lid][tgt] = k_data[lid][src], v_data[lid][src]
        k_sf[lid][tgt], v_sf[lid][tgt] = k_sf[lid][src], v_sf[lid][src]
        check_layer(f"layer {lid} after move", pool, lid, k_data[lid], k_sf[lid], v_data[lid], v_sf[lid], tgt.tolist())

    # guard: set_kv_buffer requires both K and V scales.
    for missing in ("k", "v"):
        kq, vq = to_mxfp8(rand(1)), to_mxfp8(rand(1))
        try:
            pool.set_kv_buffer(
                None, torch.tensor([PAGE], device=DEV), kq.data, vq.data,
                k_scale=None if missing == "k" else kq.scale.view(torch.float8_e8m0fnu),
                v_scale=None if missing == "v" else vq.scale.view(torch.float8_e8m0fnu),
                layer_id_override=0,
            )
            raise AssertionError(f"missing {missing}_scale accepted")
        except ValueError:
            pass
    print("  missing-scale rejection guard: OK")

    # Fused quant-store path (SGLANG_OPT_INKLING_MXFP8_FUSED_QUANT_STORE): bf16 K/V
    # in, no scales -> one kernel quantizes + scatters payload and interleaved
    # scales. Stored bytes must match the layer-side to_mxfp8 path exactly.
    pool_f = make_pool(PAGE * 15, LAYERS)
    for buf in (k_data, v_data):
        for t in buf:
            t.zero_()
    for buf in (k_sf, v_sf):
        for t in buf:
            t.view(torch.uint8).zero_()

    def write_fused(loc):
        n = loc.shape[0]
        for lid in range(LAYERS):
            kb, vb = rand(n), rand(n)
            pool_f.set_kv_buffer(None, loc, kb, vb, layer_id_override=lid)
            kq, vq = to_mxfp8(kb), to_mxfp8(vb)
            k_data[lid][loc], v_data[lid][loc] = kq.data, vq.data
            k_sf[lid][loc] = kq.scale.view(torch.float8_e8m0fnu)
            v_sf[lid][loc] = vq.scale.view(torch.float8_e8m0fnu)

    written_f = []
    seq_len = [0, 0]
    g.manual_seed(2)
    for s, chunk in [(0, 200), (1, 130)]:
        loc = torch.arange(seq_base[s], seq_base[s] + chunk, device=DEV, dtype=torch.int64)
        write_fused(loc)
        written_f.extend(loc.tolist())
        seq_len[s] = chunk
    for _ in range(70):
        loc = torch.tensor([seq_base[s] + seq_len[s] for s in range(2)], device=DEV, dtype=torch.int64)
        write_fused(loc)
        written_f.extend(loc.tolist())
        seq_len = [l + 1 for l in seq_len]
    for lid in range(LAYERS):
        check_layer(f"layer {lid} fused extend+decode", pool_f, lid, k_data[lid], k_sf[lid], v_data[lid], v_sf[lid], written_f)

    print("ALL OK")


if __name__ == "__main__":
    main()
