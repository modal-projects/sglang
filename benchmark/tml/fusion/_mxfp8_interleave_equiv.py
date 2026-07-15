"""Check triton store_sf_interleaved == host interleave_sf (pool<->kernel SF layout).

The pool writes scale factors via the triton store_sf_interleaved kernel; the FA4
kernel's bench builds them via blockscaled_utils.interleave_sf. Both must produce
the identical BlockScaledBasicChunk buffer [num_pages, nheads, 32, 4, 4], else the
kernel reads scales for the wrong tokens (silent wrong output).
"""

import torch

from blockscaled_utils import interleave_sf
from sglang.srt.kernels.mxfp8_interleave_sf import store_sf_interleaved

PAGE = 128
SF_VEC = 32
SF_DIM = 128 // SF_VEC  # 4


def check(num_pages, nheads):
    total = num_pages * PAGE
    g = torch.Generator(device="cuda").manual_seed(0)
    sf = torch.randint(0, 255, (total, nheads, SF_DIM), dtype=torch.uint8, device="cuda", generator=g)
    sf_e8m0 = sf.view(torch.float8_e8m0fnu)

    # host reference: interleave_sf wants (batch, seqlen, nheads, sf_k)
    host_in = sf_e8m0.view(num_pages, PAGE, nheads, SF_DIM)
    host_out = interleave_sf(host_in, SF_VEC)  # (num_pages, nheads, REST_M=1, REST_K=1, 32, 4, 4)
    host_flat = host_out.reshape(num_pages, nheads, 32, SF_DIM, SF_DIM).view(torch.uint8)

    # triton: scatter per-token by contiguous loc
    out = torch.zeros(num_pages, nheads, 32, SF_DIM, SF_DIM, dtype=torch.uint8, device="cuda")
    loc = torch.arange(total, dtype=torch.int64, device="cuda")
    store_sf_interleaved(sf_e8m0, out.view(torch.float8_e8m0fnu), loc, page_size=PAGE)

    ok = torch.equal(host_flat, out)
    print(f"num_pages={num_pages} nheads={nheads}: {'MATCH' if ok else 'MISMATCH'}"
          f" (mismatch count={0 if ok else (host_flat != out).sum().item()})")
    return ok


def check_roundtrip(num_pages, nheads):
    """store_sf_interleaved then gather-back (pool move_kv_cache helper) must
    recover the original per-slot scales for arbitrary (non-contiguous) loc."""
    total = num_pages * PAGE
    g = torch.Generator(device="cuda").manual_seed(1)
    sf = torch.randint(0, 255, (total, nheads, SF_DIM), dtype=torch.uint8, device="cuda", generator=g)
    sf_e8m0 = sf.view(torch.float8_e8m0fnu)
    buf = torch.zeros(num_pages, nheads, 32, SF_DIM, SF_DIM, dtype=torch.uint8, device="cuda")
    loc = torch.randperm(total, device="cuda")[: total].to(torch.int64)
    store_sf_interleaved(sf_e8m0, buf.view(torch.float8_e8m0fnu), loc, page_size=PAGE)

    # inverse gather (mirror of MHATokenToKVPoolMXFP8._read_sf_interleaved)
    buf_u32 = buf.reshape(num_pages, nheads, -1).view(torch.int32)
    off = loc % PAGE
    page = (loc // PAGE).long()
    ipos = ((off % 32) * (PAGE // 32) + (off // 32)).long()
    heads = torch.arange(nheads, device="cuda")
    gathered = buf_u32[page[:, None], heads[None, :], ipos[:, None]]
    back = gathered.reshape(total, nheads, 1).view(torch.uint8).reshape(total, nheads, SF_DIM)
    ok = torch.equal(back, sf)
    print(f"roundtrip num_pages={num_pages} nheads={nheads}: {'MATCH' if ok else 'MISMATCH'}")
    return ok


if __name__ == "__main__":
    assert torch.cuda.is_available()
    all_ok = True
    for np_, nh in [(1, 1), (4, 1), (2, 8), (7, 8)]:
        all_ok &= check(np_, nh)
    for np_, nh in [(4, 1), (8, 8)]:
        all_ok &= check_roundtrip(np_, nh)
    print("ALL", "PASS" if all_ok else "FAIL")
