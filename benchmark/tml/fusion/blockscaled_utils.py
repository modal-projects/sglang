"""Host-side MXFP8 scale-factor interleave helper for the FA4 fusion benchmarks.

Only `interleave_sf` is used (by the test_mxfp8_* / _mxfp8_* harnesses in this
directory). The former quantize/dequantize/rowwise helpers were unused and
removed; this file lives beside its only callers rather than in the shipped
sglang package.
"""

import torch


def interleave_sf(sf, sf_vec_size):
    """Interleave a K-major scale factor tensor into the BlockScaledBasicChunk atom layout.

    Input:  sf with shape (batch, seqlen, nheads, sf_k) where sf_k = ceil(hdim / sf_vec_size)
    Output: physically contiguous as (batch, nheads, REST_M, REST_K, 32, 4, 4),
            reshaped to (total_sf_elements,) for passing as a raw buffer.
            The kernel endows this with the appropriate cute layout via tile_atom_to_shape_SF.
    """
    batch, seqlen, nheads, sf_k = sf.shape

    seqlen_padded = ((seqlen + 127) // 128) * 128
    sf_k_padded = ((sf_k + 3) // 4) * 4
    rest_m = seqlen_padded // 128
    rest_k = sf_k_padded // 4

    sf_work = sf.permute(0, 2, 1, 3)  # (batch, nheads, seqlen, sf_k)

    if seqlen_padded != seqlen or sf_k_padded != sf_k:
        sf_padded = torch.zeros(
            batch,
            nheads,
            seqlen_padded,
            sf_k_padded,
            dtype=sf.dtype,
            device=sf.device,
        )
        sf_padded[:, :, :seqlen, :sf_k] = sf_work
    else:
        sf_padded = sf_work.contiguous()

    # Decompose M -> (REST_M, 4, 32), SF_K -> (REST_K, 4)
    sf_decomp = sf_padded.reshape(batch, nheads, rest_m, 4, 32, rest_k, 4)
    # Permute to (batch, nheads, REST_M, REST_K, 32, 4, 4) and make contiguous
    return sf_decomp.permute(0, 1, 2, 5, 4, 3, 6).contiguous()
