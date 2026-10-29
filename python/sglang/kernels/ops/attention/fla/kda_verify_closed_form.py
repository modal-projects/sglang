"""KDA spec verify over one draft block in closed form (ReplaySSM ring write, no state update).

With D_t = diag(exp(g_t)), G_t = D_1...D_t, and h the [K, V] state,
    u_t = beta_t (v_t - (G_t k_t)^T h0 - sum_{s<t} (k_t^T (G_t / G_s) k_s) u_s),
    o_t = (G_t q_t)^T h0 + sum_{s<=t} (q_t^T (G_t / G_s) k_s) u_s,
so a block needs two products against h0 and a lower-triangular solve instead of serial rank-1 updates.
"""

import triton
import triton.language as tl

# exp(-cumsum(g)) must stay finite in fp32 over a block.
_MAX_NEG_CUM_GATE = 80.0


@triton.jit(do_not_specialize=["T_total"])
def _kda_verify_closed_form_kernel(
    A_log,
    a,
    dt_bias,
    q,
    k,
    v,
    b,
    o,
    h0,
    h0_idx,
    stride_h0,
    cu,
    rawv,
    rawk,
    ring_g,
    ring_beta,
    ring_idx,
    s_rawv,
    s_rawk,
    s_g,
    s_beta,
    scale,
    lower_bound,
    T_total,
    stride_a,
    stride_q,
    stride_k,
    stride_v,
    stride_b,
    fa,
    wfb,
    stride_fa,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    KC: tl.constexpr,
    L: tl.constexpr,
    FUSE_FB: tl.constexpr,
    R: tl.constexpr,
):
    i_v, i_n, i_hv = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h = i_hv // (HV // H)
    bos = tl.load(cu + i_n).to(tl.int64)
    T = tl.load(cu + i_n + 1).to(tl.int64) - bos
    idx = tl.load(h0_idx + i_n).to(tl.int64)
    o_t = tl.arange(0, BT)
    o_k = tl.arange(0, K)
    o_v = i_v * BV + tl.arange(0, BV)
    m_t = o_t < T
    tok = bos + o_t

    b_q = tl.load(
        q + tok[:, None] * stride_q + i_h * K + o_k[None, :], mask=m_t[:, None], other=0
    ).to(tl.float32)
    b_k = tl.load(
        k + tok[:, None] * stride_k + i_h * K + o_k[None, :], mask=m_t[:, None], other=0
    ).to(tl.float32)
    rq = scale / tl.sqrt(tl.sum(b_q * b_q, 1) + 1e-6)
    rk = 1.0 / tl.sqrt(tl.sum(b_k * b_k, 1) + 1e-6)
    b_v = tl.load(
        v + tok[:, None] * stride_v + i_hv * V + o_v[None, :],
        mask=m_t[:, None],
        other=0,
    ).to(tl.float32)
    b_b = tl.load(b + tok * stride_b + i_hv, mask=m_t, other=0).to(tl.float32)
    b_beta = tl.where(m_t, 1.0 / (1.0 + tl.exp(-b_b)), 0.0)
    a_exp = tl.exp(tl.load(A_log + i_hv).to(tl.float32))
    slot = tl.load(ring_idx + i_n).to(tl.int64)
    m_r = m_t & (o_t < L) & (idx >= 0)
    tl.store(
        rawv + slot * s_rawv + i_hv * L * V + o_t[:, None] * V + o_v[None, :],
        b_v.to(rawv.dtype.element_ty),
        mask=m_r[:, None],
    )
    if i_v == 0:
        tl.store(ring_beta + slot * s_beta + i_hv * L + o_t, b_beta, mask=m_r)

    p = tl.zeros([BT, BV], dtype=tl.float32)
    r = tl.zeros([BT, BV], dtype=tl.float32)
    if FUSE_FB:
        o_r = tl.arange(0, R)
        b_fa = tl.load(
            fa + tok[:, None] * stride_fa + o_r[None, :], mask=m_t[:, None], other=0
        )
    akk = tl.zeros([BT, BT], dtype=tl.float32)
    aqk = tl.zeros([BT, BT], dtype=tl.float32)
    p_h = h0 + tl.maximum(idx, 0) * stride_h0 + i_hv * K * V + o_v[:, None] * K
    for c in tl.range(0, K, KC):
        o_c = c + tl.arange(0, KC)
        q_c = tl.load(
            q + tok[:, None] * stride_q + i_h * K + o_c[None, :],
            mask=m_t[:, None],
            other=0,
        ).to(tl.float32)
        k_c = tl.load(
            k + tok[:, None] * stride_k + i_h * K + o_c[None, :],
            mask=m_t[:, None],
            other=0,
        ).to(tl.float32)
        if FUSE_FB:
            # forget gate = f_a @ f_b^T, rounded to bf16 like the standalone GEMM
            w_c = tl.load(wfb + (i_hv * K + o_c)[None, :] * R + o_r[:, None])
            a_c = tl.dot(b_fa, w_c).to(tl.bfloat16).to(tl.float32)
        else:
            a_c = tl.load(
                a + tok[:, None] * stride_a + i_hv * K + o_c[None, :],
                mask=m_t[:, None],
                other=0,
            ).to(tl.float32)
        dt_c = tl.load(dt_bias + i_hv * K + o_c).to(tl.float32)
        g_c = lower_bound * tl.sigmoid(a_exp * (a_c + dt_c[None, :]))
        g_c = tl.where(m_t[:, None], g_c, 0.0)
        if i_v == 0:
            if i_hv % (HV // H) == 0:
                tl.store(
                    rawk
                    + slot * s_rawk
                    + i_h * L * K
                    + o_t[:, None] * K
                    + o_c[None, :],
                    k_c.to(rawk.dtype.element_ty),
                    mask=m_r[:, None],
                )
            tl.store(
                ring_g + slot * s_g + i_hv * L * K + o_t[:, None] * K + o_c[None, :],
                g_c,
                mask=m_r[:, None],
            )
        e = tl.exp(tl.cumsum(g_c, 0))
        qg = q_c * rq[:, None] * e
        kg = k_c * rk[:, None] * e
        ki = k_c * rk[:, None] / e
        h_c = tl.load(p_h + o_c[None, :], mask=idx >= 0, other=0.0).to(tl.float32)
        p += tl.dot(qg, tl.trans(h_c), input_precision="ieee")
        r += tl.dot(kg, tl.trans(h_c), input_precision="ieee")
        akk += tl.dot(kg, tl.trans(ki), input_precision="ieee")
        aqk += tl.dot(qg, tl.trans(ki), input_precision="ieee")

    nmat = tl.where(o_t[:, None] > o_t[None, :], b_beta[:, None] * akk, 0.0)
    eye = tl.where(o_t[:, None] == o_t[None, :], 1.0, 0.0)
    # N is strictly lower triangular, so (I + N)^-1 = (I - N)(I + N^2)(I + N^4)...
    inv = eye - nmat
    pw = nmat
    for _ in tl.static_range(1, BT.bit_length() - 1):
        pw = tl.dot(pw, pw, input_precision="ieee")
        inv = tl.dot(inv, eye + pw, input_precision="ieee")
    u = tl.dot(inv, b_beta[:, None] * (b_v - r), input_precision="ieee")
    att = tl.where(o_t[:, None] >= o_t[None, :], aqk, 0.0)
    out = p + tl.dot(att, u, input_precision="ieee")
    tl.store(
        o + (tok[:, None] * HV + i_hv) * V + o_v[None, :],
        out.to(o.dtype.element_ty),
        mask=m_t[:, None],
    )


def kda_verify_closed_form_supported(
    k,
    v,
    lower_bound,
    max_tokens_per_seq,
    intermediate_states_buffer,
    retrieve_parent_token,
):
    return (
        intermediate_states_buffer is None
        and retrieve_parent_token is None
        and lower_bound is not None
        and lower_bound < 0
        and k.shape[-1] == 128
        and v.shape[-1] == 128
        and max_tokens_per_seq <= 16
        and -lower_bound * max_tokens_per_seq <= _MAX_NEG_CUM_GATE
    )


def kda_verify_closed_form(
    A_log,
    a,
    dt_bias,
    q,
    k,
    v,
    b,
    initial_state_source,
    initial_state_indices,
    cu_seqlens,
    lower_bound,
    replayssm_rawv,
    replayssm_rawk,
    replayssm_g,
    replayssm_beta,
    ring_indices,
    scale=None,
    block_v=None,
    num_warps=4,
    num_stages=None,
    block_k=32,
    f_a=None,
    f_b_weight=None,
):
    """If f_a/f_b_weight are given, the forget gate is computed in-kernel and `a` is ignored."""
    H, K, HV, V = k.shape[2], k.shape[3], v.shape[2], v.shape[3]
    N = cu_seqlens.numel() - 1
    BV = block_v if block_v is not None else (64 if N <= 4 else 128)
    o = q.new_empty(v.shape)
    _kda_verify_closed_form_kernel[(V // BV, N, HV)](
        A_log,
        a,
        dt_bias,
        q,
        k,
        v,
        b,
        o,
        initial_state_source,
        initial_state_indices,
        initial_state_source.stride(0),
        cu_seqlens,
        replayssm_rawv,
        replayssm_rawk,
        replayssm_g,
        replayssm_beta,
        ring_indices,
        replayssm_rawv.stride(0),
        replayssm_rawk.stride(0),
        replayssm_g.stride(0),
        replayssm_beta.stride(0),
        scale if scale is not None else K**-0.5,
        lower_bound,
        v.shape[1],
        0 if f_a is not None else (a.stride(1) if a.ndim == 4 else a.stride(-2)),
        q.stride(1),
        k.stride(1),
        v.stride(1),
        b.stride(-2),
        f_a if f_a is not None else a,
        f_b_weight if f_b_weight is not None else a,
        (f_a.stride(1) if f_a.ndim == 4 else f_a.stride(-2)) if f_a is not None else 0,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=16,
        BV=BV,
        KC=block_k,
        L=replayssm_rawv.shape[-2],
        FUSE_FB=f_a is not None,
        R=f_b_weight.shape[-1] if f_b_weight is not None else 16,
        num_warps=num_warps,
        num_stages=num_stages if num_stages is not None else (2 if BV == 64 else 1),
    )
    return o
