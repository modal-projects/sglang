"""Scenario coverage for FA4 sheared bias vs the pip score_mod baseline.

The installed pip flash-attn package does not expose ``rel_bias``.  These tests
therefore compare the vendored FA4 sheared-bias path:

    ven_fn(..., rel_bias=bias)

against the pip score_mod implementation:

    pip_fn(..., score_mod=get_inkling_relative_attention_score_mod(re),
           aux_tensors=[bias])

Every active scenario below checks output and LSE bitwise equality.  The suite is
intentionally scenario-oriented rather than exhaustive fuzzing:

1. eager matrix: decode, prefill, local-window, page-edge KV lengths, forced and
   automatic split modes, plus shuffled page-table layouts.
2. ragged mixed batches: heterogeneous q/kv lengths in one batch, with each
   sequence also compared against its solo execution.
3. workspace reuse: repeated large/small rel_extent calls to catch stale
   grow-only sheared-bias workspace state.
4. CUDA graph replay: captured decode graphs replayed after mutating q, bias,
   and heterogeneous per-slot seqused_k depths.

Run:  python benchmark/tml/fusion/test_shear_bias_concurrency.py
"""

import os
from functools import lru_cache

import torch

from sglang.srt.utils.common import suppress_noisy_warnings

suppress_noisy_warnings()

from flash_attn.cute import flash_attn_varlen_func as pip_fn

from sglang.jit_kernel.flash_attn.cute import flash_attn_varlen_func as ven_fn
from sglang.srt.models.inkling_common.attn import (
    get_inkling_relative_attention_score_mod,
)

DEV = "cuda"
NH, NHK, D = 64, 8, 128
PAGE_SIZE = 128
FULL_RE = 1024
LOCAL_RE = 512
FAILS = []


@lru_cache(maxsize=None)
def score_mod(rel_extent):
    return get_inkling_relative_attention_score_mod(rel_extent)


def report(tag, ok, detail=""):
    print(f"  {tag:64s} {'OK' if ok else 'FAIL ' + detail}")
    if not ok:
        FAILS.append(tag)


def unpack(res):
    return res if isinstance(res, tuple) else (res, None)


def compare_result(tag, shear_res, pip_res):
    shear_out, shear_lse = unpack(shear_res)
    pip_out, pip_lse = unpack(pip_res)

    out_ndiff = int((shear_out != pip_out).sum().item())
    out_max = (shear_out.float() - pip_out.float()).abs().max().item()

    lse_ndiff = 0
    lse_max = 0.0
    if shear_lse is not None or pip_lse is not None:
        if shear_lse is None or pip_lse is None:
            report(tag, False, "one impl returned LSE and the other did not")
            return
        lse_ndiff = int((shear_lse != pip_lse).sum().item())
        lse_max = (shear_lse.float() - pip_lse.float()).abs().max().item()

    ok = out_ndiff == 0 and lse_ndiff == 0
    report(
        tag,
        ok,
        f"out_ndiff={out_ndiff} out_max={out_max:.3e} "
        f"lse_ndiff={lse_ndiff} lse_max={lse_max:.3e}",
    )


def make_varlen(q_lens, kv_lens, rel_extent, seed=0, page_layout="contiguous"):
    assert len(q_lens) == len(
        kv_lens
    ), "q_lens and kv_lens must describe the same batch"
    assert rel_extent % PAGE_SIZE == 0
    torch.manual_seed(seed)

    b = len(q_lens)
    total_q = sum(q_lens)
    pages_per = max((kv + PAGE_SIZE - 1) // PAGE_SIZE for kv in kv_lens)
    total_pages = b * pages_per

    kc = torch.randn(total_pages, PAGE_SIZE, NHK, D, device=DEV, dtype=torch.bfloat16)
    vc = torch.randn(total_pages, PAGE_SIZE, NHK, D, device=DEV, dtype=torch.bfloat16)

    if page_layout == "contiguous":
        page_ids = torch.arange(total_pages, dtype=torch.int32, device=DEV)
    elif page_layout == "reverse":
        page_ids = torch.arange(total_pages - 1, -1, -1, dtype=torch.int32, device=DEV)
    elif page_layout == "shuffle":
        page_ids = torch.randperm(total_pages, device=DEV).to(torch.int32)
    else:
        raise ValueError(f"unknown page_layout={page_layout!r}")

    pt = page_ids.view(b, pages_per)
    q = torch.randn(total_q, NH, D, device=DEV, dtype=torch.bfloat16)
    bias = torch.randn(total_q, NH, rel_extent, device=DEV, dtype=torch.bfloat16)
    cu = (
        torch.tensor([0] + q_lens, dtype=torch.int32, device=DEV)
        .cumsum(0)
        .to(torch.int32)
    )
    sl = torch.tensor(kv_lens, dtype=torch.int32, device=DEV)
    return q, kc, vc, bias, pt, cu, sl


def common_kwargs(pt, cu, sl, max_q, splits, windowed_extent=None):
    kw = dict(
        cu_seqlens_q=cu,
        seqused_k=sl,
        max_seqlen_q=max_q,
        page_table=pt,
        softmax_scale=1.0 / D,
        causal=True,
        num_splits=splits,
        return_lse=True,
    )
    if windowed_extent is not None:
        kw["window_size"] = (windowed_extent - 1, 0)
    return kw


def compare_shear_to_pip(
    tag,
    q,
    kc,
    vc,
    bias,
    pt,
    cu,
    sl,
    max_q,
    splits=0,
    windowed_extent=None,
):
    kw = common_kwargs(pt, cu, sl, max_q, splits, windowed_extent=windowed_extent)
    shear_res = ven_fn(q, kc, vc, rel_bias=bias, **kw)
    pip_res = pip_fn(
        q,
        kc,
        vc,
        score_mod=score_mod(bias.shape[-1]),
        aux_tensors=[bias],
        **kw,
    )
    compare_result(tag, shear_res, pip_res)


def test_eager_matrix():
    cases = [
        # tag, q_lens, kv_lens, rel_extent, windowed_extent, splits, page_layout
        ("decode tiny-kv edge", [1], [127], FULL_RE, None, 0, "contiguous"),
        ("decode exact-page edge", [1], [128], FULL_RE, None, 0, "contiguous"),
        ("decode next-page edge", [1], [129], FULL_RE, None, 0, "reverse"),
        (
            "decode large bs auto-split",
            [1] * 64,
            [4097] * 64,
            FULL_RE,
            None,
            0,
            "shuffle",
        ),
        (
            "decode large bs forced-one",
            [1] * 64,
            [4097] * 64,
            FULL_RE,
            None,
            1,
            "shuffle",
        ),
        (
            "decode large bs forced-two",
            [1] * 64,
            [4097] * 64,
            FULL_RE,
            None,
            2,
            "shuffle",
        ),
        (
            "decode local window",
            [1] * 32,
            [2048] * 32,
            LOCAL_RE,
            LOCAL_RE,
            0,
            "reverse",
        ),
        ("prefill short", [7] * 4, [517] * 4, FULL_RE, None, 1, "contiguous"),
        ("prefill chunked", [256] * 2, [8261] * 2, FULL_RE, None, 0, "shuffle"),
        (
            "prefill local chunked",
            [256] * 2,
            [8261] * 2,
            LOCAL_RE,
            LOCAL_RE,
            0,
            "shuffle",
        ),
    ]
    for tag, q_lens, kv_lens, rel_extent, windowed_extent, splits, page_layout in cases:
        q, kc, vc, bias, pt, cu, sl = make_varlen(
            q_lens, kv_lens, rel_extent, seed=len(tag), page_layout=page_layout
        )
        compare_shear_to_pip(
            f"eager {tag} splits={splits} pages={page_layout}",
            q,
            kc,
            vc,
            bias,
            pt,
            cu,
            sl,
            max(q_lens),
            splits=splits,
            windowed_extent=windowed_extent,
        )
        del q, kc, vc, bias, pt, cu, sl
        torch.cuda.empty_cache()


def test_ragged_mixed_batch():
    q_lens = [1, 7, 256, 33, 1, 130]
    kv_lens = [65537, 129, 1024, 517, 8192, 3333]
    q, kc, vc, bias, pt, cu, sl = make_varlen(
        q_lens, kv_lens, FULL_RE, seed=17, page_layout="shuffle"
    )

    for splits in (0, 1, 2):
        compare_shear_to_pip(
            f"ragged mixed batch splits={splits}",
            q,
            kc,
            vc,
            bias,
            pt,
            cu,
            sl,
            max(q_lens),
            splits=splits,
        )

        q_off = 0
        for i, (ql, kvl) in enumerate(zip(q_lens, kv_lens)):
            qi = q[q_off : q_off + ql].contiguous()
            bi = bias[q_off : q_off + ql].contiguous()
            pti = pt[i : i + 1].contiguous()
            cui = torch.tensor([0, ql], dtype=torch.int32, device=DEV)
            sli = torch.tensor([kvl], dtype=torch.int32, device=DEV)
            compare_shear_to_pip(
                f"ragged solo seq={i} q={ql} kv={kvl} splits={splits}",
                qi,
                kc,
                vc,
                bi,
                pti,
                cui,
                sli,
                ql,
                splits=splits,
            )
            q_off += ql
    torch.cuda.empty_cache()


def test_workspace_reuse():
    cases = [
        ("grow full", [256, 33], [8261, 4097], FULL_RE, None, "shuffle"),
        ("small local", [1, 7], [1211, 517], LOCAL_RE, LOCAL_RE, "reverse"),
        ("medium full", [64], [2048], FULL_RE, None, "contiguous"),
        ("tiny local", [5], [640], LOCAL_RE, LOCAL_RE, "contiguous"),
    ]
    inputs = []
    for i, (_, q_lens, kv_lens, rel_extent, windowed_extent, page_layout) in enumerate(
        cases
    ):
        inputs.append(
            make_varlen(
                q_lens, kv_lens, rel_extent, seed=100 + i, page_layout=page_layout
            )
            + (max(q_lens), windowed_extent)
        )

    for it in range(3):
        for i, (name, _, _, rel_extent, windowed_extent, _) in enumerate(cases):
            q, kc, vc, bias, pt, cu, sl, max_q, _ = inputs[i]
            compare_shear_to_pip(
                f"workspace iter={it} case={name} re={rel_extent}",
                q,
                kc,
                vc,
                bias,
                pt,
                cu,
                sl,
                max_q,
                splits=0,
                windowed_extent=windowed_extent,
            )
    torch.cuda.empty_cache()


def test_decode_graph_mixed_depth():
    b, kv_cap, rel_extent = 32, 8192, FULL_RE
    q_lens = [1] * b
    kv_lens = [kv_cap] * b
    q, kc, vc, bias, pt, cu, sl = make_varlen(
        q_lens, kv_lens, rel_extent, seed=7, page_layout="shuffle"
    )
    kw = common_kwargs(pt, cu, sl, 1, splits=0)

    outs = {}
    for name, fn, extra in (
        ("shear", ven_fn, dict(rel_bias=bias)),
        ("pip", pip_fn, dict(score_mod=score_mod(rel_extent), aux_tensors=[bias])),
    ):
        fn(q, kc, vc, **extra, **kw)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            res = fn(q, kc, vc, **extra, **kw)

        graph.replay()
        torch.cuda.synchronize()
        first = (res[0].clone(), res[1].clone())

        depths = torch.tensor(
            [129, 517, 1211, 4097, 8192, 130, 2048, 3333] * (b // 8),
            dtype=torch.int32,
            device=DEV,
        )
        torch.manual_seed(11)
        sl.copy_(depths)
        q.copy_(torch.randn_like(q))
        bias.copy_(torch.randn_like(bias))

        graph.replay()
        torch.cuda.synchronize()
        mixed_depth = (res[0].clone(), res[1].clone())

        torch.manual_seed(7)
        q0, _, _, bias0, _, _, _ = make_varlen(
            q_lens, kv_lens, rel_extent, seed=7, page_layout="shuffle"
        )
        q.copy_(q0)
        bias.copy_(bias0)
        sl.fill_(kv_cap)
        outs[name] = (first, mixed_depth)
        del graph
        torch.cuda.synchronize()

    compare_result("graph decode replay same inputs", outs["shear"][0], outs["pip"][0])
    compare_result("graph decode replay mixed depths", outs["shear"][1], outs["pip"][1])
    torch.cuda.empty_cache()


def main():
    selected = set(
        os.environ.get("SCENARIOS", "eager,ragged,workspace,graph").split(",")
    )
    if "eager" in selected:
        test_eager_matrix()
    if "ragged" in selected:
        test_ragged_mixed_batch()
    if "workspace" in selected:
        test_workspace_reuse()
    if "graph" in selected:
        test_decode_graph_mixed_depth()

    print("ALL OK" if not FAILS else f"FAILURES: {FAILS}")
    raise SystemExit(1 if FAILS else 0)


if __name__ == "__main__":
    assert torch.cuda.is_available()
    main()
