"""Reproduce the PROD decode CUDA-graph cell for FA4 sheared bias vs pip score_mod.

Covers the regime no other script exercises (the suspected AIME-regression cell):
  - page_table is the FULL max-context-width static buffer (like sglang's decode
    graph buffer), so the interface derives seqlen_k from table width and the
    auto num_splits heuristic bakes a large split count at capture
  - capture at cache_seqlens=1 (sglang's graph fill value), replay with kv
    GROWING across rel_extent / page / split boundaries
  - mixed per-row kv lengths and padded slots (seqused stays 1) inside the batch
  - prod per-rank head geometry (8 q heads / 1 kv head)
  - page-table entries beyond the valid prefix hold arbitrary in-bounds pages

Per step, compares bitwise: vendored graph replay vs pip graph replay (both
captured the same way), vendored graph vs vendored eager, and scans active rows
for NaN/Inf.

Env knobs: BS (default 8), NH/NHK (8/1), WINDOWED=0|1, MAX_TABLE_TOKENS
(default 131072), KV_SWEEP="1,127,...", SEED.

Run:  CUDA_VISIBLE_DEVICES=<free> python benchmark/tml/fusion/test_prod_decode_graph_cell.py
"""

import os

import torch

from sglang.srt.utils.common import suppress_noisy_warnings

suppress_noisy_warnings()

from flash_attn.cute import flash_attn_varlen_func as pip_fn

from sglang.jit_kernel.flash_attn.cute import flash_attn_varlen_func as ven_fn
from sglang.srt.models.inkling_common.attn import (
    get_inkling_relative_attention_score_mod,
)

DEV = "cuda"
BS = int(os.environ.get("BS", "8"))
NH = int(os.environ.get("NH", "8"))
NHK = int(os.environ.get("NHK", "1"))
D = 128
PAGE_SIZE = 128
REL_EXTENT = 1024
WINDOWED = os.environ.get("WINDOWED", "0") == "1"
MAX_TABLE_TOKENS = int(os.environ.get("MAX_TABLE_TOKENS", "131072"))
SEED = int(os.environ.get("SEED", "0"))
KV_SWEEP = [
    int(x)
    for x in os.environ.get(
        "KV_SWEEP",
        "1,127,128,129,511,1023,1024,1025,2048,4095,4096,4097,8192,16384,32767,32768",
    ).split(",")
]

FAILS = []


def build_static_inputs():
    torch.manual_seed(SEED)
    max_kv = max(KV_SWEEP)
    pages_per_row = (max_kv + PAGE_SIZE - 1) // PAGE_SIZE
    table_width = MAX_TABLE_TOKENS // PAGE_SIZE
    assert table_width >= pages_per_row
    total_pages = BS * pages_per_row
    kc = torch.randn(total_pages, PAGE_SIZE, NHK, D, device=DEV, dtype=torch.bfloat16)
    vc = torch.randn(total_pages, PAGE_SIZE, NHK, D, device=DEV, dtype=torch.bfloat16)
    # Valid prefix: row i owns pages [i*pages_per_row, (i+1)*pages_per_row).
    # Beyond the prefix: arbitrary in-bounds pages, like sglang's stale buffer.
    torch.manual_seed(SEED + 1)
    pt = torch.randint(0, total_pages, (BS, table_width), dtype=torch.int32, device=DEV)
    valid = torch.arange(total_pages, dtype=torch.int32, device=DEV).view(
        BS, pages_per_row
    )
    pt[:, :pages_per_row] = valid
    q = torch.randn(BS, NH, D, device=DEV, dtype=torch.bfloat16)
    bias = torch.randn(BS, NH, REL_EXTENT, device=DEV, dtype=torch.bfloat16)
    cu = torch.arange(0, BS + 1, dtype=torch.int32, device=DEV)
    sl = torch.ones(BS, dtype=torch.int32, device=DEV)  # capture fill value
    return q, kc, vc, bias, pt, cu, sl


def call_kwargs(cu, sl, pt):
    kw = dict(
        cu_seqlens_q=cu,
        seqused_k=sl,
        max_seqlen_q=1,
        page_table=pt,
        softmax_scale=1.0 / D,
        causal=True,
        num_splits=0,
        return_lse=True,
    )
    if WINDOWED:
        kw["window_size"] = (REL_EXTENT - 1, 0)
    return kw


def seqused_for_step(kv):
    # Mixed batch: even rows active at kv, row 1 at kv//2 (if >0), last row padded (=1).
    sl = torch.full((BS,), kv, dtype=torch.int32, device=DEV)
    if BS > 1:
        sl[1] = max(kv // 2, 1)
    if BS > 2:
        sl[BS - 1] = 1  # padded/finished slot, sglang keeps fill value
    return sl.clamp_(min=1)


def refresh_inputs(q, bias, step):
    torch.manual_seed(1000 + step)
    q.copy_(torch.randn_like(q))
    bias.copy_(torch.randn_like(bias))


def report(tag, ok, detail):
    print(f"  {tag:58s} {'OK' if ok else 'FAIL'} {detail}")
    if not ok:
        FAILS.append(tag)


def compare(tag, a, b):
    (oa, la), (ob, lb) = a, b
    od = int((oa != ob).sum().item())
    ld = int((la != lb).sum().item())
    oe = (oa.float() - ob.float()).abs().max().item()
    report(tag, od == 0 and ld == 0, f"out_diff={od} out_abs={oe:.3e} lse_diff={ld}")


def nan_scan(tag, out, lse, sl):
    active = (sl > 1).nonzero(as_tuple=True)[0]  # padded rows are discarded by sglang
    o = out[active]
    bad = int(torch.isnan(o).sum() + torch.isinf(o).sum())
    report(tag, bad == 0, f"nan_inf_active={bad}")


def main():
    sm = get_inkling_relative_attention_score_mod(REL_EXTENT)
    q, kc, vc, bias, pt, cu, sl = build_static_inputs()
    kw = call_kwargs(cu, sl, pt)

    impls = {
        "ven": (ven_fn, dict(rel_bias=bias)),
        "pip": (pip_fn, dict(score_mod=sm, aux_tensors=[bias])),
    }
    graphs, results = {}, {}
    for name, (fn, extra) in impls.items():
        fn(q, kc, vc, **extra, **kw)  # warm compile outside capture
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            res = fn(q, kc, vc, **extra, **kw)
        graphs[name] = g
        results[name] = res
    print(
        f"captured: bs={BS} heads={NH}/{NHK} table_width={pt.shape[1]} pages "
        f"(~{pt.shape[1] * PAGE_SIZE} tokens) windowed={WINDOWED}"
    )

    for step, kv in enumerate(KV_SWEEP):
        sl_step = seqused_for_step(kv)
        step_outs = {}
        for name in impls:
            refresh_inputs(q, bias, step)
            sl.copy_(sl_step)
            graphs[name].replay()
            torch.cuda.synchronize()
            o, l = results[name]
            step_outs[name] = (o.clone(), l.clone())
        # vendored eager at identical state (fresh call, auto splits at same table)
        refresh_inputs(q, bias, step)
        sl.copy_(sl_step)
        fn, extra = impls["ven"]
        oe_, le_ = fn(q, kc, vc, **extra, **kw)
        step_outs["ven_eager"] = (oe_.clone(), le_.clone())
        torch.cuda.synchronize()

        compare(
            f"kv={kv:6d} ven_graph vs pip_graph", step_outs["ven"], step_outs["pip"]
        )
        compare(
            f"kv={kv:6d} ven_graph vs ven_eager",
            step_outs["ven"],
            step_outs["ven_eager"],
        )
        nan_scan(f"kv={kv:6d} ven_graph nan scan", *step_outs["ven"], sl_step)

    print("ALL OK" if not FAILS else f"FAILURES ({len(FAILS)}): {FAILS}")
    raise SystemExit(1 if FAILS else 0)


if __name__ == "__main__":
    assert torch.cuda.is_available()
    main()
