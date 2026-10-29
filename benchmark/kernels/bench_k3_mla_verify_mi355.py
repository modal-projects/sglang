"""K3 MLA target-verify microbench (MI355X:1): k3_mla_verify_auto vs HK stage1+reduce at other
split counts. 12 heads/rank, QLEN 8, fp8 KV latent 576. Each graph replays NL calls on distinct
KV pools so the KV streams from HBM (not MALL)."""

import argparse

import torch

from sglang.kernels.ops.attention.k3_mla_verify_auto import k3_mla_verify_auto
from sglang.kernels.ops.attention.k3_mla_verify_common import _MIN_CHUNK, default_num_splits
from sglang.kernels.ops.attention.k3_mla_verify_hk import PART8_MIN_SPLITS, hk_reduce, hk_stage1

H, QL, D, DV = 12, 8, 576, 512
FP8 = torch.float8_e4m3fn


def graph_us(fn, iters=20):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(iters):
        g.replay()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) * 1e3 / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bs", default="1,2,4,6,8,9,12,16")
    ap.add_argument("--ctx", default="64000,90000")
    ap.add_argument("--nsplits", default="")  # extra split counts to try (comma), else auto set
    ap.add_argument("--nl", type=int, default=6)
    ap.add_argument("--fit-only", type=int, default=1)
    a = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(0)
    sm = 1.0 / (192**0.5)
    kv_scale = torch.ones(1, dtype=torch.float32, device=dev)
    for L in [int(x) for x in a.ctx.split(",")]:
        for bs in [int(x) for x in a.bs.split(",")]:
            NL = a.nl
            ntok = bs * L
            pools = [
                (torch.randn(ntok, 1, D, device=dev) * 0.5).to(FP8) for _ in range(NL)
            ]
            q = torch.randn(bs * QL, H, D, device=dev, dtype=torch.bfloat16)
            indptr = torch.arange(0, bs + 1, device=dev, dtype=torch.int32) * L
            idx = torch.randperm(ntok, device=dev).to(torch.int32)
            out = torch.empty(bs * QL, H, DV, device=dev, dtype=torch.bfloat16)

            def auto():
                for p in pools:
                    k3_mla_verify_auto(q, p, indptr, idx, sm, kv_scale, 1.0, out=out)

            t_auto = graph_us(auto) / NL
            ref = out.clone()
            out_fit = torch.empty_like(out)

            def auto_fit():
                for p in pools:
                    k3_mla_verify_auto(q, p, indptr, idx, sm, kv_scale, 1.0, out=out_fit, cta_fit=True)

            t_fit = graph_us(auto_fit) / NL
            # fp32 reference on the last pool, first request
            kvf = pools[-1].view(-1, D)[idx[:L].long()].float()
            qf = q[:QL].float()
            s_ = torch.einsum("thd,nd->thn", qf, kvf) * sm
            pr = torch.softmax(s_, -1)
            of = torch.einsum("thn,nd->thd", pr, kvf[:, :DV])
            e_ref = ((ref[:QL].float() - of).norm() / of.norm()).item()
            e_fit = ((out_fit[:QL].float() - of).norm() / of.norm()).item()
            e_rf = (out_fit.float() - ref.float()).abs().max().item()
            ns0 = max(2, default_num_splits(bs))
            cands = sorted(
                set([ns0] + [int(x) for x in a.nsplits.split(",") if x] + [max(1, 256 // bs), max(1, (256 + bs - 1) // bs), max(1, 2 * 256 // bs)])
            )
            line = (f"L={L} bs={bs:2d} auto {t_auto:7.2f} fit {t_fit:7.2f} us (ns {ns0}) "
                    f"err32 auto {e_ref:.1e} fit {e_fit:.1e} fit-vs-auto max {e_rf:.1e}")
            if a.fit_only:
                print(line, flush=True)
                pools.clear()
                torch.cuda.empty_cache()
                continue
            for ns in cands:
                if ns > 256:
                    continue
                op = torch.empty(bs, ns, H * QL, DV, dtype=torch.bfloat16, device=dev)
                lp = torch.empty(bs, ns, H * QL, dtype=torch.float32, device=dev)
                p8 = PART8_MIN_SPLITS if ns >= 128 else 0
                o2 = torch.empty_like(out)

                def s1(op=op, lp=lp, ns=ns, p8=p8, o2=o2):
                    for p in pools:
                        hk_stage1(q, p.view(-1, D), indptr, idx, kv_scale, op, lp, o2, ns, _MIN_CHUNK, sm, p8)

                def s12(op=op, lp=lp, ns=ns, p8=p8, o2=o2):
                    for p in pools:
                        hk_stage1(q, p.view(-1, D), indptr, idx, kv_scale, op, lp, o2, ns, _MIN_CHUNK, sm, p8)
                        if ns > 1:
                            hk_reduce(op, lp, indptr, o2, ns, _MIN_CHUNK, kv_scale, p8)

                t1 = graph_us(s1) / NL
                t12 = graph_us(s12) / NL
                d = (o2.float() - ref.float()).abs()
                rel = (d.norm() / ref.float().norm()).item()
                line += f" | ns {ns:3d} ctas {bs*ns:4d}: s1 {t1:6.2f} s1+red {t12:6.2f} rel {rel:.1e} max {d.max().item():.1e}"
            hbm = bs * L * D / 8e6
            print(line + f" | hbm {hbm:.1f}", flush=True)
            pools.clear()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
