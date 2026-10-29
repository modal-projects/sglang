"""K3 TP8 per-rank dense GEMM microbench on MI355X: BF16 prod dispatch vs static FP8, plus MLA merge."""
import argparse, json, os
import torch
from safetensors import safe_open

SNAP = "/hf/hub/models--moonshotai--Kimi-K3-MTP-Release/snapshots/8455440e191f81ff4c6338208a2012701356e46b"
P = "language_model.model.layers."


def load(names):
    idx = json.load(open(f"{SNAP}/model.safetensors.index.json"))["weight_map"]
    out = {}
    for n in names:
        with safe_open(f"{SNAP}/{idx[n]}", "pt", device="cuda") as f:
            out[n] = f.get_tensor(n)
    return out


def bf16_mm(x, w):
    from aiter.tuned_gemm import tgemm
    from sglang.srt.layers.quantization.unquant import _aiter_triton_a16w16, _prefer_triton_a16w16
    if _prefer_triton_a16w16(x, w):
        return _aiter_triton_a16w16(x, w, None, dtype=x.dtype)
    return tgemm.mm(x, w, None, otype=x.dtype)


def gtime(fn, reps=50):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        fn()
        with torch.cuda.graph(g, stream=s):
            for _ in range(reps):
                fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(5):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); g.replay(); b.record(); torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) * 1e3 / reps)
    return sorted(ts)[2]


def rel(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm()).item(), (a - b).abs().max().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms", default="9,36,144")
    ap.add_argument("--mla-layer", type=int, default=3)
    ap.add_argument("--kda-layer", type=int, default=1)
    a = ap.parse_args()
    from sglang.srt.layers.quantization.k3_rocm_dense_fp8 import StaticFP8Weight
    from sglang.kernels.ops.gemm import kimi_k3_tiny_gemm
    L, K = f"{P}{a.mla_layer}.self_attn.", f"{P}{a.kda_layer}.self_attn."
    W = load([L + "q_a_proj.weight", L + "kv_a_proj_with_mqa.weight", L + "g_proj.weight",
              L + "q_b_proj.weight", L + "o_proj.weight", K + "f_a_proj.weight", K + "f_b_proj.weight",
              K + "b_proj.weight", K + "o_proj.weight"])
    for k, v in W.items():
        print("#", k.split("layers.")[1], tuple(v.shape), v.dtype)
    bf = lambda t: t.to(torch.bfloat16).contiguous()
    qa, kva = bf(W[L + "q_a_proj.weight"]), bf(W[L + "kv_a_proj_with_mqa.weight"])
    g = bf(W[L + "g_proj.weight"][:1536]); qb = bf(W[L + "q_b_proj.weight"][:2304])
    mo = bf(W[L + "o_proj.weight"][:, :1536]); ko = bf(W[K + "o_proj.weight"][:, :1536])
    fa, fb = bf(W[K + "f_a_proj.weight"]), bf(W[K + "f_b_proj.weight"][:1536])
    qkva = torch.cat([qa, kva], 0).contiguous()
    merged = torch.cat([qa, kva, g], 0).contiguous()
    shapes = {"mla_qkv_a": qkva, "mla_g": g, "mla_qkva+g": merged, "mla_q_b": qb, "mla_o": mo, "kda_o": ko,
              "kda_f_b": fb}
    torch.manual_seed(0)
    for nm, (n, k) in {"moe_front": (6016, 7168), "kda_qkvg": (6144, 7168), "moe_up_col": (896, 3584)}.items():
        shapes[nm] = (torch.randn(n, k, device="cuda") * 0.02).to(torch.bfloat16)
    rows = []
    for m in [int(v) for v in a.ms.split(",")]:
        for nm, w in shapes.items():
            n, k = w.shape
            x = (torch.randn(m, k, device="cuda")).to(torch.bfloat16)
            if nm == "kda_f_b":
                x = bf16_mm(torch.randn(m, 7168, device="cuda").to(torch.bfloat16), fa)
                tb = gtime(lambda: kimi_k3_tiny_gemm(x, w)); ref = x.float() @ w.float().T
                e = rel(kimi_k3_tiny_gemm(x, w), ref)
                rows.append(dict(m=m, op=nm, n=n, k=k, bf16_us=round(tb, 2), err_bf16=e)); continue
            tb = gtime(lambda: bf16_mm(x, w))
            ref = x.float() @ w.float().T
            eb = rel(bf16_mm(x, w), ref)
            f8 = StaticFP8Weight(w)
            tf = gtime(lambda: f8(x)); ef = rel(f8(x), ref)
            rows.append(dict(m=m, op=nm, n=n, k=k, bf16_us=round(tb, 2), fp8_us=round(tf, 2), hbm_bf16_us=round(n * k * 2 / 8e6, 2),
                             err_bf16=[round(v, 5) for v in eb], err_fp8=[round(v, 5) for v in ef]))
        # merge exactness: merged GEMM slices vs separate GEMMs (prod BF16 dispatch)
        x = torch.randn(m, 7168, device="cuda").to(torch.bfloat16)
        ym = bf16_mm(x, merged); y1 = bf16_mm(x, qkva); y2 = bf16_mm(x, g)
        rows.append(dict(m=m, op="merge_check", bitexact_qkva=bool(torch.equal(ym[:, :2112], y1)), bitexact_g=bool(torch.equal(ym[:, 2112:], y2)),
                         err_vs_sep=rel(ym, torch.cat([y1, y2], 1)), sep_us=round(gtime(lambda: (bf16_mm(x, qkva), bf16_mm(x, g))), 2),
                         merged_us=round(gtime(lambda: bf16_mm(x, merged)), 2)))
    for r in rows:
        print(json.dumps(r))


if __name__ == "__main__":
    main()
