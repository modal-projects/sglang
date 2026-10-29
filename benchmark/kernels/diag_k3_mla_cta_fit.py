"""Accuracy of k3_mla_verify_auto split counts vs an fp32 reference (HK regime)."""

import torch

from sglang.kernels.ops.attention.k3_mla_verify_auto import k3_mla_verify_auto
from sglang.kernels.ops.attention.k3_mla_verify_common import (
    cta_fit_num_splits,
    default_num_splits,
)

H, QLEN, D, DV = 12, 8, 576, 512
SCALE = D**-0.5
KV_SCALE = 0.5


def ulp(x):
    return torch.exp2(torch.floor(torch.log2(x.abs().float().clamp_min(2.0**-126))) - 7)


def ref32(q, kv, lens, slots):
    kvf = kv.view(-1, D)
    out, start = [], 0
    for b, L in enumerate(lens):
        k = kvf[slots[start : start + L].long()].float() * KV_SCALE
        start += L
        qb = q[b * QLEN : (b + 1) * QLEN].float()
        s = torch.einsum("qhd,kd->hqk", qb, k) * SCALE
        vis = torch.arange(L, device=q.device) <= (
            L - QLEN + torch.arange(QLEN, device=q.device)[:, None]
        )
        p = s.masked_fill(~vis, float("-inf")).softmax(-1)
        out.append(torch.einsum("hqk,kd->qhd", p, k[:, :DV]))
    return torch.cat(out)


def main():
    pool = 90000 * 16 + 4096
    torch.manual_seed(0)
    dirn = torch.randn(D, device="cuda")
    for dist in ("zero_mean", "shifted"):
        kvf = torch.randn(pool, 1, D, device="cuda") * 0.5
        if dist == "shifted":
            kvf += 0.5 * dirn
        kv = kvf.to(torch.float8_e4m3fn)
        del kvf
        perm = torch.randperm(pool, device="cuda").to(torch.int32)
        ks = torch.tensor([KV_SCALE], device="cuda")
        for ctx in (2000, 32000, 64000, 90000):
            for bs in range(1, 17):
                torch.manual_seed(ctx + bs)
                lens = [ctx - 97 * b for b in range(bs)]
                indptr = (
                    torch.tensor([0] + lens, device="cuda").cumsum(0).to(torch.int32)
                )
                slots = perm[: sum(lens)].contiguous()
                q = torch.randn(bs * QLEN, H, D, device="cuda", dtype=torch.bfloat16)
                r = ref32(q, kv, lens, slots)
                o = {}
                for fit in (False, True):
                    o[fit] = k3_mla_verify_auto(
                        q, kv, indptr, slots, SCALE, ks, KV_SCALE, thresh=0, cta_fit=fit
                    ).float()
                su = ulp(r.abs().max()).item()
                d = (o[True] - o[False]).abs()
                eu = ulp(torch.maximum(o[True].abs(), o[False].abs()))
                e_def = (o[False] - r).abs().max().item()
                e_fit = (o[True] - r).abs().max().item()
                rms_def = (o[False] - r).pow(2).mean().sqrt().item()
                rms_fit = (o[True] - r).pow(2).mean().sqrt().item()
                print(
                    f"{dist:9s} L={ctx:5d} bs={bs:2d} ns {max(2, default_num_splits(bs)):3d}->{cta_fit_num_splits(bs):3d} "
                    f"|o|max={r.abs().max().item():.3f} ulp={su:.1e} fit-def max={d.max().item():.1e} ({d.max().item() / su:.2f} ulp) "
                    f"elem>1ulp={(d > eu).float().mean().item() * 100:.3f}% | vs fp32 max def {e_def:.1e} fit {e_fit:.1e} "
                    f"rms def {rms_def:.2e} fit {rms_fit:.2e}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
