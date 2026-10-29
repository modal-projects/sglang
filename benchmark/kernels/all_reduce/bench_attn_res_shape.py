"""attn_res_smallm_hip launch-shape sweep (vpt, threads) at H=7168 on one GPU."""

import json
import os

import torch

from sglang.kernels.ops.attention import attn_res_smallm_hip as m

H = 7168


def run(mod, prefix, add, pout, bank, cw, ow, out, nvb):
    mod.run(prefix, add, pout, bank, cw, ow, out, out, out, nvb, 1 | 4, 1e-6, 1e-6)


def time_graph(fn, iters=50, reps=7, inner=10):
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        for _ in range(inner):
            fn()
    best = 1e9
    for _ in range(reps):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        for _ in range(iters):
            g.replay()
        b.record()
        torch.cuda.synchronize()
        best = min(best, a.elapsed_time(b) * 1e3 / (iters * inner))
    return best


def main():
    if int(os.environ.get("RANK", "0")) != 0:
        return
    torch.manual_seed(0)
    shapes = [(2, 448), (1, 896), (4, 224), (2, 512), (4, 256), (1, 1024)]
    for T in (8, 72, 96, 128):
        for nvb in (1, 4, 8):
            bank = torch.randn(T, 9, H, device="cuda").bfloat16()
            prefix = torch.randn(T, H, device="cuda").bfloat16()
            add = torch.randn(T, H, device="cuda").bfloat16()
            cw = torch.randn(H, device="cuda").float()
            ow = (1 + 0.1 * torch.randn(H, device="cuda")).bfloat16()
            ref = torch.empty_like(prefix)
            refp = torch.empty_like(prefix)
            run(m._module(2, 448), prefix, add, refp, bank, cw, ow, ref, nvb)
            row = {"T": T, "nvb": nvb}
            for vpt, th in shapes:
                mod = m._module(vpt, th)
                out = torch.empty_like(prefix)
                pout = torch.empty_like(prefix)
                run(mod, prefix, add, pout, bank, cw, ow, out, nvb)
                torch.cuda.synchronize()
                d = (out.float() - ref.float()).abs()
                us = time_graph(
                    lambda: run(mod, prefix, add, pout, bank, cw, ow, out, nvb)
                )
                row[f"{vpt}x{th}"] = [
                    round(us, 2),
                    float(d.max()),
                    round(float((d > 0).float().mean()), 6),
                    bool(torch.equal(pout, refp)),
                ]
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
