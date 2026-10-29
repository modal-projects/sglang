"""K3HK_TS=1: per-CTA phase stamps (s_memrealtime, 100 MHz) of k3_mla_verify_hk_kernel."""

import os

os.environ["K3HK_TS"] = "1"
import torch

from sglang.kernels.ops.attention import k3_mla_verify_hk as hk

H, QLEN, D, DV, SCALE, KVS, dev = 12, 8, 576, 512, 576**-0.5, 0.5, "cuda"
pool = 90000 * 4 + 4096
kv = (torch.randn(pool, 1, D, device=dev) * 0.5).to(torch.float8_e4m3fn)
ks = torch.tensor([KVS], device=dev)
perm = torch.randperm(pool, device=dev).to(torch.int32)
buf = torch.zeros(4096 * 8, dtype=torch.int64, device=dev)
names = ["Qstage+quant", "firstTiles", "mainloop", "epilogue"]
for bs, ns in ((1, 256), (2, 128), (4, 64)):
    for L in (2000, 90000):
        lens = [L - 97 * b for b in range(bs)]
        indptr = torch.tensor([0] + lens, device=dev).cumsum(0).to(torch.int32)
        slots = perm[: sum(lens)].contiguous()
        q = torch.randn(bs * QLEN, H, D, device=dev, dtype=torch.bfloat16)
        out = torch.empty(bs * QLEN, H, DV, device=dev, dtype=torch.bfloat16)
        op = torch.empty(bs, ns, H * QLEN, DV, device=dev, dtype=torch.bfloat16)
        lp = torch.empty(bs, ns, H * QLEN, device=dev)
        for it in range(4):
            hk._module().ts_dump(buf)
            a, b = torch.cuda.Event(True), torch.cuda.Event(True)
            a.record()
            hk.hk_stage1(q, kv, indptr, slots, ks, op, lp, out, ns, 256, SCALE, 96)
            b.record()
            torch.cuda.synchronize()
            hk._module().ts_dump(buf)
            t = buf.view(4096, 8)[: bs * ns].cpu().double() * 0.01  # us
        act = t[:, 0] > 0
        t = t[act]
        t0 = t[:, 0].min()
        ph = [(t[:, i + 1] - t[:, i]) for i in range(4)]
        msg = " ".join(
            f"{n} med {p.median():.2f} max {p.max():.2f}" for n, p in zip(names, ph)
        )
        print(
            f"bs={bs} ns={ns} L={L} ctas={int(act.sum())} event_us={a.elapsed_time(b) * 1e3:.2f} "
            f"span(entry0->last end)={(t[:, 4].max() - t0):.2f} entry_spread={(t[:, 0].max() - t0):.2f} "
            f"loader(med)={(t[:, 6] - t[:, 5]).median():.2f} | {msg}",
            flush=True,
        )
