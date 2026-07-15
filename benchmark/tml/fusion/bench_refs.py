"""Shared torch references + a non-raising correctness reporter for the benches.

`report` prints PASS/FAIL + abs-max error and never raises, so a failing config
still gets timed (you see the number, the run continues).
"""

import torch


def report(tag: str, out, ref, tol: float = 1e-1) -> None:
    """Print PASS/FAIL + abs-max error for out-vs-ref. Tensors or tuples of them.
    None entries are skipped. Never raises."""
    try:
        pairs = list(zip(out, ref)) if isinstance(out, (tuple, list)) else [(out, ref)]
        err = 0.0
        for a, b in pairs:
            if a is None or b is None:
                continue
            err = max(err, (a.float() - b.float()).abs().max().item())
        status = "PASS" if err <= tol else "FAIL"
    except Exception as e:  # shape/dtype mismatch etc — report, don't crash the sweep
        print(f"[correctness] {tag}: ERROR ({e})")
        return
    print(f"[correctness] {tag}: {status} (abs_max={err:.3e})")


def rmsnorm_torch(x, w, eps):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w.float()).to(
        x.dtype
    )


def conv_torch(x, weight, cache, cidx, cmask, activation, use_residual, W):
    # ponytail: conv compute only (no in-place cache shift) — value reference
    ci = cidx.long()
    taps = [cache[ci, iw].float() for iw in range(W - 1)] + [x.float()]
    acc = sum(taps[i] * weight[:, i].float() for i in range(W))
    if activation in ("silu", "swish"):
        acc = acc * torch.sigmoid(acc)
    if use_residual:
        acc = acc + x.float()
    return acc.to(x.dtype)


def copy_torch(cache, mask, src, dst):
    # graph-safe: no boolean-mask indexing (that lowers to nonzero -> dynamic shape,
    # uncapturable). Masked-out rows map dst->src so they copy onto themselves (no-op).
    dst_eff = torch.where(mask.bool(), dst, src).long()
    cache.index_copy_(0, dst_eff, cache.index_select(0, src.long()))
    return cache


def silu_mul_torch(gateup, gammas=None):
    g = gateup.view(gateup.shape[0], -1, 2).float()
    out = (g[..., 0] * torch.sigmoid(g[..., 0])) * g[..., 1]
    if gammas is not None:
        out = out * gammas[:, None].float()
    return out.to(gateup.dtype)


def renorm_torch(logits, tk, n_shared, route_scale, gscale, topk):
    from sglang.srt.models.inkling_common.moe import (
        _renorm_topk_logits,  # lazy: heavy import
    )

    w = _renorm_topk_logits(logits, tk, n_shared, "sigmoid") * route_scale * gscale
    shared = w[..., topk:].contiguous() if n_shared > 0 else None
    return w[..., :topk].contiguous(), shared


def topk_torch(x, k):
    return torch.topk(x, k, dim=-1)
