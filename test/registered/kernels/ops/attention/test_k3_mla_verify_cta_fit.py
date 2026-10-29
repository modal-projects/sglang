"""``k3_mla_verify_auto(cta_fit=True)`` (floor(256 / bs) HK splits) vs the default
power-of-two split count, bs 1..16 x 2k..90k context, auto and forced-HK regime:
same split count -> bit-exact; otherwise within one BF16 ulp of the output scale,
except where the default merges fp8 partials (HIP reduce, part8) and cta_fit does not,
where cta_fit must be closer to an fp32 reference. With near-zero outputs (zero-mean
KV) the merge-order difference exceeds one ulp of the output, so there cta_fit must
match the default's error vs fp32 instead. Also graph capture + replay."""

import os
import unittest

import torch

from sglang.kernels.ops.attention.k3_mla_verify_auto import k3_mla_verify_auto
from sglang.kernels.ops.attention.k3_mla_verify_common import (
    _TARGET_CTAS,
    cta_fit_num_splits,
    default_num_splits,
)
from sglang.kernels.ops.attention.k3_mla_verify_hk import hip_reduce_enabled
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=240, stage="jit-kernel-unit", runner_config="amd")

H, QLEN, D, DV = 12, 8, 576, 512
SCALE = D**-0.5
KV_SCALE = 0.5
CTXS = [2000, 32000, 64000, 90000]
BATCHES = list(range(1, 17))
VERBOSE = os.environ.get("K3_MLA_CTA_FIT_VERBOSE") == "1"


def _reference(q, kv, lens, slots):
    kvf = kv.view(-1, D)
    out, start = [], 0
    for b, L in enumerate(lens):
        k = kvf[slots[start : start + L].long()].float() * KV_SCALE
        start += L
        qb = q[b * QLEN : (b + 1) * QLEN].float()
        s = torch.einsum("qhd,kd->hqk", qb, k) * SCALE
        visible = torch.arange(L, device=q.device) <= (
            L - QLEN + torch.arange(QLEN, device=q.device)[:, None]
        )
        p = s.masked_fill(~visible, float("-inf")).softmax(-1)
        out.append(torch.einsum("hqk,kd->qhd", p, k[:, :DV]))
    return torch.cat(out)


def _bf16_ulp(x: torch.Tensor) -> torch.Tensor:
    m = x.abs().float().clamp_min(2.0**-126)
    return torch.exp2(torch.floor(torch.log2(m)) - 7)


class TestK3MlaVerifyCtaFit(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.pool = max(CTXS) * max(BATCHES) + 4096
        kv = torch.randn(cls.pool, 1, D, device="cuda") * 0.5
        cls.kv_zero_mean = kv.to(torch.float8_e4m3fn)
        # A shared direction gives |o| ~ 0.8, the magnitude of real K3 MLA outputs.
        cls.kv = (kv + 0.5 * torch.randn(D, device="cuda")).to(torch.float8_e4m3fn)
        del kv
        cls.perm = torch.randperm(cls.pool, device="cuda").to(torch.int32)
        cls.kv_scale = torch.tensor([KV_SCALE], device="cuda")

    def _inputs(self, bs, ctx):
        lens = [ctx - 97 * b for b in range(bs)]
        indptr = torch.tensor([0] + lens, device="cuda").cumsum(0).to(torch.int32)
        slots = self.perm[: sum(lens)].contiguous()
        q = torch.randn(bs * QLEN, H, D, device="cuda", dtype=torch.bfloat16)
        return q, indptr, slots

    def _run(self, q, indptr, slots, cta_fit, thresh=None, out=None, kv=None):
        return k3_mla_verify_auto(
            q,
            self.kv if kv is None else kv,
            indptr,
            slots,
            SCALE,
            self.kv_scale,
            KV_SCALE,
            thresh=thresh,
            cta_fit=cta_fit,
            out=out,
        )

    def test_split_grid_fits_one_wave(self):
        for bs in range(1, _TARGET_CTAS // 2 + 1):
            ns = cta_fit_num_splits(bs)
            self.assertGreaterEqual(ns, 2)
            self.assertLessEqual(bs * ns, _TARGET_CTAS)

    def test_within_one_bf16_ulp_of_default(self):
        for ctx in CTXS:
            for bs in BATCHES:
                torch.manual_seed(ctx + bs)
                q, indptr, slots = self._inputs(bs, ctx)
                ns_def, ns_fit = max(2, default_num_splits(bs)), cta_fit_num_splits(bs)
                fp8_partials_dropped = hip_reduce_enabled(ns_def) and not (
                    hip_reduce_enabled(ns_fit)
                )
                ref32 = _reference(q, self.kv, [ctx - 97 * b for b in range(bs)], slots)
                scale_ulp = _bf16_ulp(ref32.abs().max()).item()
                for regime, thresh in (("auto", None), ("hk", 0)):
                    with self.subTest(ctx=ctx, bs=bs, regime=regime):
                        ref = self._run(q, indptr, slots, False, thresh)
                        fit = self._run(q, indptr, slots, True, thresh)
                        max_diff = (fit.float() - ref.float()).abs().max().item()
                        err_def = (ref.float() - ref32).abs().max().item()
                        err_fit = (fit.float() - ref32).abs().max().item()
                        if VERBOSE:
                            print(
                                f"ctx={ctx} bs={bs:2d} {regime:4s} ns {ns_def}->{ns_fit} "
                                f"fit-def={max_diff / scale_ulp:.2f} ulp "
                                f"err32 def={err_def:.1e} fit={err_fit:.1e}",
                                flush=True,
                            )
                        if ns_fit == ns_def:
                            self.assertEqual(max_diff, 0.0)
                        elif fp8_partials_dropped:
                            self.assertLessEqual(err_fit, err_def)
                        else:
                            self.assertLessEqual(max_diff, scale_ulp)

    def test_zero_mean_kv_matches_default_error(self):
        for ctx in CTXS[1:]:
            for bs in BATCHES:
                torch.manual_seed(ctx + bs)
                q, indptr, slots = self._inputs(bs, ctx)
                lens = [ctx - 97 * b for b in range(bs)]
                ref32 = _reference(q, self.kv_zero_mean, lens, slots)
                with self.subTest(ctx=ctx, bs=bs):
                    rms = []
                    for cta_fit in (False, True):
                        o = self._run(
                            q, indptr, slots, cta_fit, 0, kv=self.kv_zero_mean
                        )
                        rms.append((o.float() - ref32).pow(2).mean().sqrt().item())
                    self.assertLessEqual(rms[1], rms[0] * 1.05)

    def test_graph_capture_and_replay(self):
        ctx, short = 64000, 31000
        for bs in (1, 2, 3, 5, 7, 12, 16):
            with self.subTest(bs=bs):
                torch.manual_seed(bs)
                q, indptr, slots = self._inputs(bs, ctx)
                slots_buf = self.perm[: bs * ctx].clone()
                out = torch.empty(
                    bs * QLEN, H, 512, device="cuda", dtype=torch.bfloat16
                )
                eager = self._run(q, indptr, slots, True).clone()
                self._run(q, indptr, slots_buf, True, out=out)
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    self._run(q, indptr, slots_buf, True, out=out)
                g.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(out, eager, rtol=0, atol=0)
                q2, indptr2, slots2 = self._inputs(bs, short)
                q.copy_(q2)
                indptr.copy_(indptr2)
                slots_buf[: slots2.numel()].copy_(slots2)
                g.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(
                    out, self._run(q2, indptr2, slots2, True), rtol=0, atol=0
                )


if __name__ == "__main__":
    unittest.main()
