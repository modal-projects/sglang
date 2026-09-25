"""Chunked MLA attention must agree with an unsplit natural-log reference."""

import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention import flashinfer_mla_backend, trtllm_mla_backend
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _scores(q, k, scale, causal):
    scores = torch.einsum("qhd,khd->qhk", q.double(), k.double()) * scale
    if causal:
        query_positions = torch.arange(q.shape[0]) + k.shape[0] - q.shape[0]
        mask = torch.arange(k.shape[0])[None, :] > query_positions[:, None]
        scores.masked_fill_(mask[:, None, :], -torch.inf)
    return scores


def _external_log2_attention(q, k, v, *, scale, causal):
    scores = _scores(q, k, scale, causal)
    maximum = scores.amax(-1, keepdim=True)
    weights = (scores - maximum).exp()
    denominator = weights.sum(-1)
    output = torch.einsum("qhk,khd->qhd", weights, v.double())
    output /= denominator.unsqueeze(-1)
    lse = maximum.squeeze(-1) * math.log2(math.e) + denominator.log2()
    return output.float(), lse.float()


def _reference(q, k, v, *, scale, causal):
    scores = _scores(q, k, scale, causal)
    return (
        torch.einsum("qhk,khd->qhd", scores.softmax(-1), v.double()),
        scores.logsumexp(-1),
    )


def _merge(parts):
    outputs, lses = zip(*parts)
    lses = torch.stack(lses).double()
    weights = lses.softmax(dim=0)
    return (torch.stack(outputs).double() * weights.unsqueeze(-1)).sum(dim=0)


class TestMLAPrefillLSEUnits(CustomTestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(7)
        self.q = torch.randn(3, 2, 4, generator=generator)
        self.k = torch.randn(10, 2, 4, generator=generator)
        self.v = torch.randn(10, 2, 2, generator=generator)
        self.layer = SimpleNamespace(
            tp_q_head_num=2,
            tp_k_head_num=2,
            tp_v_head_num=2,
            head_dim=4,
            v_head_dim=2,
            scaling=0.5,
            logit_cap=0.0,
        )

    def _flashinfer(self, q, k, v, *, prefix, return_lse=True):
        class Wrapper:
            def forward_return_lse(self, q, k, v, *, causal, sm_scale, **kwargs):
                return _external_log2_attention(q, k, v, scale=sm_scale, causal=causal)

            def forward(self, *args, **kwargs):
                return self.forward_return_lse(*args, **kwargs)[0]

        wrapper = Wrapper()
        runner = SimpleNamespace(
            chunk_ragged_wrappers=[wrapper], ragged_wrapper=wrapper
        )
        batch = SimpleNamespace(
            attn_attend_prefix_cache=prefix,
            prefix_chunk_idx=0,
            mha_return_lse=return_lse,
        )
        return flashinfer_mla_backend.FlashInferMhaChunkKVRunner.forward(
            runner, q, k, v, self.layer, batch
        )

    def _trtllm(self, q, k, v, *, prefix, return_lse=True):
        def external(**kwargs):
            output, lse = _external_log2_attention(
                kwargs["query"],
                kwargs["key"],
                kwargs["value"],
                scale=kwargs["bmm1_scale"],
                causal=kwargs["is_causal"],
            )
            kwargs["out"].copy_(output)
            return (kwargs["out"], lse) if kwargs["return_lse"] else kwargs["out"]

        flashinfer = SimpleNamespace(
            prefill=SimpleNamespace(trtllm_ragged_attention_deepseek=external)
        )
        backend = SimpleNamespace(data_type=torch.bfloat16, workspace_buffer=None)
        with patch.object(trtllm_mla_backend, "flashinfer", flashinfer, create=True):
            return trtllm_mla_backend.TRTLLMMLABackend._run_prefill_kernel(
                backend,
                q=q,
                k=k,
                v=v,
                layer=self.layer,
                batch_size=1,
                cum_seq_lens_q=torch.tensor([0, q.shape[0]], dtype=torch.int32),
                max_q_len=q.shape[0],
                seq_lens_kv=torch.tensor([k.shape[0]], dtype=torch.int32),
                cum_seq_lens_kv=torch.tensor([0, k.shape[0]], dtype=torch.int32),
                max_kv_len=k.shape[0],
                is_causal=not prefix,
                return_lse=return_lse,
                out_buffer=torch.empty(q.shape[0], 2, 2),
            )

    def _check_backend(self, run):
        parts, references = [], []
        for start, end, prefix in ((0, 3, True), (3, 7, True), (7, 10, False)):
            q, k, v = self.q, self.k[start:end], self.v[start:end]
            parts.append(run(q, k, v, prefix=prefix))
            references.append(_reference(q, k, v, scale=0.5, causal=not prefix))
        expected, _ = _reference(self.q, self.k, self.v, scale=0.5, causal=True)
        torch.testing.assert_close(_merge(parts), expected, rtol=1e-6, atol=1e-7)
        for (output, lse), (ref_output, ref_lse) in zip(parts, references):
            torch.testing.assert_close(
                output.double(), ref_output, rtol=1e-6, atol=1e-7
            )
            torch.testing.assert_close(lse.double(), ref_lse, rtol=1e-6, atol=1e-7)
        output_only = run(self.q, self.k, self.v, prefix=False, return_lse=False)
        self.assertIsInstance(output_only, torch.Tensor)
        torch.testing.assert_close(output_only.double(), expected, rtol=1e-6, atol=1e-7)

    def test_flashinfer_prefix_and_suffix_merge_match_unsplit_attention(self):
        self._check_backend(self._flashinfer)

    def test_trtllm_prefix_and_suffix_merge_match_unsplit_attention(self):
        self._check_backend(self._trtllm)

    def test_flashinfer_prefix_returns_lse_even_without_suffix_request(self):
        output, lse = self._flashinfer(
            self.q, self.k, self.v, prefix=True, return_lse=False
        )
        expected, expected_lse = _reference(
            self.q, self.k, self.v, scale=0.5, causal=False
        )
        torch.testing.assert_close(output.double(), expected, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(lse.double(), expected_lse, rtol=1e-6, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
