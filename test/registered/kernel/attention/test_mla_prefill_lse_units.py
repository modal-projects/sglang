"""Ragged MLA attention must merge uniform prefix/suffix states by token count."""

import math
import unittest
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

H, D_QK, D_V = 16, 192, 128


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10,
    "SM100/SM103 required",
)
class TestMLAPrefillLSEUnits(CustomTestCase):
    def setUp(self):
        self.workspace = torch.zeros(
            256 * 1024 * 1024, dtype=torch.uint8, device="cuda"
        )
        self.layer = SimpleNamespace(
            tp_q_head_num=H,
            tp_k_head_num=H,
            tp_v_head_num=H,
            head_dim=D_QK,
            v_head_dim=D_V,
            scaling=1 / math.sqrt(D_QK),
            logit_cap=0.0,
        )

    def _run_flashinfer(self, q, k, v, *, prefix):
        from flashinfer import BatchPrefillWithRaggedKVCacheWrapper

        from sglang.srt.layers.attention.flashinfer_mla_backend import (
            FlashInferMhaChunkKVRunner,
        )

        wrapper = BatchPrefillWithRaggedKVCacheWrapper(
            self.workspace, "NHD", backend="cutlass"
        )
        wrapper.plan(
            torch.tensor([0, q.shape[0]], dtype=torch.int32, device="cuda"),
            torch.tensor([0, k.shape[0]], dtype=torch.int32, device="cuda"),
            H,
            H,
            q.shape[-1],
            head_dim_vo=D_V,
            causal=not prefix,
            sm_scale=self.layer.scaling,
            q_data_type=q.dtype,
            kv_data_type=k.dtype,
        )
        runner = SimpleNamespace(
            chunk_ragged_wrappers=[wrapper], ragged_wrapper=wrapper
        )
        batch = SimpleNamespace(
            attn_attend_prefix_cache=prefix, prefix_chunk_idx=0, mha_return_lse=True
        )
        return FlashInferMhaChunkKVRunner.forward(runner, q, k, v, self.layer, batch)

    def _run_trtllm(self, q, k, v, *, prefix):
        from sglang.srt.layers.attention.trtllm_mla_backend import TRTLLMMLABackend

        backend = SimpleNamespace(data_type=q.dtype, workspace_buffer=self.workspace)
        return TRTLLMMLABackend._run_prefill_kernel(
            backend,
            q=q,
            k=k,
            v=v,
            layer=self.layer,
            batch_size=1,
            cum_seq_lens_q=torch.tensor(
                [0, q.shape[0]], dtype=torch.int32, device="cuda"
            ),
            max_q_len=q.shape[0],
            seq_lens_kv=torch.tensor([k.shape[0]], dtype=torch.int32, device="cuda"),
            cum_seq_lens_kv=torch.tensor(
                [0, k.shape[0]], dtype=torch.int32, device="cuda"
            ),
            max_kv_len=k.shape[0],
            is_causal=not prefix,
            return_lse=True,
            out_buffer=torch.empty(q.shape[0], H, D_V, dtype=q.dtype, device="cuda"),
        )

    def _check_uniform_attention(self, run, cases):
        from sglang.srt.layers.attention.merge_state import merge_state

        prefix_len, suffix_len = 32, 5
        for dtype, head_dim, causal_suffix in cases:
            with self.subTest(
                dtype=dtype, head_dim=head_dim, causal_suffix=causal_suffix
            ):
                self.layer.head_dim = head_dim
                self.layer.scaling = 1 / math.sqrt(head_dim)
                q = torch.zeros(suffix_len, H, head_dim, dtype=dtype, device="cuda")
                k_prefix = torch.zeros(
                    prefix_len, H, head_dim, dtype=dtype, device="cuda"
                )
                v_prefix = torch.full(
                    (prefix_len, H, D_V), 2.0, dtype=dtype, device="cuda"
                )
                k_suffix = torch.zeros_like(q)
                v_suffix = torch.full(
                    (suffix_len, H, D_V), -2.0, dtype=dtype, device="cuda"
                )
                prefix_out, prefix_lse = run(q, k_prefix, v_prefix, prefix=True)
                suffix_out, suffix_lse = run(
                    q, k_suffix, v_suffix, prefix=not causal_suffix
                )
                merged, _ = merge_state(prefix_out, prefix_lse, suffix_out, suffix_lse)

                # Zero scores give equal probability to each visible token.
                visible_suffix = (
                    torch.arange(1, suffix_len + 1, device="cuda").float()
                    if causal_suffix
                    else torch.full(
                        (suffix_len,), suffix_len, device="cuda", dtype=torch.float32
                    )
                )
                expected = (2 * prefix_len - 2 * visible_suffix) / (
                    prefix_len + visible_suffix
                )
                expected = expected[:, None, None].expand(-1, H, D_V)
                torch.testing.assert_close(merged.float(), expected, rtol=0, atol=0.01)
                torch.testing.assert_close(
                    prefix_lse,
                    torch.full_like(prefix_lse, math.log(prefix_len)),
                    rtol=0,
                    atol=1e-5,
                )
                torch.testing.assert_close(
                    suffix_lse,
                    visible_suffix.log()[:, None].expand(-1, H),
                    rtol=0,
                    atol=1e-5,
                )

    def test_flashinfer_cutlass_uniform_chunk_merge(self):
        self._check_uniform_attention(
            self._run_flashinfer,
            [(torch.bfloat16, D_QK, True), (torch.float16, D_QK, True)],
        )

    def test_trtllm_bf16_causal_chunk_merge(self):
        self._check_uniform_attention(self._run_trtllm, [(torch.bfloat16, D_QK, True)])

    def test_trtllm_fp16_dense_chunk_merge(self):
        # FlashInfer 0.6.18 exports native FP16 SeparateQkv at 128/128 for dense masks.
        self._check_uniform_attention(self._run_trtllm, [(torch.float16, 128, False)])


if __name__ == "__main__":
    unittest.main()
