"""Prefix packing must apply checkpoint scales once and preserve fallback layouts."""

import unittest
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention import (
    tokenspeed_mla_backend,
    triton_backend,
    trtllm_mla_backend,
)
from sglang.srt.models.deepseek_common.attention_forward_methods import forward_mha
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

FP8 = torch.float8_e4m3fn


def _pack_on_cpu(k_nope, k_pe, v, *, k_scale_inv=1.0, v_scale_inv=1.0, **kwargs):
    k = torch.cat((k_nope, k_pe.expand(-1, k_nope.shape[1], -1)), dim=-1)
    return (k.float() * k_scale_inv).to(FP8), (v.float() * v_scale_inv).to(FP8)


def _quantize_on_cpu(value, scale):
    return (value.float() / scale).to(FP8), scale


def _attention(q, k, v):
    scores = torch.einsum("qhd,khd->qhk", q.float(), k.float()) * 0.5
    output = torch.einsum("qhk,khd->qhd", scores.softmax(-1), v.float())
    return output, scores.logsumexp(-1)


def _merge_on_cpu(prefix, prefix_lse, suffix, suffix_lse, output, output_lse):
    combined = torch.logaddexp(prefix_lse, suffix_lse)
    output.copy_(
        prefix * (prefix_lse - combined).exp().unsqueeze(-1)
        + suffix * (suffix_lse - combined).exp().unsqueeze(-1)
    )
    output_lse.copy_(combined)


class TestPrefixKVPack(CustomTestCase):
    def test_prepacked_values_keep_their_descales(self):
        layer = SimpleNamespace(
            k_scale_float=2.0,
            v_scale_float=0.5,
            k_scale=torch.tensor(2.0),
            v_scale=torch.tensor(0.5),
        )
        q = torch.ones(1, 2, 4, dtype=torch.bfloat16)
        k = torch.full((2, 2, 4), 3.0).to(FP8)
        v = torch.full((2, 2, 2), 4.0).to(FP8)
        with patch.object(trtllm_mla_backend, "scaled_fp8_quant", _quantize_on_cpu):
            _, packed_k, packed_v, k_scale, v_scale = (
                trtllm_mla_backend._quantize_fp8_qkv(q, k, v, layer)
            )
        self.assertEqual((k_scale, v_scale), (2.0, 0.5))
        torch.testing.assert_close(
            packed_k.float() * k_scale,
            torch.full_like(q, 6.0).expand(2, -1, -1).float(),
        )
        torch.testing.assert_close(
            packed_v.float() * v_scale, torch.full_like(v.float(), 2.0)
        )
        self.assertIs(packed_k, k)
        self.assertIs(packed_v, v)

    def test_chunked_and_packed_prefix_attention(self):
        providers = (
            (trtllm_mla_backend, trtllm_mla_backend.TRTLLMMLABackend, False),
            (
                tokenspeed_mla_backend,
                tokenspeed_mla_backend.TokenspeedMLABackend,
                False,
            ),
            (triton_backend, triton_backend.TritonAttnBackend, True),
        )
        for module, provider, pack_all in providers:
            for enabled in (False, True):
                with self.subTest(provider=provider.__name__, enabled=enabled):
                    self._check_prefix(module, provider, pack_all, enabled)
                    if pack_all:
                        self._check_prefix(
                            module, provider, pack_all, enabled, fused=True
                        )
                    if provider is trtllm_mla_backend.TRTLLMMLABackend:
                        self._check_prefix(
                            module, provider, pack_all, enabled, dtype=torch.float16
                        )

    def _check_prefix(
        self, module, provider, pack_all, enabled, *, fused=False, dtype=torch.bfloat16
    ):
        is_trt = provider is trtllm_mla_backend.TRTLLMMLABackend
        backend = SimpleNamespace(data_type=FP8, pack_all_prefix_chunks=pack_all)
        backend.pack_prefix_chunk_kv = MethodType(
            provider.pack_prefix_chunk_kv, backend
        )
        latents = torch.arange(24, dtype=dtype).reshape(3, 8) / 16
        rope = torch.arange(6, dtype=dtype).reshape(3, 1, 2) / 8
        q = torch.full((1, 2, 4), 0.25, dtype=dtype)
        if not is_trt:
            q = q.to(FP8)
        projection_dtypes = []

        class Attention:
            k_scale_float = 2.0
            v_scale_float = 0.5
            k_scale = torch.tensor(2.0)
            v_scale = torch.tensor(0.5)

            def __call__(self, q, k, v, batch, **kwargs):
                if batch.fused_prefix_k is not None:
                    k = torch.cat((batch.fused_prefix_k.float(), k.float()))
                    v = torch.cat((batch.fused_prefix_v.float(), v.float()))
                    return _attention(q, k, v)[0]
                if is_trt:
                    q, k, v, ks, vs = trtllm_mla_backend._quantize_fp8_qkv(
                        q, k, v, self
                    )
                    return _attention(q, k.float() * ks, v.float() * vs)
                return _attention(q, k, v)

        class Model(forward_mha.DeepseekMHAForwardMixin):
            num_local_heads = 2
            qk_nope_head_dim = 2
            qk_rope_head_dim = 2
            v_head_dim = 2
            attn_mha = Attention()

            def _get_mla_kv_buffer(self, indices, dtype, batch):
                projection_dtypes.append(dtype)
                return latents[indices].to(dtype), rope[indices].to(dtype)

            def kv_b_proj(self, value):
                return value, None

        batch = SimpleNamespace(
            extend_prefix_lens_cpu=[3],
            get_max_chunk_capacity=lambda: 8,
            prefix_all_kv_indices=torch.arange(3),
            num_prefix_chunks=1 if fused else 2,
            fused_prefix_k=None,
            fused_prefix_v=None,
            set_attn_attend_prefix_cache=lambda value: None,
            prefix_chunk_num_tokens=[1, 2],
            prefix_chunk_kv_indices=[torch.tensor([0]), torch.tensor([1, 2])],
            prefix_chunk_seq_lens_cpu=[[1], [2]],
            prefix_chunk_starts_cpu=[[0], [1]],
            set_prefix_chunk_idx=lambda index: None,
        )
        with (
            envs.SGLANG_OPT_TRTLLM_MLA_FUSED_CHUNK_KV_PACK.override(enabled),
            patch.object(module, "mla_kv_pack_quantize_fp8", _pack_on_cpu),
            patch.object(trtllm_mla_backend, "scaled_fp8_quant", _quantize_on_cpu),
            patch.object(forward_mha, "resolve_attn_backend", return_value=backend),
            patch.object(
                forward_mha,
                "get_parallel",
                return_value=SimpleNamespace(dcp_enabled=False),
            ),
            patch.object(
                forward_mha,
                "all_gather_kv_cache_for_mha_chunk_extend",
                side_effect=lambda k, r, *args: (k, r),
            ),
            patch.dict(
                "sys.modules",
                {
                    "sglang.srt.layers.attention.merge_state": SimpleNamespace(
                        merge_state=_merge_on_cpu
                    )
                },
            ),
        ):
            if fused:
                output = Model()._fused_prefix_extend_attn_mha(
                    q, torch.zeros_like(q), torch.zeros(1, 2, 2), batch
                )
            else:
                output = Model()._chunked_prefix_attn_mha(
                    q,
                    torch.zeros(1, 2, 2),
                    torch.full((1, 2), -torch.inf),
                    batch,
                )
        projected = latents.reshape(3, 2, 4)
        ks, vs = (2.0, 0.5) if is_trt else (1.0, 1.0)
        k, v = _pack_on_cpu(
            projected[..., :2],
            rope,
            projected[..., 2:],
            k_scale_inv=1 / ks,
            v_scale_inv=1 / vs,
        )
        if fused:
            k = torch.cat((k.float(), torch.zeros_like(q).float()))
            v = torch.cat((v.float(), torch.zeros(1, 2, 2)))
            self.assertIsNone(batch.fused_prefix_k)
            self.assertIsNone(batch.fused_prefix_v)
        expected, _ = _attention(q, k.float() * ks, v.float() * vs)
        torch.testing.assert_close(output, expected)
        self.assertTrue(projection_dtypes)
        self.assertEqual(set(projection_dtypes), {dtype})

    def test_non_fp8_backend_declines_packing(self):
        backend = SimpleNamespace(data_type=torch.bfloat16)
        with envs.SGLANG_OPT_TRTLLM_MLA_FUSED_CHUNK_KV_PACK.override(True):
            result = trtllm_mla_backend.TRTLLMMLABackend.pack_prefix_chunk_kv(
                backend, None, None, None, None
            )
        self.assertIsNone(result)

    def test_unsupported_pack_layout_keeps_the_fallback(self):
        backend = SimpleNamespace(data_type=FP8)
        for dtype, rope_dim, tokens in (
            (torch.float32, 2, 1),
            (torch.bfloat16, 0, 1),
            (torch.bfloat16, 3, 1),
            (torch.bfloat16, 2, 0),
        ):
            with (
                self.subTest(dtype=dtype, rope_dim=rope_dim, tokens=tokens),
                envs.SGLANG_OPT_TRTLLM_MLA_FUSED_CHUNK_KV_PACK.override(True),
            ):
                k_nope = torch.empty(tokens, 2, 2, dtype=dtype)
                k_pe = torch.empty(tokens, 1, rope_dim, dtype=dtype)
                v = torch.empty_like(k_nope)
                packed = trtllm_mla_backend.TRTLLMMLABackend.pack_prefix_chunk_kv(
                    backend, None, k_nope, k_pe, v
                )
                self.assertIsNone(packed)


if __name__ == "__main__":
    unittest.main()
