"""Unit tests for fast routed_experts extraction, uint8 host cache, rank-0 guard, and zero-copy b64."""

from __future__ import annotations

import types
import unittest
from unittest import mock

import numpy as np
import pybase64
import torch
from sglang.srt.managers.detokenizer_manager import DetokenizerManager
from sglang.srt.managers.io_struct import (
    BatchTokenIDOutput,
    GenerateReqInput,
    TokenizedGenerateReqInput,
)
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.state_capturer import routed_experts as re_mod
from sglang.srt.state_capturer.base import (
    BaseDeviceCache,
    BaseTopkCapturer,
    encode_tensor_b64,
)
from sglang.srt.state_capturer.routed_experts import (
    RoutedExpertsCapturer,
    extract_routed_experts_from_meta_info,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


def _make_capturer(
    num_tokens: int, num_layers: int, topk_size: int, host_dtype: torch.dtype
):
    cap = object.__new__(BaseTopkCapturer)
    cap.num_layers = num_layers
    cap.topk_size = topk_size
    buf = torch.empty((num_tokens, num_layers, topk_size), dtype=host_dtype)
    if host_dtype == torch.uint8:
        buf.copy_(
            (torch.arange(buf.numel(), dtype=torch.int64) % 256)
            .to(torch.uint8)
            .reshape_as(buf)
        )
    elif host_dtype == torch.int16:
        buf.copy_(
            (torch.arange(buf.numel(), dtype=torch.int64) % 512)
            .to(torch.int16)
            .reshape_as(buf)
        )
    else:
        buf.copy_((torch.arange(buf.numel(), dtype=torch.int32) % 512).reshape_as(buf))
    cap.host_cache = types.SimpleNamespace(buffer=buf)
    cap.device_cache = BaseDeviceCache(
        max_batch_size=16,
        num_layers=num_layers,
        topk_size=topk_size,
        device="cpu",
        name="test_routed_experts",
    )
    return cap


class FastGetTopkTest(unittest.TestCase):
    def test_get_topk_patterns_and_dtypes(self):
        num_tokens, L, K = 512, 3, 4
        patterns = {
            "empty_seqlen1": ([], 1, 0),
            "empty_start_eq_end": ([5, 6, 7], 4, 3),
            "single": ([42], 2, 0),
            "contiguous": (list(range(10, 74)), 65, 0),
            "contiguous_with_start_len": (list(range(10, 74)), 65, 12),
            "multi_run": (
                list(range(32)) + list(range(100, 140)) + list(range(300, 348)),
                121,
                0,
            ),
            "scattered": ([i * 3 for i in range(64)], 65, 0),
        }

        for host_dtype in (torch.uint8, torch.int16, torch.int32):
            cap = _make_capturer(num_tokens, L, K, host_dtype)
            for name, (slots, seqlen, start_len) in patterns.items():
                pool_row = torch.zeros(
                    max(len(slots) + 1, seqlen + 1), dtype=torch.int32
                )
                if slots:
                    pool_row[: len(slots)] = torch.tensor(slots, dtype=torch.int32)
                req_to_token_pool = types.SimpleNamespace(
                    req_to_token=pool_row.unsqueeze(0)
                )

                for req_dtype in (None, torch.int32, torch.int16, torch.uint8):
                    with self.subTest(
                        host_dtype=host_dtype, pattern=name, req_dtype=req_dtype
                    ):
                        got = cap.get_topk(
                            req_pool_idx=0,
                            seqlen=seqlen,
                            req_to_token_pool=req_to_token_pool,
                            start_len=start_len,
                            dtype=req_dtype,
                        )
                        if req_dtype is None or req_dtype == torch.uint8 and host_dtype != torch.uint8:
                            expected_dtype = host_dtype
                        else:
                            expected_dtype = req_dtype
                        self.assertEqual(got.dtype, expected_dtype)

                        end_len = max(0, seqlen - 1)
                        s_len = min(start_len, end_len)
                        expected_idx = pool_row[s_len:end_len].to(torch.int64)
                        expected = cap.host_cache.buffer[expected_idx].to(
                            expected_dtype
                        )
                        self.assertEqual(tuple(got.shape), (len(expected_idx), L, K))
                        self.assertTrue(torch.equal(got, expected))

                        # Mutating returned tensor must never corrupt host_cache.
                        if got.numel() > 0:
                            orig_val = cap.host_cache.buffer[expected_idx[0]].clone()
                            got[0].fill_(77)
                            self.assertTrue(
                                torch.equal(
                                    cap.host_cache.buffer[expected_idx[0]], orig_val
                                )
                            )

    def test_on_forward_end_casts_to_uint8_and_drops_shared_cols(self):
        cap = _make_capturer(32, 2, 4, torch.uint8)
        cap.device_cache = BaseDeviceCache(
            max_batch_size=8,
            num_layers=2,
            topk_size=5,  # 4 routed + 1 fused shared (-1)
            device="cpu",
            name="test_routed_experts",
        )
        cap.device_cache.buffer.fill_(-1)
        routed = torch.tensor(
            [
                [[0, 1, 254, 255], [10, 20, 30, 40]],
                [[5, 6, 7, 8], [100, 150, 200, 255]],
            ],
            dtype=torch.int32,
        )
        cap.device_cache.buffer[:2, :, :4] = routed
        fb = types.SimpleNamespace(
            out_cache_loc=torch.tensor([4, 9], dtype=torch.int64),
            out_cache_loc_virtual=None,
        )

        # Overlap path (no_copy_to_cpu=True)
        out = cap.on_forward_end(
            fb, can_run_graph=False, cuda_graph_batch=None, no_copy_to_cpu=True
        )
        self.assertIsNotNone(out)
        self.assertEqual(out.topk.dtype, torch.uint8)
        out.finalize()
        self.assertTrue(
            torch.equal(cap.host_cache.buffer[4], routed[0].to(torch.uint8))
        )
        self.assertTrue(
            torch.equal(cap.host_cache.buffer[9], routed[1].to(torch.uint8))
        )

        # Inline sync path (no_copy_to_cpu=False)
        fb2 = types.SimpleNamespace(
            out_cache_loc=torch.tensor([12, 15], dtype=torch.int64),
            out_cache_loc_virtual=None,
        )
        ret = cap.on_forward_end(
            fb2, can_run_graph=False, cuda_graph_batch=None, no_copy_to_cpu=False
        )
        self.assertIsNone(ret)
        self.assertTrue(
            torch.equal(cap.host_cache.buffer[12], routed[0].to(torch.uint8))
        )
        self.assertTrue(
            torch.equal(cap.host_cache.buffer[15], routed[1].to(torch.uint8))
        )


class RoutedExpertsHostDtypeSelectionTest(unittest.TestCase):
    def test_host_and_device_dtype_selected_from_expert_count(self):
        cases = [
            (types.SimpleNamespace(n_routed_experts=256), 0, torch.uint8, torch.int16),
            (types.SimpleNamespace(num_experts=128), 0, torch.uint8, torch.int16),
            (types.SimpleNamespace(num_local_experts=8), 0, torch.uint8, torch.int16),
            (types.SimpleNamespace(n_routed_experts=288), 0, torch.int16, torch.int16),
            (
                types.SimpleNamespace(n_routed_experts=256, zero_expert_num=16),
                0,
                torch.int16,
                torch.int16,
            ),
            (
                types.SimpleNamespace(n_routed_experts=32767),
                1,
                torch.int16,
                torch.int32,
            ),
            (
                types.SimpleNamespace(n_routed_experts=40000),
                0,
                torch.int32,
                torch.int32,
            ),
            (types.SimpleNamespace(), 0, torch.int32, torch.int32),
        ]
        for text_cfg, num_shared, expected_host_dtype, expected_device_dtype in cases:
            text_cfg.num_experts_per_tok = 4
            text_cfg.num_hidden_layers = 2
            model_cfg = types.SimpleNamespace(hf_text_config=text_cfg)
            with (
                mock.patch.object(
                    re_mod,
                    "get_schedule",
                    return_value=types.SimpleNamespace(chunked_prefill_size=4),
                ),
                mock.patch.object(
                    re_mod,
                    "get_parallel",
                    return_value=types.SimpleNamespace(num_dp_ranks=1, attn_tp_size=1),
                ),
                mock.patch.object(
                    re_mod, "_is_scattered_a2a_backend", return_value=False
                ),
                mock.patch(
                    "sglang.srt.state_capturer.base.ALLOC_MEMORY_FUNCS",
                    {
                        "cpu": lambda shape, dtype, device, **kw: torch.empty(
                            shape, dtype=dtype, device=device
                        )
                    },
                ),
            ):
                cap = RoutedExpertsCapturer(
                    model_config=model_cfg,
                    num_tokens=8,
                    max_running_requests=2,
                    num_fused_shared_experts=num_shared,
                    device="cpu",
                )
            self.assertEqual(cap.host_cache.buffer.dtype, expected_host_dtype)
            self.assertEqual(cap.device_cache.buffer.dtype, expected_device_dtype)


class RankZeroGuardTest(unittest.TestCase):
    def test_non_output_ranks_skip_get_topk(self):
        processor = object.__new__(SchedulerBatchResultProcessor)
        object.__setattr__(processor, "req_to_token_pool", types.SimpleNamespace())

        for pp_rank, attn_tp_rank, attn_cp_rank, should_collect in [
            (0, 0, 0, True),
            (0, 1, 0, False),
            (1, 0, 0, False),
            (0, 0, 1, False),
        ]:
            with self.subTest(pp=pp_rank, tp=attn_tp_rank, cp=attn_cp_rank):
                req = types.SimpleNamespace(
                    rid="r1",
                    return_routed_experts=True,
                    routed_experts_start_len=0,
                    routed_experts_dtype="uint8",
                    origin_input_ids=[1, 2, 3],
                    output_ids_through_stop=[4],
                    seqlen=4,
                    cached_tokens=0,
                    kv=types.SimpleNamespace(req_pool_idx=0),
                    routed_experts=None,
                )
                capturer = mock.MagicMock()
                capturer.get_topk.return_value = torch.zeros(
                    (3, 2, 4), dtype=torch.uint8
                )
                parallel = types.SimpleNamespace(
                    pp_rank=pp_rank,
                    attn_tp_rank=attn_tp_rank,
                    attn_cp_rank=attn_cp_rank,
                )
                with (
                    mock.patch(
                        "sglang.srt.managers.scheduler_components.batch_result_processor.get_parallel",
                        return_value=parallel,
                    ),
                    mock.patch(
                        "sglang.srt.managers.scheduler_components.batch_result_processor.get_global_experts_capturer",
                        return_value=capturer,
                    ),
                ):
                    processor._maybe_collect_routed_experts(req)

                if should_collect:
                    capturer.get_topk.assert_called_once()
                    self.assertIsNotNone(req.routed_experts)
                else:
                    capturer.get_topk.assert_not_called()
                    self.assertIsNone(req.routed_experts)

    def test_create_skips_non_rank0_unless_scattered_a2a(self):
        features = types.SimpleNamespace(enable_return_routed_experts=True)
        model = types.SimpleNamespace(
            shared_experts_config=types.SimpleNamespace(num_fused_shared_experts=0)
        )
        text_cfg = types.SimpleNamespace(
            n_routed_experts=128,
            num_experts_per_tok=4,
            num_hidden_layers=2,
        )
        model_cfg = types.SimpleNamespace(hf_text_config=text_cfg)

        cases = [
            (0, 0, 0, False, True),
            (0, 1, 0, False, False),
            (1, 0, 0, False, False),
            (0, 0, 1, False, False),
            (0, 1, 0, True, True),
            (1, 0, 0, True, True),
            (0, 0, 1, True, True),
        ]
        for pp_rank, attn_tp_rank, attn_cp_rank, scattered, expect_created in cases:
            with self.subTest(
                pp=pp_rank,
                tp=attn_tp_rank,
                cp=attn_cp_rank,
                scattered=scattered,
            ):
                parallel = types.SimpleNamespace(
                    pp_rank=pp_rank,
                    attn_tp_rank=attn_tp_rank,
                    attn_cp_rank=attn_cp_rank,
                    num_dp_ranks=1,
                    attn_tp_size=2,
                )
                with (
                    mock.patch.object(
                        re_mod,
                        "get_exec",
                        return_value=types.SimpleNamespace(features=features),
                    ),
                    mock.patch.object(
                        re_mod,
                        "get_schedule",
                        return_value=types.SimpleNamespace(chunked_prefill_size=4),
                    ),
                    mock.patch.object(re_mod, "get_parallel", return_value=parallel),
                    mock.patch.object(
                        re_mod, "_is_scattered_a2a_backend", return_value=scattered
                    ),
                    mock.patch.object(
                        re_mod, "is_dp_attention_enabled", return_value=False
                    ),
                    mock.patch(
                        "sglang.srt.state_capturer.base.ALLOC_MEMORY_FUNCS",
                        {
                            "cpu": lambda shape, dtype, device, **kw: torch.empty(
                                shape, dtype=dtype, device=device
                            )
                        },
                    ),
                ):
                    cap = RoutedExpertsCapturer.create(
                        model=model,
                        model_config=model_cfg,
                        num_tokens=8,
                        max_running_requests=2,
                        device="cpu",
                    )
                if expect_created:
                    self.assertIsInstance(cap, RoutedExpertsCapturer)
                else:
                    self.assertIsNone(cap)


class ZeroCopyB64AndDtypePropagationTest(unittest.TestCase):
    def test_zero_copy_b64_roundtrip_all_shapes_and_dtypes(self):
        for dtype, np_dtype, dtype_str in (
            (torch.int32, np.int32, "int32"),
            (torch.uint8, np.uint8, "uint8"),
        ):
            for shape in ((0, 3, 4), (1, 3, 4), (16, 3, 4)):
                t = (
                    (torch.arange(int(np.prod(shape)), dtype=torch.int64) % 256)
                    .to(dtype)
                    .reshape(shape)
                )
                expected_b64 = pybase64.b64encode(t.numpy().tobytes()).decode("utf-8")
                got_b64 = encode_tensor_b64(t)
                self.assertEqual(got_b64, expected_b64)

                # Verify DetokenizerManager helper and extract_routed_experts_from_meta_info roundtrip.
                encoded_list = DetokenizerManager._b64_encode_per_request([t, None])
                self.assertEqual(encoded_list, [expected_b64, None])
                decoded = extract_routed_experts_from_meta_info(
                    {
                        "meta_info": {
                            "routed_experts": encoded_list[0],
                            "routed_experts_dtype": dtype_str,
                        }
                    }
                )
                self.assertEqual(decoded.dtype, np_dtype)
                np.testing.assert_array_equal(decoded, t.numpy().reshape(-1))

            # Non-contiguous tensor
            t_nc = torch.arange(24, dtype=dtype).reshape(2, 3, 4).transpose(0, 1)
            self.assertFalse(t_nc.is_contiguous())
            self.assertEqual(
                encode_tensor_b64(t_nc),
                pybase64.b64encode(t_nc.contiguous().numpy().tobytes()).decode("utf-8"),
            )

    def test_routed_experts_dtype_struct_and_detokenizer_propagation(self):
        # 1) GenerateReqInput -> __getitem__ -> TokenizedGenerateReqInput -> Req
        gen = GenerateReqInput(
            text=["a", "b"],
            return_routed_experts=True,
            routed_experts_dtype="uint8",
        )
        gen.normalize_batch_and_arguments()
        item0 = gen[0]
        self.assertEqual(item0.routed_experts_dtype, "uint8")

        tok_req = TokenizedGenerateReqInput(
            rid="r0",
            input_text="a",
            input_ids=[1, 2],
            mm_inputs=None,
            sampling_params=None,
            return_logprob=False,
            logprob_start_len=0,
            top_logprobs_num=0,
            token_ids_logprob=None,
            stream=False,
            input_embeds=None,
            token_type_ids=None,
            return_routed_experts=True,
            routed_experts_dtype="uint8",
        )
        self.assertEqual(tok_req.routed_experts_dtype, "uint8")

        # 2) DetokenizerManager preserves per-request dtypes across mixed batch
        t_u8 = torch.tensor([[[1, 2], [3, 4]]], dtype=torch.uint8)
        t_i16 = torch.tensor([[[10, 20], [30, 40]]], dtype=torch.int16)
        detok = object.__new__(DetokenizerManager)
        detok.tokenizer = None
        detok.disable_tokenizer_batch_decode = False
        detok.trim_matched_stop = False
        detok._decode_batch_token_id_output = lambda obj: ["a", "b", "c"]

        batch_out = BatchTokenIDOutput(
            rids=["r0", "r1", "r2"],
            http_worker_ipcs=[None, None, None],
            finished_reasons=[None, None, None],
            decoded_texts=["", "", ""],
            decode_ids=[[1], [2], [3]],
            read_offsets=[0, 0, 0],
            output_ids=[[1], [2], [3]],
            skip_special_tokens=[True, True, True],
            spaces_between_special_tokens=[True, True, True],
            no_stop_trim=[False, False, False],
            prompt_tokens=[1, 1, 1],
            reasoning_tokens=[0, 0, 0],
            completion_tokens=[1, 1, 1],
            cached_tokens=[0, 0, 0],
            input_token_logprobs_val=None,
            input_token_logprobs_idx=None,
            output_token_logprobs_val=None,
            output_token_logprobs_idx=None,
            input_top_logprobs_val=None,
            input_top_logprobs_idx=None,
            output_top_logprobs_val=None,
            output_top_logprobs_idx=None,
            input_token_ids_logprobs_val=None,
            input_token_ids_logprobs_idx=None,
            output_token_ids_logprobs_val=None,
            output_token_ids_logprobs_idx=None,
            output_token_entropy_val=None,
            output_token_sampling_mask=None,
            output_hidden_states=None,
            routed_experts=[t_u8, t_i16, None],
            indexer_topk=None,
            routed_experts_num_layers=2,
            routed_experts_dtypes=["uint8", "int16", None],
            placeholder_tokens_idx=None,
            placeholder_tokens_val=None,
        )
        str_out = detok.handle_batch_token_id_out(batch_out)
        self.assertEqual(str_out.routed_experts_dtypes, ["uint8", "int16", None])
        self.assertEqual(str_out.routed_experts[0], encode_tensor_b64(t_u8))
        self.assertEqual(str_out.routed_experts[1], encode_tensor_b64(t_i16))
        self.assertIsNone(str_out.routed_experts[2])

        # Omitting routed_experts_dtypes defaults to backwards-compatible int32 wire encoding
        batch_out.routed_experts_dtypes = None
        str_out2 = detok.handle_batch_token_id_out(batch_out)
        self.assertIsNone(str_out2.routed_experts_dtypes)
        self.assertEqual(
            str_out2.routed_experts[0], encode_tensor_b64(t_u8.to(torch.int32))
        )
        self.assertEqual(
            str_out2.routed_experts[1], encode_tensor_b64(t_i16.to(torch.int32))
        )

    def test_output_streamer_num_layers_falls_back_to_model_config_when_capturer_is_none(
        self,
    ):
        from sglang.srt.disaggregation.utils import DisaggregationMode
        from sglang.srt.managers.scheduler_components.output_streamer import (
            _GenerationStreamAccumulator,
        )
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        model_config = types.SimpleNamespace(
            hf_text_config=types.SimpleNamespace(num_hidden_layers=106)
        )
        acc = _GenerationStreamAccumulator(
            return_logprob=False,
            return_hidden_states=False,
            return_routed_experts=True,
            return_indexer_topk=False,
            spec_algorithm=SpeculativeAlgorithm.NONE,
            disaggregation_mode=DisaggregationMode.NULL,
            default_stream_interval=1,
            default_force_stream_interval=100,
            get_cached_tokens_details=lambda req: None,
            is_request_level_customized_info_key=lambda key: False,
            current_weight_version=None,
            model_config=model_config,
        )
        acc.rids.append("r0")
        with mock.patch(
            "sglang.srt.state_capturer.routed_experts.get_global_experts_capturer",
            return_value=None,
        ):
            payload = acc.to_payload(dp_rank=0, is_idle_batch=False)
        self.assertIsNotNone(payload)
        self.assertEqual(payload.routed_experts_num_layers, 106)

    def test_multi_tokenizer_batch_split_preserves_routed_experts_metadata(self):
        from sglang.srt.managers.multi_tokenizer_mixin import _handle_output_by_indices

        t0 = torch.tensor([[[1, 2]]], dtype=torch.uint8)
        t1 = torch.tensor([[[3, 4]]], dtype=torch.int16)
        t2 = torch.tensor([[[5, 6]]], dtype=torch.uint8)
        batch_out = BatchTokenIDOutput(
            rids=["r0", "r1", "r2"],
            http_worker_ipcs=[None, None, None],
            finished_reasons=[None, None, None],
            decoded_texts=["", "", ""],
            decode_ids=[[1], [2], [3]],
            read_offsets=[0, 0, 0],
            output_ids=[[1], [2], [3]],
            skip_special_tokens=[True, True, True],
            spaces_between_special_tokens=[True, True, True],
            no_stop_trim=[False, False, False],
            prompt_tokens=[1, 1, 1],
            reasoning_tokens=[0, 0, 0],
            completion_tokens=[1, 1, 1],
            cached_tokens=[0, 0, 0],
            input_token_logprobs_val=None,
            input_token_logprobs_idx=None,
            output_token_logprobs_val=None,
            output_token_logprobs_idx=None,
            input_top_logprobs_val=None,
            input_top_logprobs_idx=None,
            output_top_logprobs_val=None,
            output_top_logprobs_idx=None,
            input_token_ids_logprobs_val=None,
            input_token_ids_logprobs_idx=None,
            output_token_ids_logprobs_val=None,
            output_token_ids_logprobs_idx=None,
            output_token_entropy_val=None,
            output_token_sampling_mask=None,
            output_hidden_states=None,
            routed_experts=[t0, t1, t2],
            indexer_topk=None,
            routed_experts_num_layers=61,
            routed_experts_dtypes=["uint8", "int16", "uint8"],
            placeholder_tokens_idx=None,
            placeholder_tokens_val=None,
        )
        sub = _handle_output_by_indices(batch_out, [0, 2])
        self.assertEqual(sub.rids, ["r0", "r2"])
        self.assertEqual(sub.routed_experts_num_layers, 61)
        self.assertEqual(sub.routed_experts_dtypes, ["uint8", "uint8"])
        self.assertEqual(len(sub.routed_experts), 2)
        self.assertTrue(torch.equal(sub.routed_experts[0], t0))
        self.assertTrue(torch.equal(sub.routed_experts[1], t2))


if __name__ == "__main__":
    unittest.main()
