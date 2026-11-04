"""Unit tests for the flat sampling mask response format."""

import asyncio
import base64
import unittest

import numpy as np

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.tokenizer_manager import (
    ReqState,
    TokenizerManager,
    _build_flat_sampling_mask_fields,
)
from sglang.srt.observability.req_time_stats import APIServerReqTimeStats
from sglang.srt.sampling.sampling_mask import SamplingMaskChunk

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

FLAT_KEYS = (
    "output_token_sampling_mask_flat",
    "output_token_sampling_mask_lengths",
    "output_token_sampling_logprobs_flat",
)


def _chunk(rows, support_logprobs):
    """A chunk from per-row token ids; logprobs are exact in float32."""
    token_ids = [t for row in rows for t in row]
    count = len(token_ids) if support_logprobs else len(rows)
    return SamplingMaskChunk(
        lengths=np.asarray([len(row) for row in rows], dtype=np.int32),
        token_ids=np.asarray(token_ids, dtype=np.int32),
        logprobs=-np.arange(1, count + 1, dtype=np.float32) / 8,
    )


def _make_state(**obj_kwargs) -> ReqState:
    obj = GenerateReqInput(text="hello", return_sampling_mask=True, **obj_kwargs)
    obj.normalize_batch_and_arguments()
    return ReqState(
        out_list=[],
        finished=False,
        event=asyncio.Event(),
        obj=obj,
        time_stats=APIServerReqTimeStats(),
    )


def _feed(state, chunks):
    """Each chunk as the tokenizer receives it; the sampled token is each row's first id."""
    meta_info = {}
    for chunk in chunks:
        starts = np.concatenate(([0], np.cumsum(chunk.lengths)[:-1]))
        TokenizerManager._add_sampling_mask_output(
            meta_info=meta_info,
            state=state,
            chunk=chunk,
            output_ids=chunk.token_ids[starts].tolist(),
        )
    return meta_info


def _decode(fields, name):
    return np.frombuffer(
        base64.b64decode(fields[f"{name}_b64"]), dtype=fields[f"{name}_b64_dtype"]
    )


ROWS = [[[5, 7, 9], [2]], [[4, 1], [8, 6, 3, 0]], [[11]]]


class TestFlatSamplingMaskValidation(CustomTestCase):
    def test_flags_default_off_and_valid(self):
        for kwargs in ({}, {"return_flat_sampling_mask": True}):
            GenerateReqInput(text="hello", **kwargs).normalize_batch_and_arguments()

    def test_rejects_stream(self):
        req = GenerateReqInput(
            text="hello", stream=True, return_flat_sampling_mask=True
        )
        with self.assertRaisesRegex(ValueError, "stream"):
            req.normalize_batch_and_arguments()

    def test_rejects_output_store(self):
        req = GenerateReqInput(
            text="hello", return_flat_sampling_mask=True, return_outputs_via_store=True
        )
        with self.assertRaisesRegex(ValueError, "return_outputs_via_store"):
            req.normalize_batch_and_arguments()

    def test_b64_requires_flat_flag(self):
        req = GenerateReqInput(text="hello", return_flat_sampling_mask_b64=True)
        with self.assertRaisesRegex(ValueError, "return_flat_sampling_mask"):
            req.normalize_batch_and_arguments()

    def test_flags_propagate_to_batch_items(self):
        req = GenerateReqInput(
            text=["a", "b"],
            return_flat_sampling_mask=True,
            return_flat_sampling_mask_b64=True,
        )
        req.normalize_batch_and_arguments()
        for i in range(2):
            self.assertTrue(req[i].return_flat_sampling_mask)
            self.assertTrue(req[i].return_flat_sampling_mask_b64)


class TestFlatSamplingMaskFields(CustomTestCase):
    def test_flat_matches_nested_rows(self):
        for mode in ("selected", "support"):
            chunks = [_chunk(rows, mode == "support") for rows in ROWS]
            nested = _feed(_make_state(sampling_logprobs_mode=mode), chunks)
            flat_state = _make_state(
                sampling_logprobs_mode=mode, return_flat_sampling_mask=True
            )
            flat_meta = _feed(flat_state, chunks)
            with self.subTest(mode=mode):
                self.assertEqual(flat_meta, {})
                for b64 in (False, True):
                    fields = _build_flat_sampling_mask_fields(
                        flat_state.output_token_sampling_mask_chunks, return_b64=b64
                    )
                    ids, lengths, logprobs = (
                        _decode(fields, k) if b64 else np.asarray(fields[k])
                        for k in FLAT_KEYS
                    )
                    rows = np.split(ids, np.cumsum(lengths)[:-1])
                    self.assertEqual(
                        [r.tolist() for r in rows], nested["output_token_sampling_mask"]
                    )
                    self.assertEqual(
                        fields["output_token_sampling_mask_length"],
                        nested["output_token_sampling_mask_length"],
                    )
                    expected = nested["output_token_sampling_logprobs"]
                    if mode == "support":
                        expected = [v for row in expected for v in row]
                    self.assertEqual(logprobs.tolist(), expected)

    def test_support_mode_selected_logprobs_are_in_the_arrays(self):
        chunks = [_chunk(rows, True) for rows in ROWS]
        nested = _feed(_make_state(sampling_logprobs_mode="support"), chunks)
        fields = _build_flat_sampling_mask_fields(chunks, return_b64=True)
        ids, lengths, logprobs = (_decode(fields, k) for k in FLAT_KEYS)
        sampled = np.concatenate(
            [c.token_ids[np.cumsum(c.lengths) - c.lengths] for c in chunks]
        )
        self.assertEqual(
            logprobs[ids == np.repeat(sampled, lengths)].tolist(),
            nested["output_token_sampling_selected_logprobs"],
        )

    def test_b64_dtypes(self):
        fields = _build_flat_sampling_mask_fields(
            [_chunk(ROWS[0], True)], return_b64=True
        )
        self.assertEqual(
            [fields[f"{k}_b64_dtype"] for k in FLAT_KEYS], ["int32", "int32", "float32"]
        )
        self.assertFalse(set(FLAT_KEYS) & fields.keys())

    def test_no_rows(self):
        for b64 in (False, True):
            fields = _build_flat_sampling_mask_fields([], return_b64=b64)
            self.assertEqual(fields["output_token_sampling_mask_length"], 0)
            for key in FLAT_KEYS:
                values = _decode(fields, key) if b64 else fields[key]
                self.assertEqual(len(values), 0)


class TestSelectedLogprobsWithoutMask(CustomTestCase):
    def test_selected_mode_alone_returns_only_the_sampled_tokens_logprobs(self):
        """Without the mask the response carries the logprobs per output token and no
        support, even when flat mask arrays were asked for."""
        obj = GenerateReqInput(
            text="hello",
            sampling_logprobs_mode="selected",
            return_flat_sampling_mask=True,
        )
        obj.normalize_batch_and_arguments()
        state = ReqState(
            out_list=[],
            finished=False,
            event=asyncio.Event(),
            obj=obj,
            time_stats=APIServerReqTimeStats(),
        )
        meta_info = {}
        for logprobs in ([-0.5, -0.25], [-1.0]):
            TokenizerManager._add_sampling_mask_output(
                meta_info=meta_info,
                state=state,
                chunk=SamplingMaskChunk(
                    lengths=np.zeros(len(logprobs), dtype=np.int32),
                    token_ids=np.empty(0, dtype=np.int32),
                    logprobs=np.asarray(logprobs, dtype=np.float32),
                ),
                output_ids=[0] * len(logprobs),
            )
        self.assertEqual(
            meta_info, {"output_token_sampling_logprobs": [-0.5, -0.25, -1.0]}
        )
        self.assertEqual(state.output_token_sampling_mask_chunks, [])


if __name__ == "__main__":
    unittest.main()
