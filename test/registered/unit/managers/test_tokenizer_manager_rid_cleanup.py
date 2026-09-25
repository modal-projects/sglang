"""
Unit tests for rid_to_state cleanup in TokenizerManager.

Verifies that request IDs are properly removed from rid_to_state after
completion or abort, allowing resubmission with the same rid without
triggering "Duplicate request ID detected" errors.

Covers:
  - _handle_abort_req cleans up rid_to_state
  - _handle_batch_output cleans up rid_to_state on finished requests
  - _init_req_state rejects duplicate rids
  - Resubmission succeeds after cleanup
  - Handler failures clean up pending and dispatched requests
"""

import asyncio
import copy
import gc
import pickle
import threading
import unittest
import weakref
from array import array
from collections import deque
from contextlib import nullcontext
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import msgspec
from fastapi import HTTPException

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import (  # noqa: E402
    AbortReq,
    BatchStrOutput,
    BatchTokenizedGenerateReqInput,
    EmbeddingReqInput,
    EncoderDispatchErrorReq,
    GenerateReqInput,
    TokenizedGenerateReqInput,
)
from sglang.srt.managers.tokenizer_manager import (  # noqa: E402
    ReqState,
    TokenizerManager,
)
from sglang.srt.observability.req_time_stats import (  # noqa: E402
    APIServerReqTimeStats,
)
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.sampling_params import SamplingParams

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


_NOT_FINISHED = object()  # Sentinel: request has not finished yet

# ---------------------------------------------------------------------------
# Per-request field defaults for BatchStrOutput construction.
# Categorised by value shape so that _make_batch_str_output can assign
# type-appropriate defaults without hardcoding every field name.
# When a field is renamed upstream, the old name simply won't appear in
# msgspec.structs.fields() and the new name will fall through to the
# pattern-matching or safe fallback — no test breakage.
# ---------------------------------------------------------------------------

_PER_REQUEST_INT_FIELDS = frozenset(
    {
        "prompt_tokens",
        "completion_tokens",
        "reasoning_tokens",
        "cached_tokens",
        "retraction_counts",
        # Speculative-decoding int-scalar fields (current and historical names)
        "spec_verify_ct",
        "spec_accepted_drafts",
        "spec_num_correct_drafts",
    }
)

_PER_REQUEST_FLOAT_FIELDS = frozenset(
    {
        "output_token_entropy_val",
    }
)

_PER_REQUEST_NESTED_LIST_FIELDS = frozenset(
    {
        "output_ids",
        # Logprob fields
        "input_token_logprobs_val",
        "input_token_logprobs_idx",
        "output_token_logprobs_val",
        "output_token_logprobs_idx",
        "input_top_logprobs_val",
        "input_top_logprobs_idx",
        "output_top_logprobs_val",
        "output_top_logprobs_idx",
        "input_token_ids_logprobs_val",
        "input_token_ids_logprobs_idx",
        "output_token_ids_logprobs_val",
        "output_token_ids_logprobs_idx",
        # Speculative-decoding histogram fields (current and historical names)
        "spec_acceptance_histogram",
        "spec_correct_drafts_histogram",
    }
)

_PER_REQUEST_OPTIONAL_FIELDS = frozenset(
    {
        "output_hidden_states",
        "routed_experts",
        "indexer_topk",
        "placeholder_tokens_idx",
        "placeholder_tokens_val",
    }
)


def _make_tokenizer_manager(case) -> TokenizerManager:
    """Create a TokenizerManager with mocked dependencies, bypassing __init__.

    The config it reads comes from the bags, so the stand-in needs a published
    config rather than attributes on a mock.
    """
    override = get_context().override_server_args(speculative_algorithm=None)
    override.install()
    case.addCleanup(override.restore)
    tm = TokenizerManager.__new__(TokenizerManager)
    tm.server_args = MagicMock()
    tm._config_updates = []
    tm.server_args.enable_trace = False
    tm.server_args.enable_metrics = False
    tm.server_args.enable_lora = False
    tm.server_args.speculative_algorithm = None
    tm.server_args.incremental_streaming_output = False
    tm.server_args.skip_tokenizer_init = False
    tm.server_args.batch_notify_size = 1
    tm.server_args.weight_version = "1"
    tm.server_args.crash_dump_folder = ""
    tm.server_args.dp_size = 1
    tm.disaggregation_mode = "none"
    tm.rid_to_state = {}
    tm.mm_processor = None
    tm.encoder_dispatch_ready = {}
    tm.enable_metrics = False
    tm.enable_trace = False
    tm.enable_lora = False
    tm.incremental_streaming_output = False
    tm.allow_auto_truncate = False
    tm.skip_tokenizer_init = False
    tm.dump_requests_folder = ""
    tm.crash_dump_folder = ""
    tm.send_to_scheduler = MagicMock()
    tm._async_dispatch_to_scheduler = AsyncMock()
    return tm


def _make_req_state(rid: str = "test_rid") -> ReqState:
    """Create a minimal ReqState for testing."""
    obj = Mock(spec=GenerateReqInput)
    obj.rid = rid
    obj.stream = False
    obj.return_logprob = False
    obj.lora_path = None
    obj.log_metrics = False
    return ReqState(
        out_list=[],
        finished=False,
        event=asyncio.Event(),
        obj=obj,
        time_stats=APIServerReqTimeStats(),
    )


def _make_abort_req(rid: str, abort_message: str = "Aborted") -> AbortReq:
    """Create an AbortReq for testing."""
    return AbortReq(
        rid=rid,
        abort_all=False,
        finished_reason={"type": "abort", "message": abort_message},
        abort_message=abort_message,
    )


def _make_batch_str_output(rid: str, finished_reason=None) -> BatchStrOutput:
    """Create a minimal BatchStrOutput for a single request.

    Uses struct field introspection so that new or renamed fields in
    BatchStrOutput don't break this test.  Only the fields that matter for
    test logic (rids, finished_reasons, output_strs) are set explicitly;
    all others receive type-appropriate defaults based on naming patterns.
    Fields with class-level defaults are left alone automatically.
    """
    if finished_reason is _NOT_FINISHED:
        fr = None
    elif finished_reason is None:
        fr = {"type": "length"}
    else:
        fr = finished_reason

    kwargs = {}
    for f in msgspec.structs.fields(BatchStrOutput):
        if f.name == "rids":
            kwargs[f.name] = [rid]
        elif f.name == "finished_reasons":
            kwargs[f.name] = [fr]
        elif f.name == "output_strs":
            kwargs[f.name] = ["hello"]
        elif f.name in _PER_REQUEST_INT_FIELDS:
            kwargs[f.name] = [0]
        elif f.name in _PER_REQUEST_FLOAT_FIELDS:
            kwargs[f.name] = [0.0]
        elif f.name in _PER_REQUEST_NESTED_LIST_FIELDS:
            kwargs[f.name] = [[]]
        elif f.name in _PER_REQUEST_OPTIONAL_FIELDS:
            kwargs[f.name] = [None]
        # Fields with class defaults — skip, let the default be used
        elif (
            f.default is not msgspec.NODEFAULT
            or f.default_factory is not msgspec.NODEFAULT
        ):
            continue
        # Unknown required field — provide a safe per-request default.
        # Most BatchStrOutput fields are per-request lists; [[]] works for
        # List[List[...]] and is unlikely to crash on [i] indexing for
        # List[int] either (the inner [] just means "no data").
        else:
            kwargs[f.name] = [[]]

    return BatchStrOutput(**kwargs)


class TestRidToStateCleanupOnAbort(CustomTestCase):
    """Test that _handle_abort_req removes rid from rid_to_state."""

    def test_abort_removes_rid_from_state(self):
        """After _handle_abort_req, rid should be removed from rid_to_state."""
        tm = _make_tokenizer_manager(self)
        rid = "abort_test_rid"
        state = _make_req_state(rid)
        tm.rid_to_state[rid] = state

        abort_req = _make_abort_req(rid)
        tm._handle_abort_req(abort_req)

        self.assertNotIn(rid, tm.rid_to_state)

    def test_abort_allows_resubmit_same_rid(self):
        """After abort, _init_req_state should accept the same rid again."""
        tm = _make_tokenizer_manager(self)
        rid = "resubmit_after_abort_rid"
        state = _make_req_state(rid)
        tm.rid_to_state[rid] = state

        abort_req = _make_abort_req(rid)
        tm._handle_abort_req(abort_req)

        # Resubmit with the same rid — should not raise
        obj = Mock(spec=GenerateReqInput)
        obj.rid = rid
        obj.is_single = True
        obj.received_time = 0.0
        obj.external_trace_header = None
        obj.bootstrap_room = None
        tm._init_req_state(obj)

        self.assertIn(rid, tm.rid_to_state)

    def test_abort_sets_finished_and_notifies(self):
        """_handle_abort_req should mark state as finished and set the event."""
        tm = _make_tokenizer_manager(self)
        rid = "abort_notify_rid"
        state = _make_req_state(rid)
        tm.rid_to_state[rid] = state

        abort_req = _make_abort_req(rid)
        tm._handle_abort_req(abort_req)

        self.assertTrue(state.finished)
        self.assertTrue(state.event.is_set())
        self.assertEqual(len(state.out_list), 1)
        self.assertEqual(
            state.out_list[0]["meta_info"]["finish_reason"]["type"], "abort"
        )


class TestAbortOutputPayload(CustomTestCase):
    """An abort chunk is often the only thing a client sees;
    it must carry the same optional fields as a normal finish chunk."""

    def test_abort_includes_prompt_token_ids_only_when_requested(self):
        """The abort chunk carries prompt_token_ids captured at tokenization,
        and omits the field when the request did not ask for them."""
        tm = _make_tokenizer_manager(self)
        with_ids = _make_req_state("abort_prompt_ids_rid")
        with_ids.prompt_token_ids = [1, 2, 3]
        without_ids = _make_req_state("abort_no_prompt_ids_rid")

        for state in (with_ids, without_ids):
            tm.rid_to_state[state.obj.rid] = state
            tm._handle_abort_req(_make_abort_req(state.obj.rid))

        self.assertEqual(with_ids.out_list[0]["prompt_token_ids"], [1, 2, 3])
        self.assertNotIn("prompt_token_ids", without_ids.out_list[0])

    def test_abort_output_ids_match_the_streaming_mode(self):
        """Only incremental streaming collapses the abort chunk to the last
        token; cumulative chunks supersede, so they carry the whole generation.
        """
        cases = [
            ("incremental stream", True, True, [7]),
            ("cumulative stream", True, False, [5, 6, 7]),
            ("non-stream", False, False, [5, 6, 7]),
        ]
        for name, is_stream, incremental, expected in cases:
            with self.subTest(name):
                tm = _make_tokenizer_manager(self)
                tm.incremental_streaming_output = incremental
                rid = f"abort_output_ids_{name}"
                state = _make_req_state(rid)
                state.obj.stream = is_stream
                state.output_ids = [5, 6, 7]
                tm.rid_to_state[rid] = state

                tm._handle_abort_req(_make_abort_req(rid))

                out = state.out_list[0]
                self.assertEqual(out["output_ids"], expected)
                self.assertEqual(out["meta_info"]["completion_tokens"], 3)


class TestRidToStateCleanupOnBatchOutput(CustomTestCase):
    """Test that _handle_batch_output removes rid from rid_to_state on completion."""

    def test_batch_output_removes_rid_on_finish(self):
        """When a request finishes in _handle_batch_output, rid should be removed."""
        tm = _make_tokenizer_manager(self)
        rid = "batch_finish_rid"
        state = _make_req_state(rid)
        tm.rid_to_state[rid] = state

        batch_output = _make_batch_str_output(rid)
        asyncio.run(tm._handle_batch_output(batch_output))

        self.assertNotIn(rid, tm.rid_to_state)

    def test_batch_output_allows_resubmit_after_finish(self):
        """After a request finishes, the same rid can be resubmitted."""
        tm = _make_tokenizer_manager(self)
        rid = "batch_resubmit_rid"
        state = _make_req_state(rid)
        tm.rid_to_state[rid] = state

        batch_output = _make_batch_str_output(rid)
        asyncio.run(tm._handle_batch_output(batch_output))

        # Resubmit with the same rid — should not raise
        obj = Mock(spec=GenerateReqInput)
        obj.rid = rid
        obj.is_single = True
        obj.received_time = 0.0
        obj.external_trace_header = None
        obj.bootstrap_room = None
        tm._init_req_state(obj)

        self.assertIn(rid, tm.rid_to_state)

    def test_batch_output_keeps_rid_when_not_finished(self):
        """When a request is not yet finished, rid should remain in rid_to_state."""
        tm = _make_tokenizer_manager(self)
        rid = "batch_ongoing_rid"
        state = _make_req_state(rid)
        tm.rid_to_state[rid] = state

        # finished_reason=_NOT_FINISHED means the request is still ongoing
        batch_output = _make_batch_str_output(rid, finished_reason=_NOT_FINISHED)
        asyncio.run(tm._handle_batch_output(batch_output))

        self.assertIn(rid, tm.rid_to_state)


class TestInitReqStateDuplicateDetection(CustomTestCase):
    """Test that _init_req_state raises ValueError for duplicate rids."""

    def test_duplicate_rid_raises_error(self):
        """_init_req_state should raise ValueError if rid already exists."""
        tm = _make_tokenizer_manager(self)
        rid = "duplicate_rid"
        state = _make_req_state(rid)
        tm.rid_to_state[rid] = state

        obj = Mock(spec=GenerateReqInput)
        obj.rid = rid
        obj.is_single = True
        obj.received_time = 0.0
        obj.external_trace_header = None
        obj.bootstrap_room = None

        with self.assertRaises(ValueError) as ctx:
            tm._init_req_state(obj)
        self.assertIn("Duplicate request ID", str(ctx.exception))

    def test_unique_rid_succeeds(self):
        """_init_req_state should succeed with a unique rid."""
        tm = _make_tokenizer_manager(self)
        rid = "unique_rid"

        obj = Mock(spec=GenerateReqInput)
        obj.rid = rid
        obj.is_single = True
        obj.received_time = 0.0
        obj.external_trace_header = None
        obj.bootstrap_room = None

        tm._init_req_state(obj)
        self.assertIn(rid, tm.rid_to_state)

    def test_rejected_batch_does_not_register_or_modify_any_owner(self):
        for request_type, collision in product(
            (GenerateReqInput, EmbeddingReqInput), ("batch", "existing")
        ):
            with self.subTest(request_type=request_type.__name__, collision=collision):
                tm = _make_tm_for_generate(self)
                tm._dispatch_to_scheduler = Mock()
                tm._tokenize_one_request = AsyncMock()
                tm._batch_tokenize_and_process = AsyncMock()
                existing = request_type(input_ids=[9], rid="occupied")
                existing.normalize_batch_and_arguments()
                tm._init_req_state(existing)
                owner = tm.rid_to_state["occupied"]
                owner.dispatched = True
                ready = owner.encoder_dispatch_ready = threading.Event()
                tm.encoder_dispatch_ready["occupied"] = ready
                rids = (
                    ["fresh-a", "fresh-b", "fresh-b"]
                    if collision == "batch"
                    else ["fresh-a", "fresh-b", "occupied"]
                )
                request = request_type(input_ids=[[1], [2], [3]], rid=rids)
                callback = (
                    tm.create_abort_task(request)
                    if request_type is GenerateReqInput
                    else None
                )

                async def reject():
                    generator = tm.generate_request(request)
                    with self.assertRaisesRegex(ValueError, "Duplicate request ID"):
                        await generator.__anext__()
                    await generator.aclose()
                    if callback is not None:
                        with patch(
                            "sglang.srt.managers.tokenizer_manager.asyncio.sleep",
                            AsyncMock(),
                        ):
                            await callback()

                with (
                    patch(
                        "sglang.srt.managers.tokenizer_manager.ReqState",
                        wraps=ReqState,
                    ) as make_state,
                    patch(
                        "sglang.srt.managers.tokenizer_manager.APIServerReqTimeStats",
                        wraps=APIServerReqTimeStats,
                    ) as make_time_stats,
                ):
                    asyncio.run(reject())
                self.assertEqual(set(tm.rid_to_state), {"occupied"})
                self.assertIs(tm.rid_to_state["occupied"], owner)
                self.assertTrue(owner.dispatched)
                self.assertFalse(owner.abort_sent)
                self.assertFalse(ready.is_set())
                self.assertIs(tm.encoder_dispatch_ready["occupied"], ready)
                self.assertFalse(tm._request_state_bindings)
                make_state.assert_not_called()
                make_time_stats.assert_not_called()
                tm._dispatch_to_scheduler.assert_not_called()
                tm._tokenize_one_request.assert_not_awaited()
                tm._batch_tokenize_and_process.assert_not_awaited()
                tm.request_logger.log_received_request.assert_not_called()

                retry = request_type(input_ids=[4], rid="fresh-a")
                retry.normalize_batch_and_arguments()
                tm._init_req_state(retry)
                self.assertIs(tm.rid_to_state["fresh-a"].obj, retry)

    def test_unique_batch_registers_each_normalized_request(self):
        for request_type in (GenerateReqInput, EmbeddingReqInput):
            with self.subTest(request_type=request_type.__name__):
                tm = _make_tokenizer_manager(self)
                request = request_type(
                    input_ids=[[1], [2], [3]], rid=["first", "second", "third"]
                )
                request.normalize_batch_and_arguments()
                tm._init_req_state(request)
                self.assertEqual(set(tm.rid_to_state), set(request.rid))
                for index, rid in enumerate(request.rid):
                    self.assertIs(tm.rid_to_state[rid].obj, request[index])
                    self.assertFalse(tm.rid_to_state[rid].dispatched)

    def test_registration_checks_duplicate_ids_after_normalization(self):
        """Internal registration must validate all IDs even after normalization."""
        for request_type in (GenerateReqInput, EmbeddingReqInput):
            with self.subTest(request_type=request_type.__name__):
                tm = _make_tokenizer_manager(self)
                request = request_type(
                    input_ids=[[1], [2], [3]], rid=["first", "second", "third"]
                )
                request.normalize_batch_and_arguments()
                request.rid[-1] = "second"
                with self.assertRaisesRegex(ValueError, "Duplicate request ID"):
                    tm._init_req_state(request)
                self.assertFalse(tm.rid_to_state)


class TestResubmitAfterCompletion(CustomTestCase):
    """End-to-end test: complete a request, then resubmit with the same rid."""

    def test_complete_then_resubmit_same_rid(self):
        """A request that completes normally should allow resubmission with the same rid."""
        tm = _make_tokenizer_manager(self)
        rid = "complete_resubmit_rid"

        # Phase 1: simulate a request in rid_to_state, then complete it
        state = _make_req_state(rid)
        tm.rid_to_state[rid] = state

        batch_output = _make_batch_str_output(rid, finished_reason={"type": "length"})
        asyncio.run(tm._handle_batch_output(batch_output))

        # rid should be cleaned up
        self.assertNotIn(rid, tm.rid_to_state)

        # Phase 2: resubmit with the same rid — should succeed
        obj = Mock(spec=GenerateReqInput)
        obj.rid = rid
        obj.is_single = True
        obj.received_time = 0.0
        obj.external_trace_header = None
        obj.bootstrap_room = None
        tm._init_req_state(obj)

        self.assertIn(rid, tm.rid_to_state)

    def test_abort_then_resubmit_same_rid(self):
        """An aborted request should allow resubmission with the same rid."""
        tm = _make_tokenizer_manager(self)
        rid = "abort_resubmit_rid"

        # Phase 1: simulate a request, then abort it
        state = _make_req_state(rid)
        tm.rid_to_state[rid] = state

        abort_req = _make_abort_req(rid)
        tm._handle_abort_req(abort_req)

        self.assertNotIn(rid, tm.rid_to_state)

        # Phase 2: resubmit with the same rid — should succeed
        obj = Mock(spec=GenerateReqInput)
        obj.rid = rid
        obj.is_single = True
        obj.received_time = 0.0
        obj.external_trace_header = None
        obj.bootstrap_room = None
        tm._init_req_state(obj)

        self.assertIn(rid, tm.rid_to_state)


class _DummyAsyncCM:
    """Reusable no-op async context manager (stands in for an RW lock)."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def _make_tm_for_generate(case) -> TokenizerManager:
    """Augment the mocked TokenizerManager with what generate_request needs."""
    tm = _make_tokenizer_manager(case)
    tm.server_args.language_only = False
    tm.server_args.tokenizer_worker_num = 1
    tm.server_args.enable_strict_thinking = False
    tm.auto_create_handle_loop = Mock()
    tm._set_default_priority = Mock()
    tm.request_logger = Mock()
    tm.tokenizer = None
    tm.model_config = SimpleNamespace(
        vocab_size=32000,
        hf_text_config=SimpleNamespace(vocab_size=32000),
    )
    tm.is_pause = False
    tm.is_pause_cond = asyncio.Condition()
    tm.model_update_lock = Mock()
    tm.model_update_lock.reader_lock = _DummyAsyncCM()
    tm._validate_and_resolve_lora = AsyncMock(return_value=None)
    return tm


def _make_generate_obj(rid, is_single):
    obj = MagicMock(spec=GenerateReqInput)
    obj.routed_dp_rank = None
    obj.is_single = is_single
    obj.rid = rid
    obj.received_time = 0.0
    obj.external_trace_header = None
    obj.bootstrap_room = None
    obj.max_thinking_tokens = None
    obj.normalize_batch_and_arguments = Mock()
    if not is_single:
        obj.__getitem__.side_effect = lambda i: Mock()
    return obj


class TestReleaseReqStatesOnFailure(CustomTestCase):
    """Direct tests for _release_req_states_on_failure."""

    def test_undelivered_single_is_dropped(self):
        tm = _make_tokenizer_manager(self)
        rid = "d_single"
        state = _make_req_state(rid)
        tm.rid_to_state[rid] = state
        tm._release_req_states_on_failure({rid: state})
        self.assertNotIn(rid, tm.rid_to_state)

    def test_undelivered_batch_removes_all(self):
        tm = _make_tokenizer_manager(self)
        rids = ["d0", "d1", "d2"]
        for r in rids:
            tm.rid_to_state[r] = _make_req_state(r)
        tm._release_req_states_on_failure(dict(tm.rid_to_state))
        for r in rids:
            self.assertNotIn(r, tm.rid_to_state)

    def test_ignores_already_removed(self):
        """A rid that is no longer present must not raise."""
        tm = _make_tokenizer_manager(self)
        tm.rid_to_state["p1"] = _make_req_state("p1")
        tm._release_req_states_on_failure(
            {**tm.rid_to_state, "already_gone": _make_req_state("already_gone")}
        )
        self.assertNotIn("p1", tm.rid_to_state)

    def test_dispatched_single_is_aborted_and_state_kept(self):
        tm = _make_tokenizer_manager(self)
        tm.server_args.tokenizer_worker_num = 1
        tm._dispatch_to_scheduler = Mock()
        tm.enable_metrics = True
        tm.metrics_collector = MagicMock()
        rid = "d_live"
        state = _make_req_state(rid)
        state.dispatched = True
        tm.rid_to_state[rid] = state
        tm._release_req_states_on_failure({rid: state})
        tm._release_req_states_on_failure({rid: state})

        sent = [c.args[0] for c in tm._dispatch_to_scheduler.call_args_list]
        self.assertEqual(
            [type(m) for m in sent], [AbortReq], "expected exactly one AbortReq"
        )
        self.assertEqual(sent[0].rid, rid)
        self.assertIn(rid, tm.rid_to_state)
        self.assertTrue(state.abort_sent)
        tm.metrics_collector.observe_one_aborted_request.assert_called_once()

    def test_dispatched_batch_aborts_delivered_and_drops_rest(self):
        tm = _make_tokenizer_manager(self)
        tm.server_args.tokenizer_worker_num = 1
        tm._dispatch_to_scheduler = Mock()
        delivered, undelivered = "d_delivered", "d_undelivered"
        live = _make_req_state(delivered)
        live.dispatched = True
        tm.rid_to_state[delivered] = live
        tm.rid_to_state[undelivered] = _make_req_state(undelivered)
        tm._release_req_states_on_failure(dict(tm.rid_to_state))

        sent = [c.args[0] for c in tm._dispatch_to_scheduler.call_args_list]
        self.assertEqual([type(m) for m in sent], [AbortReq])
        self.assertEqual(sent[0].rid, delivered)
        self.assertIn(delivered, tm.rid_to_state)
        self.assertNotIn(undelivered, tm.rid_to_state)

    def test_abort_failure_does_not_stop_cleanup(self):
        tm = _make_tokenizer_manager(self)
        tm.server_args.tokenizer_worker_num = 1
        tm._dispatch_to_scheduler = Mock(side_effect=RuntimeError("send failed"))
        delivered, undelivered = "live", "pending"
        live = _make_req_state(delivered)
        live.dispatched = True
        tm.rid_to_state[delivered] = live
        tm.rid_to_state[undelivered] = _make_req_state(undelivered)

        with self.assertLogs(level="ERROR"):
            tm._release_req_states_on_failure(dict(tm.rid_to_state))

        self.assertIn(delivered, tm.rid_to_state)
        self.assertFalse(live.abort_sent)
        self.assertNotIn(undelivered, tm.rid_to_state)


class TestParallelStreamTaskCleanup(CustomTestCase):
    def test_failing_choice_cancels_and_closes_sibling_waiters(self):
        tm = _make_tokenizer_manager(self)

        async def drive():
            sibling_closed = asyncio.Event()

            async def failing_choice():
                await asyncio.sleep(0)
                raise RuntimeError("choice failed")
                yield  # pragma: no cover

            async def blocked_choice():
                try:
                    await asyncio.Event().wait()
                    yield  # pragma: no cover
                finally:
                    sibling_closed.set()

            stream = tm._stream_batch_responses(
                [failing_choice(), blocked_choice()],
                ["choice-0", "choice-1"],
            )
            with self.assertRaisesRegex(RuntimeError, "choice failed"):
                await stream.__anext__()
            self.assertTrue(sibling_closed.is_set())

        asyncio.run(drive())

    def test_failing_non_stream_choice_cancels_and_closes_sibling_waiters(self):
        tm = _make_tokenizer_manager(self)

        async def drive():
            sibling_closed = asyncio.Event()

            async def failing_choice():
                await asyncio.sleep(0)
                raise RuntimeError("choice failed")
                yield  # pragma: no cover

            async def blocked_choice():
                try:
                    await asyncio.Event().wait()
                    yield  # pragma: no cover
                finally:
                    sibling_closed.set()

            with self.assertRaisesRegex(RuntimeError, "choice failed"):
                await tm._collect_batch_responses([failing_choice(), blocked_choice()])
            self.assertTrue(sibling_closed.is_set())

        asyncio.run(drive())


class TestGenerateRequestCleanupOnDispatchFailure(CustomTestCase):
    """generate_request must not leak rid_to_state when dispatch fails.

    Regression guard: _init_req_state creates rid_to_state entries up front,
    and the only remover is the scheduler-response path. A failure before the
    request reaches the scheduler (e.g. input-length validation rejecting an
    over-context request) used to leak those entries permanently.
    """

    def test_single_failure_before_dispatch_cleans_up(self):
        tm = _make_tm_for_generate(self)
        rid = "single_overlen"
        obj = _make_generate_obj(rid, is_single=True)
        # Simulate over-length rejection during tokenization/validation.
        tm._tokenize_one_request = AsyncMock(side_effect=ValueError("input too long"))
        tm._send_one_request = Mock()

        async def drive():
            await tm.generate_request(obj).__anext__()

        with self.assertRaises(ValueError):
            asyncio.run(drive())

        # Got past _init_req_state (which created the entry) ...
        tm._tokenize_one_request.assert_awaited_once()
        tm._send_one_request.assert_not_called()
        # ... and the entry was cleaned up rather than leaked.
        self.assertNotIn(rid, tm.rid_to_state)

    def test_batch_failure_before_dispatch_cleans_up_all(self):
        tm = _make_tm_for_generate(self)
        rids = ["b0", "b1", "b2"]
        obj = _make_generate_obj(list(rids), is_single=False)

        # One over-length sub-request makes the whole batch dispatch raise.
        async def _boom(*args, **kwargs):
            raise ValueError("input too long")
            yield  # pragma: no cover  (marks this an async generator)

        tm._handle_batch_request = _boom

        async def drive():
            await tm.generate_request(obj).__anext__()

        with self.assertRaises(ValueError):
            asyncio.run(drive())

        # All sub-request entries created by _init_req_state are cleaned up.
        for r in rids:
            self.assertNotIn(r, tm.rid_to_state)

    def test_parallel_sampling_failure_cleans_generated_rid(self):
        """Both prefix and sample states must join their handler's cleanup ownership."""
        for fail_at_sample in (False, True):
            with self.subTest(fail_at_sample=fail_at_sample):
                tm = _make_tm_for_generate(self)
                tm.request_metrics_exporter_manager = Mock()
                tm.request_metrics_exporter_manager.exporter_enabled.return_value = (
                    False
                )
                obj = GenerateReqInput(
                    text=["hello"],
                    rid=["base"],
                    sampling_params={"n": 2},
                )
                tokenized = MagicMock()
                tokenized.mm_inputs = None
                tokenized.sampling_params = SimpleNamespace(max_new_tokens=1)
                tm._tokenize_one_request = AsyncMock(return_value=tokenized)

                async def send(request):
                    if fail_at_sample and request.sampling_params.max_new_tokens == 0:
                        await tm._handle_batch_output(
                            _make_batch_str_output(request.rid)
                        )
                    else:
                        raise RuntimeError("dispatch failed")

                tm._send_one_request = send

                async def drive():
                    await tm.generate_request(obj).__anext__()

                with self.assertRaisesRegex(RuntimeError, "dispatch failed"):
                    asyncio.run(drive())

                self.assertFalse(tm.rid_to_state)

    def test_thinking_budget_rejects_runtime_without_strict_thinking(self):
        tm = _make_tm_for_generate(self)
        obj = GenerateReqInput(
            text="hello",
            rid="thinking-budget",
            sampling_params={},
            max_thinking_tokens=32,
        )

        async def drive():
            await tm.generate_request(obj).__anext__()

        with self.assertRaisesRegex(ValueError, "--enable-strict-thinking"):
            asyncio.run(drive())

        self.assertFalse(tm.rid_to_state)


class TestWaitOneResponseAfterStateFreed(CustomTestCase):
    """A waiter built before its request finishes must still deliver the output.

    Batch dispatch builds every waiter before advancing any, and the
    scheduler-response path drops rid_to_state as soon as a request finishes.
    """

    def test_generator_built_before_finish_still_delivers_output(self):
        tm = _make_tokenizer_manager(self)
        tm.request_logger = Mock()
        tm.request_metrics_exporter_manager = MagicMock()
        tm.request_metrics_exporter_manager.exporter_enabled.return_value = False
        rid = "freed_state_rid"
        state = _make_req_state(rid)
        state.obj.background = True  # skip the fastapi disconnect probe
        tm.rid_to_state[rid] = state

        async def drive():
            waiter = tm._wait_one_response(state.obj, None)
            await tm._handle_batch_output(_make_batch_str_output(rid))
            self.assertNotIn(rid, tm.rid_to_state)
            return await waiter.__anext__()

        out = asyncio.run(drive())
        self.assertEqual(out["meta_info"]["id"], rid)
        self.assertEqual(out["text"], "hello")

    def test_terminal_output_during_dispatch_still_reaches_request_caller(self):
        """A fast terminal response can free request state before send returns."""
        for mode in ("single", "batch", "sequential", "parallel"):
            with self.subTest(mode=mode):
                tm = _make_tm_for_generate(self)
                tm.request_metrics_exporter_manager = Mock()
                tm.request_metrics_exporter_manager.exporter_enabled.return_value = (
                    False
                )
                tm._should_use_batch_tokenization = Mock(return_value=mode == "batch")
                obj = GenerateReqInput(
                    input_ids=[1] if mode == "single" else [[1], [2]],
                    sampling_params={"n": 2 if mode == "parallel" else 1},
                    return_prompt_token_ids=True,
                )

                async def tokenize(request):
                    return SimpleNamespace(
                        rid=request.rid,
                        input_ids=request.input_ids,
                        mm_inputs=None,
                        sampling_params=SimpleNamespace(max_new_tokens=1),
                        time_stats=tm.rid_to_state[request.rid].time_stats,
                    )

                async def tokenize_batch(batch_size, request):
                    return [await tokenize(request[i]) for i in range(batch_size)]

                async def send_one(request):
                    await tm._handle_batch_output(_make_batch_str_output(request.rid))
                    self.assertNotIn(request.rid, tm.rid_to_state)

                async def send_batch(requests):
                    for request in requests:
                        await send_one(request)

                tm._tokenize_one_request = tokenize
                tm._batch_tokenize_and_process = tokenize_batch
                tm._send_one_request = send_one
                tm._send_batch_request = send_batch

                async def drive():
                    return [result async for result in tm.generate_request(obj)]

                results = asyncio.run(drive())
                outputs = results if mode == "single" else results[0]
                expected_count = {
                    "single": 1,
                    "batch": 2,
                    "sequential": 2,
                    "parallel": 4,
                }
                self.assertEqual(len(outputs), expected_count[mode])
                self.assertTrue(all(output["text"] == "hello" for output in outputs))
                self.assertTrue(
                    all(
                        output["prompt_token_ids"] == [1]
                        or output["prompt_token_ids"] == [2]
                        for output in outputs
                    )
                )
                for output in outputs:
                    self.assertNotIn(output["meta_info"]["id"], tm.rid_to_state)


class TestFailureCleanupAfterRequestIdReuse(CustomTestCase):
    def test_old_send_failure_preserves_replacement_state(self):
        """A terminal reply and RID reuse can precede the old send's completion."""
        for mode, failure, dispatched in product(
            ("single", "batch", "parallel_prefix", "parallel_sample"),
            ("cancel", "bookkeeping", "terminal_abort"),
            (False, True),
        ):
            with self.subTest(mode=mode, failure=failure, dispatched=dispatched):
                self._check_reused_state(mode, failure, dispatched)

    def _check_reused_state(self, mode, failure, replacement_dispatched):
        tm = _make_tm_for_generate(self)
        tm.request_metrics_exporter_manager = Mock()
        tm.request_metrics_exporter_manager.exporter_enabled.return_value = False
        tm.cuda_vmm_feature_transport = Mock()
        tm.cuda_vmm_feature_transport.prepare_for_dispatch_async = AsyncMock(
            return_value=[]
        )
        tm._dispatch_to_scheduler = Mock()
        tm._should_use_batch_tokenization = Mock(return_value=mode == "batch")
        obj = GenerateReqInput(
            input_ids=[1] if mode == "single" else [[1]],
            sampling_params={"n": 2 if mode.startswith("parallel") else 1},
        )
        send_error = RuntimeError("dispatch bookkeeping failed")
        old_ready = threading.Event()
        replacement_ready = threading.Event()
        replaced = {}

        async def tokenize(request):
            result = TokenizedGenerateReqInput(
                rid=request.rid,
                input_text=None,
                input_ids=array("q", [1]),
                input_embeds=None,
                mm_inputs=None,
                token_type_ids=None,
                sampling_params=SamplingParams(),
                return_logprob=False,
                logprob_start_len=-1,
                top_logprobs_num=0,
                token_ids_logprob=None,
                stream=False,
            )
            result.time_stats = tm.rid_to_state[request.rid].time_stats
            return result

        async def tokenize_batch(batch_size, request):
            return [await tokenize(request[i]) for i in range(batch_size)]

        async def dispatch(request):
            requests = (
                request.batch
                if isinstance(request, BatchTokenizedGenerateReqInput)
                else [request]
            )
            target = requests[0]
            if replaced:
                for item in requests:
                    await tm._handle_batch_output(_make_batch_str_output(item.rid))
                return
            if mode == "parallel_sample" and target.sampling_params.max_new_tokens == 0:
                await tm._handle_batch_output(_make_batch_str_output(target.rid))
                return
            original = tm.rid_to_state[target.rid]
            original.encoder_dispatch_ready = old_ready
            tm.encoder_dispatch_ready[target.rid] = old_ready
            if failure == "bookkeeping":
                target.time_stats.set_api_server_dispatch_finish_time = Mock(
                    side_effect=send_error
                )
            finish = (
                {"type": "abort", "status_code": 503, "message": "old terminal failure"}
                if failure == "terminal_abort"
                else None
            )
            for item in requests:
                await tm._handle_batch_output(
                    _make_batch_str_output(item.rid, finished_reason=finish)
                )
            replacement_obj = GenerateReqInput(input_ids=[2], rid=target.rid)
            replacement_obj.normalize_batch_and_arguments()
            tm._init_req_state(replacement_obj)
            replacement = tm.rid_to_state[target.rid]
            replacement.dispatched = replacement_dispatched
            replacement.encoder_dispatch_ready = replacement_ready
            tm.encoder_dispatch_ready[target.rid] = replacement_ready
            replaced.update(rid=target.rid, state=replacement)
            if failure == "cancel":
                task.cancel()
                await asyncio.sleep(0)

        tm._tokenize_one_request = tokenize
        tm._batch_tokenize_and_process = tokenize_batch
        tm._async_dispatch_to_scheduler = dispatch

        async def drive():
            nonlocal task
            task = asyncio.create_task(tm.generate_request(obj).__anext__())
            error_type = {
                "cancel": asyncio.CancelledError,
                "bookkeeping": RuntimeError,
                "terminal_abort": HTTPException,
            }[failure]
            with self.assertRaises(error_type) as raised:
                await task
            if failure == "bookkeeping":
                self.assertIs(raised.exception, send_error)
            elif failure == "terminal_abort":
                self.assertEqual(raised.exception.status_code, 503)

        task = None
        asyncio.run(drive())
        self.assertEqual(list(tm.rid_to_state), [replaced["rid"]])
        self.assertIs(tm.rid_to_state[replaced["rid"]], replaced["state"])
        self.assertFalse(replaced["state"].abort_sent)
        self.assertEqual(replaced["state"].dispatched, replacement_dispatched)
        self.assertTrue(old_ready.is_set())
        self.assertFalse(replacement_ready.is_set())
        self.assertIs(tm.encoder_dispatch_ready[replaced["rid"]], replacement_ready)
        self.assertEqual(tm._dispatch_to_scheduler.call_args_list, [])

    def test_old_encoder_callback_cannot_target_reused_request_id(self):
        """Releasing an old encoder waiter must not forward its error to a new RID owner."""
        tm = _make_tm_for_generate(self)
        tm.request_metrics_exporter_manager = Mock()
        tm.request_metrics_exporter_manager.exporter_enabled.return_value = False
        tm._dispatch_to_scheduler = Mock()
        events = [threading.Event(), threading.Event()]
        callbacks = []

        def encode(request, *, time_stats_json, on_dispatch_error):
            callbacks.append(on_dispatch_error)
            return events[len(callbacks) - 1]

        tm.mm_receiver = SimpleNamespace(send_encode_request=encode)
        override = get_context().override_server_args(
            enable_adaptive_dispatch_to_encoder=False,
            encoder_transfer_backend="zmq_to_scheduler",
        )
        override.install()
        self.addCleanup(override.restore)

        def create_owner():
            obj = GenerateReqInput(
                input_ids=[1], image_data=["synthetic"], rid="encoder-reuse"
            )
            obj.normalize_batch_and_arguments()
            tm._init_req_state(obj)
            tm._handle_epd_disaggregation_encode_request(obj)
            return tm.rid_to_state[obj.rid]

        async def drive():
            tm.event_loop = asyncio.get_running_loop()
            original = create_owner()
            error = EncoderDispatchErrorReq(
                rid="encoder-reuse", error_msg="old encoder failed", error_code=502
            )
            callbacks[0](error)
            await asyncio.sleep(0)
            self.assertEqual(
                [call.args[0] for call in tm._dispatch_to_scheduler.call_args_list],
                [error],
            )
            tm._dispatch_to_scheduler.reset_mock()
            await tm._handle_batch_output(_make_batch_str_output("encoder-reuse"))
            replacement = create_owner()
            tm._release_req_states_on_failure({"encoder-reuse": original})
            callbacks[0](error)
            await asyncio.sleep(0)
            self.assertIs(tm.rid_to_state["encoder-reuse"], replacement)
            self.assertIs(tm.encoder_dispatch_ready["encoder-reuse"], events[1])
            self.assertTrue(events[0].is_set())
            self.assertFalse(events[1].is_set())
            self.assertEqual(tm._dispatch_to_scheduler.call_args_list, [])
            callbacks[1](error)
            await asyncio.sleep(0)
            self.assertEqual(
                [call.args[0] for call in tm._dispatch_to_scheduler.call_args_list],
                [error],
            )

        asyncio.run(drive())


class TestDelayedAbortGenerationBinding(CustomTestCase):
    def _manager(self, mode="single"):
        tm = _make_tm_for_generate(self)
        tm.request_metrics_exporter_manager = Mock()
        tm.request_metrics_exporter_manager.exporter_enabled.return_value = False
        tm._dispatch_to_scheduler = Mock()
        tm._should_use_batch_tokenization = Mock(return_value=mode == "batch")
        sent = asyncio.Queue()

        async def tokenize(obj):
            return SimpleNamespace(
                rid=obj.rid,
                input_ids=obj.input_ids,
                mm_inputs=None,
                sampling_params=SamplingParams(max_new_tokens=1),
                time_stats=tm.rid_to_state[obj.rid].time_stats,
            )

        async def send(obj):
            tm._mark_state_dispatched(tm.rid_to_state[obj.rid])
            if obj.sampling_params.max_new_tokens == 0:
                await tm._handle_batch_output(_make_batch_str_output(obj.rid))
            else:
                sent.put_nowait(obj.rid)

        async def send_batch(objects):
            for obj in objects:
                await send(obj)

        async def tokenize_batch(size, obj):
            return [await tokenize(obj[index]) for index in range(size)]

        tm._tokenize_one_request = tokenize
        tm._batch_tokenize_and_process = tokenize_batch
        tm._send_one_request = send
        tm._send_batch_request = send_batch
        return tm, sent

    async def _background(self, task):
        delay = AsyncMock()
        with patch("sglang.srt.managers.tokenizer_manager.asyncio.sleep", delay):
            await task()
        delay.assert_awaited_once_with(2)

    async def _finish(self, tm, generator, first, rids):
        for rid in rids:
            if rid in tm.rid_to_state:
                await tm._handle_batch_output(_make_batch_str_output(rid))
        if not first.done():
            await first
        await generator.aclose()

    def test_background_captures_pre_registration_and_first_chunk_owners(self):
        """Native and OpenAI callback timing must abort the actual dispatched owners."""
        for mode, timing in product(
            ("single", "batch", "parallel"), ("before", "after")
        ):
            with self.subTest(mode=mode, timing=timing):

                async def drive():
                    tm, sent = self._manager(mode)
                    obj = GenerateReqInput(
                        input_ids=[1] if mode == "single" else [[1], [2]],
                        sampling_params={"n": 2 if mode == "parallel" else 1},
                        stream=True,
                    )
                    callback = tm.create_abort_task(obj) if timing == "before" else None
                    generator = tm.generate_request(obj)
                    first = asyncio.create_task(generator.__anext__())
                    count = {"single": 1, "batch": 2, "parallel": 4}[mode]
                    rids = [await sent.get() for _ in range(count)]
                    owners = {rid: tm.rid_to_state[rid] for rid in rids}
                    try:
                        if timing == "after":
                            await tm._handle_batch_output(
                                _make_batch_str_output(rids[0], _NOT_FINISHED)
                            )
                            await first
                            callback = tm.create_abort_task(obj)
                        await self._background(callback)
                        self.assertCountEqual(
                            [
                                call.args[0].rid
                                for call in tm._dispatch_to_scheduler.call_args_list
                            ],
                            rids,
                        )
                        for rid, owner in owners.items():
                            self.assertIs(tm.rid_to_state[rid], owner)
                            self.assertTrue(owner.abort_sent)
                    finally:
                        await self._finish(tm, generator, first, rids)
                    self.assertFalse(tm.rid_to_state)
                    self.assertNotIn("_tokenizer_request_state_binding", vars(obj))
                    self.assertFalse(tm._request_state_bindings)

                asyncio.run(drive())

    def test_old_background_cannot_abort_reused_rid_or_same_input_object(self):
        """A late callback keeps its first generation even after input-object reuse."""
        for timing, same_object in product(("before", "after"), (False, True)):
            with self.subTest(timing=timing, same_object=same_object):

                async def drive():
                    tm, sent = self._manager()
                    original_obj = GenerateReqInput(
                        input_ids=[1], rid="reused", stream=True
                    )
                    callback = (
                        tm.create_abort_task(original_obj)
                        if timing == "before"
                        else None
                    )
                    original_generator = tm.generate_request(original_obj)
                    original_first = asyncio.create_task(original_generator.__anext__())
                    await sent.get()
                    original_state = tm.rid_to_state["reused"]
                    await tm._handle_batch_output(_make_batch_str_output("reused"))
                    await original_first
                    if timing == "after":
                        callback = tm.create_abort_task(original_obj)
                    replacement_obj = (
                        original_obj
                        if same_object
                        else GenerateReqInput(input_ids=[2], rid="reused", stream=True)
                    )
                    replacement_generator = tm.generate_request(replacement_obj)
                    replacement_first = asyncio.create_task(
                        replacement_generator.__anext__()
                    )
                    await sent.get()
                    replacement_state = tm.rid_to_state["reused"]
                    self.assertIsNot(original_state, replacement_state)
                    try:
                        # Finishing an older scope must not detach the new scope.
                        await original_generator.aclose()
                        await self._background(callback)
                        self.assertIs(tm.rid_to_state["reused"], replacement_state)
                        self.assertFalse(replacement_state.abort_sent)
                        tm._dispatch_to_scheduler.assert_not_called()
                        current_callback = tm.create_abort_task(replacement_obj)
                        await self._background(current_callback)
                        self.assertTrue(replacement_state.abort_sent)
                        tm._dispatch_to_scheduler.assert_called_once()
                    finally:
                        await self._finish(
                            tm, replacement_generator, replacement_first, ["reused"]
                        )
                        await original_generator.aclose()
                    self.assertNotIn(
                        "_tokenizer_request_state_binding", vars(original_obj)
                    )
                    self.assertNotIn(
                        "_tokenizer_request_state_binding", vars(replacement_obj)
                    )
                    self.assertFalse(tm._request_state_bindings)

                asyncio.run(drive())

    def test_unregistered_callback_expires_without_touching_another_owner(self):
        """An unused native callback must neither retain its binding nor claim a RID."""

        async def drive():
            tm, _ = self._manager()
            original = GenerateReqInput(input_ids=[1], rid="unused", stream=True)
            callback = tm.create_abort_task(original)
            replacement = GenerateReqInput(input_ids=[2], rid="unused", stream=True)
            replacement.normalize_batch_and_arguments()
            tm._init_req_state(replacement)
            state = tm.rid_to_state["unused"]
            state.dispatched = True
            await self._background(callback)
            self.assertIs(tm.rid_to_state["unused"], state)
            self.assertFalse(state.abort_sent)
            tm._dispatch_to_scheduler.assert_not_called()
            self.assertNotIn("_tokenizer_request_state_binding", vars(original))
            self.assertFalse(tm._request_state_bindings)

        asyncio.run(drive())

    def test_validation_failure_does_not_bind_an_old_callback_to_next_generation(self):
        """A callback created before failed registration cannot own the later retry."""

        async def drive():
            tm, sent = self._manager()
            obj = GenerateReqInput(
                input_ids=[1], rid="retry", stream=True, max_thinking_tokens=1
            )
            callback = tm.create_abort_task(obj)
            failed = tm.generate_request(obj)
            with self.assertRaisesRegex(ValueError, "--enable-strict-thinking"):
                await failed.__anext__()
            self.assertNotIn("_tokenizer_request_state_binding", vars(obj))
            self.assertFalse(tm._request_state_bindings)
            obj.max_thinking_tokens = None
            current_callback = tm.create_abort_task(obj)
            generator = tm.generate_request(obj)
            first = asyncio.create_task(generator.__anext__())
            await sent.get()
            owner = tm.rid_to_state["retry"]
            try:
                await self._background(callback)
                self.assertIs(tm.rid_to_state["retry"], owner)
                self.assertFalse(owner.abort_sent)
                tm._dispatch_to_scheduler.assert_not_called()
                await self._background(current_callback)
                self.assertTrue(owner.abort_sent)
                tm._dispatch_to_scheduler.assert_called_once()
            finally:
                await self._finish(tm, generator, first, ["retry"])
            self.assertNotIn("_tokenizer_request_state_binding", vars(obj))
            self.assertFalse(tm._request_state_bindings)

        asyncio.run(drive())

    def test_pending_generation_is_retired_locally_without_scheduler_abort(self):
        """A callback may retire registered work that has not reached dispatch."""

        async def drive():
            tm, _ = self._manager()
            tm.is_pause = True
            registered = asyncio.Event()
            init = tm._init_req_state

            def register(*args, **kwargs):
                init(*args, **kwargs)
                registered.set()

            tm._init_req_state = register
            obj = GenerateReqInput(input_ids=[1], rid="pending", stream=True)
            callback = tm.create_abort_task(obj)
            generator = tm.generate_request(obj)
            first = asyncio.create_task(generator.__anext__())
            await registered.wait()
            owner = tm.rid_to_state[obj.rid]
            self.assertFalse(owner.dispatched)
            await self._background(callback)
            self.assertNotIn(obj.rid, tm.rid_to_state)
            self.assertFalse(owner.abort_sent)
            tm._dispatch_to_scheduler.assert_not_called()
            first.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await first
            await generator.aclose()
            self.assertNotIn("_tokenizer_request_state_binding", vars(obj))
            self.assertFalse(tm._request_state_bindings)

        asyncio.run(drive())

    def test_request_copy_does_not_share_the_original_pending_binding(self):
        """Shallow request copies must not inherit a different callback's owner."""

        async def drive():
            tm, sent = self._manager()
            original = GenerateReqInput(input_ids=[1], rid="copy", stream=True)
            old_callback = tm.create_abort_task(original)
            copied = copy.copy(original)
            generator = tm.generate_request(copied)
            first = asyncio.create_task(generator.__anext__())
            await sent.get()
            owner = tm.rid_to_state["copy"]
            try:
                await self._background(old_callback)
                self.assertIs(tm.rid_to_state["copy"], owner)
                self.assertFalse(owner.abort_sent)
                tm._dispatch_to_scheduler.assert_not_called()
                await self._background(tm.create_abort_task(copied))
                self.assertTrue(owner.abort_sent)
                tm._dispatch_to_scheduler.assert_called_once()
            finally:
                await self._finish(tm, generator, first, ["copy"])
            self.assertNotIn("_tokenizer_request_state_binding", vars(original))
            self.assertNotIn("_tokenizer_request_state_binding", vars(copied))
            self.assertFalse(tm._request_state_bindings)

        asyncio.run(drive())

    def test_active_request_dump_excludes_runtime_owners(self):
        """A request waiting for output remains serializable through crash dumping."""

        async def drive(folder):
            tm, sent = self._manager()
            tm.server_args = {}
            tm._dump_config_snapshot = Mock(return_value={})
            tm.crash_dump_folder = folder
            tm.crash_dump_performed = False
            tm.crash_dump_request_list = deque()
            obj = GenerateReqInput(input_ids=[1], rid="dump-owner", stream=True)
            generator = tm.generate_request(obj)
            first = asyncio.create_task(generator.__anext__())
            await sent.get()
            owner = tm.rid_to_state[obj.rid]
            # A pending wait contains an unpicklable Future. An idle Event would
            # not expose runtime ownership leaking into the request payload.
            await asyncio.sleep(0)
            self.assertTrue(owner.event._waiters)
            try:
                replay = pickle.loads(pickle.dumps(obj))
                self.assertEqual(replay.rid, obj.rid)
                self.assertEqual(replay.input_ids, [1])
                with (
                    patch(
                        "sglang.srt.managers.tokenizer_manager.envs."
                        "SGLANG_PYSPY_DUMP_BEFORE_CRASH.get",
                        return_value=False,
                    ),
                    patch(
                        "sglang.srt.managers.tokenizer_manager.envs."
                        "SGLANG_CUDA_COREDUMP_BEFORE_CRASH.get",
                        return_value=False,
                    ),
                ):
                    tm.dump_requests_before_crash(hostname="unit")
                paths = list(Path(folder).glob("unit/*.pkl"))
                self.assertEqual(len(paths), 1)
                with paths[0].open("rb") as handle:
                    dump = pickle.load(handle)
                self.assertEqual(dump["server_args"], {})
                self.assertEqual(dump["requests"][0][0].input_ids, [1])
                self.assertEqual(dump["requests"][0][0].rid, obj.rid)
            finally:
                await self._finish(tm, generator, first, [obj.rid])
            self.assertFalse(tm._request_state_bindings)

        with TemporaryDirectory() as folder:
            asyncio.run(drive(folder))

    def test_finished_crash_snapshot_does_not_retain_runtime_owners(self):
        """The real output path may retain replay inputs without retaining runtime state."""

        async def drive():
            tm, sent = self._manager()
            tm.crash_dump_folder = "enabled"
            tm.crash_dump_request_list = deque()
            obj = GenerateReqInput(input_ids=[1], rid="snapshot-owner", stream=True)
            generator = tm.generate_request(obj)
            first = asyncio.create_task(generator.__anext__())
            await sent.get()
            owner_ref = weakref.ref(tm.rid_to_state[obj.rid])
            binding_ref = weakref.ref(tm._request_state_bindings[id(obj)])
            try:
                await tm._handle_batch_output(_make_batch_str_output(obj.rid))
                await first
                with self.assertRaises(StopAsyncIteration):
                    await generator.__anext__()
            finally:
                await generator.aclose()
            self.assertEqual(len(tm.crash_dump_request_list), 1)
            snapshot = tm.crash_dump_request_list[0][0]
            self.assertIsNot(snapshot, obj)
            self.assertEqual(snapshot.input_ids, (1,))
            gc.collect()
            self.assertIsNone(owner_ref())
            self.assertIsNone(binding_ref())
            self.assertFalse(tm._request_state_bindings)
            replay = pickle.loads(pickle.dumps(snapshot))
            self.assertEqual(replay.rid, obj.rid)
            self.assertEqual(replay.input_ids, (1,))

        asyncio.run(drive())

    def test_dropped_callbacks_do_not_retain_or_unindex_another_generation(self):
        """A weak index neither retains abandoned callbacks nor deletes newer owners."""

        async def drive():
            tm, sent = self._manager()
            obj = GenerateReqInput(input_ids=[1], rid="weak-owner", stream=True)
            abandoned = tm.create_abort_task(obj)
            abandoned_ref = weakref.ref(tm._request_state_bindings[id(obj)])
            del abandoned
            gc.collect()
            self.assertIsNone(abandoned_ref())
            self.assertFalse(tm._request_state_bindings)

            old_callback = tm.create_abort_task(obj)
            old_ref = weakref.ref(tm._request_state_bindings[id(obj)])
            original = tm.generate_request(obj)
            original_first = asyncio.create_task(original.__anext__())
            await sent.get()
            await tm._handle_batch_output(_make_batch_str_output(obj.rid))
            await original_first
            replacement = tm.generate_request(obj)
            replacement_first = asyncio.create_task(replacement.__anext__())
            await sent.get()
            current_ref = weakref.ref(tm._request_state_bindings[id(obj)])
            self.assertIsNot(current_ref(), old_ref())
            try:
                await original.aclose()
                del old_callback
                gc.collect()
                self.assertIsNone(old_ref())
                self.assertIs(tm._request_state_bindings[id(obj)], current_ref())
                await self._background(tm.create_abort_task(obj))
                self.assertTrue(tm.rid_to_state[obj.rid].abort_sent)
                tm._dispatch_to_scheduler.assert_called_once()
            finally:
                await self._finish(tm, replacement, replacement_first, [obj.rid])
                await original.aclose()
            gc.collect()
            self.assertIsNone(current_ref())
            self.assertFalse(tm._request_state_bindings)

        asyncio.run(drive())

    def test_disconnect_await_does_not_abort_a_replacement(self):
        """Disconnect probes must recheck the waiter owner after awaiting the client."""
        for timeout, replace_owner in product((False, True), (False, True)):
            with self.subTest(timeout=timeout, replace_owner=replace_owner):

                async def drive():
                    tm, _ = self._manager()
                    obj = GenerateReqInput(
                        input_ids=[1], rid="disconnect", stream=False
                    )
                    obj.normalize_batch_and_arguments()
                    tm._init_req_state(obj)
                    original = tm.rid_to_state[obj.rid]
                    original.dispatched = True
                    if not timeout:
                        original.out_list = [{"text": "partial", "meta_info": {}}]
                        original.event.set()
                    replacement = None

                    async def disconnected():
                        nonlocal replacement
                        if replace_owner:
                            await tm._handle_batch_output(
                                _make_batch_str_output(obj.rid)
                            )
                            fresh = GenerateReqInput(input_ids=[2], rid=obj.rid)
                            fresh.normalize_batch_and_arguments()
                            tm._init_req_state(fresh)
                            replacement = tm.rid_to_state[obj.rid]
                            replacement.dispatched = True
                        return True

                    async def timed_out(awaitable, *, timeout):
                        awaitable.close()
                        raise asyncio.TimeoutError

                    waiter = tm._wait_one_response(
                        obj, SimpleNamespace(is_disconnected=disconnected)
                    )
                    context = (
                        patch(
                            "sglang.srt.managers.tokenizer_manager.asyncio.wait_for",
                            timed_out,
                        )
                        if timeout
                        else nullcontext()
                    )
                    with context, self.assertRaisesRegex(ValueError, "disconnected"):
                        await waiter.__anext__()
                    if replace_owner:
                        self.assertIs(tm.rid_to_state[obj.rid], replacement)
                        self.assertFalse(replacement.abort_sent)
                        tm._dispatch_to_scheduler.assert_not_called()
                    else:
                        self.assertIs(tm.rid_to_state[obj.rid], original)
                        self.assertTrue(original.abort_sent)
                        tm._dispatch_to_scheduler.assert_called_once()

                asyncio.run(drive())


class TestDisconnectAfterDispatchAbortsRequest(CustomTestCase):
    """Cancellation after dispatch must stop the scheduler request."""

    @patch(
        "sglang.srt.managers.tokenizer_manager.wrap_shm_features",
        side_effect=lambda obj: obj,
    )
    def test_cancel_after_dispatch_sends_abort_and_keeps_state(self, _wrap_shm):
        tm = _make_tm_for_generate(self)
        tm.cuda_vmm_feature_transport = Mock()
        tm.cuda_vmm_feature_transport.prepare_for_dispatch_async = AsyncMock(
            return_value=[]
        )
        tm._dispatch_to_scheduler = Mock()
        rid = "disconnect_zombie"
        obj = _make_generate_obj(rid, is_single=True)
        obj.return_prompt_token_ids = False
        tokenized = MagicMock()
        tokenized.rid = rid
        tokenized.mm_inputs = None
        tm._tokenize_one_request = AsyncMock(return_value=tokenized)

        async def drive():
            task = asyncio.create_task(tm.generate_request(obj).__anext__())
            for _ in range(100):
                await asyncio.sleep(0)
                if tm.rid_to_state[rid].dispatched:
                    break
            self.assertTrue(
                tm._async_dispatch_to_scheduler.called, "request never dispatched"
            )
            state = tm.rid_to_state.get(rid)
            self.assertIsNotNone(state)
            self.assertTrue(state.dispatched)

            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

        asyncio.run(drive())

        sent = [c.args[0] for c in tm._dispatch_to_scheduler.call_args_list]
        aborts = [m for m in sent if isinstance(m, AbortReq) and m.rid == rid]
        self.assertTrue(aborts, "disconnect must send an AbortReq to the scheduler")
        self.assertIn(rid, tm.rid_to_state)


if __name__ == "__main__":
    unittest.main(verbosity=2)
