import copy
import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.kernels.ops.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE
from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape
from sglang.srt.disaggregation.decode import DecodePreallocQueue
from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
from sglang.srt.disaggregation.utils import (
    FAKE_BOOTSTRAP_HOST,
    DisaggregationMode,
    KVPoll,
)
from sglang.srt.managers.io_struct import (
    CloseSessionReqInput,
    SessionParams,
    TokenizedGenerateReqInput,
)
from sglang.srt.managers.schedule_batch import (
    FINISH_ABORT,
    FINISH_LENGTH,
    ScheduleBatch,
    release_req,
)
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.invariant_checker import (
    SchedulerInvariantChecker,
)
from sglang.srt.managers.scheduler_components.new_token_ratio_tracker import (
    NewTokenRatioTracker,
)
from sglang.srt.managers.scheduler_components.pool_stats_observer import (
    SchedulerPoolStatsObserver,
)
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.chunk_cache import ChunkCache
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.mem_cache.memory_pool import (
    HybridLinearKVPool,
    HybridReqToTokenPool,
    MHATokenToKVPool,
    ReqToTokenPool,
)
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.srt.session.session_controller import Session, SessionController
from sglang.srt.session.streaming_session import SessionSlot, StreamingSession
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

PAGE = 1
KV_SIZE = 512
MAMBA_SIZE = 8
VOCAB_SIZE = 32000


def _build():
    server_args = ServerArgs(model_path="dummy", page_size=PAGE)
    server_args._mamba_cache_chunk_size = max(FLA_CHUNK_SIZE, PAGE)
    server_args.max_mamba_cache_size = MAMBA_SIZE
    server_args.mamba_max_states_per_path = 2
    publish(server_args, role="scheduler")

    full_attention_layer_ids = [3, 7]
    mamba_layer_ids = [i for i in range(8) if i not in full_attention_layer_ids]
    shape = Mamba2StateShape.create(
        tp_world_size=1,
        intermediate_size=64,
        n_groups=2,
        num_heads=4,
        head_dim=16,
        state_size=16,
        conv_kernel=4,
    )
    cache_params = Mamba2CacheParams(shape=shape, layers=mamba_layer_ids)
    with torch.device("cpu"):
        req_to_token_pool = HybridReqToTokenPool(
            size=8,
            mamba_size=MAMBA_SIZE,
            mamba_spec_state_size=8,
            max_context_len=512,
            device="cpu",
            enable_memory_saver=False,
            cache_params=cache_params,
            mamba_layer_ids=mamba_layer_ids,
            enable_mamba_extra_buffer=True,
            enable_mamba_extra_buffer_lazy=True,
            speculative_num_draft_tokens=None,
        )
    kv_pool = HybridLinearKVPool(
        size=KV_SIZE,
        dtype=torch.bfloat16,
        page_size=PAGE,
        head_num=2,
        head_dim=16,
        full_attention_layer_ids=full_attention_layer_ids,
        device="cpu",
        enable_memory_saver=False,
        mamba_pool=req_to_token_pool.mamba_pool,
    )
    allocator = TokenToKVPoolAllocator(
        size=KV_SIZE,
        dtype=torch.bfloat16,
        device="cpu",
        kvcache=kv_pool,
        need_sort=False,
    )
    cache = UnifiedRadixCache(
        params=CacheInitParams(
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=allocator,
            page_size=PAGE,
            disable=False,
            tree_components=(ComponentType.FULL, ComponentType.MAMBA),
            enable_mamba_extra_buffer=True,
            enable_mamba_extra_buffer_lazy=True,
            eviction_policy="lru",
        )
    )
    sessions = SimpleNamespace(sessions={})
    observer = SchedulerPoolStatsObserver(
        tree_cache=cache,
        token_to_kv_pool_allocator=allocator,
        req_to_token_pool=req_to_token_pool,
        session_controller=sessions,
        hisparse_coordinator=None,
        is_hybrid_swa=False,
        is_hybrid_ssm=True,
        enable_hisparse=False,
        full_tokens_per_layer=None,
        swa_tokens_per_layer=None,
        max_total_num_tokens=KV_SIZE,
        get_last_batch=lambda: None,
        get_running_batch=lambda: None,
    )
    checker = SchedulerInvariantChecker(
        is_hybrid_swa=False,
        is_hybrid_ssm=True,
        disaggregation_mode=DisaggregationMode.NULL,
        page_size=PAGE,
        full_tokens_per_layer=None,
        swa_tokens_per_layer=None,
        max_total_num_tokens=KV_SIZE,
        tree_cache=cache,
        token_to_kv_pool_allocator=allocator,
        req_to_token_pool=req_to_token_pool,
        pool_stats_observer=observer,
        get_last_batch=lambda: None,
        get_running_batch=lambda: None,
        scheduler_stage_metrics=None,
    )
    return server_args, cache, allocator, req_to_token_pool, observer, checker


def _recv(rid, input_ids, parent_rid=None, session_id="session-a"):
    return TokenizedGenerateReqInput(
        rid=rid,
        input_text=None,
        input_ids=array("q", input_ids),
        input_embeds=None,
        mm_inputs=None,
        token_type_ids=None,
        sampling_params=SamplingParams(temperature=0, max_new_tokens=4),
        return_logprob=False,
        logprob_start_len=0,
        top_logprobs_num=0,
        token_ids_logprob=None,
        stream=True,
        session_params=SessionParams(
            id=session_id,
            rid=parent_rid,
            offset=None,
            replace=False,
            drop_previous_output=False,
        ),
        lora_id=None,
        custom_logit_processor=None,
        return_sampling_mask=False,
        require_reasoning=False,
        return_hidden_states=False,
        return_routed_experts=False,
        routed_experts_start_len=0,
        priority=None,
        routing_key=None,
        extra_key=None,
        http_worker_ipc=None,
        time_stats=None,
    )


def _prefill(req, cache, allocator, req_to_token_pool):
    key = RadixKey(array("q", req.origin_input_ids + req.output_ids))
    match = cache.match_prefix(MatchPrefixParams(key=key, req=req, cow_mamba=True))
    prefix_len = len(match.device_indices)
    req.lock_receipt = cache.inc_lock_ref(match.last_device_node).to_dec_params()
    if req.kv.req_pool_idx is None:
        req_to_token_pool.alloc([req])
    total = len(req.origin_input_ids) + len(req.output_ids)
    if prefix_len:
        req_to_token_pool.write(
            (req.kv.req_pool_idx, slice(0, prefix_len)), match.device_indices
        )
    if total > prefix_len:
        req_to_token_pool.write(
            (req.kv.req_pool_idx, slice(prefix_len, total)),
            allocator.alloc(total - prefix_len),
        )
    req.prefix_indices = match.device_indices
    req.last_node = match.last_device_node
    req.kv.cache_protected_len = (
        match.cache_protected_len
        if match.cache_protected_len is not None
        else prefix_len
    )
    req.kv.kv_committed_len = total
    req.kv.kv_allocated_len = total
    req.full_untruncated_fill_ids = array("q", req.origin_input_ids + req.output_ids)
    req.set_extend_range(prefix_len, total)
    req.kv.mamba_last_track_seqlen = 0
    if req.kv.mamba_next_track_idx is None:
        req.kv.mamba_next_track_idx = 0
    cache.cache_unfinished_req(req)


def _decode_step(req, allocator, req_to_token_pool, token):
    pos = req.kv.kv_allocated_len
    idx = allocator.alloc(1)
    req_to_token_pool.write((req.kv.req_pool_idx, slice(pos, pos + 1)), idx)
    req.output_ids.append(token)
    req._refresh_fill_ids()
    req.kv.kv_committed_len += 1
    req.kv.kv_allocated_len += 1


def _finish_turn(req, cache, finished_len):
    req.finished_reason = FINISH_LENGTH(length=finished_len)
    req.finished_len = finished_len
    release_kv_cache(req, cache)


class TestSessionRetractCheckpoint(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        parallel = patch(
            "sglang.srt.managers.schedule_batch.get_parallel",
            return_value=SimpleNamespace(tp_rank=0),
        )
        parallel.start()
        self.addCleanup(parallel.stop)

    def _setup_first_turn(self):
        (
            server_args,
            cache,
            allocator,
            req_to_token_pool,
            observer,
            checker,
        ) = _build()
        session = Session(capacity_of_str_len=0, session_id="session-a", streaming=True)
        req1 = session.create_req(
            _recv("turn-1", list(range(16))),
            tokenizer=None,
            vocab_size=VOCAB_SIZE,
        )
        _prefill(req1, cache, allocator, req_to_token_pool)
        _finish_turn(req1, cache, finished_len=0)
        self.assertTrue(cache.session.has_slot(session.session_id))
        return (
            server_args,
            cache,
            allocator,
            req_to_token_pool,
            observer,
            checker,
            session,
        )

    def _assert_idle(self, observer, checker):
        pool_stats = observer.get_pool_stats()
        mamba_leak, mamba_msg = checker._check_mamba_pool(pool_stats)
        self.assertFalse(mamba_leak, mamba_msg)
        all_leak, all_messages = checker._check_all_pools(pool_stats)
        self.assertFalse(all_leak, all_messages)

    @staticmethod
    def _pool_counts(cache, allocator, pool):
        return (
            allocator.available_size(),
            pool.available_size(),
            pool.mamba_allocator.available_size(),
            cache.session.session_held_tokens(),
            cache.session.session_held_mamba_slots(),
        )

    def test_preabort_releases_borrower_alias_without_releasing_checkpoint(self):
        """A pre-aborted reader must not free the retained row during raw cleanup."""
        for later_owner in (False, True):
            with self.subTest(later_owner=later_owner):
                _, cache, allocator, pool, observer, checker, session = (
                    self._setup_first_turn()
                )
                slot = cache.session.slots[session.session_id]
                saved_kv = slot.kv
                saved_row = saved_kv.req_pool_idx
                saved_indices = pool.req_to_token[
                    saved_row, : saved_kv.kv_allocated_len
                ].clone()
                saved_mamba = saved_kv.mamba_pool_idx.clone()
                saved_tracking = saved_kv.mamba_ping_pong_track_buffer.clone()
                rejected = session.create_req(
                    _recv("rejected", [32, 33]), None, VOCAB_SIZE
                )
                rejected.init_next_round_input(cache, cow_mamba=False)
                self.assertIs(rejected.kv, saved_kv)
                active = None
                if later_owner:
                    session.abort_req(rejected.rid)
                    active = session.create_req(_recv("active", [99]), None, VOCAB_SIZE)
                    active.init_next_round_input(cache, cow_mamba=False)
                    self.assertIs(active.kv, saved_kv)
                before = self._pool_counts(cache, allocator, pool)
                rejected.to_finish = FINISH_ABORT("cancelled before admission")
                rejected.init_next_round_input(cache, cow_mamba=False)

                self.assertIsNone(rejected.session)
                self.assertIsNot(rejected.kv, saved_kv)
                self.assertFalse(rejected.kv.holds_kv)
                self.assertFalse(rejected.kv.holds_mamba)
                self.assertIs(cache.session.slots[session.session_id], slot)
                self.assertIs(slot.kv, saved_kv)
                self.assertEqual(slot.kv.req_pool_idx, saved_row)
                torch.testing.assert_close(
                    pool.req_to_token[saved_row, : len(saved_indices)], saved_indices
                )
                torch.testing.assert_close(slot.kv.mamba_pool_idx, saved_mamba)
                torch.testing.assert_close(
                    slot.kv.mamba_ping_pong_track_buffer, saved_tracking
                )
                self.assertEqual(self._pool_counts(cache, allocator, pool), before)
                self.assertEqual(
                    session._inflight_rid, "active" if later_owner else None
                )
                self.assertEqual(set(session.req_nodes), {"turn-1"})

                # Cleanup may be reached again after session context is detached.
                self.assertIsNone(cache.session.find_active_slot(rejected))
                rejected.finished_reason = rejected.to_finish
                rejected.to_finish = None
                release_kv_cache(rejected, cache, is_insert=False)
                self.assertEqual(self._pool_counts(cache, allocator, pool), before)
                if active is None:
                    active = session.create_req(_recv("active", [99]), None, VOCAB_SIZE)
                    active.init_next_round_input(cache, cow_mamba=False)
                self.assertIs(active.kv, saved_kv)
                self.assertEqual(list(active.origin_input_ids), list(range(16)) + [99])
                _prefill(active, cache, allocator, pool)
                _finish_turn(active, cache, 0)
                controller = SessionController(cache)
                controller.sessions[session.session_id] = session
                controller.close(CloseSessionReqInput(session_id=session.session_id))
                self.assertEqual(controller.sessions, {})
                self.assertFalse(cache.session.has_slot(session.session_id))
                self._assert_idle(observer, checker)

    def test_preabort_keeps_request_owned_allocations_for_raw_cleanup(self):
        """An absent or empty retained slot must not hide a first turn's allocations."""
        for empty_slot in (False, True):
            with self.subTest(empty_slot=empty_slot):
                _, cache, allocator, pool, observer, checker = _build()
                session = Session(0, "session-a", streaming=True)
                req = session.create_req(_recv("first", [1, 2]), None, VOCAB_SIZE)
                _prefill(req, cache, allocator, pool)
                if empty_slot:
                    cache.session.slots[session.session_id] = SessionSlot()
                own_record = req.kv
                before = self._pool_counts(cache, allocator, pool)
                req.to_finish = FINISH_ABORT("cancelled first turn")
                self.assertIsNone(cache.session.find_active_slot(req))
                self.assertIsNone(req.session)
                self.assertIs(req.kv, own_record)
                self.assertTrue(req.kv.holds_kv)
                self.assertTrue(req.kv.holds_mamba)
                self.assertEqual(self._pool_counts(cache, allocator, pool), before)
                req.finished_reason = req.to_finish
                req.to_finish = None
                release_kv_cache(req, cache, is_insert=False)
                self.assertFalse(req.kv.holds_kv)
                self.assertFalse(req.kv.holds_mamba)
                cache.release_session(session.session_id)
                self._assert_idle(observer, checker)

    def test_oom_retract_of_last_streaming_turn_aborts_session_turn(self):
        (
            server_args,
            cache,
            allocator,
            req_to_token_pool,
            observer,
            checker,
            session,
        ) = self._setup_first_turn()
        req = session.create_req(
            _recv("turn-2", list(range(32, 48))),
            tokenizer=None,
            vocab_size=VOCAB_SIZE,
        )
        req.init_next_round_input(cache)
        _prefill(req, cache, allocator, req_to_token_pool)

        batch = ScheduleBatch(reqs=[req])
        batch.req_to_token_pool = req_to_token_pool
        batch.token_to_kv_pool_allocator = allocator
        batch.tree_cache = cache
        batch.hisparse_coordinator = None
        batch.spec_algorithm = SimpleNamespace(is_none=lambda: True)

        with patch.object(batch, "check_decode_mem", return_value=False):
            with patch.object(batch, "filter_batch"):
                with patch.object(
                    NewTokenRatioTracker,
                    "estimate_new_token_ratio_after_retract",
                    return_value=0.0,
                ):
                    retracted, _ratio, reqs_to_abort = batch.retract_decode()

        self.assertEqual(reqs_to_abort, [req])
        self.assertEqual(retracted, [])
        self.assertIsInstance(req.to_finish, FINISH_ABORT)
        self.assertIsNone(req.kv.mamba_pool_idx)
        self.assertIsNone(req.kv.req_pool_idx)
        self.assertFalse(session.has_unfinished_request())
        self.assertFalse(cache.session.has_slot(session.session_id))
        self.assertEqual(set(session.req_nodes), {"turn-1"})
        self._assert_idle(observer, checker)
        follow_up = session.create_req(
            _recv("turn-3", [99]),
            tokenizer=None,
            vocab_size=VOCAB_SIZE,
        )
        self.assertIsNone(follow_up.finished_reason)

    def test_retract_backup_failure_aborts_session_turn(self):
        (
            _server_args,
            cache,
            allocator,
            req_to_token_pool,
            observer,
            checker,
            session,
        ) = self._setup_first_turn()
        req = session.create_req(
            _recv("turn-2", list(range(32, 48))),
            tokenizer=None,
            vocab_size=VOCAB_SIZE,
        )
        req.init_next_round_input(cache)
        _prefill(req, cache, allocator, req_to_token_pool)

        # A bystander decode request the retraction order prefers to keep
        # (more output tokens), so the loop retracts the session turn.
        bystander_session = Session(
            capacity_of_str_len=0, session_id="session-b", streaming=True
        )
        bystander = bystander_session.create_req(
            _recv("bystander", list(range(48, 64)), session_id="session-b"),
            tokenizer=None,
            vocab_size=VOCAB_SIZE,
        )
        _prefill(bystander, cache, allocator, req_to_token_pool)
        _decode_step(bystander, allocator, req_to_token_pool, 100)

        batch = ScheduleBatch(reqs=[req, bystander])
        batch.req_to_token_pool = req_to_token_pool
        batch.token_to_kv_pool_allocator = allocator
        batch.tree_cache = cache
        batch.hisparse_coordinator = None
        batch.spec_algorithm = SimpleNamespace(is_none=lambda: True)

        fake_disagg = SimpleNamespace(
            disaggregation_mode="decode",
            disaggregation_decode_retraction_backup=None,
        )
        # check_decode_mem True throughout: the first_iter pass retracts
        # exactly one request and the last-request OOM branch never fires.
        with patch.object(batch, "check_decode_mem", return_value=True):
            with patch.object(batch, "filter_batch"):
                with patch.object(
                    NewTokenRatioTracker,
                    "estimate_new_token_ratio_after_retract",
                    return_value=0.0,
                ):
                    with patch(
                        "sglang.srt.managers.schedule_batch.get_disagg",
                        return_value=fake_disagg,
                    ):
                        with patch(
                            "sglang.srt.managers.schedule_batch.retraction_backup",
                            return_value=False,
                        ) as backup:
                            retracted, _ratio, reqs_to_abort = batch.retract_decode()

        backup.assert_called_once()
        self.assertEqual(retracted, [])
        self.assertEqual(reqs_to_abort, [req])
        self.assertIsInstance(req.to_finish, FINISH_ABORT)
        self.assertIn("Retraction host KV pool exhausted", req.to_finish.message)
        # The retract nuke ran (turn KV + slot dropped) and the terminal
        # abort cleared the session inflight marker.
        self.assertIsNone(req.kv.req_pool_idx)
        self.assertIsNone(req.kv.mamba_pool_idx)
        self.assertFalse(session.has_unfinished_request())
        self.assertFalse(cache.session.has_slot(session.session_id))
        self.assertEqual(set(session.req_nodes), {"turn-1"})
        # The bystander was never retracted and stays decode-active.
        self.assertIsNone(bystander.to_finish)
        self.assertTrue(bystander_session.has_unfinished_request())

        _finish_turn(bystander, cache, finished_len=1)
        cache.session.release_session(bystander_session.session_id)
        self._assert_idle(observer, checker)
        follow_up = session.create_req(
            _recv("turn-3", [99]),
            tokenizer=None,
            vocab_size=VOCAB_SIZE,
        )
        self.assertIsNone(follow_up.finished_reason)

    def test_bootstrap_failure_before_allocation_aborts_session_turn(self):
        (
            _server_args,
            cache,
            _allocator,
            _req_to_token_pool,
            observer,
            checker,
            session,
        ) = self._setup_first_turn()
        req = session.create_req(
            _recv("turn-2", list(range(32, 48))),
            tokenizer=None,
            vocab_size=VOCAB_SIZE,
        )
        self.assertIsNone(req.kv.req_pool_idx)
        self.assertIsNone(req.kv.mamba_pool_idx)
        req.disagg_kv_sender = SimpleNamespace(failure_exception=lambda: None)
        req.bootstrap_room = 0
        req.time_stats = SimpleNamespace(
            trace_ctx=SimpleNamespace(abort=lambda **kwargs: None)
        )
        scheduler = SimpleNamespace(
            tree_cache=cache,
            ps=SimpleNamespace(tp_rank=0),
            enable_hicache_storage=False,
            disagg_prefill_pending_chunk_rids=set(),
            req_to_metadata_buffer_idx_allocator=None,
            output_streamer=SimpleNamespace(stream_output=lambda reqs, rl: None),
            metrics_reporter=SimpleNamespace(enable_metrics=False),
        )
        scheduler.clear_pending_chunk_send = (
            SchedulerDisaggregationPrefillMixin.clear_pending_chunk_send.__get__(
                scheduler
            )
        )

        scheduler._release_dropped_waiting_req_mm_inputs = (
            Scheduler._release_dropped_waiting_req_mm_inputs.__get__(scheduler)
        )
        scheduler._release_aborted_request = Scheduler._release_aborted_request.__get__(
            scheduler
        )
        SchedulerDisaggregationPrefillMixin.handle_bootstrap_failure(scheduler, req)

        self.assertTrue(req.finished())
        self.assertIsInstance(req.finished_reason, FINISH_ABORT)
        self.assertFalse(session.has_unfinished_request())
        self.assertEqual(set(session.req_nodes), {"turn-1"})
        self._assert_idle(observer, checker)
        follow_up = session.create_req(
            _recv("turn-3", [99]),
            tokenizer=None,
            vocab_size=VOCAB_SIZE,
        )
        self.assertIsNone(follow_up.finished_reason)

    def test_retract_readmit_finish_matches_no_retract(self):
        worlds = []
        for retract in (False, True):
            (
                _server_args,
                cache,
                allocator,
                req_to_token_pool,
                observer,
                checker,
                session,
            ) = self._setup_first_turn()
            req = session.create_req(
                _recv("turn-2", list(range(32, 48))),
                tokenizer=None,
                vocab_size=VOCAB_SIZE,
            )
            req.init_next_round_input(cache)
            _prefill(req, cache, allocator, req_to_token_pool)
            _decode_step(req, allocator, req_to_token_pool, 100)
            _decode_step(req, allocator, req_to_token_pool, 101)
            if retract:
                release_req(
                    req=req,
                    remaing_req_count=1,
                    req_to_token_pool=req_to_token_pool,
                    token_to_kv_pool_allocator=allocator,
                    tree_cache=cache,
                    hisparse_coordinator=None,
                    offload_kv=False,
                )
                self.assertFalse(cache.session.has_slot(session.session_id))
                self.assertEqual(set(session.req_nodes), {"turn-1"})
                self.assertTrue(session.has_unfinished_request())
                self.assertEqual(cache.session.session_held_mamba_slots(), 0)
                self._assert_idle(observer, checker)
                req.init_next_round_input(cache)
                _prefill(req, cache, allocator, req_to_token_pool)
                _decode_step(req, allocator, req_to_token_pool, 102)
                _decode_step(req, allocator, req_to_token_pool, 103)
            else:
                _decode_step(req, allocator, req_to_token_pool, 102)
                _decode_step(req, allocator, req_to_token_pool, 103)
            _finish_turn(req, cache, finished_len=4)
            worlds.append((cache, observer, checker, session, req))

        cache_a, observer_a, checker_a, session_a, req_a = worlds[0]
        cache_b, observer_b, checker_b, session_b, req_b = worlds[1]
        self.assertEqual(set(session_a.req_nodes), set(session_b.req_nodes))
        self.assertEqual(list(req_a.output_ids), list(req_b.output_ids))
        self.assertEqual(session_a.committed_origin_len, session_b.committed_origin_len)
        self.assertEqual(
            session_a.committed_unpadded_len, session_b.committed_unpadded_len
        )
        self.assertEqual(session_a.committed_fill_len, session_b.committed_fill_len)
        slot_a = cache_a.session.slots[session_a.session_id]
        slot_b = cache_b.session.slots[session_b.session_id]
        self.assertEqual(slot_a.kv.kv_committed_len, slot_b.kv.kv_committed_len)
        self.assertEqual(slot_a.kv.kv_allocated_len, slot_b.kv.kv_allocated_len)
        self.assertFalse(session_a.has_unfinished_request())
        self.assertFalse(session_b.has_unfinished_request())
        self.assertEqual(
            cache_a.session.session_held_mamba_slots(),
            cache_b.session.session_held_mamba_slots(),
        )
        self._assert_idle(observer_a, checker_a)
        self._assert_idle(observer_b, checker_b)
        next_a = session_a.create_req(
            _recv("turn-3", [99], parent_rid="turn-2"),
            tokenizer=None,
            vocab_size=VOCAB_SIZE,
        )
        next_b = session_b.create_req(
            _recv("turn-3", [99], parent_rid="turn-2"),
            tokenizer=None,
            vocab_size=VOCAB_SIZE,
        )
        self.assertEqual(list(next_a.origin_input_ids), list(next_b.origin_input_ids))

    def test_readmitted_abort_rolls_back_and_keeps_later_owner(self):
        """An aborted retry must not replace the last successful turn's history."""
        _, cache, allocator, pool, observer, checker, session = self._setup_first_turn()
        req = session.create_req(_recv("turn-2", [32, 33]), None, VOCAB_SIZE)
        req.init_next_round_input(cache)
        _prefill(req, cache, allocator, pool)
        _decode_step(req, allocator, pool, 100)
        release_req(
            req=req,
            remaing_req_count=1,
            req_to_token_pool=pool,
            token_to_kv_pool_allocator=allocator,
            tree_cache=cache,
            hisparse_coordinator=None,
            offload_kv=False,
        )
        req.init_next_round_input(cache)
        _prefill(req, cache, allocator, pool)
        req.finished_reason = FINISH_ABORT("cancel retry")
        release_kv_cache(req, cache, is_insert=False)
        self._assert_idle(observer, checker)
        self.assertEqual(set(session.req_nodes), {"turn-1"})
        next_req = session.create_req(_recv("turn-3", [99]), None, VOCAB_SIZE)
        self.assertIsNone(next_req.to_finish)
        self.assertEqual(list(next_req.origin_input_ids), list(range(16)) + [99])
        session.abort_req(req.rid)
        rejected = session.create_req(_recv("turn-4", [101]), None, VOCAB_SIZE)
        self.assertIsInstance(rejected.to_finish, FINISH_ABORT)
        self.assertEqual(session._inflight_rid, next_req.rid)

    def test_first_turn_retract_does_not_create_history(self):
        _, cache, allocator, pool, observer, checker = _build()
        session = Session(0, "session-a", streaming=True)
        req = session.create_req(_recv("turn-1", [1, 2]), None, VOCAB_SIZE)
        _prefill(req, cache, allocator, pool)
        _decode_step(req, allocator, pool, 3)
        release_req(
            req=req,
            remaing_req_count=1,
            req_to_token_pool=pool,
            token_to_kv_pool_allocator=allocator,
            tree_cache=cache,
            hisparse_coordinator=None,
            offload_kv=False,
        )
        self.assertEqual(session.req_nodes, {})
        self.assertEqual(session._inflight_rid, req.rid)
        self._assert_idle(observer, checker)
        req.init_next_round_input(cache)
        _prefill(req, cache, allocator, pool)
        _finish_turn(req, cache, 1)
        next_req = session.create_req(_recv("turn-2", [4]), None, VOCAB_SIZE)
        self.assertEqual(list(next_req.origin_input_ids), [1, 2, 3, 4])
        cache.release_session(session.session_id)
        self._assert_idle(observer, checker)

    def test_optimistic_retry_preserves_turn_and_advances_cache_attempt(self):
        _, cache, allocator, pool, observer, checker, session = self._setup_first_turn()
        req = session.create_req(_recv("turn-2", [32, 33]), None, VOCAB_SIZE)
        req.init_next_round_input(cache)
        _prefill(req, cache, allocator, pool)
        previous_handle = req.cache_request_handle
        scheduler = SimpleNamespace(
            tree_cache=cache,
            _release_aborted_request=Mock(),
            clear_pending_chunk_send=Mock(),
            metrics_reporter=SimpleNamespace(enable_metrics=False),
            processed_tokens_counter=0,
            waiting_queue=[],
        )
        with patch(
            "sglang.srt.disaggregation.prefill.get_disagg",
            return_value=SimpleNamespace(optimistic_prefill_attempts=2),
        ):
            SchedulerDisaggregationPrefillMixin.optimistic_release_and_requeue(
                scheduler, req
            )
        self.assertEqual(scheduler.waiting_queue, [req])
        self.assertEqual(set(session.req_nodes), {"turn-1"})
        self.assertEqual(session._inflight_rid, req.rid)
        self.assertEqual(req.cache_request_handle.rid, previous_handle.rid)
        self.assertEqual(
            req.cache_request_handle.attempt_id, previous_handle.attempt_id + 1
        )
        self._assert_idle(observer, checker)
        req.init_next_round_input(cache)
        _prefill(req, cache, allocator, pool)
        _finish_turn(req, cache, 0)
        self.assertEqual(set(session.req_nodes), {"turn-2"})
        cache.release_session(session.session_id)
        self._assert_idle(observer, checker)

    def test_prefill_failure_does_not_commit_allocated_turn(self):
        for method in ("handle_bootstrap_failure", "handle_inflight_transfer_failure"):
            with self.subTest(method=method):
                _, cache, allocator, pool, observer, checker, session = (
                    self._setup_first_turn()
                )
                req = session.create_req(_recv("turn-2", [32, 33]), None, VOCAB_SIZE)
                req.init_next_round_input(cache)
                _prefill(req, cache, allocator, pool)
                req.disagg_kv_sender = SimpleNamespace(failure_exception=lambda: None)
                scheduler = SimpleNamespace(
                    ps=SimpleNamespace(tp_rank=0),
                    tree_cache=cache,
                    clear_pending_chunk_send=Mock(),
                    _release_aborted_request=Mock(),
                    req_to_metadata_buffer_idx_allocator=None,
                    output_streamer=SimpleNamespace(stream_output=Mock()),
                    metrics_reporter=SimpleNamespace(enable_metrics=False),
                    enable_hicache_storage=False,
                )
                scheduler._release_dropped_waiting_req_mm_inputs = (
                    Scheduler._release_dropped_waiting_req_mm_inputs.__get__(scheduler)
                )
                getattr(SchedulerDisaggregationPrefillMixin, method)(scheduler, req)
                self.assertEqual(set(session.req_nodes), {"turn-1"})
                self.assertFalse(session.has_unfinished_request())
                self.assertIsInstance(req.finished_reason, FINISH_ABORT)
                self._assert_idle(observer, checker)

    def test_decode_resume_restores_session_backup_once(self):
        """Discarding a session row must not discard the live request's host backup."""
        _, cache, allocator, pool, observer, checker, session = self._setup_first_turn()
        req = session.create_req(_recv("turn-2", [32, 33]), None, VOCAB_SIZE)
        req.init_next_round_input(cache)
        _prefill(req, cache, allocator, pool)
        _decode_step(req, allocator, pool, 100)
        kv_pool = allocator.get_kvcache().full_kv_pool
        old_indices = pool.req_to_token[req.kv.req_pool_idx, : req.seqlen - 1].clone()
        for layer, (key, value) in enumerate(zip(kv_pool.k_buffer, kv_pool.v_buffer)):
            key[old_indices] = 10 + layer
            value[old_indices] = 20 + layer
        old_mamba = req.kv.mamba_pool_idx
        for conv in pool.mamba_pool.mamba_cache.conv:
            conv[:, old_mamba] = 30
        pool.mamba_pool.mamba_cache.temporal[:, old_mamba] = 40
        disagg = SimpleNamespace(
            disaggregation_mode="decode",
            disaggregation_decode_retraction_backup="cpu_tensor",
        )
        copy_to_host = allocator.get_cpu_copy
        # CPU-to-CPU .to() may alias the source; model an independent host copy.
        with (
            patch("sglang.srt.managers.schedule_batch.get_disagg", return_value=disagg),
            patch.object(
                allocator,
                "get_cpu_copy",
                side_effect=lambda *args, **kwargs: copy.deepcopy(
                    copy_to_host(*args, **kwargs)
                ),
            ),
        ):
            release_req(
                req=req,
                remaing_req_count=1,
                req_to_token_pool=pool,
                token_to_kv_pool_allocator=allocator,
                tree_cache=cache,
                hisparse_coordinator=None,
            )
        self._assert_idle(observer, checker)
        for key, value in zip(kv_pool.k_buffer, kv_pool.v_buffer):
            key.zero_()
            value.zero_()
        for conv in pool.mamba_pool.mamba_cache.conv:
            conv.zero_()
        pool.mamba_pool.mamba_cache.temporal.zero_()
        queue = SimpleNamespace(
            retracted_queue=[req],
            req_to_token_pool=pool,
            token_to_kv_pool_allocator=allocator,
            tree_cache=cache,
            _uses_swa_tail_prealloc=lambda: False,
            _allocatable_token_budgets=lambda **kwargs: KV_SIZE,
            _prealloc_required_tokens=lambda req: (req.seqlen, 0),
            _pre_alloc=lambda req: _prefill(req, cache, allocator, pool),
        )
        with (
            patch("sglang.srt.disaggregation.decode.get_disagg", return_value=disagg),
            patch.object(
                allocator, "load_cpu_copy", wraps=allocator.load_cpu_copy
            ) as load,
        ):
            self.assertEqual(DecodePreallocQueue.resume_retracted_reqs(queue), [req])
            self.assertEqual(DecodePreallocQueue.resume_retracted_reqs(queue), [])
            load.assert_called_once()
        new_indices = pool.req_to_token[req.kv.req_pool_idx, : req.seqlen - 1]
        for layer, (key, value) in enumerate(zip(kv_pool.k_buffer, kv_pool.v_buffer)):
            self.assertTrue(torch.all(key[new_indices] == 10 + layer))
            self.assertTrue(torch.all(value[new_indices] == 20 + layer))
        for conv in pool.mamba_pool.mamba_cache.conv:
            self.assertTrue(torch.all(conv[:, req.kv.mamba_pool_idx] == 30))
        self.assertTrue(
            torch.all(
                pool.mamba_pool.mamba_cache.temporal[:, req.kv.mamba_pool_idx] == 40
            )
        )
        self.assertIsNone(req.kv.retraction_backup)
        self.assertEqual(session._inflight_rid, req.rid)
        _finish_turn(req, cache, 1)
        cache.release_session(session.session_id)
        self._assert_idle(observer, checker)

    def test_successful_prefill_transfer_commits_the_turn(self):
        _, cache, allocator, pool, observer, checker, session = self._setup_first_turn()
        req = session.create_req(_recv("turn-2", [32, 33]), None, VOCAB_SIZE)
        req.init_next_round_input(cache)
        _prefill(req, cache, allocator, pool)
        req.disagg_kv_sender = SimpleNamespace(clear=Mock())
        req.bootstrap_host = FAKE_BOOTSTRAP_HOST
        scheduler = SimpleNamespace(
            disagg_prefill_inflight_queue=[req],
            attn_cp_cpu_group=None,
            attn_tp_cpu_group=None,
            tree_cache=cache,
            req_to_metadata_buffer_idx_allocator=None,
            output_streamer=SimpleNamespace(stream_output=Mock()),
            scheduler_stage_metrics=None,
        )
        with patch(
            "sglang.srt.disaggregation.prefill.poll_and_all_reduce_attn_cp_tp_group",
            return_value=[KVPoll.Success],
        ):
            done = SchedulerDisaggregationPrefillMixin.process_disagg_prefill_inflight_queue(
                scheduler
            )
        self.assertEqual(done, [req])
        self.assertEqual(scheduler.disagg_prefill_inflight_queue, [])
        self.assertEqual(set(session.req_nodes), {"turn-2"})
        self.assertFalse(session.has_unfinished_request())
        next_req = session.create_req(_recv("turn-3", [99]), None, VOCAB_SIZE)
        self.assertEqual(
            list(next_req.origin_input_ids), list(range(16)) + [32, 33, 99]
        )
        cache.release_session(session.session_id)
        self._assert_idle(observer, checker)

    def test_release_intent_through_chunk_and_radix_adapters(self):
        """Raw caches must still release rows; wrapped caches must retain checkpoints."""
        publish(ServerArgs(model_path="dummy"), role="scheduler")
        for cache_type in (ChunkCache, RadixCache):
            for streaming in (False, True):
                with self.subTest(cache=cache_type.__name__, streaming=streaming):
                    pool = ReqToTokenPool(
                        size=8,
                        max_context_len=128,
                        device="cpu",
                        enable_memory_saver=False,
                    )
                    kv_pool = MHATokenToKVPool(
                        size=64,
                        page_size=1,
                        dtype=torch.float16,
                        head_num=2,
                        head_dim=8,
                        layer_num=1,
                        device="cpu",
                        enable_memory_saver=False,
                    )
                    allocator = TokenToKVPoolAllocator(
                        size=64,
                        dtype=torch.float16,
                        device="cpu",
                        kvcache=kv_pool,
                        need_sort=False,
                    )
                    cache = cache_type(
                        CacheInitParams(
                            disable=False,
                            req_to_token_pool=pool,
                            token_to_kv_pool_allocator=allocator,
                            page_size=1,
                            eviction_policy="lru",
                        )
                    )
                    session = Session(0, "session-a", streaming=streaming)
                    if streaming:
                        cache = StreamingSession(cache)
                    req = session.create_req(_recv("turn-1", [1, 2]), None, VOCAB_SIZE)
                    _prefill(req, cache, allocator, pool)
                    release_req(
                        req=req,
                        remaing_req_count=1,
                        req_to_token_pool=pool,
                        token_to_kv_pool_allocator=allocator,
                        tree_cache=cache,
                        hisparse_coordinator=None,
                        offload_kv=False,
                    )
                    self.assertEqual(pool.available_size(), 8)
                    self.assertEqual(
                        allocator.available_size() + cache.evictable_size(), 64
                    )
                    if streaming:
                        self.assertEqual(session.req_nodes, {})
                        self.assertEqual(session._inflight_rid, req.rid)
                    req.init_next_round_input(cache)
                    _prefill(req, cache, allocator, pool)
                    _finish_turn(req, cache, 0)
                    if streaming:
                        next_req = session.create_req(
                            _recv("turn-2", [3]), None, VOCAB_SIZE
                        )
                        self.assertEqual(list(next_req.origin_input_ids), [1, 2, 3])
                        cache.release_session(session.session_id)
                    self.assertEqual(pool.available_size(), 8)
                    self.assertEqual(
                        allocator.available_size() + cache.evictable_size(), 64
                    )

    def test_unflagged_release_of_unfinished_req_is_a_normal_finish(self):
        # A cache release without retract intent keeps its existing finish
        # contract, even if a caller has not stamped the finish reason yet.
        (
            _server_args,
            cache,
            allocator,
            req_to_token_pool,
            observer,
            checker,
            session,
        ) = self._setup_first_turn()
        req = session.create_req(
            _recv("turn-2", list(range(32, 48))),
            tokenizer=None,
            vocab_size=VOCAB_SIZE,
        )
        req.init_next_round_input(cache)
        _prefill(req, cache, allocator, req_to_token_pool)
        self.assertIsNone(req.finished_reason)

        release_kv_cache(req, cache)
        req.finished_reason = FINISH_LENGTH(length=0)
        req.finished_len = 0

        self.assertTrue(cache.session.has_slot(session.session_id))
        self.assertFalse(session.has_unfinished_request())
        self.assertEqual(set(session.req_nodes), {"turn-2"})
        self.assertIs(session.req_nodes["turn-2"].req, req)
        self.assertIsNone(req.kv.req_pool_idx)
        self.assertGreater(cache.session.session_held_mamba_slots(), 0)
        committed = list(req.origin_input_ids) + list(req.output_ids)
        follow_up = session.create_req(
            _recv("turn-3", [99], parent_rid="turn-2"),
            tokenizer=None,
            vocab_size=VOCAB_SIZE,
        )
        self.assertIsNone(follow_up.finished_reason)
        self.assertEqual(list(follow_up.origin_input_ids), committed + [99])
        cache.session.release_session(session.session_id)
        self._assert_idle(observer, checker)


if __name__ == "__main__":
    unittest.main()
