"""Mamba checkpoint-gap accounting across request admission and reporting."""

import unittest
from array import array
from functools import partial
from types import SimpleNamespace

import prometheus_client
import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.distributed.parallel_state_wrapper import ParallelState
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.schedule_policy import (
    AddReqResult,
    PrefillAdder,
    match_prefix_for_req,
)
from sglang.srt.managers.scheduler_components.kv_events_publisher import (
    SchedulerKvEventsPublisher,
)
from sglang.srt.managers.scheduler_components.metrics_reporter import (
    PrefillStats,
    SchedulerMetricsReporter,
)
from sglang.srt.managers.scheduler_components.pool_stats_observer import PoolStats
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    MatchResult,
    get_mamba_cache_miss_cause,
    get_mamba_cache_miss_tokens,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.chunk_cache import ChunkCache
from sglang.srt.observability.metrics_collector import (
    SchedulerMetricsCollector,
    SchedulerMetricsCollectorContext,
)
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.sampling_params import SamplingParams

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _MatchCache(ChunkCache):
    """Supply lookup records at the cache boundary; consumers remain real."""

    def __init__(self):
        allocator = TokenToKVPoolAllocator(
            size=4096,
            dtype=torch.float32,
            device="cpu",
            kvcache=None,
            need_sort=False,
        )
        super().__init__(
            CacheInitParams(
                disable=False,
                req_to_token_pool=None,
                token_to_kv_pool_allocator=allocator,
                page_size=1,
            )
        )
        self.result = _match()
        self.load_back_tokens = 0

    def match_prefix(self, params):
        return self.result

    def init_load_back(self, params):
        return torch.arange(self.load_back_tokens), None


def _match(*, device=32, host=0, full=128, aligned=128, evicted=False, eligible=True):
    return MatchResult(
        device_indices=torch.arange(device),
        last_device_node=None,
        last_host_node=None,
        best_match_node=None,
        host_hit_length=host,
        full_kv_hit_length=full,
        mamba_branching_seqlen=aligned,
        mamba_state_evicted_in_gap=evicted,
        mamba_cache_miss_eligible=eligible,
    )


def _req(rid="request"):
    return Req(
        rid=rid,
        origin_input_text="",
        origin_input_ids=array("q", range(257)),
        sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
    )


def _adder(cache, *, chunk_tokens=None, max_requests=None):
    return PrefillAdder(
        page_size=1,
        tree_cache=cache,
        token_to_kv_pool_allocator=cache.token_to_kv_pool_allocator,
        running_batch=None,
        new_token_ratio=1.0,
        rem_input_tokens=4096,
        rem_chunk_tokens=chunk_tokens,
        prefill_max_requests=max_requests,
    )


def _admit(adder, req):
    return adder.add_one_req(req, has_chunked_req=False, truncation_align_size=None)


class TestMambaCacheMissAdmission(CustomTestCase):
    def setUp(self):
        override = get_context().override_server_args(
            enable_metrics=False,
            enable_metrics_for_all_schedulers=False,
            enable_mfu_metrics=False,
            enable_forward_pass_metrics=False,
            kv_events_config=None,
        )
        self.server_args = override.install()
        self.addCleanup(override.restore)
        self.cache = _MatchCache()

    def test_both_lookup_callers_transport_gap_and_cause(self):
        cases = [
            (_match(evicted=True), 96, "state_evicted"),
            (_match(evicted=False), 96, "never_saved"),
            (_match(evicted=None), 96, "unknown"),
            (_match(device=32, host=64), 32, "never_saved"),
            (_match(full=111, aligned=96), 64, "never_saved"),
            (_match(full=96, aligned=128), 64, "never_saved"),
            (_match(device=96, aligned=64), 0, "never_saved"),
            (_match(aligned=None), 0, "never_saved"),
            (_match(eligible=False), 0, "never_saved"),
        ]
        for match, expected_tokens, expected_cause in cases:
            for caller in ("policy", "request"):
                with self.subTest(match=match, caller=caller):
                    req = _req()
                    self.cache.result = match
                    if caller == "policy":
                        match_prefix_for_req(self.cache, req)
                    else:
                        req.init_next_round_input(self.cache)
                    self.assertEqual(req.mamba_cache_miss_tokens, expected_tokens)
                    self.assertEqual(req.mamba_cache_miss_cause, expected_cause)
                    self.assertEqual(
                        get_mamba_cache_miss_tokens(match), expected_tokens
                    )
                    self.assertEqual(get_mamba_cache_miss_cause(match), expected_cause)
                    self.assertFalse(req._mamba_cache_miss_reported)

    def test_deferred_admission_uses_latest_lookup(self):
        req = _req()
        self.cache.result = _match(evicted=True)
        match_prefix_for_req(self.cache, req)
        req.init_next_round_input(self.cache)
        rejected = _adder(self.cache, max_requests=0)
        self.assertEqual(_admit(rejected, req), AddReqResult.OTHER)
        self.assertEqual(
            PrefillStats.from_adder(rejected, []).mamba_cache_miss_by_cause, {}
        )
        self.assertFalse(req._mamba_cache_miss_reported)

        self.cache.result = _match(device=96, evicted=False)
        req.init_next_round_input(self.cache)
        admitted = _adder(self.cache)
        _admit(admitted, req)
        self.assertEqual(admitted.can_run_list, [req])
        self.assertEqual(
            PrefillStats.from_adder(admitted, []).mamba_cache_miss_by_cause,
            {"never_saved": (1, 32)},
        )
        self.assertTrue(req._mamba_cache_miss_reported)

    def test_chunks_and_retraction_do_not_report_request_again(self):
        req = _req()
        self.cache.result = _match(evicted=True)
        req.init_next_round_input(self.cache)
        first = _adder(self.cache, chunk_tokens=32)
        _admit(first, req)
        self.assertIs(first.new_chunked_req, req)
        self.assertEqual(
            PrefillStats.from_adder(first, []).mamba_cache_miss_by_cause,
            {"state_evicted": (1, 96)},
        )

        self.cache.result = _match(device=64, evicted=False)
        req.init_next_round_input(self.cache)
        continuation = _adder(self.cache, chunk_tokens=32)
        continuation.add_chunked_req(req)
        self.assertEqual(continuation.can_run_list, [req])
        self.assertEqual(
            PrefillStats.from_adder(continuation, []).mamba_cache_miss_by_cause, {}
        )

        req.reset_for_retract()
        self.assertTrue(req._mamba_cache_miss_reported)
        self.cache.result = _match(device=0, evicted=True)
        req.init_next_round_input(self.cache)
        retracted = _adder(self.cache)
        _admit(retracted, req)
        self.assertEqual(retracted.can_run_list, [req])
        self.assertEqual(
            PrefillStats.from_adder(retracted, []).mamba_cache_miss_by_cause, {}
        )

    def test_zero_gap_first_admission_is_also_latched(self):
        req = _req()
        self.cache.result = _match(device=128)
        req.init_next_round_input(self.cache)
        first = _adder(self.cache)
        _admit(first, req)
        self.assertEqual(
            PrefillStats.from_adder(first, []).mamba_cache_miss_by_cause, {}
        )
        self.assertTrue(req._mamba_cache_miss_reported)

        req.reset_for_retract()
        self.cache.result = _match(device=0, evicted=True)
        req.init_next_round_input(self.cache)
        second = _adder(self.cache)
        _admit(second, req)
        self.assertEqual(
            PrefillStats.from_adder(second, []).mamba_cache_miss_by_cause, {}
        )

    def test_host_load_expansion_reduces_gap_before_admission(self):
        for loaded_tokens, expected in (
            (64, {"never_saved": (1, 32)}),
            (96, {}),
        ):
            with self.subTest(loaded_tokens=loaded_tokens):
                req = _req()
                self.cache.result = _match(device=32, host=32)
                self.cache.load_back_tokens = loaded_tokens
                req.init_next_round_input(self.cache)
                self.assertEqual(req.mamba_cache_miss_tokens, 64)
                adder = _adder(self.cache)
                _admit(adder, req)
                self.assertEqual(len(req.prefix_indices), 32 + loaded_tokens)
                self.assertEqual(
                    PrefillStats.from_adder(adder, []).mamba_cache_miss_by_cause,
                    expected,
                )
                self.assertTrue(req._mamba_cache_miss_reported)

    def _collector(self):
        registry = prometheus_client.CollectorRegistry()

        class Collector(SchedulerMetricsCollector):
            _counter_cls = partial(prometheus_client.Counter, registry=registry)
            _gauge_cls = partial(prometheus_client.Gauge, registry=registry)
            _histogram_cls = partial(prometheus_client.Histogram, registry=registry)
            _summary_cls = partial(prometheus_client.Summary, registry=registry)

        return Collector(
            labels={"moe_ep_rank": 0}, server_args=self.server_args
        ), registry

    def _reporter(self, collector, ps):
        pool_stats = PoolStats(
            full_num_used=0,
            full_token_usage=0,
            full_available_size=4096,
            full_evictable_size=0,
        )
        scheduler = SimpleNamespace(
            ps=ps,
            device="cpu",
            forward_ct=0,
            waiting_queue=[],
            chunked_req=None,
            grammar_manager=[],
            enable_priority_scheduling=False,
            enable_hierarchical_cache=False,
            enable_lora=False,
            disaggregation_mode=DisaggregationMode.NULL,
            pool_stats_observer=SimpleNamespace(get_pool_stats=lambda: pool_stats),
        )
        context = SchedulerMetricsCollectorContext(
            enable_metrics=True,
            is_stats_logging_rank=False,
            current_scheduler_metrics_enabled=True,
            enable_kv_cache_events=False,
            collector=collector,
        )
        reporter = SchedulerMetricsReporter(
            scheduler=scheduler,
            tp_rank=ps.tp_rank,
            pp_rank=ps.pp_rank,
            dp_rank=ps.dp_rank,
            metrics_collector_context=context,
            metrics_collector=collector,
        )
        scheduler.kv_events_publisher = SchedulerKvEventsPublisher(
            kv_events_config=None,
            ps=ps,
            attn_tp_rank=ps.attn_tp_rank,
            attn_cp_rank=ps.attn_cp_rank,
            attn_dp_rank=ps.attn_dp_rank,
            dp_rank=ps.dp_rank,
            tree_cache=self.cache,
            send_metrics_from_scheduler=None,
            max_running_requests=16,
            max_total_num_tokens=4096,
            get_stats=lambda: reporter.stats,
        )
        return reporter

    def test_admission_groups_causes_and_exports_once_per_attention_replica(self):
        adder = _adder(self.cache)
        for index, evicted in enumerate((True, True, False, None)):
            req = _req(str(index))
            self.cache.result = _match(device=32 + 16 * index, evicted=evicted)
            req.init_next_round_input(self.cache)
            _admit(adder, req)
        stats = PrefillStats.from_adder(adder, [])
        expected = {
            "state_evicted": (2, 176),
            "never_saved": (1, 64),
            "unknown": (1, 48),
        }
        self.assertEqual(stats.mamba_cache_miss_by_cause, expected)
        collector, registry = self._collector()
        for cause in expected:
            for measure in ("requests", "tokens"):
                self.assertEqual(
                    registry.get_sample_value(
                        f"sglang:mamba_cache_miss_{measure}_total",
                        {"moe_ep_rank": "0", "cause": cause},
                    ),
                    0,
                )

        for ranks in (
            {"attn_tp_rank": 1},
            {"attn_cp_rank": 1},
            {"pp_rank": 1},
        ):
            reporter = self._reporter(collector, ParallelState.trivial(**ranks))
            reporter.report_prefill_stats(None, stats, can_run_cuda_graph=False)
        reporter = self._reporter(
            collector, ParallelState.trivial(tp_rank=2, dp_rank=1, attn_dp_rank=1)
        )
        reporter.report_prefill_stats(None, stats, can_run_cuda_graph=False)
        reporter.report_prefill_stats(
            None, PrefillStats.from_adder(adder, []), can_run_cuda_graph=False
        )
        for cause, values in expected.items():
            for measure, value in zip(("requests", "tokens"), values):
                self.assertEqual(
                    registry.get_sample_value(
                        f"sglang:mamba_cache_miss_{measure}_total",
                        {"moe_ep_rank": "0", "cause": cause},
                    ),
                    value,
                )


if __name__ == "__main__":
    unittest.main()
