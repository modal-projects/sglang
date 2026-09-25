"""Page reuse must follow the entire forward, including a scheduled old owner."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _Stream:
    def __init__(self):
        self.tasks = []
        self.finished = 0

    def wait_stream(self, stream):
        pass

    def finish(self, end=None):
        end = len(self.tasks) if end is None else end
        while self.finished < end:
            task = self.tasks[self.finished]
            self.finished += 1
            task()


class _Event:
    def __init__(self):
        self.stream = None
        self.end = 0

    def record(self, stream):
        self.stream = stream
        self.end = len(stream.tasks)

    def wait(self):
        if self.stream is not None:
            self.stream.finish(self.end)


class _Pool:
    page_size = 4

    def __init__(self):
        self.data = torch.zeros(8)

    def supports_zero_pages(self):
        return True

    def zero_pages(self, pages):
        for page in pages.tolist():
            self.data[page * 4 : (page + 1) * 4] = 0


def _scheduler():
    pool = _Pool()
    allocator = PagedTokenToKVPoolAllocator(4, 4, torch.float32, "cpu", pool, False)
    indices = allocator.alloc(4)
    stream = _Stream()
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.scheduler_stage_metrics = None
    scheduler.metrics_reporter = Mock()
    scheduler.forward_ct = 0
    scheduler.processed_tokens_counter = 0
    scheduler._sched_idled = False
    scheduler.scripted_scheduler_hook = None
    scheduler.profiler_manager = SimpleNamespace(
        _profile_batch_predicate=lambda batch: None
    )
    scheduler.forward_sleep_time = None
    scheduler.disaggregation_mode = None
    scheduler.is_generation = True
    scheduler.enable_overlap = True
    scheduler.enable_unified_memory = False
    scheduler.token_to_kv_pool_allocator = allocator
    scheduler.future_map = SimpleNamespace(
        resolve_seq_lens_cpu=lambda batch: None,
        publish=lambda *args: None,
    )
    scheduler._confidence_budget_prepare = None
    scheduler.forward_stream_ctx = nullcontext()
    scheduler.forward_stream = stream
    scheduler.schedule_stream = _Stream()
    scheduler.copy_stream = _Stream()
    scheduler.copy_stream_ctx = nullcontext()
    scheduler._forward_isolation = lambda *args, **kwargs: nullcontext()
    scheduler._relay_forward_payload = lambda *args: None
    scheduler._maybe_report_active_ranks = lambda: None
    scheduler.device_module = SimpleNamespace(Event=_Event)
    scheduler.launch_batch_sample_if_needed = lambda *args: None
    scheduler._apply_war_barrier = lambda: None

    def forward(batch, **kwargs):
        # The last write is queued after the model has finished reading shared inputs.
        stream.tasks.append(lambda: pool.data.index_fill_(0, indices, 13))
        result = SimpleNamespace(extra_keep_alive_refs=None, delay_sample_func=None)
        result.copy_to_cpu = lambda **kwargs: result.copy_done.record(stream)
        return result

    scheduler.model_worker = SimpleNamespace(forward_batch_generation=forward)
    batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_prebuilt=lambda: False),
        reqs=[],
        req_pool_indices=torch.tensor([0]),
        seq_lens=torch.tensor([1]),
        out_cache_loc=indices,
        extend_num_tokens=0,
        spec_algorithm=SimpleNamespace(is_none=lambda: True),
        return_logprob=False,
        return_hidden_states=False,
    )
    batch.copy = lambda: batch
    return scheduler, batch, pool, allocator, indices, stream


class TestPageReuseForwardFence(CustomTestCase):
    def test_completion_event_includes_the_last_forward_write(self):
        scheduler, batch, pool, allocator, indices, stream = _scheduler()
        with patch("sglang.srt.managers.scheduler.resolve_forward_inputs"):
            scheduler.run_batch(batch)
        allocator.free(indices)
        reused = allocator.alloc(4)
        stream.finish()
        self.assertTrue(torch.all(pool.data[reused] == 0))

    def test_processing_before_launch_keeps_pages_fenced_in_both_loops(self):
        """A batch built before processing still writes a just-freed request."""
        for decode_loop in (False, True):
            with self.subTest(decode_loop=decode_loop):
                scheduler, batch, pool, allocator, indices, stream = _scheduler()
                scheduler.gracefully_exit = False
                scheduler._engine_paused = False
                scheduler.running_batch = batch
                scheduler.last_batch = None
                scheduler.ingest_requests = Mock(
                    side_effect=[None, None, StopIteration]
                )
                scheduler.is_disable_overlap_for_batch = lambda batch, last_batch: (
                    last_batch is not None
                )
                plan = lambda **kwargs: SimpleNamespace(
                    running_batch=batch, batch_to_run=batch
                )
                scheduler.get_next_batch_to_run = plan
                scheduler.get_next_disagg_decode_batch_to_run = plan
                scheduler.process_decode_queue = lambda: None
                scheduler.chunked_req = None
                scheduler.disagg_decode_prealloc_queue = SimpleNamespace(
                    prefetch_prefill_dp_rank_queries=lambda: None
                )
                scheduler.ngram_embedding_manager = SimpleNamespace(
                    prepare_for_forward=lambda batch, **kwargs: batch
                )

                def process(result_batch, result):
                    result.copy_done.wait()
                    allocator.free(indices)

                scheduler.process_batch_result = process
                with patch("sglang.srt.managers.scheduler.resolve_forward_inputs"):
                    with self.assertRaises(StopIteration):
                        if decode_loop:
                            scheduler.event_loop_overlap_disagg_decode()
                        else:
                            scheduler.event_loop_overlap()
                reused = allocator.alloc(4)
                stream.finish()
                self.assertTrue(torch.all(pool.data[reused] == 0))


if __name__ == "__main__":
    unittest.main()
