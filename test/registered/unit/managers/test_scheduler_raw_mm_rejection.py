"""Preprocessing and validation rejection must retire raw media before PP relay."""

import gc
import pickle
import unittest
import weakref
from array import array
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.beam_search.coordinator import BeamCoordinator
from sglang.srt.disaggregation.utils import DisaggregationMode, TransferBackend
from sglang.srt.managers import scheduler as scheduler_module
from sglang.srt.managers.io_struct import (
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
)
from sglang.srt.managers.schedule_batch import (
    FINISH_ABORT,
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
    MultimodalProcessorOutput,
)
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.multimodal.transport import cuda_ipc, memory_pool
from sglang.srt.observability.req_time_stats import APIServerReqTimeStats
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


def _request(media, *, embedding=False):
    common = dict(
        rid="rejected",
        input_text=None,
        input_ids=array("q", [1]),
        mm_inputs=media,
        token_type_ids=None,
        sampling_params=SamplingParams(max_new_tokens=1, top_k=1),
        time_stats=APIServerReqTimeStats(),
    )
    if embedding:
        return TokenizedEmbeddingReqInput(**common)
    return TokenizedGenerateReqInput(
        **common,
        input_embeds=None,
        return_logprob=False,
        logprob_start_len=-1,
        top_logprobs_num=0,
        token_ids_logprob=None,
        stream=False,
    )


def _scheduler():
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.enable_session_radix_cache = False
    scheduler.model_config = SimpleNamespace(hf_eos_token_id=[], vocab_size=32)
    scheduler.metrics_reporter = SimpleNamespace(enable_metrics=False)
    scheduler.disaggregation_mode = DisaggregationMode.NULL
    scheduler.transfer_backend = TransferBackend.MOONCAKE
    scheduler.spec_algorithm = SpeculativeAlgorithm.NONE
    scheduler.tokenizer = None
    scheduler.dllm_config = None
    scheduler.enable_overlap = True
    scheduler.server_args = SimpleNamespace(sampling_mask_max_tokens=4)
    scheduler.max_new_tokens_limit = None
    scheduler.page_size = 1
    scheduler.max_req_len = 128
    scheduler.max_total_num_tokens = 256
    scheduler.max_req_input_len = 128
    scheduler.tree_cache = Mock()
    scheduler._add_request_to_queue = Mock()
    scheduler.output_streamer = SimpleNamespace(stream_output=Mock())
    scheduler.beam_coordinator = BeamCoordinator(
        model_config=scheduler.model_config,
        spec_algorithm=SpeculativeAlgorithm.NONE,
        dllm_enabled=False,
        max_req_len=128,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
        tree_cache=scheduler.tree_cache,
        future_map=None,
    )
    return scheduler


class TestSchedulerRawMultimodalRejection(CustomTestCase):
    def setUp(self):
        override = get_context().override_server_args(speculative_algorithm=None)
        override.install()
        self.addCleanup(override.restore)
        self.tickets = {}
        self.opens = []
        self.explicit_releases = []
        self.writes = []
        self.sequence = 0
        real_empty = torch.empty
        stream = Mock()

        def cpu_empty(*args, **kwargs):
            kwargs["device"] = "cpu"
            return real_empty(*args, **kwargs)

        def open_storage(handle):
            self.opens.append(handle)
            storage = torch.zeros(128, dtype=torch.uint8).untyped_storage()
            weakref.finalize(storage, self._consume_ticket, handle)
            return storage

        patches = (
            patch.object(
                scheduler_module,
                "get_parallel",
                return_value=SimpleNamespace(attn_dcp_size=1),
            ),
            patch(
                "sglang.srt.managers.schedule_batch.get_parallel",
                return_value=SimpleNamespace(tp_rank=0),
            ),
            patch.object(cuda_ipc, "_pool_acknowledged_generations", {}),
            patch.object(cuda_ipc, "_pool_imported_generations", {}),
            patch.object(cuda_ipc, "_pool_storage_cache", {}),
            patch.object(
                cuda_ipc, "_open_pooled_storage_uncached", side_effect=open_storage
            ),
            patch.object(
                cuda_ipc, "_release_ipc_export", side_effect=self._release_ticket
            ),
            patch.object(torch, "empty", side_effect=cpu_empty),
            patch.object(torch.cuda, "device", side_effect=lambda *_: nullcontext()),
            patch.object(torch.cuda, "current_device", return_value=0),
            patch.object(torch.cuda, "current_stream", return_value=stream),
            patch.object(cuda_ipc, "resolve_consumer_rank", return_value=0),
            patch.object(memory_pool, "stream_wait_value32"),
            patch.object(
                cuda_ipc,
                "stream_write_value32",
                side_effect=lambda *args: self.writes.append(args),
            ),
            patch.object(
                cuda_ipc.CudaIpcTensorTransportProxy,
                "reconstruct_on_target_device",
                side_effect=AssertionError("rejection reconstructed a feature"),
            ),
        )
        for context in patches:
            context.start()
            self.addCleanup(context.stop)

    def _consume_ticket(self, handle):
        key = tuple(handle[4:6])
        self.tickets[key] -= 1

    def _release_ticket(self, handle):
        self.explicit_releases.append(handle)
        self._consume_ticket(handle)

    def _media(self, container, *, allocation=None, generation=1):
        self.sequence += 1
        allocation = allocation or f"allocation-{self.sequence}"
        handle = (0, allocation, 128, 0, "counter", self.sequence, "event", False)
        self.tickets[tuple(handle[4:6])] = 1
        feature = torch.tensor([7], dtype=torch.float32)
        proxy = cuda_ipc.CudaIpcTensorTransportProxy(
            data=feature.view(torch.uint8),
            info_data=feature,
            pool_ipc_handle=handle,
            pool_ipc_handles=(handle,),
            pool_id=allocation,
            pool_byte_offset=64,
            ready_byte_offset=0,
            ack_byte_offset=4,
            generation=generation,
            total_consumer_count=1,
            use_pool_handle_cache=True,
        )
        return container(
            mm_items=[MultimodalDataItem(modality=Modality.IMAGE, feature=proxy)]
        )

    def _assert_released_and_relay_safe(self, recv):
        self.assertIsNone(recv.mm_inputs)
        relayed = pickle.loads(pickle.dumps([recv]))[0]
        self.assertIsNone(relayed.mm_inputs)
        calls = (len(self.opens), len(self.explicit_releases), len(self.writes))
        scheduler_module._release_unadmitted_mm_inputs(relayed)
        self.assertEqual(
            calls, (len(self.opens), len(self.explicit_releases), len(self.writes))
        )
        cuda_ipc._pool_handle_cache_clear()
        gc.collect()
        self.assertTrue(
            all(value == 0 for value in self.tickets.values()), self.tickets
        )

    def _run_rejection(self, reason, container):
        embedding = reason.startswith("embedding")
        recv = _request(self._media(container), embedding=embedding)
        scheduler = _scheduler()
        kwargs = {}
        if reason == "beam":
            recv.sampling_params.beam_width = 2
            scheduler.spec_algorithm = SpeculativeAlgorithm.EAGLE
            scheduler.beam_coordinator.spec_algorithm = scheduler.spec_algorithm
        elif reason == "bootstrap":
            scheduler.disaggregation_mode = DisaggregationMode.PREFILL
        elif reason in ("mm_error", "embedding_mm_error"):
            kwargs["mm_input_error"] = "processing failed"
        elif reason == "dflash":
            scheduler.spec_algorithm = SpeculativeAlgorithm.DFLASH
            recv.return_hidden_states = True
        elif reason == "uno":
            scheduler.spec_algorithm = SpeculativeAlgorithm.UNO
            recv.sampling_params.min_p = 0.2
        elif reason.startswith("mask"):
            recv.return_sampling_mask = True
            if reason == "mask_pd":
                scheduler.disaggregation_mode = DisaggregationMode.PREFILL
                recv.bootstrap_room = 1
                scheduler.disagg_metadata_buffers = SimpleNamespace(
                    enable_sampling_mask=False
                )
            elif reason == "mask_top_k":
                recv.sampling_params.top_k = 8
            elif reason == "mask_spec":
                scheduler.spec_algorithm = SpeculativeAlgorithm.EAGLE
        elif reason in ("conversion", "embedding_conversion"):
            scheduler._get_multimodal_inputs = Mock(
                side_effect=scheduler_module._MultimodalInputProcessingError(
                    "conversion failed"
                )
            )
        backend = get_context().override_server_args(
            sampling_backend="ascend" if reason == "mask_ascend" else "pytorch"
        )
        with backend:
            method = (
                scheduler.handle_embedding_request
                if embedding
                else scheduler.handle_generate_request
            )
            method(recv, **kwargs)
        if scheduler._add_request_to_queue.called:
            rejected = scheduler._add_request_to_queue.call_args.args[0]
        else:
            rejected = scheduler.output_streamer.stream_output.call_args.args[0][0]
        self.assertIsInstance(
            rejected.finished_reason or rejected.to_finish, FINISH_ABORT
        )
        self._assert_released_and_relay_safe(recv)

    def test_all_pre_mm_rejections_dispose_native_tickets_before_relay(self):
        """Every early return drops its incoming reservation before PP serializes it."""
        reasons = (
            "beam",
            "bootstrap",
            "mm_error",
            "dflash",
            "uno",
            "mask_pd",
            "mask_top_k",
            "mask_spec",
            "mask_ascend",
            "conversion",
            "embedding_mm_error",
            "embedding_conversion",
        )
        for container in (MultimodalProcessorOutput, MultimodalInputs):
            for reason in reasons:
                with self.subTest(container=container.__name__, reason=reason):
                    before = len(self.writes)
                    self._run_rejection(reason, container)
                    self.assertEqual(len(self.writes), before + 1)

    def test_cached_rejection_retires_the_fresh_ticket_without_reopening(self):
        """A reused mapping owns only its first ticket; a fresh unused ticket retires."""
        requests = [
            _request(
                self._media(
                    MultimodalProcessorOutput, allocation="shared", generation=i
                )
            )
            for i in (1, 2)
        ]
        for recv in requests:
            _scheduler().handle_generate_request(
                recv, mm_input_error="processing failed"
            )
            self.assertIsNone(recv.mm_inputs)
        self.assertEqual(len(self.opens), 1)
        self.assertEqual(len(self.explicit_releases), 1)
        self.assertEqual(len(self.writes), 2)
        self._assert_released_and_relay_safe(requests[0])
        self._assert_released_and_relay_safe(requests[1])

    def test_conversion_cleanup_and_handler_cleanup_do_not_release_twice(self):
        """A converter that already retired its input leaves no second native decrement."""
        for embedding in (False, True):
            with self.subTest(embedding=embedding):
                recv = _request(
                    self._media(MultimodalProcessorOutput), embedding=embedding
                )
                scheduler = _scheduler()

                def fail_conversion(raw):
                    MultimodalInputs(mm_items=raw.mm_items).release_features()
                    raise scheduler_module._MultimodalInputProcessingError(
                        "conversion failed"
                    )

                scheduler._get_multimodal_inputs = fail_conversion
                before = len(self.writes)
                method = (
                    scheduler.handle_embedding_request
                    if embedding
                    else scheduler.handle_generate_request
                )
                method(recv)
                self.assertEqual(len(self.writes), before + 1)
                self._assert_released_and_relay_safe(recv)

    def test_output_failure_still_drops_the_unadmitted_payload(self):
        """Failure while returning a rejection must not retain raw producer ownership."""
        recv = _request(self._media(MultimodalProcessorOutput))
        recv.sampling_params.beam_width = 2
        scheduler = _scheduler()
        scheduler.beam_coordinator.spec_algorithm = SpeculativeAlgorithm.EAGLE
        scheduler.output_streamer.stream_output.side_effect = RuntimeError(
            "output failed"
        )
        with self.assertRaisesRegex(RuntimeError, "output failed"):
            scheduler.handle_generate_request(recv)
        self._assert_released_and_relay_safe(recv)
        self.assertEqual(len(self.writes), 1)


if __name__ == "__main__":
    unittest.main()
